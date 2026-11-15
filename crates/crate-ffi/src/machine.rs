// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! `crate_machine`: a virt machine built from a config, run on a thread of
//! its own, with its state reported to the app.

use std::ffi::c_int;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, Once, Weak};
use std::thread::{JoinHandle, ThreadId};
use std::time::Duration;

use crate_core::board::CpuBackend;
use crate_core::run::{RunControl, StopReason};
use crate_host::disk::{FileDisk, Flush};
use crate_host::jit::JitRegion;
use crate_host::slirp::UserNet;
use crate_jit::backend::Jit;
use crate_virt::{ConsoleConfig, InputConfig, NetworkConfig, VirtConfig, VirtMachine};

use crate::config::{
    Backend, Config, DISK_FULL_FLUSH, DISK_NO_FLUSH, DISK_READ_ONLY, StateCallback,
};
use crate::display::{self, AppDisplay};
use crate::{Status, fail, fail_with, guarded, panic_message};

// crate_state
const STATE_CREATED: c_int = 0;
const STATE_RUNNING: c_int = 1;
const STATE_STOPPED: c_int = 3;
// crate_stop_reason
const STOP_NONE: c_int = 0;
const STOP_POWER_OFF: c_int = 1;
const STOP_RESET: c_int = 2;
const STOP_KILLED: c_int = 3;
const STOP_ERROR: c_int = 4;

/// One machine per process (crate.h).
static EXISTS: AtomicBool = AtomicBool::new(false);

/// The running machine's run control, for the panic hook.
static RUNNING: Mutex<Option<Weak<RunControl>>> = Mutex::new(None);

/// A panic on any of the engine's threads stops the machine, so the other
/// CPUs leave guest code and the machine's thread reports it, rather than
/// running on without one of them.
fn install_panic_hook() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let message = panic_message(info.payload());
            let place = info
                .location()
                .map(|l| format!(" at {}:{}", l.file(), l.line()))
                .unwrap_or_default();
            crate_core::error!("panic{place}: {message}");
            let run = RUNNING.try_lock().ok().and_then(|r| r.as_ref()?.upgrade());
            if let Some(run) = run {
                run.stop(StopReason::Error(format!(
                    "the engine failed{place}: {message}"
                )));
            }
            previous(info);
        }));
    });
}

struct Current {
    state: c_int,
    reason: c_int,
    /// The state callback has returned for this state.
    reported: bool,
}

/// The machine's state, shared with its thread.
struct Shared {
    status: Mutex<Current>,
    changed: Condvar,
    callback: Option<StateCallback>,
}

impl Shared {
    /// Records a new state and reports it; waiters wake once the report
    /// has returned.
    fn set(&self, state: c_int, reason: c_int) {
        {
            let mut s = self.status.lock().unwrap();
            s.state = state;
            s.reason = reason;
            s.reported = false;
        }
        if let Some(cb) = self.callback {
            // SAFETY: the app's callback, with its own context.
            unsafe { (cb.call)(cb.context.get(), state, reason) };
        }
        self.status.lock().unwrap().reported = true;
        self.changed.notify_all();
    }

    fn state(&self) -> c_int {
        self.status.lock().unwrap().state
    }
}

pub struct Machine {
    pub(crate) virt: Arc<VirtMachine>,
    pub(crate) displays: Vec<Arc<AppDisplay>>,
    backend: Mutex<Option<Box<dyn CpuBackend + Send>>>,
    shared: Arc<Shared>,
    thread: Mutex<Option<JoinHandle<()>>>,
    thread_id: Mutex<Option<ThreadId>>,
}

fn reason_code(reason: &StopReason) -> c_int {
    match reason {
        StopReason::PowerOff => STOP_POWER_OFF,
        StopReason::Reset => STOP_RESET,
        StopReason::Killed => STOP_KILLED,
        StopReason::Error(_) => STOP_ERROR,
    }
}

/// The CPU backend the config asks for, as this process can have it.
fn backend(config: &Config) -> Result<Box<dyn CpuBackend + Send>, Status> {
    let hvf = match config.backend {
        Backend::Jit => false,
        Backend::Hvf | Backend::Auto => {
            #[cfg(target_os = "macos")]
            let available = crate_hvf::Hvf::available();
            #[cfg(not(target_os = "macos"))]
            let available = false;
            if !available && config.backend == Backend::Hvf {
                return Err(fail(
                    Status::Unsupported,
                    "hvf: this process may not use Apple's Hypervisor framework (macOS only, \
                     and the binary needs the com.apple.security.hypervisor entitlement)",
                ));
            }
            if !available && cfg!(target_os = "macos") {
                crate_core::info!("hvf was refused to this process; running on jit");
            }
            available
        }
    };
    #[cfg(target_os = "macos")]
    if hvf {
        return Ok(Box::new(crate_hvf::Hvf::new()));
    }
    let _ = hvf;
    let region = match config.jit_region {
        Some(r) => {
            // SAFETY: crate.h makes the app promise the region is what it
            // says and stays valid until the machine is freed.
            let region =
                unsafe { JitRegion::from_raw(r.exec as *mut u8, r.write as *mut u8, r.size) };
            Arc::new(region)
        }
        None => {
            let size = config
                .jit_size
                .unwrap_or(config.cpus * crate_jit::cache::MAX_REGION);
            JitRegion::shared(size).map_err(fail_with)?
        }
    };
    Ok(Box::new(Jit::new().in_region(region)))
}

fn build(config: &Config) -> Result<(VirtMachine, Vec<Arc<AppDisplay>>), Status> {
    let kernel = config
        .kernel
        .clone()
        .ok_or_else(|| fail(Status::InvalidArgument, "no kernel"))?;
    let mut disks: Vec<Arc<dyn crate_core::block::BlockBackend>> = Vec::new();
    for disk in &config.disks {
        let flush = if disk.flags & DISK_FULL_FLUSH != 0 {
            Flush::Full
        } else if disk.flags & DISK_NO_FLUSH != 0 {
            Flush::Ignore
        } else {
            Flush::Data
        };
        let file = FileDisk::open_with(&disk.path, disk.flags & DISK_READ_ONLY != 0, flush)
            .map_err(fail_with)?;
        disks.push(Arc::new(file));
    }
    let mut networks = Vec::new();
    for network in &config.networks {
        networks.push(NetworkConfig {
            backend: UserNet::with_ipv6(network.forwards.clone(), network.ipv6)
                .map_err(fail_with)?,
            mac: network.mac,
        });
    }
    let consoles = config
        .consoles
        .iter()
        .enumerate()
        .map(|(index, console)| {
            let index = index as u32;
            let console = *console;
            ConsoleConfig {
                kind: console.kind,
                output: Arc::new(move |bytes: &[u8]| {
                    if let Some(output) = console.output {
                        // SAFETY: the app's callback, with its own context.
                        unsafe {
                            output(console.context.get(), index, bytes.as_ptr(), bytes.len())
                        };
                    }
                }),
                input_ready: console.input_ready.map(|ready| {
                    Arc::new(move || {
                        // SAFETY: as above.
                        unsafe { ready(console.context.get(), index) }
                    }) as Arc<dyn Fn() + Send + Sync>
                }),
            }
        })
        .collect();
    let (sinks, displays) = display::build(config);
    let virt = VirtMachine::new(VirtConfig {
        cpus: config.cpus,
        memory: config.memory,
        kernel,
        raw_kernel: false,
        initrd: config.initrd.clone(),
        cmdline: config.cmdline.clone(),
        disks,
        networks,
        consoles,
        displays,
        input: InputConfig {
            keyboard: config.input & display::INPUT_KEYBOARD != 0,
            tablet: config.input & display::INPUT_TABLET != 0,
            mouse: config.input & display::INPUT_MOUSE != 0,
        },
    })
    .map_err(fail_with)?;
    display::serve(&sinks, &virt)?;
    Ok((virt, sinks))
}

/// `crate_machine_create`.
///
/// # Safety
///
/// `config` is NULL or a live config; `machine` is NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_machine_create(
    config: *const Config,
    machine: *mut *mut Machine,
) -> Status {
    guarded(|| {
        // SAFETY: the caller's contract.
        let (Some(config), false) = (unsafe { config.as_ref() }, machine.is_null()) else {
            return fail(
                Status::InvalidArgument,
                "the config or the machine pointer is NULL",
            );
        };
        if EXISTS.swap(true, Ordering::AcqRel) {
            return fail(Status::State, "a machine already exists in this process");
        }
        install_panic_hook();
        let made = backend(config).and_then(|backend| Ok((backend, build(config)?)));
        let (backend, (virt, displays)) = match made {
            Ok(made) => made,
            Err(status) => {
                EXISTS.store(false, Ordering::Release);
                return status;
            }
        };
        crate_core::info!(
            "{} CPUs, {} MiB, on {}",
            config.cpus,
            config.memory >> 20,
            backend.name()
        );
        let built = Box::new(Machine {
            virt: Arc::new(virt),
            displays,
            backend: Mutex::new(Some(backend)),
            shared: Arc::new(Shared {
                status: Mutex::new(Current {
                    state: STATE_CREATED,
                    reason: STOP_NONE,
                    reported: true,
                }),
                changed: Condvar::new(),
                callback: config.state,
            }),
            thread: Mutex::new(None),
            thread_id: Mutex::new(None),
        });
        // SAFETY: checked non-NULL above.
        unsafe { *machine = Box::into_raw(built) };
        Status::Ok
    })
}

/// # Safety
///
/// `machine` is NULL or a live machine.
unsafe fn machine_ref<'a>(machine: *const Machine) -> Result<&'a Machine, Status> {
    // SAFETY: the caller's contract.
    unsafe { machine.as_ref() }.ok_or_else(|| fail(Status::InvalidArgument, "the machine is NULL"))
}

/// `crate_machine_start`.
///
/// # Safety
///
/// `machine` is NULL or a live machine.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_machine_start(machine: *mut Machine) -> Status {
    guarded(|| {
        // SAFETY: the caller's contract.
        let m = match unsafe { machine_ref(machine) } {
            Ok(m) => m,
            Err(status) => return status,
        };
        let Some(backend) = m.backend.lock().unwrap().take() else {
            return fail(Status::State, "the machine has already started");
        };
        *RUNNING.lock().unwrap() = Some(Arc::downgrade(&m.virt.board().run));
        let virt = m.virt.clone();
        let shared = m.shared.clone();
        let spawned = std::thread::Builder::new()
            .name("crate-machine".into())
            .spawn(move || {
                shared.set(STATE_RUNNING, STOP_NONE);
                let result = catch_unwind(AssertUnwindSafe(|| virt.run(backend.as_ref())));
                let reason = match result {
                    Ok(Ok(reason)) => reason,
                    Ok(Err(e)) => StopReason::Error(e.to_string()),
                    Err(payload) => StopReason::Error(format!(
                        "the engine failed: {}",
                        panic_message(&*payload)
                    )),
                };
                // Whatever ended the run, the devices hear of it.
                virt.board().run.stop(reason.clone());
                let reason = virt.board().run.reason().unwrap_or(reason);
                if let StopReason::Error(message) = &reason {
                    crate_core::error!("the machine stopped: {message}");
                    // crate.h: the callback's thread can ask why.
                    fail(Status::State, message.clone());
                }
                drop(backend);
                shared.set(STATE_STOPPED, reason_code(&reason));
            });
        match spawned {
            Ok(thread) => {
                *m.thread_id.lock().unwrap() = Some(thread.thread().id());
                *m.thread.lock().unwrap() = Some(thread);
                Status::Ok
            }
            Err(e) => fail(
                Status::NoMemory,
                format!("starting the machine's thread: {e}"),
            ),
        }
    })
}

/// `crate_machine_pause`: not yet.
///
/// # Safety
///
/// `machine` is NULL or a live machine.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_machine_pause(machine: *mut Machine) -> Status {
    // SAFETY: the caller's contract.
    match unsafe { machine_ref(machine) } {
        Ok(_) => fail(Status::Unsupported, "machines cannot pause yet"),
        Err(status) => status,
    }
}

/// `crate_machine_resume`: not yet.
///
/// # Safety
///
/// `machine` is NULL or a live machine.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_machine_resume(machine: *mut Machine) -> Status {
    // SAFETY: the caller's contract.
    match unsafe { machine_ref(machine) } {
        Ok(_) => fail(Status::Unsupported, "machines cannot pause yet"),
        Err(status) => status,
    }
}

/// `crate_machine_request_shutdown`: virt has no power button yet.
///
/// # Safety
///
/// `machine` is NULL or a live machine.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_machine_request_shutdown(machine: *mut Machine) -> Status {
    // SAFETY: the caller's contract.
    match unsafe { machine_ref(machine) } {
        Ok(_) => fail(
            Status::Unsupported,
            "virt has no power button yet; ask the guest to power off",
        ),
        Err(status) => status,
    }
}

/// `crate_machine_kill`.
///
/// # Safety
///
/// `machine` is NULL or a live machine.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_machine_kill(machine: *mut Machine) -> Status {
    guarded(|| {
        // SAFETY: the caller's contract.
        let m = match unsafe { machine_ref(machine) } {
            Ok(m) => m,
            Err(status) => return status,
        };
        m.virt.kill();
        // Never started: it stops here, and can no longer start.
        if m.backend.lock().unwrap().take().is_some() {
            m.shared.set(STATE_STOPPED, STOP_KILLED);
        }
        Status::Ok
    })
}

/// `crate_machine_wait`.
///
/// # Safety
///
/// `machine` is NULL or a live machine.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_machine_wait(machine: *mut Machine, timeout_ms: i64) -> Status {
    guarded(|| {
        // SAFETY: the caller's contract.
        let m = match unsafe { machine_ref(machine) } {
            Ok(m) => m,
            Err(status) => return status,
        };
        // Not the state, which the machine's thread sets once it runs.
        if m.backend.lock().unwrap().is_some() {
            return fail(Status::State, "the machine has not started");
        }
        let status = m.shared.status.lock().unwrap();
        let running = |s: &mut Current| !(s.state == STATE_STOPPED && s.reported);
        if timeout_ms < 0 {
            drop(m.shared.changed.wait_while(status, running).unwrap());
            return Status::Ok;
        }
        let timeout = Duration::from_millis(timeout_ms as u64);
        let (_status, result) = m
            .shared
            .changed
            .wait_timeout_while(status, timeout, running)
            .unwrap();
        if result.timed_out() {
            fail(Status::Timeout, "the machine is still running")
        } else {
            Status::Ok
        }
    })
}

/// `crate_machine_state`.
///
/// # Safety
///
/// `machine` is NULL or a live machine.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_machine_state(machine: *const Machine) -> c_int {
    // SAFETY: the caller's contract.
    match unsafe { machine.as_ref() } {
        Some(m) => m.shared.state(),
        None => STATE_STOPPED,
    }
}

/// `crate_machine_free`.
///
/// # Safety
///
/// `machine` is NULL or a live machine, used by nothing else from now on.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_machine_free(machine: *mut Machine) {
    if machine.is_null() {
        return;
    }
    // SAFETY: the caller's contract; made by Box::into_raw.
    let m = unsafe { Box::from_raw(machine) };
    m.virt.kill();
    let thread = m.thread.lock().unwrap().take();
    let on_its_thread = *m.thread_id.lock().unwrap() == Some(std::thread::current().id());
    match thread {
        // From one of the machine's own callbacks it cannot wait for itself;
        // the thread finishes on its own.
        Some(_) if on_its_thread => {}
        Some(thread) => {
            let _ = thread.join();
        }
        None => {}
    }
    *RUNNING.lock().unwrap() = None;
    drop(m);
    EXISTS.store(false, Ordering::Release);
}

/// `crate_console_write`.
///
/// # Safety
///
/// `machine` is NULL or a live machine; `bytes` is NULL or `length` readable
/// bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_console_write(
    machine: *mut Machine,
    console: u32,
    bytes: *const u8,
    length: usize,
) -> usize {
    // SAFETY: the caller's contract.
    let Some(m) = (unsafe { machine.as_ref() }) else {
        return 0;
    };
    if bytes.is_null() || length == 0 || m.virt.board().run.is_stopped() {
        return 0;
    }
    // SAFETY: the caller's contract.
    let bytes = unsafe { std::slice::from_raw_parts(bytes, length) };
    catch_unwind(AssertUnwindSafe(|| {
        m.virt.console_write(console as usize, bytes)
    }))
    .unwrap_or(0)
}

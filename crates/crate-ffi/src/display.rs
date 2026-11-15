// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! crate.h's displays and input: virtio-gpu screens whose frames reach the
//! app's callback (and an optional VNC server), and the keyboard and tablet.

use std::ffi::{c_int, c_void};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate_core::display::{DisplayControl, DisplaySink, Frame, Mode};
use crate_devices::virtio::gpu::edid::MAX_MODES;
use crate_virt::{DEFAULT_MODES, DisplayConfig, VirtMachine};
use crate_vnc::VncServer;

use crate::config::{Config, c_str, with};
use crate::machine::Machine;
use crate::{Context, Status, fail, fail_with, guarded};

pub(crate) const INPUT_KEYBOARD: u32 = 1 << 0;
pub(crate) const INPUT_TABLET: u32 = 1 << 1;
pub(crate) const INPUT_MOUSE: u32 = 1 << 2;

/// `crate_frame`.
#[repr(C)]
pub struct CFrame {
    width: u32,
    height: u32,
    stride: u32,
    pixels: *const u8,
    dirty_x: u32,
    dirty_y: u32,
    dirty_width: u32,
    dirty_height: u32,
}

pub type FrameFn = unsafe extern "C" fn(*mut c_void, u32, *const CFrame);

/// `crate_display_callbacks`.
#[repr(C)]
pub struct DisplayCallbacks {
    size: usize,
    context: *mut c_void,
    frame: Option<FrameFn>,
}

/// `crate_display_mode`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DisplayMode {
    width: u32,
    height: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Gpu {
    TwoD,
    Virgl,
}

pub(crate) enum VncAddress {
    Unix(PathBuf),
    Tcp(SocketAddr),
}

/// A display as the config describes it.
pub(crate) struct Display {
    gpu: Gpu,
    context: Context,
    frame: Option<FrameFn>,
    modes: Vec<Mode>,
    vnc: Vec<VncAddress>,
}

/// Where a display's frames go: the app's callback while it says the
/// display is active, and the VNC server while it has clients.
pub(crate) struct AppDisplay {
    index: u32,
    context: Context,
    frame: Option<FrameFn>,
    active: AtomicBool,
    vnc: Option<Arc<VncServer>>,
    addresses: Vec<VncAddress>,
}

impl DisplaySink for AppDisplay {
    fn active(&self) -> bool {
        (self.frame.is_some() && self.active.load(Ordering::Acquire))
            || self.vnc.as_ref().is_some_and(|v| v.active())
    }

    fn frame(&self, frame: &Frame) {
        if let Some(vnc) = &self.vnc {
            if vnc.active() {
                vnc.frame(frame);
            }
        }
        let Some(callback) = self.frame else {
            return;
        };
        if !self.active.load(Ordering::Acquire) {
            return;
        }
        let c = CFrame {
            width: frame.width,
            height: frame.height,
            stride: frame.stride,
            pixels: frame.pixels.as_ptr(),
            dirty_x: frame.dirty.x,
            dirty_y: frame.dirty.y,
            dirty_width: frame.dirty.width,
            dirty_height: frame.dirty.height,
        };
        // SAFETY: the app's callback with its own context; the frame and its
        // pixels live for the call.
        unsafe { callback(self.context.get(), self.index, &c) };
    }
}

/// The displays' sinks and the virt machine's descriptions of them.
pub(crate) fn build(config: &Config) -> (Vec<Arc<AppDisplay>>, Vec<DisplayConfig>) {
    let mut sinks = Vec::new();
    let mut displays = Vec::new();
    for (index, d) in config.displays.iter().enumerate() {
        let preferred = d.modes[0];
        let vnc =
            (!d.vnc.is_empty()).then(|| VncServer::new("crate", preferred.width, preferred.height));
        let sink = Arc::new(AppDisplay {
            index: index as u32,
            context: d.context,
            frame: d.frame,
            active: AtomicBool::new(false),
            vnc,
            addresses: d
                .vnc
                .iter()
                .map(|a| match a {
                    VncAddress::Unix(p) => VncAddress::Unix(p.clone()),
                    VncAddress::Tcp(a) => VncAddress::Tcp(*a),
                })
                .collect(),
        });
        displays.push(DisplayConfig {
            modes: d.modes.clone(),
            sink: sink.clone(),
            renderer: (d.gpu == Gpu::Virgl).then(crate_host::virgl::factory),
        });
        sinks.push(sink);
    }
    (sinks, displays)
}

/// Starts the displays' VNC servers, now that their machine exists.
pub(crate) fn serve(sinks: &[Arc<AppDisplay>], virt: &VirtMachine) -> Result<(), Status> {
    for sink in sinks {
        let (Some(vnc), Some(control)) = (&sink.vnc, virt.display(sink.index as usize)) else {
            continue;
        };
        vnc.attach(Arc::new(control.clone()), virt.input());
        for address in &sink.addresses {
            let listening = match address {
                VncAddress::Unix(path) => vnc.listen_unix(path),
                VncAddress::Tcp(addr) => vnc.listen_tcp(*addr),
            };
            listening.map_err(fail_with)?;
        }
    }
    Ok(())
}

/// # Safety
///
/// `modes` is NULL or `count` readable modes.
unsafe fn read_modes(modes: *const DisplayMode, count: usize) -> Result<Vec<Mode>, Status> {
    if modes.is_null() || count == 0 {
        return Err(fail(Status::InvalidArgument, "no modes"));
    }
    if count > MAX_MODES {
        return Err(fail(
            Status::InvalidArgument,
            format!("{count} modes; a display takes up to {MAX_MODES}"),
        ));
    }
    // SAFETY: the caller's contract.
    let modes = unsafe { std::slice::from_raw_parts(modes, count) };
    modes
        .iter()
        .map(|m| {
            if (16..=4095).contains(&m.width) && (16..=4095).contains(&m.height) {
                Ok(Mode::new(m.width, m.height))
            } else {
                Err(fail(
                    Status::InvalidArgument,
                    format!(
                        "{}x{}: modes are 16 to 4095 pixels a side",
                        m.width, m.height
                    ),
                ))
            }
        })
        .collect()
}

/// `crate_config_add_display`.
///
/// # Safety
///
/// `config` is NULL or a live config; `callbacks` is NULL or a struct whose
/// `size` says how much of it there is; `index` is NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_config_add_display(
    config: *mut Config,
    gpu: c_int,
    callbacks: *const DisplayCallbacks,
    index: *mut u32,
) -> Status {
    // SAFETY: the caller's contract.
    unsafe {
        with(config, |c| {
            let gpu = match gpu {
                0 => Gpu::TwoD,
                1 => Gpu::Virgl,
                _ => return fail(Status::InvalidArgument, format!("no GPU kind {gpu}")),
            };
            let mut display = Display {
                gpu,
                context: Context(std::ptr::null_mut()),
                frame: None,
                modes: DEFAULT_MODES.to_vec(),
                vnc: Vec::new(),
            };
            if !callbacks.is_null() {
                // Only the fields the app's struct has: `size` says which.
                let size = (*callbacks).size;
                use std::mem::{offset_of, size_of};
                if size >= offset_of!(DisplayCallbacks, context) + size_of::<usize>() {
                    display.context = Context((*callbacks).context);
                }
                if size >= offset_of!(DisplayCallbacks, frame) + size_of::<usize>() {
                    display.frame = (*callbacks).frame;
                }
            }
            c.displays.push(display);
            if let Some(index) = index.as_mut() {
                *index = (c.displays.len() - 1) as u32;
            }
            Status::Ok
        })
    }
}

/// `crate_config_set_display_modes`.
///
/// # Safety
///
/// As for `crate_config_add_display`; `modes` is NULL or `count` readable
/// modes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_config_set_display_modes(
    config: *mut Config,
    display: u32,
    modes: *const DisplayMode,
    count: usize,
) -> Status {
    // SAFETY: the caller's contract.
    unsafe {
        with(config, |c| {
            let modes = match read_modes(modes, count) {
                Ok(modes) => modes,
                Err(status) => return status,
            };
            match c.displays.get_mut(display as usize) {
                Some(d) => {
                    d.modes = modes;
                    Status::Ok
                }
                None => fail(Status::InvalidArgument, format!("no display {display}")),
            }
        })
    }
}

/// `crate_config_add_vnc`.
///
/// # Safety
///
/// As for `crate_config_add_display`; `address` is NULL or a C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_config_add_vnc(
    config: *mut Config,
    display: u32,
    address: *const std::ffi::c_char,
) -> Status {
    // SAFETY: the caller's contract.
    unsafe {
        with(config, |c| {
            let text = match c_str(address, "VNC address") {
                Ok(s) => s.to_string_lossy().into_owned(),
                Err(status) => return status,
            };
            let parsed = if let Some(path) = text.strip_prefix("unix:") {
                if path.is_empty() {
                    return fail(Status::InvalidArgument, "an empty socket path");
                }
                VncAddress::Unix(path.into())
            } else if let Some(addr) = text.strip_prefix("tcp:") {
                match addr.parse() {
                    Ok(a) => VncAddress::Tcp(a),
                    Err(_) => {
                        return fail(
                            Status::InvalidArgument,
                            format!("{addr}: expected HOST:PORT"),
                        );
                    }
                }
            } else {
                return fail(
                    Status::InvalidArgument,
                    format!("{text}: expected unix:PATH or tcp:HOST:PORT"),
                );
            };
            match c.displays.get_mut(display as usize) {
                Some(d) => {
                    d.vnc.push(parsed);
                    Status::Ok
                }
                None => fail(Status::InvalidArgument, format!("no display {display}")),
            }
        })
    }
}

/// `crate_config_add_input`.
///
/// # Safety
///
/// As for `crate_config_add_display`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_config_add_input(config: *mut Config, devices: u32) -> Status {
    // SAFETY: the caller's contract.
    unsafe {
        with(config, |c| {
            if devices & !(INPUT_KEYBOARD | INPUT_TABLET | INPUT_MOUSE) != 0 {
                return fail(
                    Status::InvalidArgument,
                    format!("no input devices {devices:#x}"),
                );
            }
            c.input |= devices;
            Status::Ok
        })
    }
}

/// `crate_display_set_active`.
///
/// # Safety
///
/// `machine` is NULL or a live machine.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_display_set_active(
    machine: *mut Machine,
    display: u32,
    active: bool,
) {
    // SAFETY: the caller's contract.
    let Some(m) = (unsafe { machine.as_ref() }) else {
        return;
    };
    guarded(|| {
        let Some(sink) = m.displays.get(display as usize) else {
            return fail(Status::InvalidArgument, format!("no display {display}"));
        };
        if sink.active.swap(active, Ordering::AcqRel) != active && active {
            // The app has nothing to show yet: a whole frame, now.
            if let Some(control) = m.virt.display(display as usize) {
                control.refresh();
            }
        }
        Status::Ok
    });
}

/// `crate_display_set_modes`.
///
/// # Safety
///
/// `machine` is NULL or a live machine; `modes` is NULL or `count` readable
/// modes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_display_set_modes(
    machine: *mut Machine,
    display: u32,
    modes: *const DisplayMode,
    count: usize,
) -> Status {
    // SAFETY: the caller's contract.
    let Some(m) = (unsafe { machine.as_ref() }) else {
        return fail(Status::InvalidArgument, "the machine is NULL");
    };
    guarded(|| {
        // SAFETY: the caller's contract.
        let modes = match unsafe { read_modes(modes, count) } {
            Ok(modes) => modes,
            Err(status) => return status,
        };
        match m.virt.display(display as usize) {
            Some(control) => {
                control.set_modes(modes);
                Status::Ok
            }
            None => fail(Status::InvalidArgument, format!("no display {display}")),
        }
    })
}

/// `crate_input_key`.
///
/// # Safety
///
/// `machine` is NULL or a live machine.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_input_key(machine: *mut Machine, code: u16, down: bool) {
    // SAFETY: the caller's contract.
    if let Some(m) = unsafe { machine.as_ref() } {
        guarded(|| {
            m.virt.input().key(code, down);
            Status::Ok
        });
    }
}

/// `crate_input_pointer`.
///
/// # Safety
///
/// `machine` is NULL or a live machine.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_input_pointer(machine: *mut Machine, x: u32, y: u32, buttons: u32) {
    // SAFETY: the caller's contract.
    if let Some(m) = unsafe { machine.as_ref() } {
        guarded(|| {
            m.virt.input().pointer(x, y, buttons);
            Status::Ok
        });
    }
}

/// `crate_input_mouse`.
///
/// # Safety
///
/// `machine` is NULL or a live machine.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_input_mouse(machine: *mut Machine, dx: i32, dy: i32, buttons: u32) {
    // SAFETY: the caller's contract.
    if let Some(m) = unsafe { machine.as_ref() } {
        guarded(|| {
            m.virt.input().mouse(dx, dy, buttons);
            Status::Ok
        });
    }
}

/// `crate_input_scroll`.
///
/// # Safety
///
/// `machine` is NULL or a live machine.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn crate_input_scroll(machine: *mut Machine, dx: i32, dy: i32) {
    // SAFETY: the caller's contract.
    if let Some(m) = unsafe { machine.as_ref() } {
        guarded(|| {
            m.virt.input().scroll(dx, dy);
            Status::Ok
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{crate_config_free, crate_config_new};

    #[test]
    fn describes_displays() {
        // SAFETY: a config made here and freed at the end; literal strings.
        unsafe {
            let c = crate_config_new(c"virt".as_ptr());
            let mut index = 9;
            assert_eq!(
                crate_config_add_display(c, 2, std::ptr::null(), &mut index),
                Status::InvalidArgument
            );
            assert_eq!(
                crate_config_add_display(c, 1, std::ptr::null(), &mut index),
                Status::Ok
            );
            assert_eq!(index, 0);
            let modes = [
                DisplayMode {
                    width: 800,
                    height: 600,
                },
                DisplayMode {
                    width: 1280,
                    height: 960,
                },
            ];
            assert_eq!(
                crate_config_set_display_modes(c, 0, modes.as_ptr(), 2),
                Status::Ok
            );
            assert_eq!(
                crate_config_set_display_modes(c, 1, modes.as_ptr(), 2),
                Status::InvalidArgument,
                "no display 1"
            );
            assert_eq!(
                crate_config_set_display_modes(c, 0, modes.as_ptr(), 0),
                Status::InvalidArgument
            );
            assert_eq!(
                crate_config_add_vnc(c, 0, c"unix:vnc.sock".as_ptr()),
                Status::Ok
            );
            assert_eq!(
                crate_config_add_vnc(c, 0, c"tcp:127.0.0.1:5905".as_ptr()),
                Status::Ok
            );
            assert_eq!(
                crate_config_add_vnc(c, 0, c"127.0.0.1:5905".as_ptr()),
                Status::InvalidArgument
            );
            assert_eq!(crate_config_add_input(c, 7), Status::Ok);
            assert_eq!(crate_config_add_input(c, 8), Status::InvalidArgument);
            let config = &*c;
            assert_eq!(config.displays[0].gpu, Gpu::Virgl);
            assert_eq!(config.displays[0].modes[1], Mode::new(1280, 960));
            assert_eq!(config.displays[0].vnc.len(), 2);
            assert_eq!(config.input, 7);
            crate_config_free(c);
        }
    }
}

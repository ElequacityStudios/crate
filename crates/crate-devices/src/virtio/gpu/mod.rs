// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! virtio-gpu (VIRTIO 1.2, section 5.7): one scanout, 2D resources the device
//! keeps itself, a cursor, an EDID with the display's modes, and, with a host
//! renderer (`crate_core::gpu::Renderer`, virglrenderer in practice), the 3D
//! commands: capability sets, contexts, 3D resources, transfers, command
//! submission and fences. No blob resources yet.
//!
//! Every command runs on the device's own thread, the GPU thread, which owns
//! the renderer (a GL context is current on one thread). A vCPU's notify only
//! wakes it. Commands with a fence answer once the renderer has retired the
//! fence, in order, since the guest takes an answered fence as every earlier
//! one done.
//!
//! What the scanout shows reaches a `DisplaySink` as frames, R, G, B, A, the
//! cursor drawn in. A flush makes a frame only while the sink is active;
//! otherwise the device remembers the screen is stale and makes a whole frame
//! when asked (`GpuControl::refresh`).
//!
//! New modes while the guest runs (`GpuControl::set_modes`) arrive as a
//! display unplugged and plugged back in: the scanout first reads as disabled,
//! with no EDID, and comes back with the new modes once the guest has turned
//! it off, or after `REPLUG_AFTER`. A compositor reads a connector's modes
//! when it is connected, not again while it stays connected, so only a new
//! connection makes it take the new preferred mode.

use std::collections::{HashMap, VecDeque};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate_core::display::{DisplayControl, DisplaySink, Frame, Mode, Rect};
use crate_core::gpu::{Box3d, Capset, HostSlice, Renderer, RendererFactory, Resource3d, Transfer};
use crate_core::memory::{GuestMemory, HostVisibleMemory};
use crate_core::{debug, info, warn};

use super::queue::Chain;
use super::{
    Activation, F_VERSION_1, Interrupt, Queue, SharedMemoryRegion, VirtioDevice,
    read_config_bytes,
};

pub mod edid;
pub mod format;

pub const ID_GPU: u32 = 16;

const F_VIRGL: u64 = 1 << 0;
const F_EDID: u64 = 1 << 1;

const EVENT_DISPLAY: u32 = 1 << 0;

const QUEUE_SIZE: u16 = 256;
const SCANOUTS: u32 = 1;
const MAX_SCANOUTS: usize = 16;

const CMD_GET_DISPLAY_INFO: u32 = 0x0100;
const CMD_RESOURCE_CREATE_2D: u32 = 0x0101;
const CMD_RESOURCE_UNREF: u32 = 0x0102;
const CMD_SET_SCANOUT: u32 = 0x0103;
const CMD_RESOURCE_FLUSH: u32 = 0x0104;
const CMD_TRANSFER_TO_HOST_2D: u32 = 0x0105;
const CMD_RESOURCE_ATTACH_BACKING: u32 = 0x0106;
const CMD_RESOURCE_DETACH_BACKING: u32 = 0x0107;
const CMD_GET_CAPSET_INFO: u32 = 0x0108;
const CMD_GET_CAPSET: u32 = 0x0109;
const CMD_GET_EDID: u32 = 0x010a;
const CMD_CTX_CREATE: u32 = 0x0200;
const CMD_CTX_DESTROY: u32 = 0x0201;
const CMD_CTX_ATTACH_RESOURCE: u32 = 0x0202;
const CMD_CTX_DETACH_RESOURCE: u32 = 0x0203;
const CMD_RESOURCE_CREATE_3D: u32 = 0x0204;
const CMD_TRANSFER_TO_HOST_3D: u32 = 0x0205;
const CMD_TRANSFER_FROM_HOST_3D: u32 = 0x0206;
const CMD_SUBMIT_3D: u32 = 0x0207;
const CMD_UPDATE_CURSOR: u32 = 0x0300;
const CMD_MOVE_CURSOR: u32 = 0x0301;

const RESP_OK_NODATA: u32 = 0x1100;
const RESP_OK_DISPLAY_INFO: u32 = 0x1101;
const RESP_OK_CAPSET_INFO: u32 = 0x1102;
const RESP_OK_CAPSET: u32 = 0x1103;
const RESP_OK_EDID: u32 = 0x1104;
const RESP_ERR_UNSPEC: u32 = 0x1200;
const RESP_ERR_OUT_OF_MEMORY: u32 = 0x1201;
const RESP_ERR_INVALID_SCANOUT_ID: u32 = 0x1202;
const RESP_ERR_INVALID_RESOURCE_ID: u32 = 0x1203;
const RESP_ERR_INVALID_CONTEXT_ID: u32 = 0x1204;
const RESP_ERR_INVALID_PARAMETER: u32 = 0x1205;

const FLAG_FENCE: u32 = 1 << 0;
const FLAG_INFO_RING_IDX: u32 = 1 << 1;

const HEADER: usize = 24;
/// The largest request read: a command buffer of 16 MiB.
const MAX_REQUEST: usize = 16 << 20;
/// 2D resources live in host memory; this caps what the guest can make the
/// host hold.
const MAX_2D_BYTES: usize = 1 << 30;
const MAX_DIMENSION: u32 = 16384;
const MAX_BACKING_ENTRIES: u32 = 1 << 20;
/// How long the GPU thread waits between looks at the renderer's fences while
/// some are outstanding.
const FENCE_POLL: Duration = Duration::from_micros(500);
/// How long the display stays unplugged at most when the guest does not turn
/// it off, so it is never left out.
const REPLUG_AFTER: Duration = Duration::from_secs(1);

pub struct GpuConfig {
    /// Offered to the guest in this order; the first is the preferred size.
    pub modes: Vec<Mode>,
    pub sink: Arc<dyn DisplaySink>,
    /// The 3D renderer, made on the GPU thread. Without one, or when it fails,
    /// the device is 2D only.
    pub renderer: Option<RendererFactory>,
    /// Host-visible memory shared with a Venus renderer, if 3D is enabled.
    pub host_visible_memory: Option<Arc<dyn HostVisibleMemory>>,
}

/// What the device and its thread share.
struct Work {
    /// Handed from DRIVER_OK to the thread.
    activation: Option<Activation>,
    /// Queues notified since the thread last looked: bit 0 control, bit 1
    /// cursor.
    notified: u32,
    resets_asked: u64,
    resets_done: u64,
    refresh: bool,
    exit: bool,
    /// The thread has ended; nobody will finish a reset.
    gone: bool,
    events: u32,
    /// A display event to raise once the guest clears the one it has, which
    /// would clear a second one raised now along with it.
    event_again: bool,
    interrupt: Option<Arc<Interrupt>>,
    /// What the guest is offered; while the display is out, what it comes
    /// back with.
    modes: Vec<Mode>,
    plug: Plug,
}

/// Whether the display reads as connected.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Plug {
    In,
    /// Unplugged for new modes until the guest turns the scanout off, once it
    /// has read the display as disabled (`seen`), or `by` passes.
    Out {
        by: Instant,
        seen: bool,
    },
}

impl Work {
    fn raise_display_event(&mut self) {
        if self.events & EVENT_DISPLAY != 0 {
            self.event_again = true;
            return;
        }
        self.events |= EVENT_DISPLAY;
        if let Some(interrupt) = &self.interrupt {
            interrupt.signal_config();
        }
    }

    fn plug_in(&mut self) {
        self.plug = Plug::In;
        self.raise_display_event();
    }

    /// When the display must be back, while it is out.
    fn plug_by(&self) -> Option<Instant> {
        match self.plug {
            Plug::In => None,
            Plug::Out { by, .. } => Some(by),
        }
    }
}

struct Shared {
    work: Mutex<Work>,
    wake: Condvar,
    reset_done: Condvar,
}

impl Shared {
    fn poke(&self, f: impl FnOnce(&mut Work)) {
        f(&mut self.work.lock().unwrap());
        self.wake.notify_one();
    }
}

/// A viewer's handle on the display.
#[derive(Clone)]
pub struct GpuControl(Arc<Shared>);

impl GpuControl {
    /// Offers the guest new modes, the first preferred, by unplugging the
    /// display and plugging it back in with them (see the module comment).
    /// The same modes again change nothing; new ones while the display is out
    /// are what it comes back with.
    pub fn set_modes(&self, modes: Vec<Mode>) {
        if modes.is_empty() {
            return;
        }
        let mut w = self.0.work.lock().unwrap();
        if w.modes == modes {
            return;
        }
        w.modes = modes;
        // Without a driver there is nobody to tell; it reads them when it
        // starts.
        if w.interrupt.is_none() || w.plug != Plug::In {
            return;
        }
        w.plug = Plug::Out {
            by: Instant::now() + REPLUG_AFTER,
            seen: false,
        };
        w.raise_display_event();
        // The GPU thread keeps the time.
        self.0.wake.notify_one();
    }
}

impl DisplayControl for GpuControl {
    fn refresh(&self) {
        self.0.poke(|w| w.refresh = true);
    }
}

pub struct Gpu {
    features: u64,
    capsets: u32,
    shared: Arc<Shared>,
    host_visible_region: Option<SharedMemoryRegion>,
    _host_visible_memory: Option<Arc<dyn HostVisibleMemory>>,
    thread: Option<JoinHandle<()>>,
}

impl Gpu {
    pub fn new(config: GpuConfig) -> crate_core::Result<Self> {
        if config.modes.is_empty() {
            return Err(crate_core::Error::InvalidArgument(
                "a display needs at least one mode".into(),
            ));
        }
        let shared = Arc::new(Shared {
            work: Mutex::new(Work {
                activation: None,
                notified: 0,
                resets_asked: 0,
                resets_done: 0,
                refresh: false,
                exit: false,
                gone: false,
                events: 0,
                event_again: false,
                interrupt: None,
                modes: config.modes,
                plug: Plug::In,
            }),
            wake: Condvar::new(),
            reset_done: Condvar::new(),
        });
        let (ready, started) = mpsc::channel();
        let sink = config.sink;
        let factory = config.renderer;
        let host_visible_region = config.host_visible_memory.as_ref().map(|memory| {
            SharedMemoryRegion {
                addr: memory.guest_addr(),
                len: memory.len() as u64,
            }
        });
        let host_visible_memory = config.host_visible_memory;
        let thread_shared = shared.clone();
        let thread = std::thread::Builder::new()
            .name("crate-gpu".into())
            .spawn(move || {
                let renderer = factory.and_then(|make| match make() {
                    Ok(renderer) => {
                        info!("virtio-gpu: 3D on {}", renderer.description());
                        Some(renderer)
                    }
                    Err(e) => {
                        warn!("virtio-gpu: no 3D, 2D only: {e}");
                        None
                    }
                });
                let capsets = renderer.as_ref().map(|r| r.capsets()).unwrap_or_default();
                let _ = ready.send(capsets.len());
                let worker = Worker {
                    shared: thread_shared.clone(),
                    sink,
                    renderer,
                    capsets,
                    memory: None,
                    control: None,
                    cursor: None,
                    interrupt: None,
                    resources: HashMap::new(),
                    bytes_2d: 0,
                    scanouts: vec![Scanout::default(); SCANOUTS as usize],
                    fenced: VecDeque::new(),
                    trace: std::env::var_os("CRATE_GPU_TRACE").is_some(),
                };
                worker.run();
                let mut w = thread_shared.work.lock().unwrap();
                w.gone = true;
                thread_shared.reset_done.notify_all();
            })
            .map_err(|e| crate_core::Error::NoMemory(format!("the GPU thread: {e}")))?;
        let capsets = started.recv().unwrap_or(0) as u32;
        let mut features = F_VERSION_1 | F_EDID;
        if capsets > 0 {
            features |= F_VIRGL;
        }
        Ok(Gpu {
            features,
            capsets,
            shared,
            host_visible_region,
            _host_visible_memory: host_visible_memory,
            thread: Some(thread),
        })
    }

    pub fn control(&self) -> GpuControl {
        GpuControl(self.shared.clone())
    }

    /// Whether the guest is offered 3D.
    pub fn has_3d(&self) -> bool {
        self.features & F_VIRGL != 0
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        self.shared.poke(|w| w.exit = true);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl VirtioDevice for Gpu {
    fn device_id(&self) -> u32 {
        ID_GPU
    }

    fn features(&self) -> u64 {
        self.features
    }

    fn shared_memory_region(&self, id: u32) -> Option<SharedMemoryRegion> {
        (id == 1).then_some(self.host_visible_region).flatten()
    }

    fn queue_max_sizes(&self) -> &[u16] {
        &[QUEUE_SIZE, QUEUE_SIZE]
    }

    fn read_config(&self, offset: u64, data: &mut [u8]) {
        let events = self.shared.work.lock().unwrap().events;
        let mut config = [0u8; 16];
        config[0..4].copy_from_slice(&events.to_le_bytes());
        config[8..12].copy_from_slice(&SCANOUTS.to_le_bytes());
        config[12..16].copy_from_slice(&self.capsets.to_le_bytes());
        read_config_bytes(&config, offset, data);
    }

    fn write_config(&mut self, offset: u64, data: &[u8]) {
        // events_clear, at 4: the driver acknowledges the events it has seen.
        if offset == 4 && data.len() == 4 {
            let clear = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            let mut w = self.shared.work.lock().unwrap();
            w.events &= !clear;
            if clear & EVENT_DISPLAY != 0 && w.event_again {
                w.event_again = false;
                w.raise_display_event();
            }
        }
    }

    fn activate(&mut self, activation: Activation) {
        let interrupt = activation.interrupt.clone();
        self.shared.poke(|w| {
            w.interrupt = Some(interrupt);
            w.activation = Some(activation);
        });
    }

    fn notify(&mut self, index: u16) {
        if index < 2 {
            self.shared.poke(|w| w.notified |= 1 << index);
        }
    }

    fn reset(&mut self) {
        let mut w = self.shared.work.lock().unwrap();
        w.activation = None;
        w.interrupt = None;
        w.notified = 0;
        w.events = 0;
        w.event_again = false;
        w.plug = Plug::In;
        w.resets_asked += 1;
        let asked = w.resets_asked;
        self.shared.wake.notify_one();
        // The guest may reuse the queues' memory once this returns, so the
        // thread must be done with them.
        while w.resets_done < asked && !w.gone {
            w = self.shared.reset_done.wait(w).unwrap();
        }
    }
}

/// A slice of guest memory a 2D resource copies from, at its offset in the
/// backing.
#[derive(Clone, Copy)]
struct Piece {
    start: usize,
    slice: HostSlice,
}

enum Kind {
    /// A 2D resource: its pixels live here.
    Simple {
        pixels: Vec<u8>,
        backing: Vec<Piece>,
        backing_len: usize,
    },
    /// The renderer's.
    Renderer,
}

struct Resource {
    width: u32,
    height: u32,
    format: u32,
    kind: Kind,
}

#[derive(Clone, Default)]
struct Cursor {
    resource: u32,
    x: i32,
    y: i32,
    /// R, G, B, premultiplied A.
    image: Option<(u32, u32, Vec<u8>)>,
    /// Where it was last drawn, on the screen.
    drawn: Rect,
}

#[derive(Clone, Default)]
struct Scanout {
    /// 0 while off.
    resource: u32,
    /// The part of the resource shown.
    rect: Rect,
    /// What the screen shows, R, G, B, A, without the cursor.
    screen: Vec<u8>,
    /// Flushes were skipped while nobody watched.
    stale: bool,
    cursor: Cursor,
}

impl Scanout {
    fn width(&self) -> u32 {
        self.rect.width
    }

    fn height(&self) -> u32 {
        self.rect.height
    }

    fn cursor_rect(&self) -> Rect {
        let Some((w, h, _)) = &self.cursor.image else {
            return Rect::default();
        };
        if self.cursor.resource == 0 {
            return Rect::default();
        }
        let x0 = self.cursor.x.max(0) as u32;
        let y0 = self.cursor.y.max(0) as u32;
        let x1 = (self.cursor.x + *w as i32).max(0) as u32;
        let y1 = (self.cursor.y + *h as i32).max(0) as u32;
        Rect::new(x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0))
            .clip(self.width(), self.height())
    }
}

/// A request read out of a chain.
struct Request<'a> {
    kind: u32,
    flags: u32,
    fence: u64,
    context: u32,
    ring: u8,
    body: &'a [u8],
}

impl Request<'_> {
    fn u32(&self, at: usize) -> u32 {
        self.body
            .get(at..at + 4)
            .map_or(0, |b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&self, at: usize) -> u64 {
        self.u32(at) as u64 | (self.u32(at + 4) as u64) << 32
    }

    fn rect(&self, at: usize) -> Rect {
        Rect::new(
            self.u32(at),
            self.u32(at + 4),
            self.u32(at + 8),
            self.u32(at + 12),
        )
    }

    fn fenced(&self) -> bool {
        self.flags & FLAG_FENCE != 0
    }

    /// An answer: the header echoing the request's fence, then `data`.
    fn answer(&self, kind: u32, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER + data.len());
        out.extend_from_slice(&kind.to_le_bytes());
        out.extend_from_slice(&(self.flags & (FLAG_FENCE | FLAG_INFO_RING_IDX)).to_le_bytes());
        out.extend_from_slice(&self.fence.to_le_bytes());
        out.extend_from_slice(&self.context.to_le_bytes());
        out.extend_from_slice(&[self.ring, 0, 0, 0]);
        out.extend_from_slice(data);
        out
    }
}

/// An answer held back until its fence retires.
struct Fenced {
    fence: u64,
    chain: Chain,
    answer: Vec<u8>,
}

struct Worker {
    shared: Arc<Shared>,
    sink: Arc<dyn DisplaySink>,
    renderer: Option<Box<dyn Renderer>>,
    capsets: Vec<Capset>,
    memory: Option<Arc<GuestMemory>>,
    control: Option<Queue>,
    cursor: Option<Queue>,
    interrupt: Option<Arc<Interrupt>>,
    resources: HashMap<u32, Resource>,
    bytes_2d: usize,
    scanouts: Vec<Scanout>,
    fenced: VecDeque<Fenced>,
    /// CRATE_GPU_TRACE: log every control command.
    trace: bool,
}

impl Worker {
    fn run(mut self) {
        loop {
            let (notified, refresh, reset, activation) = {
                let mut w = self.shared.work.lock().unwrap();
                loop {
                    if w.exit {
                        return;
                    }
                    let plug_by = w.plug_by();
                    if plug_by.is_some_and(|by| Instant::now() >= by) {
                        w.plug_in();
                        continue;
                    }
                    if w.notified != 0
                        || w.refresh
                        || w.resets_asked != w.resets_done
                        || w.activation.is_some()
                    {
                        break;
                    }
                    if self.fenced.is_empty() && plug_by.is_none() {
                        w = self.shared.wake.wait(w).unwrap();
                        continue;
                    }
                    let mut wait = plug_by.map_or(FENCE_POLL, |by| {
                        by.saturating_duration_since(Instant::now())
                    });
                    if !self.fenced.is_empty() {
                        wait = wait.min(FENCE_POLL);
                    }
                    let (guard, waited) = self.shared.wake.wait_timeout(w, wait).unwrap();
                    w = guard;
                    if waited.timed_out() && !self.fenced.is_empty() {
                        break;
                    }
                }
                // A reset drops any activation and notifications before it,
                // so what is here now came after it.
                let reset = (w.resets_asked != w.resets_done).then_some(w.resets_asked);
                (
                    std::mem::take(&mut w.notified),
                    std::mem::take(&mut w.refresh),
                    reset,
                    w.activation.take(),
                )
            };
            if let Some(asked) = reset {
                self.reset();
                self.shared.work.lock().unwrap().resets_done = asked;
                self.shared.reset_done.notify_all();
            }
            if let Some(activation) = activation {
                self.activate(activation);
            }
            if notified & 1 != 0 {
                self.run_control();
            }
            if notified & 2 != 0 {
                self.run_cursor();
            }
            if refresh {
                for index in 0..self.scanouts.len() {
                    self.show_all(index);
                }
            }
            self.retire_fences();
        }
    }

    fn activate(&mut self, activation: Activation) {
        let mut queues = activation.queues.into_iter();
        self.control = queues.next();
        self.cursor = queues.next();
        self.memory = Some(activation.memory);
        self.interrupt = Some(activation.interrupt);
    }

    fn reset(&mut self) {
        self.control = None;
        self.cursor = None;
        self.interrupt = None;
        self.fenced.clear();
        if let Some(renderer) = &mut self.renderer {
            for (&id, resource) in &self.resources {
                if matches!(resource.kind, Kind::Renderer) {
                    renderer.destroy_resource(id);
                }
            }
            renderer.reset();
        }
        self.resources.clear();
        self.bytes_2d = 0;
        for scanout in &mut self.scanouts {
            scanout.resource = 0;
            scanout.cursor = Cursor::default();
        }
    }

    fn signal(&self) {
        if let Some(interrupt) = &self.interrupt {
            interrupt.signal_used();
        }
    }

    fn run_control(&mut self) {
        let mut used = false;
        loop {
            let Some(queue) = &mut self.control else {
                return;
            };
            let chain = match queue.pop() {
                Ok(Some(chain)) => chain,
                Ok(None) => break,
                Err(e) => {
                    warn!("virtio-gpu: control queue: {e}");
                    break;
                }
            };
            let bytes = match chain.read_all(queue.memory(), MAX_REQUEST) {
                Ok(bytes) => bytes,
                Err(e) => {
                    warn!("virtio-gpu: {e}");
                    Vec::new()
                }
            };
            let Some(request) = parse(&bytes) else {
                let mut answer = [0u8; HEADER];
                answer[0..4].copy_from_slice(&RESP_ERR_UNSPEC.to_le_bytes());
                used |= self.complete(&chain, &answer);
                continue;
            };
            let answer = self.command(&request);
            if request.fenced() {
                let fenced = match &mut self.renderer {
                    Some(renderer) => renderer
                        .create_fence(request.fence, request.context)
                        .map_err(|e| warn!("virtio-gpu: fence {}: {e}", request.fence))
                        .is_ok(),
                    None => false,
                };
                // With a renderer every fence goes through it, so they answer
                // in the order they were made.
                if fenced {
                    self.fenced.push_back(Fenced {
                        fence: request.fence,
                        chain,
                        answer,
                    });
                    continue;
                }
            }
            used |= self.complete(&chain, &answer);
        }
        if used {
            self.notify_driver();
        }
    }

    /// Writes an answer and returns the buffer; true if it was used.
    fn complete(&mut self, chain: &Chain, answer: &[u8]) -> bool {
        let Some(queue) = &mut self.control else {
            return false;
        };
        let n = chain.write_all(queue.memory(), answer).unwrap_or(0);
        queue.add_used(chain.head, n as u32).is_ok()
    }

    fn notify_driver(&mut self) {
        if self.control.as_ref().is_some_and(|q| q.needs_interrupt()) {
            self.signal();
        }
    }

    fn retire_fences(&mut self) {
        if self.fenced.is_empty() {
            return;
        }
        let Some(renderer) = &mut self.renderer else {
            return;
        };
        let retired = renderer.poll();
        let Some(&last) = retired.last() else {
            return;
        };
        if !self.fenced.iter().any(|f| f.fence == last) {
            return;
        }
        let mut used = false;
        // Retiring a fence retires every one made before it.
        while let Some(f) = self.fenced.pop_front() {
            used |= self.complete(&f.chain, &f.answer);
            if f.fence == last {
                break;
            }
        }
        if used {
            self.notify_driver();
        }
    }

    fn run_cursor(&mut self) {
        let mut used = false;
        loop {
            let Some(queue) = &mut self.cursor else {
                return;
            };
            let chain = match queue.pop() {
                Ok(Some(chain)) => chain,
                Ok(None) => break,
                Err(e) => {
                    warn!("virtio-gpu: cursor queue: {e}");
                    break;
                }
            };
            let bytes = chain.read_all(queue.memory(), 4096).unwrap_or_default();
            if queue.add_used(chain.head, 0).is_ok() {
                used = true;
            }
            if let Some(request) = parse(&bytes) {
                self.cursor_command(&request);
            }
        }
        if used && self.cursor.as_ref().is_some_and(|q| q.needs_interrupt()) {
            self.signal();
        }
    }

    fn cursor_command(&mut self, r: &Request) {
        let scanout = r.u32(0) as usize;
        let (x, y) = (r.u32(4) as i32, r.u32(8) as i32);
        if scanout >= self.scanouts.len() {
            return;
        }
        if r.kind == CMD_UPDATE_CURSOR {
            let resource = r.u32(16);
            let (hot_x, hot_y) = (r.u32(20), r.u32(24));
            let image = self.cursor_image(resource);
            let cursor = &mut self.scanouts[scanout].cursor;
            cursor.resource = resource;
            cursor.image = image;
            debug!("virtio-gpu: cursor {resource} at {x},{y}, hot spot {hot_x},{hot_y}");
        } else if r.kind != CMD_MOVE_CURSOR {
            return;
        }
        let cursor = &mut self.scanouts[scanout].cursor;
        cursor.x = x;
        cursor.y = y;
        self.show(scanout, Rect::default());
    }

    fn cursor_image(&mut self, resource: u32) -> Option<(u32, u32, Vec<u8>)> {
        if resource == 0 {
            return None;
        }
        let r = self.resources.get(&resource)?;
        let (w, h, mut pixels, format) = match &r.kind {
            Kind::Simple { pixels, .. } => (r.width, r.height, pixels.clone(), r.format),
            Kind::Renderer => {
                let (w, h, pixels) = self.renderer.as_mut()?.cursor_image(resource)?;
                (w, h, pixels, format::B8G8R8A8)
            }
        };
        if pixels.len() < (w * h * 4) as usize {
            return None;
        }
        format::to_rgba(format::with_alpha(format), &mut pixels);
        Some((w, h, pixels))
    }

    /// Runs one control command and returns its answer.
    fn command(&mut self, r: &Request) -> Vec<u8> {
        if self.trace {
            let words: Vec<u32> = (0..12).map(|i| r.u32(4 * i)).collect();
            info!(
                "gpu: {:#x} ctx {} fence {} {:?}",
                r.kind,
                r.context,
                if r.fenced() { r.fence } else { 0 },
                words
            );
        }
        let ok = || r.answer(RESP_OK_NODATA, &[]);
        let err = |kind: u32| r.answer(kind, &[]);
        let status = match r.kind {
            CMD_GET_DISPLAY_INFO => return r.answer(RESP_OK_DISPLAY_INFO, &self.display_info()),
            CMD_GET_EDID => {
                if r.u32(0) as usize >= self.scanouts.len() {
                    return err(RESP_ERR_INVALID_SCANOUT_ID);
                }
                let w = self.shared.work.lock().unwrap();
                // Unplugged: no EDID.
                let blob = if w.plug == Plug::In {
                    edid::edid(&w.modes)
                } else {
                    Vec::new()
                };
                drop(w);
                let mut data = vec![0u8; 8 + 1024];
                data[0..4].copy_from_slice(&(blob.len() as u32).to_le_bytes());
                data[8..8 + blob.len()].copy_from_slice(&blob);
                return r.answer(RESP_OK_EDID, &data);
            }
            CMD_GET_CAPSET_INFO => {
                let mut data = [0u8; 16];
                if let Some(c) = self.capsets.get(r.u32(0) as usize) {
                    data[0..4].copy_from_slice(&c.id.to_le_bytes());
                    data[4..8].copy_from_slice(&c.max_version.to_le_bytes());
                    data[8..12].copy_from_slice(&c.max_size.to_le_bytes());
                }
                return r.answer(RESP_OK_CAPSET_INFO, &data);
            }
            CMD_GET_CAPSET => {
                let (id, version) = (r.u32(0), r.u32(4));
                let Some(c) = self.capsets.iter().find(|c| c.id == id).copied() else {
                    return err(RESP_ERR_INVALID_PARAMETER);
                };
                let mut data = vec![0u8; c.max_size as usize];
                if let Some(renderer) = &mut self.renderer {
                    renderer.fill_capset(id, version, &mut data);
                }
                return r.answer(RESP_OK_CAPSET, &data);
            }
            CMD_RESOURCE_CREATE_2D => self.create_2d(r),
            CMD_RESOURCE_UNREF => self.unref(r.u32(0)),
            CMD_SET_SCANOUT => self.set_scanout(r),
            CMD_RESOURCE_FLUSH => self.flush(r),
            CMD_TRANSFER_TO_HOST_2D => self.transfer_2d(r),
            CMD_RESOURCE_ATTACH_BACKING => self.attach_backing(r),
            CMD_RESOURCE_DETACH_BACKING => self.detach_backing(r.u32(0)),
            CMD_CTX_CREATE..=CMD_SUBMIT_3D => self.command_3d(r),
            _ => {
                debug!("virtio-gpu: command {:#x} not offered", r.kind);
                Err(RESP_ERR_UNSPEC)
            }
        };
        match status {
            Ok(()) => ok(),
            Err(kind) => {
                debug!("virtio-gpu: command {:#x} failed: {kind:#x}", r.kind);
                err(kind)
            }
        }
    }

    fn display_info(&self) -> Vec<u8> {
        let mut data = vec![0u8; 24 * MAX_SCANOUTS];
        let mut w = self.shared.work.lock().unwrap();
        if let Plug::Out { seen, .. } = &mut w.plug {
            // Unplugged: every scanout disabled, and the guest now knows.
            *seen = true;
            return data;
        }
        if let Some(mode) = w.modes.first() {
            for one in data.chunks_exact_mut(24).take(self.scanouts.len()) {
                one[8..12].copy_from_slice(&mode.width.to_le_bytes());
                one[12..16].copy_from_slice(&mode.height.to_le_bytes());
                one[16..20].copy_from_slice(&1u32.to_le_bytes());
            }
        }
        data
    }

    fn create_2d(&mut self, r: &Request) -> Result<(), u32> {
        let (id, format, width, height) = (r.u32(0), r.u32(4), r.u32(8), r.u32(12));
        if id == 0 || self.resources.contains_key(&id) {
            return Err(RESP_ERR_INVALID_RESOURCE_ID);
        }
        if !format::supported(format)
            || width == 0
            || height == 0
            || width > MAX_DIMENSION
            || height > MAX_DIMENSION
        {
            return Err(RESP_ERR_INVALID_PARAMETER);
        }
        let bytes = width as usize * height as usize * 4;
        if self.bytes_2d + bytes > MAX_2D_BYTES {
            return Err(RESP_ERR_OUT_OF_MEMORY);
        }
        self.bytes_2d += bytes;
        self.resources.insert(
            id,
            Resource {
                width,
                height,
                format,
                kind: Kind::Simple {
                    pixels: vec![0; bytes],
                    backing: Vec::new(),
                    backing_len: 0,
                },
            },
        );
        Ok(())
    }

    fn unref(&mut self, id: u32) -> Result<(), u32> {
        let resource = self
            .resources
            .remove(&id)
            .ok_or(RESP_ERR_INVALID_RESOURCE_ID)?;
        match resource.kind {
            Kind::Simple { pixels, .. } => self.bytes_2d -= pixels.len(),
            Kind::Renderer => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.destroy_resource(id);
                }
            }
        }
        for scanout in &mut self.scanouts {
            if scanout.resource == id {
                scanout.resource = 0;
            }
        }
        Ok(())
    }

    fn set_scanout(&mut self, r: &Request) -> Result<(), u32> {
        let rect = r.rect(0);
        let (index, id) = (r.u32(16) as usize, r.u32(20));
        if index >= self.scanouts.len() {
            return Err(RESP_ERR_INVALID_SCANOUT_ID);
        }
        if id == 0 {
            // The guest let go of a display it read as unplugged: plug it
            // back in.
            let mut w = self.shared.work.lock().unwrap();
            if matches!(w.plug, Plug::Out { seen: true, .. }) {
                w.plug_in();
            }
            drop(w);
            // Off: the screen goes black.
            let scanout = &mut self.scanouts[index];
            scanout.resource = 0;
            scanout.screen.fill(0);
            let all = Rect::new(0, 0, scanout.width(), scanout.height());
            self.show(index, all);
            return Ok(());
        }
        let resource = self
            .resources
            .get(&id)
            .ok_or(RESP_ERR_INVALID_RESOURCE_ID)?;
        if rect.is_empty() || rect.clip(resource.width, resource.height) != rect {
            return Err(RESP_ERR_INVALID_PARAMETER);
        }
        let scanout = &mut self.scanouts[index];
        if (scanout.width(), scanout.height()) != (rect.width, rect.height) {
            scanout.screen = vec![0; rect.width as usize * rect.height as usize * 4];
            // A new size: the next flush must fill all of it.
            scanout.stale = true;
            info!(
                "virtio-gpu: scanout {index} is {}x{}",
                rect.width, rect.height
            );
        }
        scanout.resource = id;
        scanout.rect = rect;
        Ok(())
    }

    fn flush(&mut self, r: &Request) -> Result<(), u32> {
        let (rect, id) = (r.rect(0), r.u32(16));
        if !self.resources.contains_key(&id) {
            return Err(RESP_ERR_INVALID_RESOURCE_ID);
        }
        for index in 0..self.scanouts.len() {
            let scanout = &mut self.scanouts[index];
            if scanout.resource != id {
                continue;
            }
            if !self.sink.active() {
                scanout.stale = true;
                continue;
            }
            if scanout.stale {
                self.show_all(index);
                continue;
            }
            let area = rect.intersect(&scanout.rect);
            if !area.is_empty() {
                self.update(index, area);
                let origin = self.scanouts[index].rect;
                self.show(
                    index,
                    Rect::new(
                        area.x - origin.x,
                        area.y - origin.y,
                        area.width,
                        area.height,
                    ),
                );
            }
        }
        Ok(())
    }

    /// Makes a frame of the whole screen.
    fn show_all(&mut self, index: usize) {
        let scanout = &mut self.scanouts[index];
        if scanout.resource == 0 || !self.sink.active() {
            return;
        }
        scanout.stale = false;
        let rect = scanout.rect;
        self.update(index, rect);
        self.show(index, Rect::new(0, 0, rect.width, rect.height));
    }

    /// Copies `area` (in the resource) of the scanout's resource to its screen.
    fn update(&mut self, index: usize, area: Rect) {
        let scanout = &mut self.scanouts[index];
        let Some(resource) = self.resources.get(&scanout.resource) else {
            return;
        };
        let stride = scanout.width() as usize * 4;
        let (dx, dy) = (
            (area.x - scanout.rect.x) as usize,
            (area.y - scanout.rect.y) as usize,
        );
        let start = dy * stride + dx * 4;
        let row = area.width as usize * 4;
        match &resource.kind {
            Kind::Simple { pixels, .. } => {
                let source_stride = resource.width as usize * 4;
                for line in 0..area.height as usize {
                    let from = (area.y as usize + line) * source_stride + area.x as usize * 4;
                    let to = start + line * stride;
                    let out = &mut scanout.screen[to..to + row];
                    out.copy_from_slice(&pixels[from..from + row]);
                    format::to_rgba(resource.format, out);
                }
            }
            Kind::Renderer => {
                let Some(renderer) = &mut self.renderer else {
                    return;
                };
                let end = start + (area.height as usize - 1) * stride + row;
                if let Err(e) = renderer.read(
                    scanout.resource,
                    area,
                    &mut scanout.screen[start..end],
                    stride as u32,
                ) {
                    warn!("virtio-gpu: reading resource {}: {e}", scanout.resource);
                }
            }
        }
    }

    /// Hands the sink a frame with `dirty` changed, the cursor drawn in.
    fn show(&mut self, index: usize, dirty: Rect) {
        let scanout = &mut self.scanouts[index];
        if scanout.screen.is_empty() || !self.sink.active() {
            return;
        }
        let (width, height) = (scanout.width(), scanout.height());
        let stride = width as usize * 4;
        let cursor = scanout.cursor_rect();
        let dirty = dirty.union(&scanout.cursor.drawn).union(&cursor);
        if dirty.is_empty() {
            return;
        }
        // Draw the cursor over the screen, and put back what it covered once
        // the frame has been seen.
        let mut under = Vec::new();
        if let (false, Some((cw, _, image))) = (cursor.is_empty(), &scanout.cursor.image) {
            let row = cursor.width as usize * 4;
            under.reserve(row * cursor.height as usize);
            let (ox, oy) = (
                cursor.x as i64 - scanout.cursor.x as i64,
                cursor.y as i64 - scanout.cursor.y as i64,
            );
            for line in 0..cursor.height as usize {
                let at = (cursor.y as usize + line) * stride + cursor.x as usize * 4;
                let dst = &mut scanout.screen[at..at + row];
                under.extend_from_slice(dst);
                let src_at = ((oy as usize + line) * *cw as usize + ox as usize) * 4;
                let src = &image[src_at..src_at + row];
                for (d, s) in dst.chunks_exact_mut(4).zip(src.chunks_exact(4)) {
                    let keep = 255 - s[3] as u32;
                    for c in 0..3 {
                        d[c] = (s[c] as u32 + (d[c] as u32 * keep + 127) / 255).min(255) as u8;
                    }
                }
            }
        }
        self.sink.frame(&Frame {
            width,
            height,
            stride: stride as u32,
            pixels: &scanout.screen,
            dirty,
        });
        if !under.is_empty() {
            let row = cursor.width as usize * 4;
            for (line, saved) in under.chunks_exact(row).enumerate() {
                let at = (cursor.y as usize + line) * stride + cursor.x as usize * 4;
                scanout.screen[at..at + row].copy_from_slice(saved);
            }
        }
        scanout.cursor.drawn = cursor;
    }

    fn transfer_2d(&mut self, r: &Request) -> Result<(), u32> {
        let (rect, offset, id) = (r.rect(0), r.u64(16), r.u32(24));
        let resource = self
            .resources
            .get_mut(&id)
            .ok_or(RESP_ERR_INVALID_RESOURCE_ID)?;
        let Kind::Simple {
            pixels,
            backing,
            backing_len,
        } = &mut resource.kind
        else {
            // A 3D resource the guest moves with 2D commands: the renderer's
            // transfer does the same.
            let Some(renderer) = &mut self.renderer else {
                return Err(RESP_ERR_INVALID_RESOURCE_ID);
            };
            return renderer
                .transfer_to_host(&Transfer {
                    resource: id,
                    context: 0,
                    level: 0,
                    stride: 0,
                    layer_stride: 0,
                    region: Box3d {
                        x: rect.x,
                        y: rect.y,
                        z: 0,
                        w: rect.width,
                        h: rect.height,
                        d: 1,
                    },
                    offset,
                })
                .map_err(|_| RESP_ERR_INVALID_PARAMETER);
        };
        if rect.is_empty() || rect.clip(resource.width, resource.height) != rect {
            return Err(RESP_ERR_INVALID_PARAMETER);
        }
        let stride = resource.width as usize * 4;
        let row = rect.width as usize * 4;
        let last = offset as usize + (rect.height as usize - 1) * stride + row;
        if backing.is_empty() || last > *backing_len {
            return Err(RESP_ERR_INVALID_PARAMETER);
        }
        for line in 0..rect.height as usize {
            let to = (rect.y as usize + line) * stride + rect.x as usize * 4;
            copy_from_backing(
                backing,
                offset as usize + line * stride,
                &mut pixels[to..to + row],
            );
        }
        Ok(())
    }

    /// The backing's pieces as host memory, checked to be guest RAM.
    fn backing_pieces(&self, r: &Request) -> Result<(Vec<Piece>, usize), u32> {
        let count = r.u32(4);
        if count == 0 || count > MAX_BACKING_ENTRIES || r.body.len() < 8 + 16 * count as usize {
            return Err(RESP_ERR_INVALID_PARAMETER);
        }
        let memory = self.memory.as_ref().ok_or(RESP_ERR_UNSPEC)?;
        let mut pieces = Vec::with_capacity(count as usize);
        let mut total = 0usize;
        for i in 0..count as usize {
            let at = 8 + 16 * i;
            let (addr, len) = (r.u64(at), r.u32(at + 8) as usize);
            let ptr = memory
                .host_range(addr, len as u64)
                .map_err(|_| RESP_ERR_INVALID_PARAMETER)?;
            pieces.push(Piece {
                start: total,
                slice: HostSlice { ptr, len },
            });
            total += len;
        }
        Ok((pieces, total))
    }

    fn attach_backing(&mut self, r: &Request) -> Result<(), u32> {
        let id = r.u32(0);
        if !self.resources.contains_key(&id) {
            return Err(RESP_ERR_INVALID_RESOURCE_ID);
        }
        let (pieces, total) = self.backing_pieces(r)?;
        let resource = self.resources.get_mut(&id).unwrap();
        match &mut resource.kind {
            Kind::Simple {
                backing,
                backing_len,
                ..
            } => {
                *backing = pieces;
                *backing_len = total;
                Ok(())
            }
            Kind::Renderer => {
                let renderer = self.renderer.as_mut().ok_or(RESP_ERR_UNSPEC)?;
                let slices = pieces.into_iter().map(|p| p.slice).collect();
                // SAFETY: guest RAM stays mapped for the machine's life, which
                // the renderer, owned by this device, does not outlive.
                unsafe { renderer.attach_backing(id, slices) }
                    .map_err(|_| RESP_ERR_INVALID_PARAMETER)
            }
        }
    }

    fn detach_backing(&mut self, id: u32) -> Result<(), u32> {
        let resource = self
            .resources
            .get_mut(&id)
            .ok_or(RESP_ERR_INVALID_RESOURCE_ID)?;
        match &mut resource.kind {
            Kind::Simple {
                backing,
                backing_len,
                ..
            } => {
                backing.clear();
                *backing_len = 0;
            }
            Kind::Renderer => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.detach_backing(id);
                }
            }
        }
        Ok(())
    }

    fn command_3d(&mut self, r: &Request) -> Result<(), u32> {
        let Some(renderer) = &mut self.renderer else {
            return Err(RESP_ERR_UNSPEC);
        };
        let failed = |what: &str, e: String| {
            debug!("virtio-gpu: {what}: {e}");
        };
        match r.kind {
            CMD_CTX_CREATE => {
                let len = (r.u32(0) as usize).min(64);
                let name = r.body.get(8..8 + len).unwrap_or(&[]);
                renderer
                    .create_context(r.context, name)
                    .map_err(|e| failed("creating a context", e))
                    .map_err(|_| RESP_ERR_INVALID_CONTEXT_ID)
            }
            CMD_CTX_DESTROY => {
                renderer.destroy_context(r.context);
                Ok(())
            }
            CMD_CTX_ATTACH_RESOURCE | CMD_CTX_DETACH_RESOURCE => {
                let id = r.u32(0);
                // 2D resources are the device's own; the renderer never sees them.
                if let Some(Resource {
                    kind: Kind::Renderer,
                    ..
                }) = self.resources.get(&id)
                {
                    if r.kind == CMD_CTX_ATTACH_RESOURCE {
                        renderer.attach_resource(r.context, id);
                    } else {
                        renderer.detach_resource(r.context, id);
                    }
                }
                Ok(())
            }
            CMD_RESOURCE_CREATE_3D => {
                let args = Resource3d {
                    id: r.u32(0),
                    target: r.u32(4),
                    format: r.u32(8),
                    bind: r.u32(12),
                    width: r.u32(16),
                    height: r.u32(20),
                    depth: r.u32(24),
                    array_size: r.u32(28),
                    last_level: r.u32(32),
                    nr_samples: r.u32(36),
                    flags: r.u32(40),
                };
                if args.id == 0 || self.resources.contains_key(&args.id) {
                    return Err(RESP_ERR_INVALID_RESOURCE_ID);
                }
                renderer
                    .create_resource(&args)
                    .map_err(|e| failed("creating a resource", e))
                    .map_err(|_| RESP_ERR_INVALID_PARAMETER)?;
                self.resources.insert(
                    args.id,
                    Resource {
                        width: args.width,
                        height: args.height,
                        format: args.format,
                        kind: Kind::Renderer,
                    },
                );
                Ok(())
            }
            CMD_TRANSFER_TO_HOST_3D | CMD_TRANSFER_FROM_HOST_3D => {
                let transfer = Transfer {
                    region: Box3d {
                        x: r.u32(0),
                        y: r.u32(4),
                        z: r.u32(8),
                        w: r.u32(12),
                        h: r.u32(16),
                        d: r.u32(20),
                    },
                    offset: r.u64(24),
                    resource: r.u32(32),
                    level: r.u32(36),
                    stride: r.u32(40),
                    layer_stride: r.u32(44),
                    context: r.context,
                };
                if !matches!(
                    self.resources.get(&transfer.resource),
                    Some(Resource {
                        kind: Kind::Renderer,
                        ..
                    })
                ) {
                    return Err(RESP_ERR_INVALID_RESOURCE_ID);
                }
                let result = if r.kind == CMD_TRANSFER_TO_HOST_3D {
                    renderer.transfer_to_host(&transfer)
                } else {
                    renderer.transfer_from_host(&transfer)
                };
                result
                    .map_err(|e| failed("a transfer", e))
                    .map_err(|_| RESP_ERR_INVALID_PARAMETER)
            }
            CMD_SUBMIT_3D => {
                let size = r.u32(0) as usize;
                let Some(bytes) = r.body.get(8..8 + size) else {
                    return Err(RESP_ERR_INVALID_PARAMETER);
                };
                let mut words: Vec<u32> = bytes
                    .chunks_exact(4)
                    .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                    .collect();
                renderer
                    .submit(r.context, &mut words)
                    .map_err(|e| failed("a submission", e))
                    .map_err(|_| RESP_ERR_INVALID_PARAMETER)
            }
            _ => Err(RESP_ERR_UNSPEC),
        }
    }
}

/// Fills `out` from the backing, starting `offset` bytes into it.
fn copy_from_backing(backing: &[Piece], offset: usize, out: &mut [u8]) {
    // The last piece that starts at or before the offset.
    let mut i = backing
        .partition_point(|p| p.start <= offset)
        .saturating_sub(1);
    let mut done = 0;
    while done < out.len() && i < backing.len() {
        let piece = backing[i];
        let skip = (offset + done).saturating_sub(piece.start);
        if skip < piece.slice.len {
            let n = (piece.slice.len - skip).min(out.len() - done);
            // SAFETY: the piece was checked to be guest RAM when it was
            // attached, and guest RAM stays mapped; the guest may write it
            // meanwhile, which only changes the bytes copied.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    piece.slice.ptr.add(skip),
                    out[done..].as_mut_ptr(),
                    n,
                )
            };
            done += n;
        }
        i += 1;
    }
}

fn parse(bytes: &[u8]) -> Option<Request<'_>> {
    if bytes.len() < HEADER {
        return None;
    }
    let u32_at =
        |at: usize| u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
    Some(Request {
        kind: u32_at(0),
        flags: u32_at(4),
        fence: u32_at(8) as u64 | (u32_at(12) as u64) << 32,
        context: u32_at(16),
        ring: bytes[20],
        body: &bytes[HEADER..],
    })
}

#[cfg(test)]
mod tests;

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use super::super::queue::QueueConfig;
use super::super::queue::testing::*;
use super::*;
use crate_core::gpu::{BlobResource, ContextFence, RendererResult};
use crate_core::memory::{HeapRam, RamBlock};
use crate_core::irq::{IrqLine, LevelRecorder};

const CURSOR_RINGS: u64 = RAM_BASE + 0x4000;
const REQUEST: u64 = DATA;
const ANSWER: u64 = DATA + 0x1000;

/// A frame as the sink saw it: width, height, what changed, the pixels.
type Seen = (u32, u32, Rect, Vec<u8>);

#[derive(Default)]
struct Screen {
    active: AtomicBool,
    frames: Mutex<Vec<Seen>>,
}

impl DisplaySink for Screen {
    fn active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }
    fn frame(&self, frame: &Frame) {
        self.frames.lock().unwrap().push((
            frame.width,
            frame.height,
            frame.dirty,
            frame.pixels.to_vec(),
        ));
    }
}

struct Machine {
    gpu: Gpu,
    memory: Arc<GuestMemory>,
    control: Driver,
    cursor: Driver,
    screen: Arc<Screen>,
}

fn cursor_config() -> QueueConfig {
    QueueConfig {
        size: 16,
        ready: true,
        desc: CURSOR_RINGS,
        driver: CURSOR_RINGS + 0x1000,
        device: CURSOR_RINGS + 0x2000,
    }
}

fn machine(renderer: Option<RendererFactory>) -> Machine {
    machine_with_memory(renderer, memory(), None, None)
}

const HOST_VISIBLE_BASE: u64 = RAM_BASE + 0x10_0000;
const HOST_VISIBLE_SIZE: usize = 0x1_0000;

struct TestHostVisibleMemory {
    ram: HeapRam,
    guest_addr: u64,
    mappings: Mutex<std::collections::HashMap<u64, usize>>,
}

impl TestHostVisibleMemory {
    fn new(guest_addr: u64, len: usize) -> Self {
        assert_eq!(guest_addr % 4_096, 0);
        assert_eq!(len % 4_096, 0);
        Self {
            ram: HeapRam::new(len).unwrap(),
            guest_addr,
            mappings: Mutex::new(std::collections::HashMap::new()),
        }
    }
}

unsafe impl RamBlock for TestHostVisibleMemory {
    fn as_ptr(&self) -> *mut u8 {
        self.ram.as_ptr()
    }
    fn len(&self) -> usize {
        self.ram.len()
    }
}

impl HostVisibleMemory for TestHostVisibleMemory {
    fn guest_addr(&self) -> u64 {
        self.guest_addr
    }
    fn alignment(&self) -> u32 {
        4_096
    }
    fn reserve(&self, guest_addr: u64, len: usize) -> io::Result<*mut u8> {
        let offset = guest_addr
            .checked_sub(self.guest_addr)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "outside host memory"))?;
        let end = offset
            .checked_add(len)
            .filter(|end| *end <= self.ram.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "mapping out of range"))?;
        let mut mappings = self.mappings.lock().unwrap();
        if mappings.contains_key(&guest_addr) {
            return Err(io::Error::new(io::ErrorKind::AlreadyExists, "already mapped"));
        }
        mappings.insert(guest_addr, end - offset);
        // SAFETY: the checked range lies within the owned HeapRam allocation.
        Ok(unsafe { self.ram.as_ptr().add(offset) })
    }
    fn map_fd(&self, _guest_addr: u64, _fd: std::os::fd::RawFd, _len: usize) -> io::Result<()> {
        Err(io::Error::new(io::ErrorKind::Unsupported, "not used by this test"))
    }
    fn unmap(&self, guest_addr: u64) -> io::Result<()> {
        self.mappings
            .lock()
            .unwrap()
            .remove(&guest_addr)
            .map(|_| ())
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "not mapped"))
    }
}

fn machine_with_host_visible(
    renderer: Option<RendererFactory>,
    host: Arc<TestHostVisibleMemory>,
) -> Machine {
    machine_with_host_visible_features(renderer, host, None)
}

fn machine_with_host_visible_features(
    renderer: Option<RendererFactory>,
    host: Arc<TestHostVisibleMemory>,
    negotiated_features: Option<u64>,
) -> Machine {
    let ram: Arc<dyn RamBlock> = Arc::new(HeapRam::new(0x10_0000).unwrap());
    let host_ram: Arc<dyn RamBlock> = host.clone();
    let memory = Arc::new(
        GuestMemory::new(vec![(RAM_BASE, ram), (host.guest_addr(), host_ram)]).unwrap(),
    );
    let host_visible_memory: Arc<dyn HostVisibleMemory> = host;
    machine_with_memory(renderer, memory, Some(host_visible_memory), negotiated_features)
}

fn machine_with_memory(
    renderer: Option<RendererFactory>,
    memory: Arc<GuestMemory>,
    host_visible_memory: Option<Arc<dyn HostVisibleMemory>>,
    negotiated_features: Option<u64>,
) -> Machine {
    let screen = Arc::new(Screen::default());
    screen.active.store(true, Ordering::Relaxed);
    let has_host_visible_memory = host_visible_memory.is_some();
    let mut gpu = Gpu::new(GpuConfig {
        modes: vec![Mode::new(800, 600), Mode::new(640, 480)],
        sink: screen.clone(),
        renderer,
        host_visible_memory,
    })
    .unwrap();
    let features = negotiated_features.unwrap_or_else(|| {
        if has_host_visible_memory { gpu.features() } else { F_VERSION_1 }
    });
    gpu.activate(Activation {
        memory: memory.clone(),
        queues: vec![
            Queue::new(memory.clone(), config(16)),
            Queue::new(memory.clone(), cursor_config()),
        ],
        interrupt: Interrupt::new(IrqLine::new(Arc::new(LevelRecorder::default()), 0)),
        features,
    });
    Machine {
        gpu,
        control: Driver::new(memory.clone(), 16),
        cursor: Driver::at(memory.clone(), cursor_config()),
        memory,
        screen,
    }
}

fn header(kind: u32, flags: u32, fence: u64) -> Vec<u8> {
    let mut h = Vec::new();
    h.extend_from_slice(&kind.to_le_bytes());
    h.extend_from_slice(&flags.to_le_bytes());
    h.extend_from_slice(&fence.to_le_bytes());
    h.extend_from_slice(&[0; 8]);
    h
}

fn words(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn wait_used(driver: &mut Driver) -> Option<(u16, u32)> {
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(5) {
        if let Some(used) = driver.take_used() {
            return Some(used);
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    None
}

impl Machine {
    /// Sends a command and returns the answer, once there is one.
    fn send(&mut self, request: &[u8]) -> Vec<u8> {
        self.send_at(request, 0);
        let (_, len) = wait_used(&mut self.control).expect("an answer");
        let mut answer = vec![0u8; len as usize];
        self.memory.read(ANSWER, &mut answer).unwrap();
        answer
    }

    /// Offers a command without waiting for it; `slot` keeps buffers of
    /// commands in flight apart.
    fn send_at(&mut self, request: &[u8], slot: u64) {
        let at = REQUEST + 0x100 * slot;
        self.memory.write(at, request).unwrap();
        self.control.offer(&[
            (at, request.len() as u32, false),
            (ANSWER + 0x2000 * slot, 0x1000, true),
        ]);
        self.gpu.notify(0);
    }

    fn send_ok(&mut self, kind: u32, body: &[u32]) {
        let mut r = header(kind, 0, 0);
        r.extend(words(body));
        let answer = self.send(&r);
        assert_eq!(
            u32::from_le_bytes(answer[0..4].try_into().unwrap()),
            RESP_OK_NODATA,
            "command {kind:#x}"
        );
    }

    fn cursor(&mut self, kind: u32, x: u32, y: u32, resource: u32) {
        let mut r = header(kind, 0, 0);
        r.extend(words(&[0, x, y, 0, resource, 0, 0, 0]));
        let at = DATA + 0x8000;
        self.memory.write(at, &r).unwrap();
        self.cursor.offer(&[(at, r.len() as u32, false)]);
        self.gpu.notify(1);
        wait_used(&mut self.cursor).expect("the cursor command back");
    }

    /// Waits for the frame count to reach `n`, and returns the last frame.
    fn frame(&self, n: usize) -> Seen {
        let start = Instant::now();
        loop {
            let frames = self.screen.frames.lock().unwrap();
            if frames.len() >= n {
                return frames[n - 1].clone();
            }
            drop(frames);
            assert!(start.elapsed() < Duration::from_secs(5), "no frame {n}");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// A 2D resource of `pixels` (B, G, R, A), its backing in two pieces.
    fn resource(&mut self, id: u32, width: u32, height: u32, pixels: &[u8], at: u64) {
        self.send_ok(
            CMD_RESOURCE_CREATE_2D,
            &[id, format::B8G8R8A8, width, height],
        );
        let split = 12u64;
        self.memory.write(at, &pixels[..split as usize]).unwrap();
        self.memory
            .write(at + 0x800, &pixels[split as usize..])
            .unwrap();
        let len = pixels.len() as u32;
        let (lo, hi) = (at as u32, (at >> 32) as u32);
        let (lo2, hi2) = ((at + 0x800) as u32, ((at + 0x800) >> 32) as u32);
        self.send_ok(
            CMD_RESOURCE_ATTACH_BACKING,
            &[
                id,
                2,
                lo,
                hi,
                split as u32,
                0,
                lo2,
                hi2,
                len - split as u32,
                0,
            ],
        );
        self.send_ok(CMD_TRANSFER_TO_HOST_2D, &[0, 0, width, height, 0, 0, id, 0]);
    }
}

#[test]
fn offers_its_modes() {
    let mut m = machine(None);
    assert_eq!(
        m.gpu.features(),
        F_VERSION_1 | F_EDID,
        "no virgl without a renderer"
    );
    let answer = m.send(&header(CMD_GET_DISPLAY_INFO, 0, 0));
    assert_eq!(answer.len(), HEADER + 24 * MAX_SCANOUTS);
    assert_eq!(&answer[0..4], &RESP_OK_DISPLAY_INFO.to_le_bytes());
    assert_eq!(&answer[HEADER + 8..HEADER + 20], &words(&[800, 600, 1])[..]);

    let mut r = header(CMD_GET_EDID, 0, 0);
    r.extend(words(&[0, 0]));
    let answer = m.send(&r);
    assert_eq!(&answer[0..4], &RESP_OK_EDID.to_le_bytes());
    assert_eq!(&answer[HEADER..HEADER + 4], &128u32.to_le_bytes());
    assert_eq!(
        &answer[HEADER + 8..HEADER + 16],
        &[0, 255, 255, 255, 255, 255, 255, 0]
    );
}

impl Machine {
    fn events(&self) -> u32 {
        let mut events = [0u8; 4];
        self.gpu.read_config(0, &mut events);
        u32::from_le_bytes(events)
    }

    /// The first scanout's width, height and enabled flag.
    fn display(&mut self) -> [u32; 3] {
        let answer = self.send(&header(CMD_GET_DISPLAY_INFO, 0, 0));
        let at = |i: usize| u32::from_le_bytes(answer[i..i + 4].try_into().unwrap());
        [at(HEADER + 8), at(HEADER + 12), at(HEADER + 16)]
    }

    fn edid_size(&mut self) -> u32 {
        let mut r = header(CMD_GET_EDID, 0, 0);
        r.extend(words(&[0, 0]));
        let answer = self.send(&r);
        u32::from_le_bytes(answer[HEADER..HEADER + 4].try_into().unwrap())
    }
}

#[test]
fn new_modes_unplug_the_display_and_plug_it_back_in() {
    let mut m = machine(None);
    m.gpu.control().set_modes(vec![Mode::new(1024, 768)]);
    assert_eq!(m.events(), EVENT_DISPLAY, "a display event");
    assert_eq!(m.display(), [0, 0, 0], "unplugged");
    assert_eq!(m.edid_size(), 0, "no EDID while unplugged");
    m.gpu.write_config(4, &[1, 0, 0, 0]);
    assert_eq!(m.events(), 0);
    // The compositor lets go of the display it lost.
    m.send_ok(CMD_SET_SCANOUT, &[0, 0, 0, 0, 0, 0]);
    assert_eq!(m.events(), EVENT_DISPLAY, "plugged back in");
    assert_eq!(m.display(), [1024, 768, 1]);
    assert_eq!(m.edid_size(), 128);
}

#[test]
fn a_replug_before_the_guest_clears_the_unplug_is_raised_after() {
    let mut m = machine(None);
    m.gpu.control().set_modes(vec![Mode::new(1024, 768)]);
    assert_eq!(m.display(), [0, 0, 0]);
    m.send_ok(CMD_SET_SCANOUT, &[0, 0, 0, 0, 0, 0]);
    assert_eq!(m.events(), EVENT_DISPLAY);
    // Clearing the unplug's event must not lose the replug's.
    m.gpu.write_config(4, &[1, 0, 0, 0]);
    assert_eq!(m.events(), EVENT_DISPLAY, "raised again");
    m.gpu.write_config(4, &[1, 0, 0, 0]);
    assert_eq!(m.events(), 0);
    assert_eq!(m.display(), [1024, 768, 1]);
}

#[test]
fn a_display_turned_off_before_the_guest_saw_it_unplugged_stays_out() {
    let mut m = machine(None);
    m.gpu.control().set_modes(vec![Mode::new(1024, 768)]);
    m.send_ok(CMD_SET_SCANOUT, &[0, 0, 0, 0, 0, 0]);
    m.gpu.write_config(4, &[1, 0, 0, 0]);
    assert_eq!(m.events(), 0, "no replug yet");
    assert_eq!(m.display(), [0, 0, 0]);
}

#[test]
fn modes_that_arrive_while_unplugged_are_what_comes_back() {
    let mut m = machine(None);
    let control = m.gpu.control();
    control.set_modes(vec![Mode::new(800, 600), Mode::new(640, 480)]);
    assert_eq!(m.events(), 0, "the same modes change nothing");
    control.set_modes(vec![Mode::new(1024, 768)]);
    m.gpu.write_config(4, &[1, 0, 0, 0]);
    control.set_modes(vec![Mode::new(600, 800)]);
    control.set_modes(vec![Mode::new(768, 1024)]);
    assert_eq!(m.events(), 0, "one unplug");
    assert_eq!(m.display(), [0, 0, 0]);
    m.send_ok(CMD_SET_SCANOUT, &[0, 0, 0, 0, 0, 0]);
    assert_eq!(m.display(), [768, 1024, 1]);
}

#[test]
fn a_display_the_guest_keeps_comes_back_on_its_own() {
    let mut m = machine(None);
    m.gpu.control().set_modes(vec![Mode::new(1024, 768)]);
    assert_eq!(m.display(), [0, 0, 0]);
    m.gpu.write_config(4, &[1, 0, 0, 0]);
    let start = Instant::now();
    while m.events() == 0 {
        assert!(start.elapsed() < REPLUG_AFTER * 3, "never plugged back in");
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(start.elapsed() >= REPLUG_AFTER / 2);
    assert_eq!(m.display(), [1024, 768, 1]);
}

#[test]
fn modes_before_the_driver_starts_need_no_unplug() {
    let memory = memory();
    let mut gpu = Gpu::new(GpuConfig {
        modes: vec![Mode::new(800, 600)],
        sink: Arc::new(Screen::default()),
        renderer: None,
        host_visible_memory: None,
    })
    .unwrap();
    gpu.control().set_modes(vec![Mode::new(1024, 768)]);
    let mut events = [0u8; 4];
    gpu.read_config(0, &mut events);
    assert_eq!(events, [0; 4]);
    gpu.activate(Activation {
        memory: memory.clone(),
        queues: vec![
            Queue::new(memory.clone(), config(16)),
            Queue::new(memory.clone(), cursor_config()),
        ],
        interrupt: Interrupt::new(IrqLine::new(Arc::new(LevelRecorder::default()), 0)),
        features: F_VERSION_1,
    });
    let mut m = Machine {
        gpu,
        control: Driver::new(memory.clone(), 16),
        cursor: Driver::at(memory.clone(), cursor_config()),
        memory,
        screen: Arc::new(Screen::default()),
    };
    assert_eq!(m.display(), [1024, 768, 1]);
}

#[test]
fn shows_a_2d_resource_with_its_cursor() {
    let mut m = machine(None);
    // 4x2, B, G, R, A: pixel n is red n, green 10 + n, blue 20 + n.
    let pixels: Vec<u8> = (0..8u8).flat_map(|n| [20 + n, 10 + n, n, 255]).collect();
    m.resource(1, 4, 2, &pixels, DATA + 0x10000);
    m.send_ok(CMD_SET_SCANOUT, &[0, 0, 4, 2, 0, 1]);
    m.send_ok(CMD_RESOURCE_FLUSH, &[0, 0, 4, 2, 1, 0]);
    let (w, h, dirty, frame) = m.frame(1);
    assert_eq!((w, h, dirty), (4, 2, Rect::new(0, 0, 4, 2)));
    let rgba: Vec<u8> = (0..8u8).flat_map(|n| [n, 10 + n, 20 + n, 255]).collect();
    assert_eq!(frame, rgba);

    // A 2x2 cursor: an opaque white pixel, then three transparent ones.
    let mut cursor = vec![0u8; 16];
    cursor[0..4].copy_from_slice(&[255, 255, 255, 255]);
    m.resource(2, 2, 2, &cursor, DATA + 0x20000);
    m.cursor(CMD_UPDATE_CURSOR, 1, 1, 2);
    let (_, _, dirty, frame) = m.frame(2);
    assert_eq!(dirty, Rect::new(1, 1, 2, 1), "clipped to the screen");
    assert_eq!(&frame[20..24], &[255, 255, 255, 255], "pixel (1, 1)");
    assert_eq!(&frame[24..28], &rgba[24..28], "under a transparent pixel");

    m.cursor(CMD_MOVE_CURSOR, 0, 0, 0);
    let (_, _, dirty, frame) = m.frame(3);
    assert_eq!(dirty, Rect::new(0, 0, 3, 2), "where it was and where it is");
    assert_eq!(&frame[0..4], &[255, 255, 255, 255]);
    assert_eq!(&frame[20..24], &rgba[20..24], "the old place put back");

    // Nobody watching: a flush makes no frame, and the screen is caught up
    // on the next refresh.
    m.screen.active.store(false, Ordering::Relaxed);
    m.send_ok(CMD_RESOURCE_FLUSH, &[0, 0, 4, 2, 1, 0]);
    assert_eq!(m.screen.frames.lock().unwrap().len(), 3);
    m.screen.active.store(true, Ordering::Relaxed);
    m.gpu.control().refresh();
    assert_eq!(m.frame(4).2, Rect::new(0, 0, 4, 2));
}

#[test]
fn refuses_bad_commands() {
    let mut m = machine(None);
    let status = |answer: Vec<u8>| u32::from_le_bytes(answer[0..4].try_into().unwrap());
    let mut r = header(CMD_RESOURCE_CREATE_2D, 0, 0);
    r.extend(words(&[1, 999, 4, 4]));
    assert_eq!(
        status(m.send(&r)),
        RESP_ERR_INVALID_PARAMETER,
        "unknown format"
    );
    let mut r = header(CMD_SET_SCANOUT, 0, 0);
    r.extend(words(&[0, 0, 4, 2, 0, 7]));
    assert_eq!(status(m.send(&r)), RESP_ERR_INVALID_RESOURCE_ID);
    let mut r = header(CMD_RESOURCE_ATTACH_BACKING, 0, 0);
    m.send_ok(CMD_RESOURCE_CREATE_2D, &[1, format::B8G8R8X8, 4, 4]);
    r.extend(words(&[1, 1, 0x10, 0, 64, 0]));
    assert_eq!(
        status(m.send(&r)),
        RESP_ERR_INVALID_PARAMETER,
        "not guest RAM"
    );
    let mut r = header(CMD_TRANSFER_TO_HOST_2D, 0, 0);
    r.extend(words(&[0, 0, 4, 4, 0, 0, 1, 0]));
    assert_eq!(status(m.send(&r)), RESP_ERR_INVALID_PARAMETER, "no backing");
    let mut r = header(CMD_SUBMIT_3D, 0, 0);
    r.extend(words(&[0, 0]));
    assert_eq!(status(m.send(&r)), RESP_ERR_UNSPEC, "no 3D");
    // A fence without a renderer answers at once, echoing it.
    let answer = m.send(&header(CMD_GET_DISPLAY_INFO, FLAG_FENCE, 77));
    assert_eq!(&answer[4..16], &[1, 0, 0, 0, 77, 0, 0, 0, 0, 0, 0, 0]);
}

/// A renderer whose fences retire when the test says.
#[derive(Default)]
struct FakeState {
    fences: Vec<u64>,
    retire: usize,
    submitted: Vec<u32>,
    contexts: Vec<(u32, u32)>,
    context_names: Vec<Vec<u8>>,
    context_fences: Vec<(u32, u32, u64)>,
    retired_context_fences: Vec<ContextFence>,
    blobs: Vec<BlobResource>,
    attached: Vec<u32>,
    detached: Vec<u32>,
    map_data: Vec<u8>,
}

struct Fake(Arc<Mutex<FakeState>>);

impl Renderer for Fake {
    fn description(&self) -> String {
        "fake".into()
    }
    fn supports_blob(&self) -> bool {
        true
    }
    fn blob_alignment(&self) -> u32 {
        4_096
    }
    fn create_blob(&mut self, blob: &BlobResource, _backing: &[HostSlice]) -> RendererResult<()> {
        self.0.lock().unwrap().blobs.push(*blob);
        Ok(())
    }
    fn map_blob_fixed(&mut self, _resource: u32, host_addr: *mut u8) -> RendererResult<bool> {
        let data = self.0.lock().unwrap().map_data.clone();
        if !data.is_empty() {
            // SAFETY: the test maps a blob larger than this test pixel payload.
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), host_addr, data.len()) };
        }
        Ok(true)
    }
    fn blob_map_info(&mut self, _resource: u32) -> RendererResult<u32> {
        Ok(1)
    }
    fn capsets(&self) -> Vec<Capset> {
        vec![
            Capset {
                id: 1,
                max_version: 1,
                max_size: 8,
            },
            Capset {
                id: 4,
                max_version: 1,
                max_size: 32,
            },
        ]
    }
    fn fill_capset(&mut self, _id: u32, _version: u32, out: &mut [u8]) {
        out.fill(0xcc);
    }
    fn create_context(&mut self, context: u32, context_init: u32, name: &[u8]) -> RendererResult<()> {
        let mut state = self.0.lock().unwrap();
        state.contexts.push((context, context_init));
        state.context_names.push(name.to_vec());
        Ok(())
    }
    fn create_context_fence(
        &mut self,
        context: u32,
        _flags: u32,
        ring_idx: u32,
        fence: u64,
    ) -> RendererResult<()> {
        self.0.lock().unwrap().context_fences.push((context, ring_idx, fence));
        Ok(())
    }
    fn poll_context_fences(&mut self) -> Vec<ContextFence> {
        std::mem::take(&mut self.0.lock().unwrap().retired_context_fences)
    }
    fn destroy_context(&mut self, _context: u32) {}
    fn attach_resource(&mut self, _context: u32, _resource: u32) {}
    fn detach_resource(&mut self, _context: u32, _resource: u32) {}
    fn create_resource(&mut self, _resource: &Resource3d) -> RendererResult<()> {
        Ok(())
    }
    fn destroy_resource(&mut self, _resource: u32) {}
    unsafe fn attach_backing(&mut self, resource: u32, _backing: Vec<HostSlice>) -> RendererResult<()> {
        self.0.lock().unwrap().attached.push(resource);
        Ok(())
    }
    fn detach_backing(&mut self, resource: u32) {
        self.0.lock().unwrap().detached.push(resource);
    }
    fn transfer_to_host(&mut self, _t: &Transfer) -> RendererResult<()> {
        Ok(())
    }
    fn transfer_from_host(&mut self, _t: &Transfer) -> RendererResult<()> {
        Ok(())
    }
    fn submit(&mut self, _context: u32, commands: &mut [u32]) -> RendererResult<()> {
        self.0.lock().unwrap().submitted.extend_from_slice(commands);
        Ok(())
    }
    fn create_fence(&mut self, fence: u64, _context: u32) -> RendererResult<()> {
        self.0.lock().unwrap().fences.push(fence);
        Ok(())
    }
    fn poll(&mut self) -> Vec<u64> {
        let mut s = self.0.lock().unwrap();
        let retire = s.retire.min(s.fences.len());
        s.retire = 0;
        s.fences.drain(..retire).collect()
    }
    fn read(&mut self, _: u32, _: Rect, out: &mut [u8], _: u32) -> RendererResult<()> {
        out.fill(7);
        Ok(())
    }
    fn cursor_image(&mut self, _resource: u32) -> Option<(u32, u32, Vec<u8>)> {
        None
    }
    fn reset(&mut self) {}
}

#[test]
fn answers_fences_once_the_renderer_retires_them() {
    let state = Arc::new(Mutex::new(FakeState::default()));
    let fake = state.clone();
    let mut m = machine(Some(Box::new(move || {
        Ok(Box::new(Fake(fake)) as Box<dyn Renderer>)
    })));
    assert!(m.gpu.has_3d());
    let mut capsets = [0u8; 4];
    m.gpu.read_config(12, &mut capsets);
    assert_eq!(u32::from_le_bytes(capsets), 1, "Venus is hidden without host-visible blobs");
    assert_eq!(m.gpu.features & F_CONTEXT_INIT, 0);

    let mut r = header(CMD_GET_CAPSET, 0, 0);
    r.extend(words(&[1, 1]));
    let answer = m.send(&r);
    assert_eq!(&answer[0..4], &RESP_OK_CAPSET.to_le_bytes());
    assert_eq!(&answer[HEADER..], &[0xcc; 8]);

    let mut submit = header(CMD_SUBMIT_3D, FLAG_FENCE, 5);
    submit.extend(words(&[8, 0, 0xdead, 0xbeef]));
    m.send_at(&submit, 1);
    let mut second = header(CMD_SUBMIT_3D, FLAG_FENCE, 6);
    second.extend(words(&[4, 0, 0xf00d]));
    m.send_at(&second, 2);
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(m.control.take_used(), None, "held until the fence retires");
    assert_eq!(state.lock().unwrap().submitted, [0xdead, 0xbeef, 0xf00d]);
    assert_eq!(state.lock().unwrap().fences, [5, 6]);

    state.lock().unwrap().retire = 1;
    assert!(wait_used(&mut m.control).is_some());
    std::thread::sleep(Duration::from_millis(20));
    assert_eq!(m.control.take_used(), None, "the second still waits");
    state.lock().unwrap().retire = 1;
    assert!(wait_used(&mut m.control).is_some());

    // A 3D scanout is read back through the renderer.
    m.send_ok(
        CMD_RESOURCE_CREATE_3D,
        &[9, 2, format::R8G8B8A8, 0, 2, 2, 1, 1, 0, 0, 0, 0],
    );
    m.send_ok(CMD_SET_SCANOUT, &[0, 0, 2, 2, 0, 9]);
    m.send_ok(CMD_RESOURCE_FLUSH, &[0, 0, 2, 2, 9, 0]);
    assert_eq!(m.frame(1).3, vec![7; 16]);
}

#[test]
fn a_reset_forgets_everything() {
    let mut m = machine(None);
    m.send_ok(CMD_RESOURCE_CREATE_2D, &[1, format::B8G8R8X8, 4, 4]);
    m.gpu.reset();
    // A fresh start: the same id is free again.
    m.gpu.activate(Activation {
        memory: m.memory.clone(),
        queues: vec![
            Queue::new(m.memory.clone(), config(16)),
            Queue::new(m.memory.clone(), cursor_config()),
        ],
        interrupt: Interrupt::new(IrqLine::new(Arc::new(LevelRecorder::default()), 0)),
        features: F_VERSION_1,
    });
    m.control = Driver::new(m.memory.clone(), 16);
    m.memory.fill(RAM_BASE, 0x3000, 0).unwrap();
    m.send_ok(CMD_RESOURCE_CREATE_2D, &[1, format::B8G8R8X8, 4, 4]);
}

fn create_venus_context(m: &mut Machine, context: u32) {
    let mut create = header(CMD_CTX_CREATE, 0, 0);
    create[16..20].copy_from_slice(&context.to_le_bytes());
    create.extend(words(&[0, CAPSET_VENUS]));
    let answer = m.send(&create);
    assert_eq!(
        u32::from_le_bytes(answer[0..4].try_into().unwrap()),
        RESP_OK_NODATA
    );
}

fn submit_venus_fence(m: &mut Machine, context: u32, ring: u8, fence: u64, slot: u64) {
    let mut submit = header(CMD_SUBMIT_3D, FLAG_FENCE | FLAG_INFO_RING_IDX, fence);
    submit[16..20].copy_from_slice(&context.to_le_bytes());
    submit[20] = ring;
    submit.extend(words(&[4, 0, 0xfeed]));
    m.send_at(&submit, slot);
}

#[test]
fn legacy_context_name_uses_the_non_context_init_layout() {
    let state = Arc::new(Mutex::new(FakeState::default()));
    let fake = state.clone();
    let mut m = machine(Some(Box::new(move || Ok(Box::new(Fake(fake)) as Box<dyn Renderer>))));
    assert_eq!(m.gpu.features & F_CONTEXT_INIT, 0);

    let mut create = header(CMD_CTX_CREATE, 0, 0);
    create[16..20].copy_from_slice(&1u32.to_le_bytes());
    create.extend(4u32.to_le_bytes());
    create.extend_from_slice(b"test");
    let answer = m.send(&create);
    assert_eq!(u32::from_le_bytes(answer[0..4].try_into().unwrap()), RESP_OK_NODATA);
    assert_eq!(state.lock().unwrap().context_names, vec![b"test".to_vec()]);
}

#[test]
fn venus_capset_does_not_force_context_init_without_feature_negotiation() {
    let state = Arc::new(Mutex::new(FakeState::default()));
    let fake = state.clone();
    let host = Arc::new(TestHostVisibleMemory::new(HOST_VISIBLE_BASE, HOST_VISIBLE_SIZE));
    let mut m = machine_with_host_visible_features(
        Some(Box::new(move || Ok(Box::new(Fake(fake)) as Box<dyn Renderer>))),
        host,
        Some(F_VERSION_1),
    );
    assert_ne!(m.gpu.features & F_CONTEXT_INIT, 0, "the feature is offered");
    let mut create = header(CMD_CTX_CREATE, 0, 0);
    create[16..20].copy_from_slice(&1u32.to_le_bytes());
    create.extend(4u32.to_le_bytes());
    create.extend_from_slice(b"test");
    let answer = m.send(&create);
    assert_eq!(u32::from_le_bytes(answer[0..4].try_into().unwrap()), RESP_OK_NODATA);
    {
        let state = state.lock().unwrap();
        assert_eq!(state.contexts, vec![(1, 0)]);
        assert_eq!(state.context_names, vec![b"test".to_vec()]);
    }

    // The ring-index flag is invalid unless F_CONTEXT_INIT was negotiated.
    // Reject it before submitting any guest command to the renderer.
    let mut submit = header(CMD_SUBMIT_3D, FLAG_FENCE | FLAG_INFO_RING_IDX, 7);
    submit[16..20].copy_from_slice(&1u32.to_le_bytes());
    submit[20] = 3;
    submit.extend(words(&[4, 0, 0xfeed]));
    let answer = m.send(&submit);
    assert_eq!(
        u32::from_le_bytes(answer[0..4].try_into().unwrap()),
        RESP_ERR_INVALID_PARAMETER
    );
    let state = state.lock().unwrap();
    assert!(state.context_fences.is_empty());
    assert!(state.submitted.is_empty());
}

#[test]
fn venus_context_init_and_timeline_fences_are_forwarded() {
    let state = Arc::new(Mutex::new(FakeState::default()));
    let fake = state.clone();
    let host = Arc::new(TestHostVisibleMemory::new(HOST_VISIBLE_BASE, HOST_VISIBLE_SIZE));
    let mut m = machine_with_host_visible(
        Some(Box::new(move || Ok(Box::new(Fake(fake)) as Box<dyn Renderer>))),
        host,
    );
    assert_ne!(m.gpu.features & F_CONTEXT_INIT, 0);
    assert_ne!(m.gpu.features & F_RESOURCE_BLOB, 0);
    assert_ne!(m.gpu.features & F_BLOB_ALIGNMENT, 0);

    create_venus_context(&mut m, 1);
    create_venus_context(&mut m, 2);
    assert_eq!(
        state.lock().unwrap().contexts,
        vec![(1, CAPSET_VENUS), (2, CAPSET_VENUS)]
    );

    submit_venus_fence(&mut m, 1, 3, 41, 1);
    submit_venus_fence(&mut m, 1, 4, 42, 2);
    submit_venus_fence(&mut m, 2, 3, 43, 3);
    submit_venus_fence(&mut m, 1, 3, 44, 4);
    let start = Instant::now();
    while state.lock().unwrap().context_fences.len() != 4 {
        assert!(start.elapsed() < Duration::from_secs(5), "Venus fences were not submitted");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(
        state.lock().unwrap().context_fences,
        vec![(1, 3, 41), (1, 4, 42), (2, 3, 43), (1, 3, 44)]
    );

    // A completion on context 1/ring 3 retires only that timeline, through 44.
    state.lock().unwrap().retired_context_fences.push(ContextFence {
        context: 1,
        ring_idx: 3,
        fence: 44,
    });
    assert!(wait_used(&mut m.control).is_some());
    assert!(wait_used(&mut m.control).is_some());
    assert_eq!(m.control.take_used(), None, "other timelines remain pending");

    state.lock().unwrap().retired_context_fences.push(ContextFence {
        context: 1,
        ring_idx: 4,
        fence: 42,
    });
    assert!(wait_used(&mut m.control).is_some());
    assert_eq!(m.control.take_used(), None);

    state.lock().unwrap().retired_context_fences.push(ContextFence {
        context: 2,
        ring_idx: 3,
        fence: 43,
    });
    assert!(wait_used(&mut m.control).is_some());
    assert_eq!(m.control.take_used(), None);
}

#[test]
fn venus_host_blob_maps_and_scans_out() {
    let pixels = vec![1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255, 10, 11, 12, 255];
    let state = Arc::new(Mutex::new(FakeState {
        map_data: pixels.clone(),
        ..FakeState::default()
    }));
    let fake = state.clone();
    let host = Arc::new(TestHostVisibleMemory::new(HOST_VISIBLE_BASE, HOST_VISIBLE_SIZE));
    let host_check = host.clone();
    let mut m = machine_with_host_visible(
        Some(Box::new(move || Ok(Box::new(Fake(fake)) as Box<dyn Renderer>))),
        host,
    );
    assert_ne!(m.gpu.features & F_RESOURCE_BLOB, 0);
    assert_ne!(m.gpu.features & F_BLOB_ALIGNMENT, 0);

    let mut create = header(CMD_RESOURCE_CREATE_BLOB, 0, 0);
    create.extend(words(&[11, 2, 1, 0]));
    create.extend(0x1234u64.to_le_bytes());
    create.extend(16_384u64.to_le_bytes());
    let answer = m.send(&create);
    assert_eq!(u32::from_le_bytes(answer[0..4].try_into().unwrap()), RESP_OK_NODATA);

    let mut map = header(CMD_RESOURCE_MAP_BLOB, 0, 0);
    map.extend(words(&[11, 0]));
    map.extend(0u64.to_le_bytes());
    let answer = m.send(&map);
    assert_eq!(u32::from_le_bytes(answer[0..4].try_into().unwrap()), RESP_OK_MAP_INFO);
    assert_eq!(
        u32::from_le_bytes(answer[HEADER..HEADER + 4].try_into().unwrap()),
        1
    );

    m.send_ok(
        CMD_SET_SCANOUT_BLOB,
        &[0, 0, 2, 2, 0, 11, 2, 2, format::R8G8B8A8, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    );
    m.send_ok(CMD_RESOURCE_FLUSH, &[0, 0, 2, 2, 11, 0]);
    assert_eq!(m.frame(1).3, pixels);
    m.send_ok(CMD_RESOURCE_UNMAP_BLOB, &[11, 0]);
    assert!(host_check.mappings.lock().unwrap().is_empty());
    m.send_ok(CMD_RESOURCE_UNREF, &[11]);
    assert_eq!(state.lock().unwrap().blobs[0].memory, 2);
}

#[test]
fn venus_default_blob_can_attach_and_detach_backing_later() {
    let state = Arc::new(Mutex::new(FakeState::default()));
    let fake = state.clone();
    let host = Arc::new(TestHostVisibleMemory::new(HOST_VISIBLE_BASE, HOST_VISIBLE_SIZE));
    let mut m = machine_with_host_visible(
        Some(Box::new(move || Ok(Box::new(Fake(fake)) as Box<dyn Renderer>))),
        host,
    );

    let mut create = header(CMD_RESOURCE_CREATE_BLOB, 0, 0);
    create.extend(words(&[12, 3, 0, 0]));
    create.extend(0x5678u64.to_le_bytes());
    create.extend(16_384u64.to_le_bytes());
    let answer = m.send(&create);
    assert_eq!(u32::from_le_bytes(answer[0..4].try_into().unwrap()), RESP_OK_NODATA);

    let mut attach = header(CMD_RESOURCE_ATTACH_BACKING, 0, 0);
    attach.extend(words(&[12, 1]));
    attach.extend((DATA + 0x20_000).to_le_bytes());
    attach.extend(words(&[16_384, 0]));
    let answer = m.send(&attach);
    assert_eq!(u32::from_le_bytes(answer[0..4].try_into().unwrap()), RESP_OK_NODATA);
    m.send_ok(CMD_RESOURCE_DETACH_BACKING, &[12, 0]);
    m.send_ok(CMD_RESOURCE_UNREF, &[12]);

    let state = state.lock().unwrap();
    assert_eq!(state.blobs[0].memory, 3);
    assert_eq!(state.attached, vec![12]);
    assert_eq!(state.detached, vec![12]);
}

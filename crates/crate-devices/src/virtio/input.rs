// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! virtio-input (VIRTIO 1.2, section 5.8): a keyboard, an absolute tablet and
//! a relative mouse, each its own device, passing Linux input events (type, code, value) to the
//! guest's evdev.
//!
//! The configuration space answers the driver's questions one at a time: it
//! writes what it wants to know (`select`, `subsel`) and reads the size and
//! the answer. The event queue carries 8-byte events to the guest; events the
//! guest has posted no buffers for wait in a small backlog. The status queue
//! (LEDs, from the guest) is read and dropped.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate_core::display::{BUTTON_LEFT, BUTTON_MIDDLE, BUTTON_RIGHT, InputSink, POINTER_MAX};
use crate_core::warn;

use super::{Activation, F_VERSION_1, Interrupt, Queue, VirtioDevice, read_config_bytes};

pub const ID_INPUT: u32 = 18;

const QUEUE_SIZE: u16 = 64;
/// Events kept while the guest has no buffers posted; past this new ones are
/// dropped, as a full ring drops them.
const BACKLOG: usize = 1024;

const CFG_ID_NAME: u8 = 0x01;
const CFG_ID_SERIAL: u8 = 0x02;
const CFG_ID_DEVIDS: u8 = 0x03;
const CFG_PROP_BITS: u8 = 0x10;
const CFG_EV_BITS: u8 = 0x11;
const CFG_ABS_INFO: u8 = 0x12;

// Linux's input event codes (include/uapi/linux/input-event-codes.h).
pub const EV_SYN: u16 = 0x00;
pub const EV_KEY: u16 = 0x01;
pub const EV_REL: u16 = 0x02;
pub const EV_ABS: u16 = 0x03;
pub const EV_REP: u16 = 0x14;
pub const SYN_REPORT: u16 = 0;
pub const REL_X: u16 = 0x00;
pub const REL_Y: u16 = 0x01;
pub const REL_HWHEEL: u16 = 0x06;
pub const REL_WHEEL: u16 = 0x08;
pub const ABS_X: u16 = 0x00;
pub const ABS_Y: u16 = 0x01;
pub const BTN_LEFT: u16 = 0x110;
pub const BTN_RIGHT: u16 = 0x111;
pub const BTN_MIDDLE: u16 = 0x112;
/// The keyboard offers every key code from 1 (KEY_ESC) to this (KEY_MICMUTE).
const LAST_KEY: u16 = 248;

const BUS_VIRTUAL: u16 = 0x06;
const VENDOR: u16 = 0x6372; // "cr"

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputKind {
    Keyboard,
    /// An absolute pointer, 0 to POINTER_MAX on each axis, with three buttons
    /// and both wheels.
    Tablet,
    /// A relative mouse with three buttons and both wheels.
    Mouse,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Event {
    pub kind: u16,
    pub code: u16,
    pub value: i32,
}

impl Event {
    pub const fn new(kind: u16, code: u16, value: i32) -> Self {
        Event { kind, code, value }
    }

    fn bytes(&self) -> [u8; 8] {
        let mut b = [0; 8];
        b[0..2].copy_from_slice(&self.kind.to_le_bytes());
        b[2..4].copy_from_slice(&self.code.to_le_bytes());
        b[4..8].copy_from_slice(&self.value.to_le_bytes());
        b
    }
}

#[derive(Default)]
struct EventQueue {
    queue: Option<(Queue, Arc<Interrupt>)>,
    backlog: VecDeque<Event>,
}

impl EventQueue {
    /// Moves backlogged events into buffers the guest posted, while there are
    /// both.
    fn flush(&mut self) {
        let Some((queue, interrupt)) = &mut self.queue else {
            return;
        };
        let mut used = false;
        while let Some(event) = self.backlog.front() {
            let chain = match queue.pop() {
                Ok(Some(chain)) => chain,
                Ok(None) => break,
                Err(e) => {
                    warn!("virtio-input: {e}");
                    break;
                }
            };
            let n = chain.write_all(queue.memory(), &event.bytes()).unwrap_or(0);
            if queue.add_used(chain.head, n as u32).is_err() {
                break;
            }
            self.backlog.pop_front();
            used = true;
        }
        if used && queue.needs_interrupt() {
            interrupt.signal_used();
        }
    }
}

/// Sends events to one input device; any thread may.
#[derive(Clone, Default)]
pub struct EventSender(Arc<Mutex<EventQueue>>);

impl EventSender {
    /// Sends `events` followed by a SYN_REPORT, which tells the guest they
    /// belong together.
    pub fn send(&self, events: &[Event]) {
        let mut q = self.0.lock().unwrap();
        if q.queue.is_none() {
            // No driver yet: nobody would see them.
            return;
        }
        if q.backlog.len() + events.len() + 1 > BACKLOG {
            return;
        }
        q.backlog.extend(events.iter().copied());
        q.backlog.push_back(Event::new(EV_SYN, SYN_REPORT, 0));
        q.flush();
    }
}

pub struct Input {
    kind: InputKind,
    name: String,
    select: u8,
    subsel: u8,
    events: EventSender,
    status: Option<Queue>,
}

impl Input {
    pub fn new(kind: InputKind, name: &str) -> Self {
        Input {
            kind,
            name: name.into(),
            select: 0,
            subsel: 0,
            events: EventSender::default(),
            status: None,
        }
    }

    pub fn sender(&self) -> EventSender {
        self.events.clone()
    }

    /// The answer to the current question, as (bytes); empty when there is
    /// none, which the driver reads as a size of 0.
    fn answer(&self) -> Vec<u8> {
        match (self.select, self.subsel) {
            (CFG_ID_NAME, 0) => self.name.as_bytes().iter().take(128).copied().collect(),
            (CFG_ID_SERIAL, 0) => b"0".to_vec(),
            (CFG_ID_DEVIDS, 0) => {
                let product: u16 = match self.kind {
                    InputKind::Keyboard => 1,
                    InputKind::Tablet => 2,
                    InputKind::Mouse => 3,
                };
                [BUS_VIRTUAL, VENDOR, product, 1]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect()
            }
            (CFG_PROP_BITS, 0) => Vec::new(),
            (CFG_EV_BITS, kind) => bitmap(&self.codes(kind as u16)),
            (CFG_ABS_INFO, axis) if self.kind == InputKind::Tablet => {
                if axis as u16 == ABS_X || axis as u16 == ABS_Y {
                    // min, max, fuzz, flat, resolution
                    [0, POINTER_MAX, 0, 0, 0]
                        .iter()
                        .flat_map(|v: &u32| v.to_le_bytes())
                        .collect()
                } else {
                    Vec::new()
                }
            }
            _ => Vec::new(),
        }
    }

    /// The codes of event type `kind` the device can send.
    fn codes(&self, kind: u16) -> Vec<u16> {
        match (self.kind, kind) {
            (InputKind::Keyboard, EV_KEY) => (1..=LAST_KEY).collect(),
            // Any bit: the driver asks only whether the answer is empty, and
            // then the guest repeats held keys itself.
            (InputKind::Keyboard, EV_REP) => vec![0],
            (InputKind::Tablet, EV_KEY) | (InputKind::Mouse, EV_KEY) => {
                vec![BTN_LEFT, BTN_RIGHT, BTN_MIDDLE]
            }
            (InputKind::Tablet, EV_REL) => vec![REL_HWHEEL, REL_WHEEL],
            (InputKind::Mouse, EV_REL) => vec![REL_X, REL_Y, REL_HWHEEL, REL_WHEEL],
            (InputKind::Tablet, EV_ABS) => vec![ABS_X, ABS_Y],
            _ => Vec::new(),
        }
    }

    fn config(&self) -> Vec<u8> {
        let answer = self.answer();
        let mut cfg = vec![0u8; 8 + 128];
        cfg[0] = self.select;
        cfg[1] = self.subsel;
        cfg[2] = answer.len() as u8;
        cfg[8..8 + answer.len()].copy_from_slice(&answer);
        cfg
    }
}

/// A bitmap with the given bits set, trimmed after its last set byte.
fn bitmap(bits: &[u16]) -> Vec<u8> {
    let Some(&last) = bits.iter().max() else {
        return Vec::new();
    };
    let mut map = vec![0u8; last as usize / 8 + 1];
    for &bit in bits {
        map[bit as usize / 8] |= 1 << (bit % 8);
    }
    map
}

impl VirtioDevice for Input {
    fn device_id(&self) -> u32 {
        ID_INPUT
    }

    fn features(&self) -> u64 {
        F_VERSION_1
    }

    fn queue_max_sizes(&self) -> &[u16] {
        &[QUEUE_SIZE, QUEUE_SIZE]
    }

    fn read_config(&self, offset: u64, data: &mut [u8]) {
        read_config_bytes(&self.config(), offset, data);
    }

    fn write_config(&mut self, offset: u64, data: &[u8]) {
        for (i, &byte) in data.iter().enumerate() {
            match offset + i as u64 {
                0 => self.select = byte,
                1 => self.subsel = byte,
                _ => {}
            }
        }
    }

    fn activate(&mut self, activation: Activation) {
        let mut queues = activation.queues.into_iter();
        let (Some(events), Some(status)) = (queues.next(), queues.next()) else {
            return;
        };
        let mut q = self.events.0.lock().unwrap();
        q.queue = Some((events, activation.interrupt));
        q.backlog.clear();
        self.status = Some(status);
    }

    fn notify(&mut self, index: u16) {
        match index {
            0 => self.events.0.lock().unwrap().flush(),
            1 => {
                let Some(status) = &mut self.status else {
                    return;
                };
                // LED states and the like; nothing here has LEDs to light.
                while let Ok(Some(chain)) = status.pop() {
                    if status.add_used(chain.head, 0).is_err() {
                        break;
                    }
                }
            }
            _ => {}
        }
    }

    fn reset(&mut self) {
        let mut q = self.events.0.lock().unwrap();
        q.queue = None;
        q.backlog.clear();
        self.status = None;
        self.select = 0;
        self.subsel = 0;
    }
}

/// A keyboard, absolute tablet and relative mouse behind `InputSink`.
pub struct Inputs {
    keyboard: Option<EventSender>,
    tablet: Option<EventSender>,
    mouse: Option<EventSender>,
    tablet_buttons: Mutex<u32>,
    mouse_buttons: Mutex<u32>,
}

impl Inputs {
    pub fn new(
        keyboard: Option<EventSender>,
        tablet: Option<EventSender>,
        mouse: Option<EventSender>,
    ) -> Self {
        Inputs {
            keyboard,
            tablet,
            mouse,
            tablet_buttons: Mutex::new(0),
            mouse_buttons: Mutex::new(0),
        }
    }
}

impl InputSink for Inputs {
    fn key(&self, code: u16, down: bool) {
        if let Some(keyboard) = &self.keyboard {
            if code != 0 && code <= LAST_KEY {
                keyboard.send(&[Event::new(EV_KEY, code, down as i32)]);
            }
        }
    }

    fn pointer(&self, x: u32, y: u32, buttons: u32) {
        let Some(tablet) = &self.tablet else {
            return;
        };
        let mut held = self.tablet_buttons.lock().unwrap();
        let mut events = vec![
            Event::new(EV_ABS, ABS_X, x.min(POINTER_MAX) as i32),
            Event::new(EV_ABS, ABS_Y, y.min(POINTER_MAX) as i32),
        ];
        for (bit, code) in [
            (BUTTON_LEFT, BTN_LEFT),
            (BUTTON_RIGHT, BTN_RIGHT),
            (BUTTON_MIDDLE, BTN_MIDDLE),
        ] {
            if (*held ^ buttons) & bit != 0 {
                events.push(Event::new(EV_KEY, code, (buttons & bit != 0) as i32));
            }
        }
        *held = buttons;
        tablet.send(&events);
    }

    fn mouse(&self, dx: i32, dy: i32, buttons: u32) {
        let Some(mouse) = &self.mouse else {
            return;
        };
        let mut held = self.mouse_buttons.lock().unwrap();
        let mut events = Vec::new();
        if dx != 0 {
            events.push(Event::new(EV_REL, REL_X, dx));
        }
        if dy != 0 {
            events.push(Event::new(EV_REL, REL_Y, dy));
        }
        for (bit, code) in [
            (BUTTON_LEFT, BTN_LEFT),
            (BUTTON_RIGHT, BTN_RIGHT),
            (BUTTON_MIDDLE, BTN_MIDDLE),
        ] {
            if (*held ^ buttons) & bit != 0 {
                events.push(Event::new(EV_KEY, code, (buttons & bit != 0) as i32));
            }
        }
        *held = buttons;
        if !events.is_empty() {
            mouse.send(&events);
        }
    }

    fn scroll(&self, dx: i32, dy: i32) {
        let device = self.mouse.as_ref().or(self.tablet.as_ref());
        let Some(device) = device else {
            return;
        };
        let mut events = Vec::new();
        if dy != 0 {
            // Linux's wheel counts up as positive.
            events.push(Event::new(EV_REL, REL_WHEEL, dy.saturating_neg()));
        }
        if dx != 0 {
            events.push(Event::new(EV_REL, REL_HWHEEL, dx));
        }
        if !events.is_empty() {
            device.send(&events);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::queue::testing::*;
    use super::*;
    use crate_core::irq::{IrqLine, LevelRecorder};

    fn ask(dev: &mut Input, select: u8, subsel: u8) -> Vec<u8> {
        dev.write_config(0, &[select, subsel]);
        let mut size = [0u8];
        dev.read_config(2, &mut size);
        let mut data = vec![0u8; size[0] as usize];
        dev.read_config(8, &mut data);
        data
    }

    #[test]
    fn answers_the_drivers_questions() {
        let mut tablet = Input::new(InputKind::Tablet, "crate tablet");
        assert_eq!(ask(&mut tablet, CFG_ID_NAME, 0), b"crate tablet");
        assert_eq!(
            ask(&mut tablet, CFG_ID_DEVIDS, 0),
            [6, 0, 0x72, 0x63, 2, 0, 1, 0]
        );
        // EV_ABS: X and Y.
        assert_eq!(ask(&mut tablet, CFG_EV_BITS, EV_ABS as u8), [0b11]);
        let abs = ask(&mut tablet, CFG_ABS_INFO, ABS_Y as u8);
        assert_eq!(abs.len(), 20);
        assert_eq!(&abs[4..8], &32767u32.to_le_bytes());
        let keys = ask(&mut tablet, CFG_EV_BITS, EV_KEY as u8);
        assert_eq!(keys.len(), 0x112 / 8 + 1);
        assert_eq!(keys[0x110 / 8], 0b111);
        assert!(ask(&mut tablet, CFG_EV_BITS, EV_REP as u8).is_empty());

        let mut keyboard = Input::new(InputKind::Keyboard, "crate keyboard");
        let keys = ask(&mut keyboard, CFG_EV_BITS, EV_KEY as u8);
        assert_eq!(keys[0], 0xfe, "KEY_ESC is 1; 0 is reserved");
        assert_eq!(keys[30 / 8] & (1 << (30 % 8)), 1 << (30 % 8), "KEY_A");
        assert!(!ask(&mut keyboard, CFG_EV_BITS, EV_REP as u8).is_empty());
        assert!(ask(&mut keyboard, CFG_EV_BITS, EV_ABS as u8).is_empty());
        assert!(ask(&mut keyboard, CFG_ABS_INFO, 0).is_empty());

        let mut mouse = Input::new(InputKind::Mouse, "crate mouse");
        assert_eq!(ask(&mut mouse, CFG_ID_DEVIDS, 0), [6, 0, 0x72, 0x63, 3, 0, 1, 0]);
        assert_eq!(ask(&mut mouse, CFG_EV_BITS, EV_REL as u8), [0b0100_0011, 0b0000_0001]);
        let mouse_keys = ask(&mut mouse, CFG_EV_BITS, EV_KEY as u8);
        assert_eq!(mouse_keys[BTN_LEFT as usize / 8] & 0b111, 0b111);
        assert!(ask(&mut mouse, CFG_EV_BITS, EV_ABS as u8).is_empty());
    }

    #[test]
    fn delivers_events_and_keeps_a_backlog() {
        let memory = memory();
        let mut dev = Input::new(InputKind::Tablet, "t");
        let inputs = Inputs::new(None, Some(dev.sender()), None);
        inputs.pointer(1, 2, 0);
        dev.activate(Activation {
            memory: memory.clone(),
            queues: vec![
                Queue::new(memory.clone(), config(8)),
                Queue::new(memory.clone(), config(8)),
            ],
            interrupt: Interrupt::new(IrqLine::new(Arc::new(LevelRecorder::default()), 0)),
            features: F_VERSION_1,
        });
        let mut driver = Driver::new(memory.clone(), 8);
        for i in 0..2 {
            driver.offer(&[(DATA + 8 * i, 8, true)]);
        }
        // Press: X, Y, BTN_LEFT, SYN, with room for two.
        inputs.pointer(100, 200, BUTTON_LEFT);
        let mut event = [0u8; 16];
        memory.read(DATA, &mut event).unwrap();
        assert_eq!(&event[0..8], &Event::new(EV_ABS, ABS_X, 100).bytes());
        assert_eq!(&event[8..16], &Event::new(EV_ABS, ABS_Y, 200).bytes());
        for i in 2..4 {
            driver.offer(&[(DATA + 8 * i, 8, true)]);
        }
        dev.notify(0);
        memory.read(DATA + 16, &mut event).unwrap();
        assert_eq!(&event[0..8], &Event::new(EV_KEY, BTN_LEFT, 1).bytes());
        assert_eq!(&event[8..16], &Event::new(EV_SYN, SYN_REPORT, 0).bytes());
        // The same buttons again send no key events; a scroll down is a
        // negative wheel.
        for i in 4..8 {
            driver.offer(&[(DATA + 8 * i, 8, true)]);
        }
        inputs.pointer(100, 200, BUTTON_LEFT);
        inputs.scroll(0, 1);
        let mut rest = [0u8; 16];
        memory.read(DATA + 48, &mut rest).unwrap();
        assert_eq!(&rest[0..8], &Event::new(EV_SYN, SYN_REPORT, 0).bytes());
        assert_eq!(&rest[8..16], &Event::new(EV_REL, REL_WHEEL, -1).bytes());
    }
    #[test]
    fn delivers_relative_mouse_motion_and_buttons() {
        let memory = memory();
        let mut dev = Input::new(InputKind::Mouse, "crate mouse");
        let inputs = Inputs::new(None, None, Some(dev.sender()));
        dev.activate(Activation {
            memory: memory.clone(),
            queues: vec![
                Queue::new(memory.clone(), config(8)),
                Queue::new(memory.clone(), config(8)),
            ],
            interrupt: Interrupt::new(IrqLine::new(Arc::new(LevelRecorder::default()), 0)),
            features: F_VERSION_1,
        });
        let mut driver = Driver::new(memory.clone(), 8);
        for i in 0..4 {
            driver.offer(&[(DATA + 8 * i, 8, true)]);
        }
        inputs.mouse(12, -7, BUTTON_LEFT);
        let mut bytes = [0u8; 32];
        memory.read(DATA, &mut bytes).unwrap();
        let expected = [
            Event::new(EV_REL, REL_X, 12),
            Event::new(EV_REL, REL_Y, -7),
            Event::new(EV_KEY, BTN_LEFT, 1),
            Event::new(EV_SYN, SYN_REPORT, 0),
        ];
        for (i, event) in expected.iter().enumerate() {
            assert_eq!(&bytes[i * 8..(i + 1) * 8], &event.bytes());
        }
    }

}

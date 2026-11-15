// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Virtio devices (OASIS VIRTIO 1.2), split from their transport.
//!
//! A device implements [`VirtioDevice`]: its type, features, configuration
//! space and what it does when the driver notifies a queue. The transport
//! ([`mmio::VirtioMmio`] for now) owns feature negotiation, queue setup and
//! the interrupt status, and hands the device its queues once the driver says
//! DRIVER_OK. A PCI transport can later wrap the same devices.
//!
//! Queues are processed on the thread that notifies them, which is a vCPU's,
//! or on a backend's own thread for traffic that starts outside the guest
//! (network frames, console input).

use std::sync::{Arc, Mutex};

use crate_core::irq::IrqLine;
use crate_core::memory::GuestMemory;

pub mod block;
pub mod console;
pub mod gpu;
pub mod input;
pub mod mmio;
pub mod net;
pub mod queue;

pub use queue::Queue;

pub const ID_NET: u32 = 1;
pub const ID_BLOCK: u32 = 2;
pub const ID_CONSOLE: u32 = 3;

/// A transport-visible shared-memory window advertised through virtio-mmio.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedMemoryRegion {
    pub addr: u64,
    pub len: u64,
}

pub const F_INDIRECT_DESC: u64 = 1 << 28;
pub const F_VERSION_1: u64 = 1 << 32;

pub const STATUS_ACKNOWLEDGE: u32 = 1;
pub const STATUS_DRIVER: u32 = 2;
pub const STATUS_DRIVER_OK: u32 = 4;
pub const STATUS_FEATURES_OK: u32 = 8;
pub const STATUS_FAILED: u32 = 128;

const INT_USED_BUFFER: u32 = 1;
const INT_CONFIG_CHANGE: u32 = 2;

/// The interrupt status a device and its transport share, and the line it
/// drives: high while any status bit is set.
pub struct Interrupt {
    status: Mutex<u32>,
    line: IrqLine,
}

impl Interrupt {
    pub fn new(line: IrqLine) -> Arc<Self> {
        Arc::new(Interrupt {
            status: Mutex::new(0),
            line,
        })
    }

    /// Tells the driver that a queue has new used buffers.
    pub fn signal_used(&self) {
        self.raise(INT_USED_BUFFER);
    }

    pub fn signal_config(&self) {
        self.raise(INT_CONFIG_CHANGE);
    }

    fn raise(&self, bits: u32) {
        let mut status = self.status.lock().unwrap();
        if *status & bits != bits {
            *status |= bits;
            self.line.set(true);
        }
    }

    fn acknowledge(&self, bits: u32) {
        let mut status = self.status.lock().unwrap();
        *status &= !bits;
        if *status == 0 {
            self.line.set(false);
        }
    }

    fn status(&self) -> u32 {
        *self.status.lock().unwrap()
    }
}

/// Everything a device needs once the driver has set it up.
pub struct Activation {
    pub memory: Arc<GuestMemory>,
    pub queues: Vec<Queue>,
    pub interrupt: Arc<Interrupt>,
    /// The features the driver accepted.
    pub features: u64,
}

pub trait VirtioDevice: Send {
    fn device_id(&self) -> u32;
    /// Every feature the device offers, VERSION_1 included.
    fn features(&self) -> u64;
    /// The largest size of each queue; the transport offers this many queues.
    fn queue_max_sizes(&self) -> &[u16];
    /// Reads the device-specific configuration space.
    fn read_config(&self, offset: u64, data: &mut [u8]);
    /// Returns a transport-visible shared-memory region by its device-defined ID.
    fn shared_memory_region(&self, _id: u32) -> Option<SharedMemoryRegion> {
        None
    }
    fn write_config(&mut self, _offset: u64, _data: &[u8]) {}
    /// The driver set DRIVER_OK. `activation.queues` has one queue per entry
    /// of `queue_max_sizes`; one the driver did not enable is not ready and
    /// never yields a request.
    fn activate(&mut self, activation: Activation);
    /// The driver notified queue `index`.
    fn notify(&mut self, index: u16);
    /// The driver reset the device: forget the queues.
    fn reset(&mut self);
}

/// Copies `src`, a little-endian configuration structure, into `data` from
/// `offset`, leaving bytes past its end zero.
pub fn read_config_bytes(src: &[u8], offset: u64, data: &mut [u8]) {
    for (i, byte) in data.iter_mut().enumerate() {
        *byte = src.get(offset as usize + i).copied().unwrap_or(0);
    }
}

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The virtio-mmio transport, version 2 (VIRTIO 1.2, section 4.2): a register
//! window per device, described to the guest by a "virtio,mmio" device-tree
//! node, with one level-sensitive interrupt.

use std::sync::{Arc, Mutex};

use crate_core::bus::MmioDevice;
use crate_core::irq::IrqLine;
use crate_core::memory::GuestMemory;
use crate_core::warn;

use super::queue::QueueConfig;
use super::{
    Activation, F_VERSION_1, Interrupt, Queue, SharedMemoryRegion, STATUS_DRIVER_OK,
    STATUS_FEATURES_OK, VirtioDevice,
};

pub const SIZE: u64 = 0x200;

const MAGIC: u32 = 0x7472_6976; // "virt"
const VERSION: u32 = 2;
const VENDOR: u32 = 0x7461_7263; // "crat"

const REG_MAGIC: u64 = 0x000;
const REG_VERSION: u64 = 0x004;
const REG_DEVICE_ID: u64 = 0x008;
const REG_VENDOR_ID: u64 = 0x00c;
const REG_DEVICE_FEATURES: u64 = 0x010;
const REG_DEVICE_FEATURES_SEL: u64 = 0x014;
const REG_DRIVER_FEATURES: u64 = 0x020;
const REG_DRIVER_FEATURES_SEL: u64 = 0x024;
const REG_QUEUE_SEL: u64 = 0x030;
const REG_QUEUE_NUM_MAX: u64 = 0x034;
const REG_QUEUE_NUM: u64 = 0x038;
const REG_QUEUE_READY: u64 = 0x044;
const REG_QUEUE_NOTIFY: u64 = 0x050;
const REG_INTERRUPT_STATUS: u64 = 0x060;
const REG_INTERRUPT_ACK: u64 = 0x064;
const REG_STATUS: u64 = 0x070;
const REG_QUEUE_DESC_LOW: u64 = 0x080;
const REG_QUEUE_DESC_HIGH: u64 = 0x084;
const REG_QUEUE_DRIVER_LOW: u64 = 0x090;
const REG_QUEUE_DRIVER_HIGH: u64 = 0x094;
const REG_QUEUE_DEVICE_LOW: u64 = 0x0a0;
const REG_QUEUE_DEVICE_HIGH: u64 = 0x0a4;
const REG_SHM_SEL: u64 = 0x0ac;
const REG_SHM_LEN_LOW: u64 = 0x0b0;
const REG_SHM_LEN_HIGH: u64 = 0x0b4;
const REG_SHM_BASE_LOW: u64 = 0x0b8;
const REG_SHM_BASE_HIGH: u64 = 0x0bc;
const REG_CONFIG_GENERATION: u64 = 0x0fc;
const REG_CONFIG: u64 = 0x100;

struct Inner {
    device: Box<dyn VirtioDevice>,
    device_features_sel: u32,
    driver_features_sel: u32,
    driver_features: u64,
    queue_sel: u32,
    shm_sel: u32,
    queues: Vec<QueueConfig>,
    status: u32,
    active: bool,
}

impl Inner {
    fn queue(&mut self) -> Option<&mut QueueConfig> {
        self.queues.get_mut(self.queue_sel as usize)
    }

    fn reset(&mut self) {
        self.device.reset();
        self.device_features_sel = 0;
        self.driver_features_sel = 0;
        self.driver_features = 0;
        self.queue_sel = 0;
        self.shm_sel = 0;
        for q in &mut self.queues {
            *q = QueueConfig::default();
        }
        self.status = 0;
        self.active = false;
    }
}

pub struct VirtioMmio {
    inner: Mutex<Inner>,
    interrupt: Arc<Interrupt>,
    memory: Arc<GuestMemory>,
}

impl VirtioMmio {
    pub fn new(device: Box<dyn VirtioDevice>, memory: Arc<GuestMemory>, irq: IrqLine) -> Arc<Self> {
        let queues = vec![QueueConfig::default(); device.queue_max_sizes().len()];
        Arc::new(VirtioMmio {
            inner: Mutex::new(Inner {
                device,
                device_features_sel: 0,
                driver_features_sel: 0,
                driver_features: 0,
                queue_sel: 0,
                shm_sel: 0,
                queues,
                status: 0,
                active: false,
            }),
            interrupt: Interrupt::new(irq),
            memory,
        })
    }

    fn set_status(&self, inner: &mut Inner, value: u32) {
        if value == 0 {
            inner.reset();
            self.interrupt.acknowledge(u32::MAX);
            return;
        }
        let mut value = value;
        if value & STATUS_FEATURES_OK != 0 && inner.status & STATUS_FEATURES_OK == 0 {
            let offered = inner.device.features();
            let accepted = inner.driver_features;
            if accepted & !offered != 0 || accepted & F_VERSION_1 == 0 {
                // Leaving FEATURES_OK clear tells the driver the set was refused.
                value &= !STATUS_FEATURES_OK;
            }
        }
        inner.status = value;
        if value & STATUS_DRIVER_OK != 0 && !inner.active {
            inner.active = true;
            let queues = inner
                .queues
                .iter()
                .map(|&config| Queue::new(self.memory.clone(), config))
                .collect();
            let activation = Activation {
                memory: self.memory.clone(),
                queues,
                interrupt: self.interrupt.clone(),
                features: inner.driver_features,
            };
            inner.device.activate(activation);
        }
    }
}

fn set_low(value: &mut u64, low: u64) {
    *value = (*value & !0xffff_ffff) | (low & 0xffff_ffff);
}

fn set_high(value: &mut u64, high: u64) {
    *value = (*value & 0xffff_ffff) | (high << 32);
}

impl MmioDevice for VirtioMmio {
    fn read(&self, offset: u64, size: u8) -> u64 {
        let inner = self.inner.lock().unwrap();
        if offset >= REG_CONFIG {
            let mut data = [0u8; 8];
            let size = (size as usize).min(8);
            inner
                .device
                .read_config(offset - REG_CONFIG, &mut data[..size]);
            return u64::from_le_bytes(data);
        }
        let value: u32 = match offset {
            REG_MAGIC => MAGIC,
            REG_VERSION => VERSION,
            REG_DEVICE_ID => inner.device.device_id(),
            REG_VENDOR_ID => VENDOR,
            REG_DEVICE_FEATURES => match inner.device_features_sel {
                0 => inner.device.features() as u32,
                1 => (inner.device.features() >> 32) as u32,
                _ => 0,
            },
            REG_QUEUE_NUM_MAX => inner
                .device
                .queue_max_sizes()
                .get(inner.queue_sel as usize)
                .map_or(0, |&n| n as u32),
            REG_QUEUE_READY => inner
                .queues
                .get(inner.queue_sel as usize)
                .is_some_and(|q| q.ready) as u32,
            REG_INTERRUPT_STATUS => self.interrupt.status(),
            REG_STATUS => inner.status,
            REG_SHM_SEL => inner.shm_sel,
            REG_SHM_LEN_LOW => inner
                .device
                .shared_memory_region(inner.shm_sel)
                .map_or(u32::MAX, |region| region.len as u32),
            REG_SHM_LEN_HIGH => inner
                .device
                .shared_memory_region(inner.shm_sel)
                .map_or(u32::MAX, |region| (region.len >> 32) as u32),
            REG_SHM_BASE_LOW => inner
                .device
                .shared_memory_region(inner.shm_sel)
                .map_or(0, |region| region.addr as u32),
            REG_SHM_BASE_HIGH => inner
                .device
                .shared_memory_region(inner.shm_sel)
                .map_or(0, |region| (region.addr >> 32) as u32),
            REG_CONFIG_GENERATION => 0,
            _ => 0,
        };
        value as u64
    }

    fn write(&self, offset: u64, size: u8, value: u64) {
        let mut inner = self.inner.lock().unwrap();
        if offset >= REG_CONFIG {
            let bytes = value.to_le_bytes();
            inner
                .device
                .write_config(offset - REG_CONFIG, &bytes[..(size as usize).min(8)]);
            return;
        }
        let v = value & 0xffff_ffff;
        match offset {
            REG_DEVICE_FEATURES_SEL => inner.device_features_sel = v as u32,
            REG_DRIVER_FEATURES_SEL => inner.driver_features_sel = v as u32,
            REG_DRIVER_FEATURES => match inner.driver_features_sel {
                0 => set_low(&mut inner.driver_features, v),
                1 => set_high(&mut inner.driver_features, v),
                _ => {}
            },
            REG_QUEUE_SEL => inner.queue_sel = v as u32,
            REG_QUEUE_NUM => {
                let max = inner
                    .device
                    .queue_max_sizes()
                    .get(inner.queue_sel as usize)
                    .copied();
                if let (Some(max), Some(q)) = (max, inner.queue()) {
                    // Split queues must be a power of two in size.
                    if v as u16 as u64 == v && v <= max as u64 && (v as u16).is_power_of_two() {
                        q.size = v as u16;
                    } else {
                        warn!("virtio-mmio: queue size {v} refused (largest {max})");
                    }
                }
            }
            REG_QUEUE_READY => {
                if let Some(q) = inner.queue() {
                    q.ready = v & 1 != 0;
                }
            }
            REG_QUEUE_DESC_LOW => inner
                .queue()
                .into_iter()
                .for_each(|q| set_low(&mut q.desc, v)),
            REG_QUEUE_DESC_HIGH => inner
                .queue()
                .into_iter()
                .for_each(|q| set_high(&mut q.desc, v)),
            REG_QUEUE_DRIVER_LOW => inner
                .queue()
                .into_iter()
                .for_each(|q| set_low(&mut q.driver, v)),
            REG_QUEUE_DRIVER_HIGH => inner
                .queue()
                .into_iter()
                .for_each(|q| set_high(&mut q.driver, v)),
            REG_QUEUE_DEVICE_LOW => inner
                .queue()
                .into_iter()
                .for_each(|q| set_low(&mut q.device, v)),
            REG_QUEUE_DEVICE_HIGH => inner
                .queue()
                .into_iter()
                .for_each(|q| set_high(&mut q.device, v)),
            REG_QUEUE_NOTIFY => {
                if inner.active {
                    inner.device.notify(v as u16);
                }
            }
            REG_INTERRUPT_ACK => self.interrupt.acknowledge(v as u32),
            REG_STATUS => self.set_status(&mut inner, v as u32),
            REG_SHM_SEL => inner.shm_sel = v as u32,
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::queue::testing::*;
    use super::super::{STATUS_ACKNOWLEDGE, STATUS_DRIVER};
    use super::*;
    use crate_core::irq::LevelRecorder;

    /// A device that echoes each readable buffer into the writable ones.
    struct Echo {
        active: Option<Activation>,
        resets: usize,
        shm_region: Option<SharedMemoryRegion>,
    }

    impl VirtioDevice for Echo {
        fn device_id(&self) -> u32 {
            42
        }
        fn features(&self) -> u64 {
            F_VERSION_1 | 1 << 3
        }
        fn queue_max_sizes(&self) -> &[u16] {
            &[16, 8]
        }
        fn read_config(&self, offset: u64, data: &mut [u8]) {
            super::super::read_config_bytes(&[1, 2, 3, 4, 5, 6, 7, 8], offset, data);
        }
        fn shared_memory_region(&self, id: u32) -> Option<SharedMemoryRegion> {
            (id == 1).then_some(self.shm_region).flatten()
        }
        fn activate(&mut self, activation: Activation) {
            self.active = Some(activation);
        }
        fn notify(&mut self, index: u16) {
            let a = self.active.as_mut().unwrap();
            let q = &mut a.queues[index as usize];
            while let Some(chain) = q.pop().unwrap() {
                let data = chain.read_all(&a.memory, 1 << 20).unwrap();
                let n = chain.write_all(&a.memory, &data).unwrap();
                q.add_used(chain.head, n as u32).unwrap();
            }
            a.interrupt.signal_used();
        }
        fn reset(&mut self) {
            self.active = None;
            self.resets += 1;
        }
    }

    fn transport() -> (Arc<VirtioMmio>, Arc<LevelRecorder>, Arc<GuestMemory>) {
        let memory = memory();
        let chip = Arc::new(LevelRecorder::default());
        let dev = VirtioMmio::new(
            Box::new(Echo {
                active: None,
                resets: 0,
                shm_region: None,
            }),
            memory.clone(),
            IrqLine::new(chip.clone(), 16),
        );
        (dev, chip, memory)
    }

    fn set_up(dev: &VirtioMmio) {
        dev.write(REG_STATUS, 4, (STATUS_ACKNOWLEDGE | STATUS_DRIVER) as u64);
        dev.write(REG_DRIVER_FEATURES_SEL, 4, 1);
        dev.write(REG_DRIVER_FEATURES, 4, 1);
        dev.write(
            REG_STATUS,
            4,
            (STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK) as u64,
        );
        dev.write(REG_QUEUE_SEL, 4, 0);
        dev.write(REG_QUEUE_NUM, 4, 8);
        dev.write(REG_QUEUE_DESC_LOW, 4, DESC & 0xffff_ffff);
        dev.write(REG_QUEUE_DESC_HIGH, 4, DESC >> 32);
        dev.write(REG_QUEUE_DRIVER_LOW, 4, AVAIL);
        dev.write(REG_QUEUE_DEVICE_LOW, 4, USED);
        dev.write(REG_QUEUE_READY, 4, 1);
        dev.write(
            REG_STATUS,
            4,
            (STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK | STATUS_DRIVER_OK) as u64,
        );
    }

    #[test]
    fn identifies_itself() {
        let (dev, _, _) = transport();
        assert_eq!(dev.read(REG_MAGIC, 4), 0x7472_6976);
        assert_eq!(dev.read(REG_VERSION, 4), 2);
        assert_eq!(dev.read(REG_DEVICE_ID, 4), 42);
        assert_eq!(dev.read(REG_DEVICE_FEATURES, 4), 1 << 3);
        dev.write(REG_DEVICE_FEATURES_SEL, 4, 1);
        assert_eq!(dev.read(REG_DEVICE_FEATURES, 4), 1);
        dev.write(REG_QUEUE_SEL, 4, 1);
        assert_eq!(dev.read(REG_QUEUE_NUM_MAX, 4), 8);
        dev.write(REG_QUEUE_SEL, 4, 2);
        assert_eq!(dev.read(REG_QUEUE_NUM_MAX, 4), 0, "no third queue");
        assert_eq!(dev.read(REG_CONFIG + 2, 2), 0x0403);
        assert_eq!(dev.read(REG_CONFIG + 4, 4), 0x0807_0605);
        assert_eq!(dev.read(REG_SHM_LEN_LOW, 4), 0xffff_ffff);
    }

    #[test]
    fn exposes_selected_shared_memory_region() {
        let memory = memory();
        let chip = Arc::new(LevelRecorder::default());
        let dev = VirtioMmio::new(
            Box::new(Echo {
                active: None,
                resets: 0,
                shm_region: Some(SharedMemoryRegion {
                    addr: 0x1234_5678_9abc_0000,
                    len: 0x0000_0001_2345_0000,
                }),
            }),
            memory,
            IrqLine::new(chip, 16),
        );
        dev.write(REG_SHM_SEL, 4, 1);
        assert_eq!(dev.read(REG_SHM_SEL, 4), 1);
        assert_eq!(dev.read(REG_SHM_LEN_LOW, 4), 0x2345_0000);
        assert_eq!(dev.read(REG_SHM_LEN_HIGH, 4), 1);
        assert_eq!(dev.read(REG_SHM_BASE_LOW, 4), 0x9abc_0000);
        assert_eq!(dev.read(REG_SHM_BASE_HIGH, 4), 0x1234_5678);
        dev.write(REG_SHM_SEL, 4, 2);
        assert_eq!(dev.read(REG_SHM_LEN_LOW, 4), u32::MAX as u64);
        assert_eq!(dev.read(REG_SHM_LEN_HIGH, 4), u32::MAX as u64);
    }

    #[test]
    fn refuses_features_it_did_not_offer() {
        let (dev, _, _) = transport();
        dev.write(REG_DRIVER_FEATURES_SEL, 4, 0);
        dev.write(REG_DRIVER_FEATURES, 4, 1 << 4);
        dev.write(REG_DRIVER_FEATURES_SEL, 4, 1);
        dev.write(REG_DRIVER_FEATURES, 4, 1);
        dev.write(REG_STATUS, 4, STATUS_FEATURES_OK as u64 | 3);
        assert_eq!(dev.read(REG_STATUS, 4) as u32 & STATUS_FEATURES_OK, 0);
    }

    #[test]
    fn refuses_a_driver_without_version_1() {
        let (dev, _, _) = transport();
        dev.write(REG_DRIVER_FEATURES, 4, 1 << 3);
        dev.write(REG_STATUS, 4, STATUS_FEATURES_OK as u64 | 3);
        assert_eq!(dev.read(REG_STATUS, 4) as u32 & STATUS_FEATURES_OK, 0);
    }

    #[test]
    fn runs_a_queue_and_interrupts() {
        let (dev, chip, memory) = transport();
        set_up(&dev);
        assert_eq!(
            dev.read(REG_STATUS, 4) as u32 & STATUS_FEATURES_OK,
            STATUS_FEATURES_OK
        );
        memory.write(DATA, b"ping").unwrap();
        let mut driver = Driver::new(memory.clone(), 8);
        let head = driver.offer(&[(DATA, 4, false), (DATA + 0x100, 16, true)]);
        dev.write(REG_QUEUE_NOTIFY, 4, 0);
        assert_eq!(driver.take_used(), Some((head, 4)));
        let mut echoed = [0; 4];
        memory.read(DATA + 0x100, &mut echoed).unwrap();
        assert_eq!(&echoed, b"ping");
        assert!(chip.level(16));
        assert_eq!(dev.read(REG_INTERRUPT_STATUS, 4), 1);
        dev.write(REG_INTERRUPT_ACK, 4, 1);
        assert!(!chip.level(16));
        assert_eq!(dev.read(REG_INTERRUPT_STATUS, 4), 0);
    }

    #[test]
    fn rejects_queue_sizes_that_are_not_powers_of_two() {
        let (dev, _, _) = transport();
        dev.write(REG_QUEUE_SEL, 4, 0);
        dev.write(REG_QUEUE_NUM, 4, 12);
        dev.write(REG_QUEUE_NUM, 4, 32);
        assert_eq!(dev.inner.lock().unwrap().queues[0].size, 0);
    }

    #[test]
    fn status_zero_resets() {
        let (dev, chip, memory) = transport();
        set_up(&dev);
        let mut driver = Driver::new(memory, 8);
        driver.offer(&[(DATA, 4, false)]);
        dev.write(REG_QUEUE_NOTIFY, 4, 0);
        assert!(chip.level(16));
        dev.write(REG_STATUS, 4, 0);
        assert!(!chip.level(16));
        assert_eq!(dev.read(REG_STATUS, 4), 0);
        assert_eq!(dev.read(REG_QUEUE_READY, 4), 0);
        // A notify before the next DRIVER_OK goes nowhere.
        dev.write(REG_QUEUE_NOTIFY, 4, 0);
    }
}

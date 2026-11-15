// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The `virt` machine: a generic arm64 board for Linux guests. Its memory map,
//! the device tree it writes for the guest, and Linux's boot protocol.
//!
//! The board: a GICv3, the architected timers, PSCI over HVC, a PL011 UART, a
//! PL031 RTC, and virtio devices on virtio-mmio, all described by a device
//! tree, with RAM from 1 GiB up. virtio-mmio rather than PCIe because it needs
//! no config space, BARs or MSIs to emulate, and Linux finds it from the
//! device tree alone; PCIe arrives when something needs it (a GPU's BARs, or
//! guests that only know PCI).

use std::path::PathBuf;
use std::sync::Arc;

use crate_core::block::BlockBackend;
use crate_core::board::{Board, BootState, CpuBackend};
use crate_core::bus::MmioBus;
use crate_core::display::{DisplaySink, InputSink, Mode};
use crate_core::gpu::RendererFactory;
use crate_core::irq::{InterruptController, IrqLine};
use crate_core::memory::{GuestMemory, RamBlock};
use crate_core::net::NetBackend;
use crate_core::psci::Psci;
use crate_core::run::{RunControl, StopReason};
use crate_core::{Error, Result, debug};
use crate_devices::gicv3::{self, GicV3};
use crate_devices::virtio::console::{Console, ConsoleInput};
use crate_devices::virtio::gpu::{Gpu, GpuConfig, GpuControl};
use crate_devices::virtio::input::{Input, InputKind, Inputs};
use crate_devices::virtio::mmio::VirtioMmio;
use crate_devices::virtio::{VirtioDevice, block::Block, net::Net};
use crate_devices::{pl011, pl011::Pl011, pl031, pl031::Pl031};
use crate_host::ram::AnonymousRam;

pub mod boot;
pub mod fdt;
pub mod layout;

use layout::*;

pub type Output = Arc<dyn Fn(&[u8]) + Send + Sync>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsoleKind {
    /// The PL011: ttyAMA0, live from the kernel's first line. One at most.
    Serial,
    /// A virtio console: hvc0, hvc1, ... in the order they are added.
    Virtio,
}

pub struct ConsoleConfig {
    pub kind: ConsoleKind,
    /// Gets what the guest writes, on whichever thread it was written.
    pub output: Output,
    /// Told when the console can take bytes again after `console_write`
    /// took fewer than it was offered.
    pub input_ready: Option<Arc<dyn Fn() + Send + Sync>>,
}

pub struct NetworkConfig {
    pub backend: Arc<dyn NetBackend>,
    /// Without one, network n gets 02:63:72:61:74:65+n, the same every run.
    pub mac: Option<[u8; 6]>,
}

pub struct DisplayConfig {
    /// Offered to the guest in this order; the first is the preferred size.
    pub modes: Vec<Mode>,
    pub sink: Arc<dyn DisplaySink>,
    /// A 3D renderer for virgl; without one, or if it fails to start, the
    /// GPU is 2D only.
    pub renderer: Option<RendererFactory>,
}

/// Which input devices the machine has.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InputConfig {
    pub keyboard: bool,
    /// An absolute pointer.
    pub tablet: bool,
    /// A relative mouse.
    pub mouse: bool,
}

/// The modes a display offers unless told otherwise: the preferred first.
pub const DEFAULT_MODES: [Mode; 4] = [
    Mode::new(1024, 768),
    Mode::new(640, 480),
    Mode::new(800, 600),
    Mode::new(1280, 960),
];

pub struct VirtConfig {
    pub cpus: usize,
    /// Bytes of RAM, a multiple of 2 MiB.
    pub memory: u64,
    pub kernel: PathBuf,
    /// Load the kernel as a raw, position-independent binary at 2 MiB into RAM
    /// rather than parsing it as a Linux Image (for bare-metal payloads such as
    /// kvm-unit-tests' `.flat` images).
    pub raw_kernel: bool,
    pub initrd: Option<PathBuf>,
    pub cmdline: String,
    /// virtio block devices: vda, vdb, ... in this order.
    pub disks: Vec<Arc<dyn BlockBackend>>,
    pub networks: Vec<NetworkConfig>,
    pub consoles: Vec<ConsoleConfig>,
    /// virtio-gpu devices, each with one screen.
    pub displays: Vec<DisplayConfig>,
    pub input: InputConfig,
}

enum ConsoleHandle {
    Serial(Arc<Pl011>),
    Virtio(ConsoleInput),
}

pub struct VirtMachine {
    board: Board,
    consoles: Vec<ConsoleHandle>,
    displays: Vec<GpuControl>,
    input: Arc<Inputs>,
    dtb: Vec<u8>,
    _ram: Arc<dyn RamBlock>,
}

impl VirtMachine {
    pub fn new(mut config: VirtConfig) -> Result<Self> {
        if config.cpus == 0 || config.cpus > MAX_CPUS {
            return Err(Error::InvalidArgument(format!(
                "{} CPUs; virt takes 1 to {MAX_CPUS}",
                config.cpus
            )));
        }
        if config.memory < MIN_MEMORY
            || config.memory > MAX_MEMORY
            || config.memory % (2 << 20) != 0
        {
            return Err(Error::InvalidArgument(format!(
                "{} bytes of RAM; virt takes a multiple of 2 MiB from {} MiB to {} GiB",
                config.memory,
                MIN_MEMORY >> 20,
                MAX_MEMORY >> 30
            )));
        }
        let serials = config
            .consoles
            .iter()
            .filter(|c| c.kind == ConsoleKind::Serial)
            .count();
        if serials > 1 {
            return Err(Error::InvalidArgument("virt has one serial port".into()));
        }
        let slots = config.disks.len()
            + config.networks.len()
            + config.displays.len()
            + config.input.keyboard as usize
            + config.input.tablet as usize
            + config
                .consoles
                .iter()
                .filter(|c| c.kind == ConsoleKind::Virtio)
                .count();
        if slots > VIRTIO_SLOTS {
            return Err(Error::InvalidArgument(format!(
                "{slots} virtio devices; virt has room for {VIRTIO_SLOTS}"
            )));
        }

        let read = |path: &PathBuf, what: &str| {
            std::fs::read(path)
                .map_err(|e| Error::io(format!("reading the {what} {}", path.display()), e))
        };
        let kernel = if config.raw_kernel {
            let data = read(&config.kernel, "kernel")?;
            // A raw payload loads 2 MiB into RAM; it relocates itself and finds
            // RAM and the device tree from x0 and the tree, so no header.
            boot::KernelImage {
                image_size: data.len() as u64,
                text_offset: 2 << 20,
                data,
            }
        } else {
            boot::parse_image(read(&config.kernel, "kernel")?)?
        };
        let initrd = config
            .initrd
            .as_ref()
            .map(|p| read(p, "initrd"))
            .transpose()?;
        let placement = boot::place(
            RAM_BASE,
            config.memory,
            &kernel,
            initrd.as_ref().map(|i| i.len() as u64),
        )?;

        let ram: Arc<dyn RamBlock> = Arc::new(AnonymousRam::new(config.memory as usize)?);
        let memory = Arc::new(GuestMemory::new(vec![(RAM_BASE, ram.clone())])?);
        let load = |addr: u64, data: &[u8]| {
            memory
                .write(addr, data)
                .map_err(|e| Error::Guest(format!("loading: {e}")))
        };
        load(placement.kernel, &kernel.data)?;
        if let (Some(data), Some((start, _))) = (&initrd, placement.initrd) {
            load(start, data)?;
        }

        let mpidrs: Vec<u64> = (0..config.cpus as u64).map(mpidr).collect();
        let run = Arc::new(RunControl::new());
        let psci = Psci::new(mpidrs.clone(), run.clone());
        let gic = GicV3::new(&mpidrs, SPIS);
        let lines: Arc<dyn InterruptController> = gic.clone();
        let mut mmio = MmioBus::new();
        mmio.insert(GICD_BASE, gicv3::DISTRIBUTOR_SIZE, gic.distributor())?;
        mmio.insert(
            GICR_BASE,
            gicv3::REDISTRIBUTOR_SIZE * config.cpus as u64,
            gic.redistributors(),
        )?;

        let serial_output = config
            .consoles
            .iter()
            .find(|c| c.kind == ConsoleKind::Serial)
            .map(|c| c.output.clone());
        let uart = Pl011::new(
            IrqLine::new(lines.clone(), UART_SPI),
            serial_output.unwrap_or_else(|| Arc::new(|_: &[u8]| {})),
        );
        mmio.insert(UART_BASE, pl011::SIZE, uart.clone())?;
        mmio.insert(
            RTC_BASE,
            pl031::SIZE,
            Pl031::new(IrqLine::new(lines.clone(), RTC_SPI)),
        )?;

        let mut devices: Vec<Box<dyn VirtioDevice>> = Vec::new();
        for (i, disk) in config.disks.iter().enumerate() {
            devices.push(Box::new(Block::new(
                disk.clone(),
                &format!("crate-vd{}", (b'a' + i as u8) as char),
            )));
        }
        for (i, net) in config.networks.iter().enumerate() {
            devices.push(Box::new(Net::new(
                net.backend.clone(),
                net.mac
                    .unwrap_or([0x02, 0x63, 0x72, 0x61, 0x74, 0x65 + i as u8]),
            )));
        }
        let mut consoles = Vec::new();
        for console in &config.consoles {
            match console.kind {
                ConsoleKind::Serial => {
                    if let Some(ready) = &console.input_ready {
                        uart.set_input_ready(ready.clone());
                    }
                    consoles.push(ConsoleHandle::Serial(uart.clone()))
                }
                ConsoleKind::Virtio => {
                    let device = Console::new(console.output.clone());
                    if let Some(ready) = &console.input_ready {
                        device.input().set_input_ready(ready.clone());
                    }
                    consoles.push(ConsoleHandle::Virtio(device.input()));
                    devices.push(Box::new(device));
                }
            }
        }
        let mut displays = Vec::new();
        for display in config.displays.drain(..) {
            let gpu = Gpu::new(GpuConfig {
                modes: display.modes,
                sink: display.sink,
                renderer: display.renderer,
            })?;
            displays.push(gpu.control());
            devices.push(Box::new(gpu));
        }
        let mut keyboard = None;
        if config.input.keyboard {
            let device = Input::new(InputKind::Keyboard, "crate keyboard");
            keyboard = Some(device.sender());
            devices.push(Box::new(device));
        }
        let mut tablet = None;
        if config.input.tablet {
            let device = Input::new(InputKind::Tablet, "crate tablet");
            tablet = Some(device.sender());
            devices.push(Box::new(device));
        }
        let mut mouse = None;
        if config.input.mouse {
            let device = Input::new(InputKind::Mouse, "crate mouse");
            mouse = Some(device.sender());
            devices.push(Box::new(device));
        }
        let virtio_count = devices.len();
        for (slot, device) in devices.into_iter().enumerate() {
            let irq = IrqLine::new(lines.clone(), VIRTIO_SPI_BASE + slot as u32);
            let transport = VirtioMmio::new(device, memory.clone(), irq);
            mmio.insert(
                virtio_base(slot),
                crate_devices::virtio::mmio::SIZE,
                transport,
            )?;
        }

        let dtb = device_tree(&DeviceTree {
            cpus: config.cpus,
            memory: config.memory,
            cmdline: &config.cmdline,
            initrd: placement.initrd,
            virtio_slots: virtio_count,
        });
        if dtb.len() as u64 > boot::DTB_MAX {
            return Err(Error::InvalidArgument(
                "the device tree is over 2 MB".into(),
            ));
        }
        load(placement.dtb, &dtb)?;
        debug!(
            "kernel at {:#x}, initrd at {:x?}, device tree at {:#x} ({} bytes)",
            placement.kernel,
            placement.initrd,
            placement.dtb,
            dtb.len()
        );

        let board = Board {
            memory,
            mmio,
            interrupts: gic,
            lines,
            mpidrs,
            boot: BootState {
                pc: placement.kernel,
                x: [placement.dtb, 0, 0, 0],
            },
            vtimer_ppi: VTIMER_PPI,
            ptimer_ppi: PTIMER_PPI,
            psci,
            run,
        };
        Ok(VirtMachine {
            board,
            consoles,
            displays,
            input: Arc::new(Inputs::new(keyboard, tablet, mouse)),
            dtb,
            _ram: ram,
        })
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn device_tree(&self) -> &[u8] {
        &self.dtb
    }

    /// Runs the machine on `backend` until it stops.
    pub fn run(&self, backend: &dyn CpuBackend) -> Result<StopReason> {
        backend.run(&self.board)
    }

    /// Stops the machine at once, as pulling the plug does.
    pub fn kill(&self) {
        self.board.run.stop(StopReason::Killed);
    }

    /// Display `index`'s controls, in the order the config listed them.
    pub fn display(&self, index: usize) -> Option<&GpuControl> {
        self.displays.get(index)
    }

    /// The keyboard and tablet; input for devices the machine lacks goes
    /// nowhere.
    pub fn input(&self) -> Arc<dyn InputSink> {
        self.input.clone()
    }

    /// Bytes for console `index` (in the order the config listed them) to
    /// read. Never blocks: returns how many were taken.
    pub fn console_write(&self, index: usize, bytes: &[u8]) -> usize {
        match self.consoles.get(index) {
            Some(ConsoleHandle::Serial(uart)) => uart.receive(bytes),
            Some(ConsoleHandle::Virtio(input)) => input.write(bytes),
            None => 0,
        }
    }
}

struct DeviceTree<'a> {
    cpus: usize,
    memory: u64,
    cmdline: &'a str,
    initrd: Option<(u64, u64)>,
    virtio_slots: usize,
}

const PHANDLE_GIC: u32 = 1;
const PHANDLE_CLOCK: u32 = 2;

const IRQ_SPI: u32 = 0;
const IRQ_PPI: u32 = 1;
const IRQ_LEVEL_HIGH: u32 = 4;

fn device_tree(t: &DeviceTree) -> Vec<u8> {
    let mut w = fdt::FdtWriter::new();
    let uart = format!("/pl011@{UART_BASE:x}");
    w.begin_node("");
    w.property_strings("compatible", &["crate,virt"]);
    w.property_string("model", "crate virt");
    w.property_u32("#address-cells", 2);
    w.property_u32("#size-cells", 2);
    w.property_u32("interrupt-parent", PHANDLE_GIC);

    w.begin_node("chosen");
    w.property_string("bootargs", t.cmdline);
    w.property_string("stdout-path", &uart);
    if let Some((start, end)) = t.initrd {
        w.property_u64("linux,initrd-start", start);
        w.property_u64("linux,initrd-end", end);
    }
    if let Some(seed) = random_seed() {
        w.property("rng-seed", &seed);
    }
    w.end_node();

    w.begin_node("aliases");
    w.property_string("serial0", &uart);
    w.end_node();

    w.begin_node(&format!("memory@{RAM_BASE:x}"));
    w.property_string("device_type", "memory");
    w.property_reg(&[(RAM_BASE, t.memory)]);
    w.end_node();

    w.begin_node("cpus");
    w.property_u32("#address-cells", 1);
    w.property_u32("#size-cells", 0);
    for cpu in 0..t.cpus {
        let reg = (mpidr(cpu as u64) & 0xff_ffff) as u32;
        w.begin_node(&format!("cpu@{reg:x}"));
        w.property_string("device_type", "cpu");
        w.property_string("compatible", "arm,armv8");
        w.property_u32("reg", reg);
        w.property_string("enable-method", "psci");
        w.end_node();
    }
    w.end_node();

    w.begin_node("psci");
    w.property_strings("compatible", &["arm,psci-1.0", "arm,psci-0.2"]);
    w.property_string("method", "hvc");
    w.end_node();

    w.begin_node("timer");
    w.property_string("compatible", "arm,armv8-timer");
    let mut cells = Vec::new();
    for ppi in [SECURE_PTIMER_PPI, PTIMER_PPI, VTIMER_PPI, HYP_TIMER_PPI] {
        cells.extend_from_slice(&[IRQ_PPI, ppi, IRQ_LEVEL_HIGH]);
    }
    w.property_cells("interrupts", &cells);
    w.property_empty("always-on");
    w.end_node();

    w.begin_node(&format!("interrupt-controller@{GICD_BASE:x}"));
    w.property_string("compatible", "arm,gic-v3");
    w.property_u32("#interrupt-cells", 3);
    w.property_empty("interrupt-controller");
    w.property_reg(&[
        (GICD_BASE, gicv3::DISTRIBUTOR_SIZE),
        (GICR_BASE, gicv3::REDISTRIBUTOR_SIZE * t.cpus as u64),
    ]);
    w.property_u32("phandle", PHANDLE_GIC);
    w.end_node();

    w.begin_node("apb-pclk");
    w.property_string("compatible", "fixed-clock");
    w.property_u32("#clock-cells", 0);
    w.property_u32("clock-frequency", 24_000_000);
    w.property_string("clock-output-names", "clk24mhz");
    w.property_u32("phandle", PHANDLE_CLOCK);
    w.end_node();

    w.begin_node(&uart[1..]);
    w.property_strings("compatible", &["arm,pl011", "arm,primecell"]);
    w.property_reg(&[(UART_BASE, pl011::SIZE)]);
    w.property_cells("interrupts", &[IRQ_SPI, UART_SPI, IRQ_LEVEL_HIGH]);
    w.property_cells("clocks", &[PHANDLE_CLOCK, PHANDLE_CLOCK]);
    w.property_strings("clock-names", &["uartclk", "apb_pclk"]);
    w.end_node();

    w.begin_node(&format!("pl031@{RTC_BASE:x}"));
    w.property_strings("compatible", &["arm,pl031", "arm,primecell"]);
    w.property_reg(&[(RTC_BASE, pl031::SIZE)]);
    w.property_cells("interrupts", &[IRQ_SPI, RTC_SPI, IRQ_LEVEL_HIGH]);
    w.property_u32("clocks", PHANDLE_CLOCK);
    w.property_string("clock-names", "apb_pclk");
    w.end_node();

    for slot in 0..t.virtio_slots {
        let base = virtio_base(slot);
        w.begin_node(&format!("virtio_mmio@{base:x}"));
        w.property_string("compatible", "virtio,mmio");
        w.property_reg(&[(base, crate_devices::virtio::mmio::SIZE)]);
        w.property_cells(
            "interrupts",
            &[IRQ_SPI, VIRTIO_SPI_BASE + slot as u32, IRQ_LEVEL_HIGH],
        );
        w.property_empty("dma-coherent");
        w.end_node();
    }

    w.end_node();
    w.finish(0)
}

/// 64 random bytes for the kernel's entropy pool, so it does not start empty.
fn random_seed() -> Option<[u8; 64]> {
    use std::io::Read;
    let mut seed = [0; 64];
    std::fs::File::open("/dev/urandom")
        .ok()?
        .read_exact(&mut seed)
        .ok()?;
    Some(seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree(cpus: usize, virtio_slots: usize) -> fdt::Node {
        let blob = device_tree(&DeviceTree {
            cpus,
            memory: 2 << 30,
            cmdline: "console=ttyAMA0 root=/dev/vda",
            initrd: Some((0x8000_0000, 0x8100_0000)),
            virtio_slots,
        });
        fdt::parse(&blob).unwrap()
    }

    #[test]
    fn describes_the_board() {
        let root = tree(4, 3);
        assert_eq!(root.cells("#address-cells"), Some(vec![2]));
        assert_eq!(root.cells("interrupt-parent"), Some(vec![PHANDLE_GIC]));
        let chosen = root.find("/chosen").unwrap();
        assert_eq!(
            chosen.strings("bootargs").unwrap(),
            ["console=ttyAMA0 root=/dev/vda"]
        );
        assert_eq!(
            chosen.cells("linux,initrd-start"),
            Some(vec![0, 0x8000_0000])
        );
        assert_eq!(chosen.cells("linux,initrd-end"), Some(vec![0, 0x8100_0000]));
        assert_eq!(chosen.property("rng-seed").map(<[u8]>::len), Some(64));
        assert_eq!(chosen.strings("stdout-path").unwrap(), ["/pl011@9000000"]);
        assert!(
            root.find("/pl011@9000000").is_some(),
            "stdout-path names a real node"
        );

        let memory = root.find("/memory@40000000").unwrap();
        assert_eq!(
            memory.cells("reg"),
            Some(vec![0, 0x4000_0000, 0, 0x8000_0000])
        );

        let cpus = root.find("/cpus").unwrap();
        assert_eq!(cpus.children.len(), 4);
        for (i, cpu) in cpus.children.iter().enumerate() {
            assert_eq!(cpu.name, format!("cpu@{i}"));
            assert_eq!(cpu.cells("reg"), Some(vec![i as u32]));
            assert_eq!(cpu.strings("enable-method").unwrap(), ["psci"]);
        }
        assert_eq!(
            root.find("/psci").unwrap().strings("method").unwrap(),
            ["hvc"]
        );

        let gic = root.find("/interrupt-controller@8000000").unwrap();
        assert_eq!(gic.strings("compatible").unwrap(), ["arm,gic-v3"]);
        assert_eq!(gic.cells("#interrupt-cells"), Some(vec![3]));
        assert_eq!(
            gic.cells("reg").unwrap()[4..],
            [0, GICR_BASE as u32, 0, 4 * 0x2_0000]
        );

        let timer = root.find("/timer").unwrap();
        assert_eq!(
            timer.cells("interrupts").unwrap()[6..9],
            [IRQ_PPI, 11, IRQ_LEVEL_HIGH],
            "virtual timer"
        );

        let uart = root.find("/pl011@9000000").unwrap();
        assert_eq!(
            uart.strings("compatible").unwrap(),
            ["arm,pl011", "arm,primecell"]
        );
        assert_eq!(
            uart.cells("clocks"),
            Some(vec![PHANDLE_CLOCK, PHANDLE_CLOCK])
        );
        assert_eq!(
            root.find("/apb-pclk").unwrap().cells("phandle"),
            Some(vec![PHANDLE_CLOCK])
        );

        let virtio: Vec<&fdt::Node> = root
            .children
            .iter()
            .filter(|c| c.name.starts_with("virtio_mmio@"))
            .collect();
        assert_eq!(virtio.len(), 3);
        assert_eq!(virtio[1].name, "virtio_mmio@a000200");
        assert_eq!(
            virtio[1].cells("interrupts"),
            Some(vec![IRQ_SPI, 17, IRQ_LEVEL_HIGH])
        );
        assert!(virtio[1].property("dma-coherent").is_some());
    }

    #[test]
    fn interrupts_fit_the_gic() {
        // Every SPI the tree hands out must exist in the GIC.
        let root = tree(1, VIRTIO_SLOTS);
        let gic = GicV3::new(&[mpidr(0)], SPIS);
        for node in &root.children {
            if let Some(cells) = node.cells("interrupts") {
                for irq in cells.chunks(3) {
                    if irq[0] == IRQ_SPI {
                        assert!(
                            (irq[1] as usize) < gic.spi_count(),
                            "{} uses SPI {}",
                            node.name,
                            irq[1]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn refuses_bad_configurations() {
        let config = |cpus, memory| VirtConfig {
            raw_kernel: false,
            cpus,
            memory,
            kernel: "/nonexistent".into(),
            initrd: None,
            cmdline: String::new(),
            disks: Vec::new(),
            networks: Vec::new(),
            consoles: Vec::new(),
            displays: Vec::new(),
            input: InputConfig::default(),
        };
        assert!(matches!(
            VirtMachine::new(config(0, 1 << 30)),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            VirtMachine::new(config(1, (1 << 30) + 4096)),
            Err(Error::InvalidArgument(_))
        ));
        assert!(
            matches!(VirtMachine::new(config(1, 1 << 30)), Err(Error::Io { .. })),
            "no kernel file"
        );
    }
}

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Guest physical memory: RAM regions at guest addresses, each backed by host
//! memory that a host backend provides.
//!
//! The guest's CPUs write this memory while devices read it, with no lock
//! between them, exactly as a real device's DMA races the CPUs. Devices
//! therefore copy in and out through raw pointers rather than holding Rust
//! references into guest memory, and order their accesses with the atomic
//! loads and stores below where a protocol (a virtqueue's indices) needs it.

use std::alloc::{self, Layout};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU16, Ordering};

use crate::{Error, Result};

/// Host memory that backs a region of guest RAM.
///
/// # Safety
///
/// `as_ptr` must return the same pointer for the block's whole life, valid for
/// reads and writes of `len` bytes, and aligned to at least the host's page
/// size; a hypervisor maps it into the guest as it is.
pub unsafe trait RamBlock: Send + Sync {
    fn as_ptr(&self) -> *mut u8;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A guest-physical shared-memory window backed by renderer-exported file descriptors.
pub trait HostVisibleMemory: RamBlock {
    fn guest_addr(&self) -> u64;
    fn map_fd(&self, guest_addr: u64, fd: std::os::fd::RawFd, len: usize) -> std::io::Result<()>;
    fn unmap(&self, guest_addr: u64) -> std::io::Result<()>;
}

/// RAM from the Rust heap, zeroed. For tests and for hosts with nothing better;
/// real machines use a host backend's mapping.
pub struct HeapRam {
    ptr: *mut u8,
    layout: Layout,
}

// SAFETY: the allocation is owned by this value and only reached through raw
// pointers, whose users accept that guest memory is shared.
unsafe impl Send for HeapRam {}
// SAFETY: as for Send.
unsafe impl Sync for HeapRam {}

impl HeapRam {
    pub const ALIGN: usize = 16384;

    pub fn new(len: usize) -> Result<Self> {
        let layout = Layout::from_size_align(len.max(1), Self::ALIGN)
            .map_err(|_| Error::InvalidArgument(format!("RAM size {len:#x}")))?;
        // SAFETY: the layout has a non-zero size.
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(Error::NoMemory(format!("{len:#x} bytes of guest RAM")));
        }
        Ok(HeapRam { ptr, layout })
    }
}

impl Drop for HeapRam {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout.
        unsafe { alloc::dealloc(self.ptr, self.layout) }
    }
}

// SAFETY: the allocation never moves, spans `len` bytes and is aligned to
// ALIGN, which is the largest page size of the hosts crate supports.
unsafe impl RamBlock for HeapRam {
    fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }
    fn len(&self) -> usize {
        self.layout.size()
    }
}

pub struct MemoryRegion {
    guest_base: u64,
    /// The block's host pointer and length, read once (they never change).
    ptr: *mut u8,
    len: u64,
    /// Keeps the host memory alive as long as the region.
    _block: Arc<dyn RamBlock>,
}

// SAFETY: the pointer is the block's own, valid for the block's life (which
// the region holds); guest memory is shared by design and only reached
// through raw pointers.
unsafe impl Send for MemoryRegion {}
// SAFETY: as for Send.
unsafe impl Sync for MemoryRegion {}

impl MemoryRegion {
    pub fn guest_base(&self) -> u64 {
        self.guest_base
    }
    pub fn size(&self) -> u64 {
        self.len
    }
    pub fn host_ptr(&self) -> *mut u8 {
        self.ptr
    }
    fn contains(&self, addr: u64, len: u64) -> bool {
        addr.wrapping_sub(self.guest_base) < self.len
            && addr
                .checked_add(len)
                .is_some_and(|end| end <= self.guest_base + self.len)
    }
}

/// A guest access outside RAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryError {
    pub addr: u64,
    pub len: u64,
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} bytes at {:#x} are not guest RAM",
            self.len, self.addr
        )
    }
}

impl std::error::Error for MemoryError {}

pub type MemoryResult<T> = std::result::Result<T, MemoryError>;

pub struct GuestMemory {
    regions: Vec<MemoryRegion>,
}

impl GuestMemory {
    pub fn new(regions: Vec<(u64, Arc<dyn RamBlock>)>) -> Result<Self> {
        let mut regions: Vec<MemoryRegion> = regions
            .into_iter()
            .map(|(guest_base, block)| MemoryRegion {
                guest_base,
                ptr: block.as_ptr(),
                len: block.len() as u64,
                _block: block,
            })
            .collect();
        regions.sort_by_key(|r| r.guest_base);
        for pair in regions.windows(2) {
            if pair[0].guest_base + pair[0].size() > pair[1].guest_base {
                return Err(Error::InvalidArgument(format!(
                    "RAM regions at {:#x} and {:#x} overlap",
                    pair[0].guest_base, pair[1].guest_base
                )));
            }
        }
        Ok(GuestMemory { regions })
    }

    pub fn regions(&self) -> &[MemoryRegion] {
        &self.regions
    }

    /// The host address of `len` bytes at guest address `addr`, if they all lie
    /// in one region.
    pub fn host_range(&self, addr: u64, len: u64) -> MemoryResult<*mut u8> {
        let region = self
            .regions
            .iter()
            .find(|r| r.contains(addr, len))
            .ok_or(MemoryError { addr, len })?;
        // SAFETY: `contains` checked that the offset lies inside the block.
        Ok(unsafe { region.host_ptr().add((addr - region.guest_base) as usize) })
    }

    pub fn read(&self, addr: u64, buf: &mut [u8]) -> MemoryResult<()> {
        let src = self.host_range(addr, buf.len() as u64)?;
        // SAFETY: host_range checked the source span; `buf` is ours.
        unsafe { std::ptr::copy_nonoverlapping(src, buf.as_mut_ptr(), buf.len()) };
        Ok(())
    }

    pub fn write(&self, addr: u64, buf: &[u8]) -> MemoryResult<()> {
        let dst = self.host_range(addr, buf.len() as u64)?;
        // SAFETY: host_range checked the destination span.
        unsafe { std::ptr::copy_nonoverlapping(buf.as_ptr(), dst, buf.len()) };
        Ok(())
    }

    pub fn fill(&self, addr: u64, len: u64, byte: u8) -> MemoryResult<()> {
        let dst = self.host_range(addr, len)?;
        // SAFETY: host_range checked the destination span.
        unsafe { std::ptr::write_bytes(dst, byte, len as usize) };
        Ok(())
    }

    pub fn read_u16(&self, addr: u64) -> MemoryResult<u16> {
        let mut b = [0; 2];
        self.read(addr, &mut b)?;
        Ok(u16::from_le_bytes(b))
    }

    pub fn read_u32(&self, addr: u64) -> MemoryResult<u32> {
        let mut b = [0; 4];
        self.read(addr, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    pub fn read_u64(&self, addr: u64) -> MemoryResult<u64> {
        let mut b = [0; 8];
        self.read(addr, &mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    pub fn write_u16(&self, addr: u64, value: u16) -> MemoryResult<()> {
        self.write(addr, &value.to_le_bytes())
    }

    pub fn write_u32(&self, addr: u64, value: u32) -> MemoryResult<()> {
        self.write(addr, &value.to_le_bytes())
    }

    pub fn write_u64(&self, addr: u64, value: u64) -> MemoryResult<()> {
        self.write(addr, &value.to_le_bytes())
    }

    fn atomic_u16(&self, addr: u64) -> MemoryResult<&AtomicU16> {
        if addr % 2 != 0 {
            return Err(MemoryError { addr, len: 2 });
        }
        let ptr = self.host_range(addr, 2)?;
        // SAFETY: the span is inside RAM, aligned, and lives as long as self.
        Ok(unsafe { AtomicU16::from_ptr(ptr.cast()) })
    }

    /// A little-endian u16 read with acquire ordering: later reads see what the
    /// guest wrote before it stored this value with release ordering.
    pub fn load_u16_acquire(&self, addr: u64) -> MemoryResult<u16> {
        Ok(u16::from_le(self.atomic_u16(addr)?.load(Ordering::Acquire)))
    }

    /// A little-endian u16 store with release ordering: the guest sees every
    /// earlier write once it sees this one.
    pub fn store_u16_release(&self, addr: u64, value: u16) -> MemoryResult<()> {
        self.atomic_u16(addr)?
            .store(value.to_le(), Ordering::Release);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory() -> GuestMemory {
        let a: Arc<dyn RamBlock> = Arc::new(HeapRam::new(0x4000).unwrap());
        let b: Arc<dyn RamBlock> = Arc::new(HeapRam::new(0x4000).unwrap());
        GuestMemory::new(vec![(0x10000, b), (0x4000, a)]).unwrap()
    }

    #[test]
    fn reads_back_what_it_wrote() {
        let mem = memory();
        mem.write_u32(0x4010, 0xdead_beef).unwrap();
        assert_eq!(mem.read_u32(0x4010).unwrap(), 0xdead_beef);
        assert_eq!(mem.read_u16(0x4010).unwrap(), 0xbeef);
        mem.store_u16_release(0x10002, 7).unwrap();
        assert_eq!(mem.load_u16_acquire(0x10002).unwrap(), 7);
    }

    #[test]
    fn refuses_access_outside_or_across_regions() {
        let mem = memory();
        assert!(mem.read_u32(0x0).is_err());
        assert!(
            mem.read_u32(0x7ffe).is_err(),
            "straddles the end of a region"
        );
        assert!(mem.read_u32(0x8000).is_err(), "in the gap");
        assert!(mem.read_u64(0x13ff8).is_ok());
        assert!(mem.read_u64(0x13ffc).is_err());
        assert!(mem.load_u16_acquire(0x10001).is_err(), "unaligned");
        assert!(mem.host_range(u64::MAX - 1, 4).is_err(), "wraps");
    }

    #[test]
    fn refuses_overlapping_regions() {
        let a: Arc<dyn RamBlock> = Arc::new(HeapRam::new(0x8000).unwrap());
        let b: Arc<dyn RamBlock> = Arc::new(HeapRam::new(0x4000).unwrap());
        assert!(GuestMemory::new(vec![(0x0, a), (0x4000, b)]).is_err());
    }
}

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Guest RAM as an anonymous private mapping: pages cost nothing until the
//! guest touches them.

use crate_core::memory::RamBlock;
use crate_core::{Error, Result};

pub struct AnonymousRam {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the mapping belongs to this value and is only reached through raw
// pointers, whose users accept that guest memory is shared.
unsafe impl Send for AnonymousRam {}
// SAFETY: as for Send.
unsafe impl Sync for AnonymousRam {}

impl AnonymousRam {
    /// `len` is rounded up to the host's page size.
    pub fn new(len: usize) -> Result<Self> {
        // SAFETY: sysconf has no preconditions.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let len = len.div_ceil(page) * page;
        // SAFETY: a fresh anonymous mapping aliases nothing.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(Error::NoMemory(format!(
                "mapping {} MiB of guest RAM: {}",
                len >> 20,
                std::io::Error::last_os_error()
            )));
        }
        Ok(AnonymousRam {
            ptr: ptr.cast(),
            len,
        })
    }
}

impl Drop for AnonymousRam {
    fn drop(&mut self) {
        // SAFETY: mapped in `new` with this length and not unmapped since.
        unsafe { libc::munmap(self.ptr.cast(), self.len) };
    }
}

// SAFETY: the mapping never moves, spans `len` bytes and is page-aligned.
unsafe impl RamBlock for AnonymousRam {
    fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }
    fn len(&self) -> usize {
        self.len
    }
}


/// Anonymous guest RAM reserved for virtio-gpu host-visible blob mappings.
/// Individual ranges can be replaced with MAP_SHARED mappings of renderer-exported
/// POSIX shared-memory FDs while preserving the window's stable host virtual address.
pub struct HostVisiblePool {
    ram: AnonymousRam,
    guest_addr: u64,
    page_size: usize,
    mappings: std::sync::Mutex<std::collections::BTreeMap<u64, usize>>,
}

impl HostVisiblePool {
    pub fn new(guest_addr: u64, len: usize) -> Result<Self> {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page <= 0 {
            return Err(Error::NoMemory("could not determine host page size".into()));
        }
        let page_size = page as usize;
        if len == 0 || len % page_size != 0 || guest_addr % page_size as u64 != 0 {
            return Err(Error::InvalidArgument(
                "host-visible region must be nonzero and page aligned".into(),
            ));
        }
        Ok(Self {
            ram: AnonymousRam::new(len)?,
            guest_addr,
            page_size,
            mappings: std::sync::Mutex::new(std::collections::BTreeMap::new()),
        })
    }

    fn range(&self, guest_addr: u64, len: usize) -> Result<(*mut libc::c_void, usize)> {
        let offset = guest_addr
            .checked_sub(self.guest_addr)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| Error::InvalidArgument("address precedes host-visible region".into()))?;
        let mapped_len = len
            .checked_add(self.page_size - 1)
            .map(|n| n / self.page_size * self.page_size)
            .ok_or_else(|| Error::InvalidArgument("mapping length overflow".into()))?;
        if len == 0
            || guest_addr % self.page_size as u64 != 0
            || offset.checked_add(mapped_len).is_none_or(|end| end > self.ram.len())
        {
            return Err(Error::InvalidArgument(
                "blob mapping is empty, unaligned, or outside the host-visible region".into(),
            ));
        }
        let ptr = unsafe { self.ram.as_ptr().add(offset).cast::<libc::c_void>() };
        Ok((ptr, mapped_len))
    }
}

// SAFETY: the wrapped anonymous mapping is stable for the pool's lifetime and
// is page-aligned; MAP_FIXED only replaces pages within that same reserved range.
unsafe impl RamBlock for HostVisiblePool {
    fn as_ptr(&self) -> *mut u8 {
        self.ram.as_ptr()
    }

    fn len(&self) -> usize {
        self.ram.len()
    }
}

impl crate_core::memory::HostVisibleMemory for HostVisiblePool {
    fn guest_addr(&self) -> u64 {
        self.guest_addr
    }

    fn map_fd(&self, guest_addr: u64, fd: std::os::fd::RawFd, len: usize) -> std::io::Result<()> {
        use std::ops::Bound::{Excluded, Unbounded};

        let (ptr, mapped_len) = self
            .range(guest_addr, len)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e.to_string()))?;
        let end = guest_addr
            .checked_add(mapped_len as u64)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "mapping address overflow"))?;
        let mut mappings = self.mappings.lock().unwrap();
        if mappings
            .range(..=guest_addr)
            .next_back()
            .is_some_and(|(start, size)| *start + *size as u64 > guest_addr)
            || mappings
                .range((Excluded(guest_addr), Unbounded))
                .next()
                .is_some_and(|(start, _)| *start < end)
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::AlreadyExists,
                "host-visible blob mapping overlaps an existing mapping",
            ));
        }
        let mapped = unsafe {
            libc::mmap(
                ptr,
                mapped_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_FIXED,
                fd,
                0,
            )
        };
        if mapped == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        if mapped != ptr {
            return Err(std::io::Error::other("MAP_FIXED returned an unexpected address"));
        }
        mappings.insert(guest_addr, mapped_len);
        Ok(())
    }

    fn unmap(&self, guest_addr: u64) -> std::io::Result<()> {
        let mapped_len = self
            .mappings
            .lock()
            .unwrap()
            .remove(&guest_addr)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "blob mapping not found"))?;
        let offset = usize::try_from(guest_addr - self.guest_addr)
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid blob address"))?;
        let ptr = unsafe { self.ram.as_ptr().add(offset).cast::<libc::c_void>() };
        let mapped = unsafe {
            libc::mmap(
                ptr,
                mapped_len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON | libc::MAP_FIXED,
                -1,
                0,
            )
        };
        if mapped == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }
        Ok(())
    }
}

#[cfg(test)]
mod host_visible_pool_tests {
    use super::*;
    use crate_core::memory::HostVisibleMemory;
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::os::fd::AsRawFd;

    #[test]
    fn maps_and_unmaps_shared_fd_at_guest_physical_address() {
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let guest_base = 0x4000_0000;
        let pool = HostVisiblePool::new(guest_base, page * 4).unwrap();

        // macOS POSIX shm descriptors do not behave like ordinary files for
        // pwrite on every runner version. An unlinked regular file exercises
        // the same mmap(MAP_SHARED) FD contract without that platform quirk.
        let path = std::env::temp_dir().join(format!(
            "crate-host-visible-{}-{}.tmp",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        std::fs::remove_file(&path).unwrap();
        file.set_len(page as u64).unwrap();
        file.write_all(&[0x5a]).unwrap();
        file.flush().unwrap();

        let blob_addr = guest_base + page as u64;
        pool.map_fd(blob_addr, file.as_raw_fd(), page).unwrap();
        let mapped = unsafe { pool.as_ptr().add(page) };
        assert_eq!(unsafe { *mapped }, 0x5a);
        unsafe { *mapped = 0xa5 };
        file.seek(SeekFrom::Start(0)).unwrap();
        let mut readback = [0_u8];
        file.read_exact(&mut readback).unwrap();
        assert_eq!(readback[0], 0xa5);

        pool.unmap(blob_addr).unwrap();
        assert_eq!(unsafe { *mapped }, 0);
    }
}

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The host's 3D renderer behind a virtio-gpu: what the device's 3D commands
//! need, in the shape virglrenderer gives them (contexts that run the guest's
//! command streams, resources the guest names by number, fences that retire
//! in the order they were made).
//!
//! A renderer belongs to one thread, the GPU's: a GL context is current on one
//! thread at a time, so the device makes the renderer there (`RendererFactory`)
//! and calls it from nowhere else.

use std::os::fd::OwnedFd;

use crate::display::Rect;

/// A span of host memory the renderer may read and write: a piece of guest
/// RAM, for a resource's backing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostSlice {
    pub ptr: *mut u8,
    pub len: usize,
}

/// One of the renderer's capability sets, which the guest's driver reads to
/// learn what the host GPU can do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capset {
    pub id: u32,
    pub max_version: u32,
    pub max_size: u32,
}

/// A virtio-gpu host3D blob passed to the renderer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BlobResource {
    pub id: u32,
    pub context: u32,
    pub blob_id: u64,
    pub memory: u32,
    pub flags: u32,
    pub size: u64,
}

/// RESOURCE_CREATE_3D's arguments, as the guest gave them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Resource3d {
    pub id: u32,
    pub target: u32,
    pub format: u32,
    pub bind: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub array_size: u32,
    pub last_level: u32,
    pub nr_samples: u32,
    pub flags: u32,
}

/// A box in a resource, in texels.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Box3d {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub w: u32,
    pub h: u32,
    pub d: u32,
}

/// A copy between a resource and its backing in guest memory.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Transfer {
    pub resource: u32,
    pub context: u32,
    pub level: u32,
    pub stride: u32,
    pub layer_stride: u32,
    pub region: Box3d,
    /// Where in the backing the box starts.
    pub offset: u64,
}

/// A failure, as the negative errno the renderer gave or a description.
pub type RendererResult<T> = std::result::Result<T, String>;

pub trait Renderer {
    /// Whether this renderer implements the virtio-gpu resource-blob protocol.
    fn supports_blob(&self) -> bool {
        false
    }

    /// Alignment required for guest mappings of host-visible blobs.
    fn blob_alignment(&self) -> u32 {
        0
    }

    /// Creates a host3D blob resource.
    fn create_blob(&mut self, _blob: &BlobResource, _backing: &[HostSlice]) -> RendererResult<()> {
        Err("blob resources are unsupported".into())
    }

    /// Maps a blob into a host-visible guest RAM address when the renderer can
    /// provide a fixed mapping. Returns false when the caller should try an FD.
    fn map_blob_fixed(&mut self, _resource: u32, _host_addr: *mut u8) -> RendererResult<bool> {
        Ok(false)
    }

    /// Exports a blob's shared-memory file descriptor and its handle type.
    fn export_blob(&mut self, _resource: u32) -> RendererResult<(u32, OwnedFd)> {
        Err("blob export is unsupported".into())
    }

    /// Returns the guest mapping cache policy for a blob.
    fn blob_map_info(&mut self, _resource: u32) -> RendererResult<u32> {
        Err("blob mapping is unsupported".into())
    }

    /// Notifies the renderer that a guest blob mapping has been released.
    fn unmap_blob(&mut self, _resource: u32) -> RendererResult<()> {
        Err("blob unmapping is unsupported".into())
    }
    /// For log messages: the renderer and what it runs on.
    fn description(&self) -> String;

    fn capsets(&self) -> Vec<Capset>;
    /// Fills `out` with capability set `id` at `version`; `out` is the set's
    /// `max_size`.
    fn fill_capset(&mut self, id: u32, version: u32, out: &mut [u8]);

    fn create_context(&mut self, context: u32, context_init: u32, name: &[u8]) -> RendererResult<()>;
    fn destroy_context(&mut self, context: u32);
    fn attach_resource(&mut self, context: u32, resource: u32);
    fn detach_resource(&mut self, context: u32, resource: u32);

    fn create_resource(&mut self, resource: &Resource3d) -> RendererResult<()>;
    /// Frees the resource, and its backing if it still has one.
    fn destroy_resource(&mut self, resource: u32);
    /// Gives the resource its backing, the pieces in order.
    ///
    /// # Safety
    ///
    /// Every slice stays valid for reads and writes until the backing is
    /// detached or the resource destroyed.
    unsafe fn attach_backing(
        &mut self,
        resource: u32,
        backing: Vec<HostSlice>,
    ) -> RendererResult<()>;
    fn detach_backing(&mut self, resource: u32);

    /// Backing to resource.
    fn transfer_to_host(&mut self, transfer: &Transfer) -> RendererResult<()>;
    /// Resource to backing.
    fn transfer_from_host(&mut self, transfer: &Transfer) -> RendererResult<()>;
    /// Runs a context's command stream.
    fn submit(&mut self, context: u32, commands: &mut [u32]) -> RendererResult<()>;

    /// A fence after everything submitted so far. Fences retire in the order
    /// they were made.
    fn create_fence(&mut self, fence: u64, context: u32) -> RendererResult<()>;
    /// Looks for finished work, and returns the fences retired since the last
    /// call, oldest first.
    fn poll(&mut self) -> Vec<u64>;

    /// Copies `rect` of a 2D resource into `out`, `stride` bytes a row, as
    /// R, G, B, A.
    fn read(
        &mut self,
        resource: u32,
        rect: Rect,
        out: &mut [u8],
        stride: u32,
    ) -> RendererResult<()>;
    /// A cursor resource's image: its width, height and pixels (B, G, R, A).
    fn cursor_image(&mut self, resource: u32) -> Option<(u32, u32, Vec<u8>)>;

    /// Forgets every context and resource, as when the guest resets the device.
    fn reset(&mut self);
}

/// Makes a renderer on the thread that will use it.
pub type RendererFactory = Box<dyn FnOnce() -> crate::Result<Box<dyn Renderer>> + Send>;

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The 3D renderer behind virtio-gpu: virglrenderer, which turns the guest's
//! Gallium command streams into OpenGL ES, on EGL contexts made here through
//! ANGLE (Vulkan by default, where it gives the guest OpenGL ES 3.1).
//!
//! Both libraries are loaded when a machine asks for 3D, not linked, so crate
//! builds and runs without them; a GPU that cannot start is 2D. They are found
//! by name: `CRATE_VIRGLRENDERER` and `CRATE_EGL` override the defaults
//! (`libvirglrenderer.1.dylib` and ANGLE's `EGL.framework` on a Mac, which
//! dyld finds through `DYLD_LIBRARY_PATH` and `DYLD_FRAMEWORK_PATH`; the
//! frameworks an iOS app embeds; the usual sonames elsewhere). ANGLE's
//! back end is `CRATE_ANGLE_PLATFORM` (vulkan, metal or default), or ANGLE's
//! own `ANGLE_DEFAULT_PLATFORM` when that is set.
//!
//! Everything here runs on the GPU's thread, which made the renderer:
//! virglrenderer calls back into this module for its GL contexts and fences on
//! that thread, from inside the calls made to it.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::ptr;
use std::os::fd::{FromRawFd, OwnedFd};

use crate_core::display::Rect;
use crate_core::gpu::{
    BlobResource, Box3d, Capset, ContextFence, HostSlice, Renderer, RendererFactory, RendererResult,
    Resource3d, Transfer,
};
use crate_core::{Error, Result, debug, info, warn};

/// A renderer for a virtio-gpu, made on the GPU's thread.
pub fn factory() -> RendererFactory {
    Box::new(|| Virgl::new().map(|v| Box::new(v) as Box<dyn Renderer>))
}

type EglDisplay = *mut c_void;
type EglConfig = *mut c_void;
type EglContext = *mut c_void;
type EglSurface = *mut c_void;

const EGL_NONE: i32 = 0x3038;
const EGL_SUCCESS: i32 = 0x3000;
const EGL_VENDOR: i32 = 0x3053;
const EGL_EXTENSIONS: i32 = 0x3055;
const EGL_OPENGL_ES_API: u32 = 0x30a0;
const EGL_RENDERABLE_TYPE: i32 = 0x3040;
const EGL_OPENGL_ES3_BIT: i32 = 0x40;
const EGL_SURFACE_TYPE: i32 = 0x3033;
const EGL_PBUFFER_BIT: i32 = 0x1;
const EGL_RED_SIZE: i32 = 0x3024;
const EGL_GREEN_SIZE: i32 = 0x3023;
const EGL_BLUE_SIZE: i32 = 0x3022;
const EGL_ALPHA_SIZE: i32 = 0x3021;
const EGL_WIDTH: i32 = 0x3057;
const EGL_HEIGHT: i32 = 0x3056;
const EGL_CONTEXT_MAJOR_VERSION: i32 = 0x3098;
const EGL_CONTEXT_MINOR_VERSION: i32 = 0x30fb;
const EGL_CONTEXT_OPENGL_DEBUG: i32 = 0x31b0;
const GL_DEBUG_OUTPUT: u32 = 0x92e0;
const GL_DEBUG_OUTPUT_SYNCHRONOUS: u32 = 0x8242;
// EGL_ANGLE_platform_angle and its Vulkan and Metal extensions.
const EGL_PLATFORM_ANGLE_ANGLE: u32 = 0x3202;
const EGL_PLATFORM_ANGLE_TYPE_ANGLE: i32 = 0x3203;
const EGL_PLATFORM_ANGLE_TYPE_VULKAN_ANGLE: i32 = 0x3450;
const EGL_PLATFORM_ANGLE_TYPE_METAL_ANGLE: i32 = 0x3489;

const EGL_DRAW: i32 = 0x3059;
const EGL_READ: i32 = 0x305a;

const GL_RENDERER: u32 = 0x1f01;
const GL_VERSION: u32 = 0x1f02;
const GL_READ_FRAMEBUFFER: u32 = 0x8ca8;
const GL_COLOR_ATTACHMENT0: u32 = 0x8ce0;
const GL_TEXTURE_2D: u32 = 0x0de1;
const GL_FRAMEBUFFER_COMPLETE: u32 = 0x8cd5;
const GL_PACK_ALIGNMENT: u32 = 0x0d05;
const GL_PACK_ROW_LENGTH: u32 = 0x0d02;
const GL_RGBA: u32 = 0x1908;
const GL_UNSIGNED_BYTE: u32 = 0x1401;

// virglrenderer.h
const VIRGL_RENDERER_USE_GLES: c_int = 1 << 4;
const VIRGL_RENDERER_VENUS: c_int = 1 << 6;
const VIRGL_RENDERER_RENDER_SERVER: c_int = 1 << 9;
const VIRGL_RENDERER_CALLBACKS_VERSION: c_int = 4;
const CAPSET_VIRGL: u32 = 1;
const CAPSET_VIRGL2: u32 = 2;
const CAPSET_VENUS: u32 = 4;

#[repr(C)]
struct GlContextParam {
    version: c_int,
    shared: bool,
    major: c_int,
    minor: c_int,
    compat: c_int,
}

#[repr(C)]
struct Callbacks {
    version: c_int,
    write_fence: extern "C" fn(*mut c_void, u32),
    create_gl_context: extern "C" fn(*mut c_void, c_int, *mut GlContextParam) -> *mut c_void,
    destroy_gl_context: extern "C" fn(*mut c_void, *mut c_void),
    make_current: extern "C" fn(*mut c_void, c_int, *mut c_void) -> c_int,
    get_drm_fd: Option<extern "C" fn(*mut c_void) -> c_int>,
    write_context_fence: Option<extern "C" fn(*mut c_void, u32, u32, u64)>,
    get_server_fd: Option<extern "C" fn(*mut c_void, u32) -> c_int>,
    get_egl_display: extern "C" fn(*mut c_void) -> *mut c_void,
}

/// virgl_renderer_resource_info.
#[repr(C)]
#[derive(Default)]
struct ResourceInfo {
    handle: u32,
    virgl_format: u32,
    width: u32,
    height: u32,
    depth: u32,
    flags: u32,
    tex_id: u32,
    stride: u32,
    drm_fourcc: c_int,
    fd: c_int,
}

#[repr(C)]
struct VirglResourceArgs {
    handle: u32,
    target: u32,
    format: u32,
    bind: u32,
    width: u32,
    height: u32,
    depth: u32,
    array_size: u32,
    last_level: u32,
    nr_samples: u32,
    flags: u32,
}

#[repr(C)]
struct VirglBlobArgs {
    res_handle: u32,
    ctx_id: u32,
    blob_mem: u32,
    blob_flags: u32,
    blob_id: u64,
    size: u64,
    iovecs: *const libc::iovec,
    num_iovs: u32,
}

/// Optional entry points provided by the Venus-enabled virglrenderer build.
struct BlobFunctions {
    create: unsafe extern "C" fn(*const VirglBlobArgs) -> c_int,
    export: unsafe extern "C" fn(u32, *mut u32, *mut c_int) -> c_int,
    get_map_info: unsafe extern "C" fn(u32, *mut u32) -> c_int,
    map_fixed: Option<unsafe extern "C" fn(u32, *mut c_void) -> c_int>,
}

impl BlobFunctions {
    fn load(library: &Library) -> Option<Self> {
        let create = library.symbol("virgl_renderer_resource_create_blob").ok()?;
        let export = library.symbol("virgl_renderer_resource_export_blob").ok()?;
        let get_map_info = library.symbol("virgl_renderer_resource_get_map_info").ok()?;
        let map_fixed = library.symbol("virgl_renderer_resource_map_fixed").ok().map(|p| unsafe {
            std::mem::transmute::<*mut c_void, unsafe extern "C" fn(u32, *mut c_void) -> c_int>(p)
        });
        Some(Self {
            create: unsafe { std::mem::transmute(create) },
            export: unsafe { std::mem::transmute(export) },
            get_map_info: unsafe { std::mem::transmute(get_map_info) },
            map_fixed,
        })
    }
}

/// A library opened with dlopen, closed never: its functions may still be
/// on some thread's stack, and libraries like these do not unload cleanly.
struct Library(*mut c_void);

impl Library {
    fn open(names: &[String]) -> Result<Library> {
        let mut tried = Vec::new();
        for name in names {
            let c =
                CString::new(name.as_str()).map_err(|_| Error::InvalidArgument(name.clone()))?;
            // SAFETY: dlopen with a valid string; RTLD_NOW resolves everything
            // now rather than failing later on a missing symbol.
            let handle = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL) };
            if !handle.is_null() {
                debug!("virgl: loaded {name}");
                return Ok(Library(handle));
            }
            // SAFETY: dlerror returns NULL or a message valid until the next call.
            let why = unsafe { libc::dlerror() };
            let why = if why.is_null() {
                String::from("not found")
            } else {
                // SAFETY: as above.
                unsafe { CStr::from_ptr(why) }
                    .to_string_lossy()
                    .into_owned()
            };
            tried.push(format!("{name}: {why}"));
        }
        Err(Error::Unsupported(format!(
            "cannot load {}",
            tried.join("; ")
        )))
    }

    fn symbol(&self, name: &str) -> Result<*mut c_void> {
        let c = CString::new(name).unwrap();
        // SAFETY: a handle from dlopen and a valid string.
        let f = unsafe { libc::dlsym(self.0, c.as_ptr()) };
        if f.is_null() {
            Err(Error::Unsupported(format!("{name} is missing")))
        } else {
            Ok(f)
        }
    }
}

/// Declares a struct of C functions, and how to load it from something that
/// finds each by name.
macro_rules! functions {
    ($name:ident { $($f:ident: fn($($arg:ty),*) $(-> $ret:ty)?;)* }) => {
        #[allow(non_snake_case)]
        struct $name {
            $($f: unsafe extern "C" fn($($arg),*) $(-> $ret)?,)*
        }

        impl $name {
            fn load(find: impl Fn(&str) -> Result<*mut c_void>) -> Result<Self> {
                Ok($name {
                    // SAFETY: each is the C function of that name, whose
                    // signature the declaration above follows.
                    $($f: unsafe {
                        std::mem::transmute::<*mut c_void, unsafe extern "C" fn($($arg),*) $(-> $ret)?>(
                            find(stringify!($f))?,
                        )
                    },)*
                })
            }
        }
    };
}

functions!(EglFunctions {
    eglGetProcAddress: fn(*const c_char) -> *mut c_void;
    eglGetDisplay: fn(*mut c_void) -> EglDisplay;
    eglInitialize: fn(EglDisplay, *mut i32, *mut i32) -> u32;
    eglBindAPI: fn(u32) -> u32;
    eglChooseConfig: fn(EglDisplay, *const i32, *mut EglConfig, i32, *mut i32) -> u32;
    eglCreateContext: fn(EglDisplay, EglConfig, EglContext, *const i32) -> EglContext;
    eglDestroyContext: fn(EglDisplay, EglContext) -> u32;
    eglCreatePbufferSurface: fn(EglDisplay, EglConfig, *const i32) -> EglSurface;
    eglMakeCurrent: fn(EglDisplay, EglSurface, EglSurface, EglContext) -> u32;
    eglGetError: fn() -> i32;
    eglGetCurrentContext: fn() -> EglContext;
    eglGetCurrentSurface: fn(i32) -> EglSurface;
    eglQueryString: fn(EglDisplay, i32) -> *const c_char;
    eglReleaseThread: fn() -> u32;
});

functions!(VirglFunctions {
    virgl_renderer_init: fn(*mut c_void, c_int, *mut Callbacks) -> c_int;
    virgl_renderer_poll: fn();
    virgl_renderer_get_cursor_data: fn(u32, *mut u32, *mut u32) -> *mut c_void;
    virgl_renderer_resource_create: fn(*mut VirglResourceArgs, *mut libc::iovec, u32) -> c_int;
    virgl_renderer_resource_unref: fn(u32);
    virgl_renderer_context_create: fn(u32, u32, *const c_char) -> c_int;
    virgl_renderer_context_destroy: fn(u32);
    virgl_renderer_submit_cmd: fn(*mut c_void, c_int, c_int) -> c_int;
    virgl_renderer_transfer_read_iov: fn(u32, u32, u32, u32, u32, *mut Box3d, u64, *mut libc::iovec, c_int) -> c_int;
    virgl_renderer_transfer_write_iov: fn(u32, u32, c_int, u32, u32, *mut Box3d, u64, *mut libc::iovec, u32) -> c_int;
    virgl_renderer_get_cap_set: fn(u32, *mut u32, *mut u32);
    virgl_renderer_fill_caps: fn(u32, u32, *mut c_void);
    virgl_renderer_resource_attach_iov: fn(c_int, *mut libc::iovec, c_int) -> c_int;
    virgl_renderer_resource_detach_iov: fn(c_int, *mut *mut libc::iovec, *mut c_int);
    virgl_renderer_create_fence: fn(c_int, u32) -> c_int;
    virgl_renderer_ctx_attach_resource: fn(c_int, c_int);
    virgl_renderer_ctx_detach_resource: fn(c_int, c_int);
    virgl_renderer_resource_get_info: fn(c_int, *mut ResourceInfo) -> c_int;
    virgl_renderer_reset: fn();
});

// The OpenGL ES functions this module calls itself, from eglGetProcAddress.
functions!(GlFunctions {
    glGenFramebuffers: fn(i32, *mut u32);
    glBindFramebuffer: fn(u32, u32);
    glFramebufferTexture2D: fn(u32, u32, u32, u32, i32);
    glCheckFramebufferStatus: fn(u32) -> u32;
    glPixelStorei: fn(u32, i32);
    glReadPixels: fn(i32, i32, i32, i32, u32, u32, *mut c_void);
    glGetError: fn() -> u32;
    glGetString: fn(u32) -> *const c_char;
});

impl EglFunctions {
    /// A GLES or EGL extension function, by name.
    fn proc(&self, name: &str) -> Result<*mut c_void> {
        let c = CString::new(name).unwrap();
        // SAFETY: an EGL query.
        let f = unsafe { (self.eglGetProcAddress)(c.as_ptr()) };
        if f.is_null() {
            Err(Error::Unsupported(format!("{name} is missing")))
        } else {
            Ok(f)
        }
    }
}

type LogCallback = extern "C" fn(c_int, *const c_char, *mut c_void);
type SetLogCallback =
    unsafe extern "C" fn(Option<LogCallback>, *mut c_void, Option<extern "C" fn(*mut c_void)>);
type ContextCreateWithFlags = unsafe extern "C" fn(u32, u32, u32, *const c_char) -> c_int;
type ContextCreateFence = unsafe extern "C" fn(u32, u32, u32, u64) -> c_int;

/// What virglrenderer's callbacks reach: the EGL display and contexts.
struct Gl {
    egl: EglFunctions,
    display: EglDisplay,
    config: EglConfig,
    /// The context virglrenderer's own share with; also kept current while
    /// none of theirs is.
    primary: EglContext,
    /// A 1x1 surface for EGLs without surfaceless contexts; null otherwise.
    surface: EglSurface,
    /// Fences virglrenderer reported done, not yet collected.
    retired: RefCell<Vec<u32>>,
    /// Venus fences retired on their individual context timelines.
    retired_context: RefCell<Vec<ContextFence>>,
    /// CRATE_GL_DEBUG: debug contexts, their messages logged.
    debug: bool,
}

impl Gl {
    fn error(&self) -> String {
        // SAFETY: a plain EGL query.
        format!("EGL error {:#x}", unsafe { (self.egl.eglGetError)() })
    }

    fn create_context(&self, share: EglContext, major: i32, minor: i32) -> EglContext {
        let attribs = [
            EGL_CONTEXT_MAJOR_VERSION,
            major,
            EGL_CONTEXT_MINOR_VERSION,
            minor,
            EGL_CONTEXT_OPENGL_DEBUG,
            self.debug as i32,
            EGL_NONE,
        ];
        // SAFETY: a valid display and config, attributes ending in EGL_NONE.
        let context = unsafe {
            (self.egl.eglCreateContext)(self.display, self.config, share, attribs.as_ptr())
        };
        if self.debug && !context.is_null() {
            self.log_messages(context);
        }
        context
    }

    /// Has `context` report its GL errors and warnings to the log.
    fn log_messages(&self, context: EglContext) {
        let previous = self.current();
        if !self.make_current(context) {
            return;
        }
        if let (Ok(set), Ok(on)) = (
            self.egl.proc("glDebugMessageCallbackKHR"),
            self.egl.proc("glEnable"),
        ) {
            type SetCallback = unsafe extern "C" fn(DebugCallback, *const c_void);
            type Enable = unsafe extern "C" fn(u32);
            // SAFETY: the GLES functions of those names, called while
            // `context` is current; the callback lives forever.
            unsafe {
                let set = std::mem::transmute::<*mut c_void, SetCallback>(set);
                let on = std::mem::transmute::<*mut c_void, Enable>(on);
                set(gl_message, ptr::null());
                on(GL_DEBUG_OUTPUT);
                on(GL_DEBUG_OUTPUT_SYNCHRONOUS);
            }
        }
        self.restore(previous);
    }

    /// What is current now, to put back with `restore`.
    fn current(&self) -> (EglContext, EglSurface, EglSurface) {
        // SAFETY: plain EGL queries.
        unsafe {
            (
                (self.egl.eglGetCurrentContext)(),
                (self.egl.eglGetCurrentSurface)(EGL_DRAW),
                (self.egl.eglGetCurrentSurface)(EGL_READ),
            )
        }
    }

    fn restore(&self, (context, draw, read): (EglContext, EglSurface, EglSurface)) {
        // SAFETY: what `current` returned.
        unsafe { (self.egl.eglMakeCurrent)(self.display, draw, read, context) };
    }

    fn make_current(&self, context: EglContext) -> bool {
        // SAFETY: the display's own surface (or none) and context.
        unsafe { (self.egl.eglMakeCurrent)(self.display, self.surface, self.surface, context) != 0 }
    }
}

type DebugCallback = extern "C" fn(u32, u32, u32, u32, i32, *const c_char, *const c_void);

extern "C" fn gl_message(
    _source: u32,
    kind: u32,
    _id: u32,
    severity: u32,
    _length: i32,
    message: *const c_char,
    _: *const c_void,
) {
    if message.is_null() {
        return;
    }
    // SAFETY: GL passes a C string.
    let text = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    warn!("GL ({kind:#x}, severity {severity:#x}): {text}");
}

extern "C" fn write_fence(cookie: *mut c_void, fence: u32) {
    // SAFETY: the cookie is the renderer's Gl, alive while virglrenderer is.
    let gl = unsafe { &*(cookie as *const Gl) };
    gl.retired.borrow_mut().push(fence);
}

extern "C" fn write_context_fence(cookie: *mut c_void, context: u32, ring_idx: u32, fence: u64) {
    // SAFETY: the cookie is the renderer's Gl, alive while virglrenderer is.
    let gl = unsafe { &*(cookie as *const Gl) };
    gl.retired_context.borrow_mut().push(ContextFence { context, ring_idx, fence });
}

extern "C" fn create_gl_context(
    cookie: *mut c_void,
    _scanout: c_int,
    param: *mut GlContextParam,
) -> *mut c_void {
    // SAFETY: as in write_fence; virglrenderer passes a valid parameter block.
    let (gl, param) = unsafe { (&*(cookie as *const Gl), &*param) };
    // Every context shares with the primary, whatever `shared` says: fences
    // and textures cross between virglrenderer's contexts, and its fences
    // are waited on in another context than the one that made them.
    let context = gl.create_context(gl.primary, param.major, param.minor);
    if gl.debug {
        debug!(
            "virgl: created {context:?} ({}.{}, shared {})",
            param.major, param.minor, param.shared
        );
    }
    if context.is_null() {
        debug!(
            "virgl: no OpenGL ES {}.{} context: {}",
            param.major,
            param.minor,
            gl.error()
        );
    }
    context
}

extern "C" fn destroy_gl_context(cookie: *mut c_void, context: *mut c_void) {
    // SAFETY: as in write_fence.
    let gl = unsafe { &*(cookie as *const Gl) };
    // SAFETY: a context made by create_gl_context.
    unsafe { (gl.egl.eglDestroyContext)(gl.display, context) };
}

extern "C" fn make_current(cookie: *mut c_void, _scanout: c_int, context: *mut c_void) -> c_int {
    // SAFETY: as in write_fence.
    let gl = unsafe { &*(cookie as *const Gl) };
    if gl.debug {
        debug!("virgl: current {context:?}");
    }
    if gl.make_current(context) {
        0
    } else {
        warn!("virgl: eglMakeCurrent({context:?}): {}", gl.error());
        -1
    }
}

extern "C" fn get_egl_display(cookie: *mut c_void) -> *mut c_void {
    // SAFETY: as in write_fence.
    unsafe { &*(cookie as *const Gl) }.display
}

extern "C" fn log_message(level: c_int, message: *const c_char, _: *mut c_void) {
    if message.is_null() {
        return;
    }
    // SAFETY: virglrenderer passes a C string.
    let text = unsafe { CStr::from_ptr(message) }.to_string_lossy();
    let text = text.trim_end();
    // virgl_log_level_flags: debug, info, warning, error.
    match level {
        3 => warn!("virglrenderer: {text}"),
        2 => info!("virglrenderer: {text}"),
        _ => debug!("virglrenderer: {text}"),
    }
}

fn library_names(variable: &str, defaults: &[&str]) -> Vec<String> {
    match std::env::var(variable) {
        Ok(name) if !name.is_empty() => vec![name],
        _ => defaults.iter().map(|s| s.to_string()).collect(),
    }
}

#[cfg(target_os = "ios")]
const VIRGL_NAMES: &[&str] = &[
    "@rpath/virglrenderer.venus.framework/virglrenderer.venus",
    "@rpath/virglrenderer.1.framework/virglrenderer.1",
];
#[cfg(target_os = "ios")]
const EGL_NAMES: &[&str] = &["@rpath/EGL.framework/EGL"];
#[cfg(target_os = "macos")]
const VIRGL_NAMES: &[&str] = &["libvirglrenderer.1.dylib"];
#[cfg(target_os = "macos")]
const EGL_NAMES: &[&str] = &["EGL.framework/Versions/Current/EGL", "libEGL.dylib"];
#[cfg(not(target_vendor = "apple"))]
const VIRGL_NAMES: &[&str] = &["libvirglrenderer.so.1"];
#[cfg(not(target_vendor = "apple"))]
const EGL_NAMES: &[&str] = &["libEGL.so.1"];

/// virglrenderer and the EGL state it runs on.
struct Inner {
    virgl: VirglFunctions,
    functions: GlFunctions,
    /// The primary context's framebuffer for reading resources back.
    framebuffer: u32,
    /// Boxed: virglrenderer keeps pointers to both.
    gl: Box<Gl>,
    _callbacks: Box<Callbacks>,
    description: String,
    /// The backing handed to each resource, which is ours to free.
    backings: HashMap<u32, Box<[libc::iovec]>>,
    blob_backings: HashMap<u32, Box<[libc::iovec]>>,
    blobs: Option<BlobFunctions>,
    context_create_with_flags: Option<ContextCreateWithFlags>,
    context_create_fence: Option<ContextCreateFence>,
    /// Fences made and not yet retired, oldest first.
    fences: VecDeque<u64>,
    _libraries: [Library; 2],
}

impl Inner {
    /// Loads the libraries and starts virglrenderer.
    fn start() -> Result<Self> {
        let egl_library = Library::open(&library_names("CRATE_EGL", EGL_NAMES))?;
        let virgl_library = Library::open(&library_names("CRATE_VIRGLRENDERER", VIRGL_NAMES))?;
        let egl = EglFunctions::load(|name| egl_library.symbol(name))?;
        let virgl = VirglFunctions::load(|name| virgl_library.symbol(name))?;
        let blobs = BlobFunctions::load(&virgl_library);
        let context_create_with_flags = virgl_library.symbol("virgl_renderer_context_create_with_flags").ok().map(|symbol| unsafe {
            std::mem::transmute::<*mut c_void, ContextCreateWithFlags>(symbol)
        });
        let context_create_fence = virgl_library.symbol("virgl_renderer_context_create_fence").ok().map(|symbol| unsafe {
            std::mem::transmute::<*mut c_void, ContextCreateFence>(symbol)
        });
        if let Ok(set_log) = virgl_library.symbol("virgl_set_log_callback") {
            // SAFETY: virglrenderer's function of that name.
            let set_log = unsafe { std::mem::transmute::<*mut c_void, SetLogCallback>(set_log) };
            // SAFETY: a callback valid for the life of the process.
            unsafe { set_log(Some(log_message), ptr::null_mut(), None) };
        }

        let (display, platform) = open_display(&egl)?;
        let (mut major, mut minor) = (0, 0);
        // SAFETY: a display from EGL and two out-parameters.
        if unsafe { (egl.eglInitialize)(display, &mut major, &mut minor) } == 0 {
            // SAFETY: as above.
            let error = unsafe { (egl.eglGetError)() };
            return Err(Error::Unsupported(format!("eglInitialize: {error:#x}")));
        }
        let query = |name: i32| {
            // SAFETY: an initialized display.
            let s = unsafe { (egl.eglQueryString)(display, name) };
            if s.is_null() {
                String::new()
            } else {
                // SAFETY: EGL returns a C string.
                unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
            }
        };
        let extensions = query(EGL_EXTENSIONS);
        debug!(
            "EGL {major}.{minor} from {}: {extensions}",
            query(EGL_VENDOR)
        );
        // SAFETY: a plain EGL call.
        unsafe { (egl.eglBindAPI)(EGL_OPENGL_ES_API) };
        let config_attribs = [
            EGL_RENDERABLE_TYPE,
            EGL_OPENGL_ES3_BIT,
            EGL_SURFACE_TYPE,
            EGL_PBUFFER_BIT,
            EGL_RED_SIZE,
            8,
            EGL_GREEN_SIZE,
            8,
            EGL_BLUE_SIZE,
            8,
            EGL_ALPHA_SIZE,
            8,
            EGL_NONE,
        ];
        let mut config = ptr::null_mut();
        let mut count = 0;
        // SAFETY: an initialized display, attributes ending in EGL_NONE, room
        // for one config.
        let chose = unsafe {
            (egl.eglChooseConfig)(display, config_attribs.as_ptr(), &mut config, 1, &mut count)
        };
        if chose == 0 || count == 0 {
            return Err(Error::Unsupported("no EGL config for OpenGL ES 3".into()));
        }
        let mut gl = Box::new(Gl {
            egl,
            display,
            config,
            primary: ptr::null_mut(),
            surface: ptr::null_mut(),
            retired: RefCell::new(Vec::new()),
            retired_context: RefCell::new(Vec::new()),
            debug: std::env::var_os("CRATE_GL_DEBUG").is_some(),
        });
        if !extensions
            .split(' ')
            .any(|e| e == "EGL_KHR_surfaceless_context")
        {
            let attribs = [EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE];
            // SAFETY: an initialized display and its config.
            gl.surface =
                unsafe { (gl.egl.eglCreatePbufferSurface)(display, config, attribs.as_ptr()) };
            if gl.surface.is_null() {
                return Err(Error::Unsupported(format!("no surface: {}", gl.error())));
            }
        }
        gl.primary = [(3, 2), (3, 1), (3, 0)]
            .iter()
            .map(|&(major, minor)| gl.create_context(ptr::null_mut(), major, minor))
            .find(|c| !c.is_null())
            .ok_or_else(|| Error::Unsupported(format!("no OpenGL ES 3 context: {}", gl.error())))?;
        if !gl.make_current(gl.primary) {
            return Err(Error::Unsupported(format!(
                "eglMakeCurrent: {}",
                gl.error()
            )));
        }
        let functions = GlFunctions::load(|name| gl.egl.proc(name))?;
        let renderer = gl_string(&functions, GL_RENDERER);
        let version = gl_string(&functions, GL_VERSION);
        let mut framebuffer = 0;
        // SAFETY: the primary context is current.
        unsafe { (functions.glGenFramebuffers)(1, &mut framebuffer) };

        let mut callbacks = Box::new(Callbacks {
            version: VIRGL_RENDERER_CALLBACKS_VERSION,
            write_fence,
            create_gl_context,
            destroy_gl_context,
            make_current,
            get_drm_fd: None,
            write_context_fence: Some(write_context_fence),
            get_server_fd: None,
            get_egl_display,
        });
        let cookie = &*gl as *const Gl as *mut c_void;
        // SAFETY: the cookie and callbacks are boxed and live as long as the
        // renderer, which cleans up before dropping them.
        let status = unsafe {
            (virgl.virgl_renderer_init)(
                cookie,
                VIRGL_RENDERER_USE_GLES | VIRGL_RENDERER_VENUS | VIRGL_RENDERER_RENDER_SERVER,
                &mut *callbacks,
            )
        };
        if status != 0 {
            gl.make_current(ptr::null_mut());
            // SAFETY: the context made above.
            unsafe { (gl.egl.eglDestroyContext)(display, gl.primary) };
            return Err(Error::Unsupported(format!(
                "virgl_renderer_init failed ({status})"
            )));
        }
        Ok(Inner {
            virgl,
            functions,
            framebuffer,
            gl,
            _callbacks: callbacks,
            description: format!("virglrenderer, ANGLE {platform}: {renderer}, {version}"),
            backings: HashMap::new(),
            blob_backings: HashMap::new(),
            blobs,
            context_create_with_flags,
            context_create_fence,
            fences: VecDeque::new(),
            _libraries: [egl_library, virgl_library],
        })
    }

    fn detach(&mut self, resource: u32) {
        if self.backings.contains_key(&resource) {
            let (mut iov, mut count) = (ptr::null_mut(), 0);
            // SAFETY: the resource has the backing this module attached,
            // freed below once virglrenderer has let go of it.
            unsafe {
                (self.virgl.virgl_renderer_resource_detach_iov)(
                    resource as c_int,
                    &mut iov,
                    &mut count,
                )
            };
            self.backings.remove(&resource);
        }
    }
}

/// A string from the current context.
fn gl_string(gl: &GlFunctions, name: u32) -> String {
    // SAFETY: a context is current.
    let s = unsafe { (gl.glGetString)(name) };
    if s.is_null() {
        String::new()
    } else {
        // SAFETY: GL returns a C string.
        unsafe { CStr::from_ptr(s) }.to_string_lossy().into_owned()
    }
}

/// The display: ANGLE on the platform asked for, or EGL's default.
fn open_display(egl: &EglFunctions) -> Result<(EglDisplay, &'static str)> {
    let wanted = std::env::var("CRATE_ANGLE_PLATFORM").unwrap_or_default();
    let angle_env = std::env::var_os("ANGLE_DEFAULT_PLATFORM").is_some();
    let platform = match wanted.as_str() {
        "metal" => Some((EGL_PLATFORM_ANGLE_TYPE_METAL_ANGLE, "Metal")),
        "default" => None,
        "vulkan" => Some((EGL_PLATFORM_ANGLE_TYPE_VULKAN_ANGLE, "Vulkan")),
        _ if angle_env => None,
        _ => Some((EGL_PLATFORM_ANGLE_TYPE_VULKAN_ANGLE, "Vulkan")),
    };
    if let Some((kind, name)) = platform {
        if let Ok(get) = egl.proc("eglGetPlatformDisplayEXT") {
            type GetPlatformDisplay =
                unsafe extern "C" fn(u32, *mut c_void, *const i32) -> EglDisplay;
            // SAFETY: EGL_EXT_platform_base's signature, whose attributes are
            // EGLint (EGL 1.5's eglGetPlatformDisplay takes EGLAttrib).
            let get = unsafe { std::mem::transmute::<*mut c_void, GetPlatformDisplay>(get) };
            let attribs = [EGL_PLATFORM_ANGLE_TYPE_ANGLE, kind, EGL_NONE];
            // SAFETY: ANGLE's platform with attributes ending in EGL_NONE.
            let display =
                unsafe { get(EGL_PLATFORM_ANGLE_ANGLE, ptr::null_mut(), attribs.as_ptr()) };
            if !display.is_null() {
                return Ok((display, name));
            }
            warn!("virgl: ANGLE has no {name} display; trying EGL's default");
        }
    }
    // SAFETY: EGL_DEFAULT_DISPLAY.
    let display = unsafe { (egl.eglGetDisplay)(ptr::null_mut()) };
    if display.is_null() {
        // SAFETY: a plain EGL call.
        let error = unsafe { (egl.eglGetError)() };
        if error != EGL_SUCCESS {
            return Err(Error::Unsupported(format!("no EGL display: {error:#x}")));
        }
        return Err(Error::Unsupported("no EGL display".into()));
    }
    Ok((
        display,
        if angle_env {
            "(ANGLE_DEFAULT_PLATFORM)"
        } else {
            "default"
        },
    ))
}

impl Inner {
    /// Leaves the renderer ready for another machine, on another thread:
    /// every guest context and resource gone, no context current here.
    fn park(&mut self) -> Parked {
        self.reset();
        let current = self.gl.current();
        self.gl.make_current(ptr::null_mut());
        // SAFETY: a plain EGL call; this thread is done with EGL.
        unsafe { (self.gl.egl.eglReleaseThread)() };
        Parked(current)
    }

    /// Takes the renderer up on this thread: what was current when it was
    /// parked is current again, as virglrenderer believes.
    fn resume(&mut self, parked: Parked) {
        self.gl.restore(parked.0);
        self.gl.retired.borrow_mut().clear();
    }
}

/// What was current when a renderer was parked.
struct Parked((EglContext, EglSurface, EglSurface));

/// A parked renderer, waiting for the next machine.
struct Kept(Inner, Parked);

// SAFETY: a kept renderer is used by no thread: it was parked with no context
// current, and the thread that takes it resumes it before any use.
unsafe impl Send for Kept {}

/// virglrenderer stays for the life of the process once started:
/// virgl_renderer_cleanup waits for the thread of its render server, which
/// this build starts even without venus and which never finishes, so freeing
/// a machine would hang. A machine's renderer is parked here instead, and the
/// next machine gets it back.
static KEPT: std::sync::Mutex<Option<Kept>> = std::sync::Mutex::new(None);

/// The renderer a virtio-gpu uses: virglrenderer, started on the first use
/// and kept afterwards.
pub struct Virgl {
    inner: Option<Inner>,
}

impl Virgl {
    pub fn new() -> Result<Self> {
        let kept = KEPT.lock().unwrap_or_else(|e| e.into_inner()).take();
        let inner = match kept {
            Some(Kept(mut inner, parked)) => {
                inner.resume(parked);
                inner
            }
            None => Inner::start()?,
        };
        Ok(Virgl { inner: Some(inner) })
    }

    fn inner(&mut self) -> &mut Inner {
        self.inner.as_mut().expect("a renderer")
    }
}

impl Drop for Virgl {
    fn drop(&mut self) {
        if let Some(mut inner) = self.inner.take() {
            let parked = inner.park();
            *KEPT.lock().unwrap_or_else(|e| e.into_inner()) = Some(Kept(inner, parked));
        }
    }
}

fn status(what: &str, code: c_int) -> RendererResult<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(format!("{what}: {code}"))
    }
}

impl Renderer for Virgl {
    fn supports_blob(&self) -> bool {
        self.inner.as_ref().is_some_and(|inner| inner.blobs.is_some())
    }

    fn blob_alignment(&self) -> u32 {
        if self.supports_blob() { 16384 } else { 0 }
    }

    fn create_blob(&mut self, blob: &BlobResource, backing: &[HostSlice]) -> RendererResult<()> {
        let inner = self.inner();
        let functions = inner.blobs.as_ref().ok_or_else(|| "Venus blob API is unavailable".to_string())?;
        let iovecs: Box<[libc::iovec]> = backing
            .iter()
            .map(|slice| libc::iovec {
                iov_base: slice.ptr.cast(),
                iov_len: slice.len,
            })
            .collect();
        let args = VirglBlobArgs {
            res_handle: blob.id,
            ctx_id: blob.context,
            blob_mem: blob.memory,
            blob_flags: blob.flags,
            blob_id: blob.blob_id,
            size: blob.size,
            iovecs: if iovecs.is_empty() { ptr::null() } else { iovecs.as_ptr() },
            num_iovs: iovecs.len() as u32,
        };
        // SAFETY: args and iovec array remain valid for the call; guest backing
        // remains valid until this resource is destroyed.
        let code = unsafe { (functions.create)(&args) };
        status("resource_create_blob", code)?;
        if !iovecs.is_empty() {
            inner.blob_backings.insert(blob.id, iovecs);
        }
        Ok(())
    }

    fn map_blob_fixed(&mut self, resource: u32, host_addr: *mut u8) -> RendererResult<bool> {
        let inner = self.inner();
        let Some(functions) = &inner.blobs else { return Ok(false) };
        let Some(map_fixed) = functions.map_fixed else { return Ok(false) };
        // SAFETY: host_addr is a reserved, page-aligned range in the guest's
        // host-visible RAM window, held for the lifetime of the mapping.
        let code = unsafe { map_fixed(resource, host_addr.cast()) };
        if code == 0 {
            Ok(true)
        } else if code == -libc::EOPNOTSUPP || code == -libc::ENOTSUP {
            Ok(false)
        } else {
            Err(format!("resource_map_fixed: {code}"))
        }
    }

    fn export_blob(&mut self, resource: u32) -> RendererResult<(u32, OwnedFd)> {
        let inner = self.inner();
        let functions = inner.blobs.as_ref().ok_or_else(|| "Venus blob API is unavailable".to_string())?;
        let (mut fd_type, mut fd) = (0, -1);
        // SAFETY: fd_type and fd are valid output parameters.
        let code = unsafe { (functions.export)(resource, &mut fd_type, &mut fd) };
        status("resource_export_blob", code)?;
        if fd < 0 {
            return Err("resource_export_blob returned an invalid FD".into());
        }
        // SAFETY: successful export transfers ownership of this new FD.
        Ok((fd_type, unsafe { OwnedFd::from_raw_fd(fd) }))
    }

    fn blob_map_info(&mut self, resource: u32) -> RendererResult<u32> {
        let inner = self.inner();
        let functions = inner.blobs.as_ref().ok_or_else(|| "Venus blob API is unavailable".to_string())?;
        let mut info = 0;
        // SAFETY: info is the API's output parameter.
        let code = unsafe { (functions.get_map_info)(resource, &mut info) };
        status("resource_get_map_info", code)?;
        Ok(info)
    }

    fn description(&self) -> String {
        self.inner
            .as_ref()
            .map(|i| i.description.clone())
            .unwrap_or_default()
    }

    fn capsets(&self) -> Vec<Capset> {
        self.inner.as_ref().map(Inner::capsets).unwrap_or_default()
    }

    fn fill_capset(&mut self, id: u32, version: u32, out: &mut [u8]) {
        self.inner().fill_capset(id, version, out)
    }

    fn create_context(&mut self, context: u32, context_init: u32, name: &[u8]) -> RendererResult<()> {
        self.inner().create_context(context, context_init, name)
    }

    fn destroy_context(&mut self, context: u32) {
        self.inner().destroy_context(context)
    }

    fn attach_resource(&mut self, context: u32, resource: u32) {
        self.inner().attach_resource(context, resource)
    }

    fn detach_resource(&mut self, context: u32, resource: u32) {
        self.inner().detach_resource(context, resource)
    }

    fn create_resource(&mut self, resource: &Resource3d) -> RendererResult<()> {
        self.inner().create_resource(resource)
    }

    fn destroy_resource(&mut self, resource: u32) {
        self.inner().destroy_resource(resource)
    }

    unsafe fn attach_backing(
        &mut self,
        resource: u32,
        backing: Vec<HostSlice>,
    ) -> RendererResult<()> {
        // SAFETY: the caller's promise, passed on.
        unsafe { self.inner().attach_backing(resource, backing) }
    }

    fn detach_backing(&mut self, resource: u32) {
        self.inner().detach_backing(resource)
    }

    fn transfer_to_host(&mut self, transfer: &Transfer) -> RendererResult<()> {
        self.inner().transfer_to_host(transfer)
    }

    fn transfer_from_host(&mut self, transfer: &Transfer) -> RendererResult<()> {
        self.inner().transfer_from_host(transfer)
    }

    fn submit(&mut self, context: u32, commands: &mut [u32]) -> RendererResult<()> {
        self.inner().submit(context, commands)
    }

    fn create_fence(&mut self, fence: u64, context: u32) -> RendererResult<()> {
        self.inner().create_fence(fence, context)
    }

    fn create_context_fence(
        &mut self,
        context: u32,
        flags: u32,
        ring_idx: u32,
        fence: u64,
    ) -> RendererResult<()> {
        self.inner().create_context_fence(context, flags, ring_idx, fence)
    }

    fn poll(&mut self) -> Vec<u64> {
        self.inner().poll()
    }

    fn poll_context_fences(&mut self) -> Vec<ContextFence> {
        self.inner().poll_context_fences()
    }

    fn read(
        &mut self,
        resource: u32,
        rect: Rect,
        out: &mut [u8],
        stride: u32,
    ) -> RendererResult<()> {
        self.inner().read(resource, rect, out, stride)
    }

    fn cursor_image(&mut self, resource: u32) -> Option<(u32, u32, Vec<u8>)> {
        self.inner().cursor_image(resource)
    }

    fn reset(&mut self) {
        self.inner().reset()
    }
}

/// The renderer's work, which `Virgl` passes on.
impl Inner {
    fn capsets(&self) -> Vec<Capset> {
        [CAPSET_VIRGL, CAPSET_VIRGL2, CAPSET_VENUS]
            .iter()
            .filter_map(|&id| {
                if id == CAPSET_VENUS
                    && (self.context_create_with_flags.is_none() || self.context_create_fence.is_none())
                {
                    return None;
                }
                let (mut max_version, mut max_size) = (0, 0);
                // SAFETY: two out-parameters.
                unsafe {
                    (self.virgl.virgl_renderer_get_cap_set)(id, &mut max_version, &mut max_size)
                };
                (max_size > 0).then_some(Capset {
                    id,
                    max_version,
                    max_size,
                })
            })
            .collect()
    }

    fn fill_capset(&mut self, id: u32, version: u32, out: &mut [u8]) {
        let (mut max_version, mut max_size) = (0, 0);
        // SAFETY: two out-parameters.
        unsafe { (self.virgl.virgl_renderer_get_cap_set)(id, &mut max_version, &mut max_size) };
        if max_size as usize > out.len() || max_size == 0 {
            return;
        }
        // SAFETY: `out` holds the set's max_size bytes.
        unsafe { (self.virgl.virgl_renderer_fill_caps)(id, version, out.as_mut_ptr().cast()) };
    }

    fn create_context(&mut self, context: u32, context_init: u32, name: &[u8]) -> RendererResult<()> {
        // virglrenderer makes the new context's GL context current to set it
        // up, yet goes on believing the one before is current: the next
        // command for the earlier context would then run in the new GL
        // context, which has none of its vertex arrays and framebuffers. So
        // whatever was current goes back.
        let previous = self.gl.current();
        let code = if context_init == 0 {
            // Preserve the legacy virgl2 default for guests that don't use
            // VIRTIO_GPU_F_CONTEXT_INIT.
            unsafe {
                (self.virgl.virgl_renderer_context_create)(
                    context,
                    name.len() as u32,
                    name.as_ptr().cast(),
                )
            }
        } else if let Some(create) = self.context_create_with_flags {
            // SAFETY: UTM's virglrenderer exports this API for capset-selected
            // contexts, including Venus (capset 4).
            unsafe { create(context, context_init, name.len() as u32, name.as_ptr().cast()) }
        } else {
            return Err("virglrenderer does not support context_init".into());
        };
        self.gl.restore(previous);
        status("context_create", code)
    }

    fn destroy_context(&mut self, context: u32) {
        // SAFETY: virglrenderer checks the id.
        unsafe { (self.virgl.virgl_renderer_context_destroy)(context) };
    }

    fn attach_resource(&mut self, context: u32, resource: u32) {
        // SAFETY: virglrenderer checks both ids.
        unsafe {
            (self.virgl.virgl_renderer_ctx_attach_resource)(context as c_int, resource as c_int)
        };
    }

    fn detach_resource(&mut self, context: u32, resource: u32) {
        // SAFETY: as above.
        unsafe {
            (self.virgl.virgl_renderer_ctx_detach_resource)(context as c_int, resource as c_int)
        };
    }

    fn create_resource(&mut self, r: &Resource3d) -> RendererResult<()> {
        let mut args = VirglResourceArgs {
            handle: r.id,
            target: r.target,
            format: r.format,
            bind: r.bind,
            width: r.width,
            height: r.height,
            depth: r.depth,
            array_size: r.array_size,
            last_level: r.last_level,
            nr_samples: r.nr_samples,
            flags: r.flags,
        };
        // SAFETY: valid arguments and no backing yet.
        let code =
            unsafe { (self.virgl.virgl_renderer_resource_create)(&mut args, ptr::null_mut(), 0) };
        status("resource_create", code)
    }

    fn destroy_resource(&mut self, resource: u32) {
        self.detach(resource);
        self.blob_backings.remove(&resource);
        // SAFETY: virglrenderer checks the id.
        unsafe { (self.virgl.virgl_renderer_resource_unref)(resource) };
    }

    unsafe fn attach_backing(
        &mut self,
        resource: u32,
        backing: Vec<HostSlice>,
    ) -> RendererResult<()> {
        self.detach(resource);
        let mut iov: Box<[libc::iovec]> = backing
            .iter()
            .map(|s| libc::iovec {
                iov_base: s.ptr.cast(),
                iov_len: s.len,
            })
            .collect();
        // SAFETY: the caller keeps the slices valid until detach; the iovec
        // array is kept in `backings` until then.
        let code = unsafe {
            (self.virgl.virgl_renderer_resource_attach_iov)(
                resource as c_int,
                iov.as_mut_ptr(),
                iov.len() as c_int,
            )
        };
        status("attach_iov", code)?;
        self.backings.insert(resource, iov);
        Ok(())
    }

    fn detach_backing(&mut self, resource: u32) {
        self.detach(resource);
    }

    fn transfer_to_host(&mut self, t: &Transfer) -> RendererResult<()> {
        let mut region = t.region;
        // SAFETY: no iovec: the resource's attached backing is used.
        let code = unsafe {
            (self.virgl.virgl_renderer_transfer_write_iov)(
                t.resource,
                t.context,
                t.level as c_int,
                t.stride,
                t.layer_stride,
                &mut region,
                t.offset,
                ptr::null_mut(),
                0,
            )
        };
        status("transfer_write", code)
    }

    fn transfer_from_host(&mut self, t: &Transfer) -> RendererResult<()> {
        let mut region = t.region;
        // SAFETY: as above.
        let code = unsafe {
            (self.virgl.virgl_renderer_transfer_read_iov)(
                t.resource,
                t.context,
                t.level,
                t.stride,
                t.layer_stride,
                &mut region,
                t.offset,
                ptr::null_mut(),
                0,
            )
        };
        status("transfer_read", code)
    }

    fn submit(&mut self, context: u32, commands: &mut [u32]) -> RendererResult<()> {
        // SAFETY: a 4-byte-aligned buffer of that many words.
        let code = unsafe {
            (self.virgl.virgl_renderer_submit_cmd)(
                commands.as_mut_ptr().cast(),
                context as c_int,
                commands.len() as c_int,
            )
        };
        status("submit_cmd", code)
    }

    fn create_fence(&mut self, fence: u64, context: u32) -> RendererResult<()> {
        // virglrenderer's fences are 32 bits; poll matches them back up.
        // SAFETY: a plain call.
        let code =
            unsafe { (self.virgl.virgl_renderer_create_fence)(fence as u32 as c_int, context) };
        status("create_fence", code)?;
        self.fences.push_back(fence);
        Ok(())
    }

    fn create_context_fence(
        &mut self,
        context: u32,
        flags: u32,
        ring_idx: u32,
        fence: u64,
    ) -> RendererResult<()> {
        let Some(create) = self.context_create_fence else {
            return Err("virglrenderer does not support context fences".into());
        };
        // SAFETY: the context, ring, and fence are guest virtio-gpu values.
        let code = unsafe { create(context, flags, ring_idx, fence) };
        status("context_create_fence", code)
    }

    fn poll_context_fences(&mut self) -> Vec<ContextFence> {
        std::mem::take(&mut *self.gl.retired_context.borrow_mut())
    }

    fn poll(&mut self) -> Vec<u64> {
        // SAFETY: a plain call; it may call write_fence.
        unsafe { (self.virgl.virgl_renderer_poll)() };
        let retired = std::mem::take(&mut *self.gl.retired.borrow_mut());
        let mut done = Vec::new();
        for low in retired {
            // In order: everything up to the one retired.
            if !self.fences.iter().any(|&f| f as u32 == low) {
                continue;
            }
            while let Some(f) = self.fences.pop_front() {
                done.push(f);
                if f as u32 == low {
                    break;
                }
            }
        }
        done
    }

    fn read(
        &mut self,
        resource: u32,
        rect: Rect,
        out: &mut [u8],
        stride: u32,
    ) -> RendererResult<()> {
        let mut info = ResourceInfo::default();
        // SAFETY: an out-parameter of the right type.
        let code =
            unsafe { (self.virgl.virgl_renderer_resource_get_info)(resource as c_int, &mut info) };
        status("resource_get_info", code)?;
        if info.tex_id == 0 {
            return Err(format!("resource {resource} has no texture"));
        }
        let gl = &self.gl;
        let f = &self.functions;
        // virglrenderer keeps track of which of its contexts is current, so
        // whatever was current goes back afterwards.
        let previous = gl.current();
        if !gl.make_current(gl.primary) {
            return Err(format!("eglMakeCurrent: {}", gl.error()));
        }
        // SAFETY: the primary context is current, its framebuffer reads the
        // resource's texture (shared with virglrenderer's contexts), and
        // `out` holds the rows at `stride`.
        let result = unsafe {
            (f.glBindFramebuffer)(GL_READ_FRAMEBUFFER, self.framebuffer);
            (f.glFramebufferTexture2D)(
                GL_READ_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D,
                info.tex_id,
                0,
            );
            let complete = (f.glCheckFramebufferStatus)(GL_READ_FRAMEBUFFER);
            let result = if complete != GL_FRAMEBUFFER_COMPLETE {
                Err(format!("framebuffer status {complete:#x}"))
            } else {
                (f.glPixelStorei)(GL_PACK_ALIGNMENT, 4);
                (f.glPixelStorei)(GL_PACK_ROW_LENGTH, (stride / 4) as i32);
                (f.glReadPixels)(
                    rect.x as i32,
                    rect.y as i32,
                    rect.width as i32,
                    rect.height as i32,
                    GL_RGBA,
                    GL_UNSIGNED_BYTE,
                    out.as_mut_ptr().cast(),
                );
                (f.glPixelStorei)(GL_PACK_ROW_LENGTH, 0);
                match (f.glGetError)() {
                    0 => Ok(()),
                    e => Err(format!("glReadPixels: GL error {e:#x}")),
                }
            };
            (f.glFramebufferTexture2D)(
                GL_READ_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D,
                0,
                0,
            );
            (f.glBindFramebuffer)(GL_READ_FRAMEBUFFER, 0);
            result
        };
        gl.restore(previous);
        result
    }

    fn cursor_image(&mut self, resource: u32) -> Option<(u32, u32, Vec<u8>)> {
        let (mut width, mut height) = (0, 0);
        // Reading the cursor back leaves another context current than the one
        // virglrenderer believes is, and its next command then runs in the
        // wrong context (whose vertex arrays it does not have). Whatever was
        // current goes back.
        let previous = self.gl.current();
        // SAFETY: two out-parameters; the data is malloc'd for the caller.
        let data = unsafe {
            (self.virgl.virgl_renderer_get_cursor_data)(resource, &mut width, &mut height)
        };
        self.gl.restore(previous);
        if data.is_null() {
            return None;
        }
        let len = width as usize * height as usize * 4;
        // SAFETY: width x height pixels of 4 bytes, then freed as asked.
        let pixels = unsafe { std::slice::from_raw_parts(data as *const u8, len) }.to_vec();
        // SAFETY: as above.
        unsafe { libc::free(data) };
        Some((width, height, pixels))
    }

    fn reset(&mut self) {
        let resources: Vec<u32> = self.backings.keys().copied().collect();
        for resource in resources {
            self.detach(resource);
        }
        self.fences.clear();
        self.gl.retired.borrow_mut().clear();
        self.gl.retired_context.borrow_mut().clear();
        // SAFETY: a plain call.
        unsafe { (self.virgl.virgl_renderer_reset)() };
    }
}

//! Metal compute utilities for GPU programming.

// Only Apple computers with M-series chips are supported.
#![cfg(target_os = "macos")]

mod error;

use std::marker::PhantomData;
use std::path::Path;

use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue};
use objc2_metal::{MTLComputeCommandEncoder, MTLDevice, MTLLibrary};

pub use error::Error;
pub use objc2;
pub use objc2_foundation;
pub use objc2_metal;

/// A handle to a GPU device.
pub struct Device(objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>);

impl Device {
    /// Acquires a handle to the system's default GPU.
    pub fn acquire() -> Result<Self, Error> {
        let device = objc2_metal::MTLCreateSystemDefaultDevice().ok_or(Error::DeviceNotFound)?;
        Ok(Self(device))
    }

    /// Creates a new command queue for submitting work to this device.
    pub fn create_queue(&self) -> Result<CommandQueue, Error> {
        let inner = self.0.newCommandQueue().ok_or(Error::QueueCreation)?;
        Ok(CommandQueue { inner })
    }

    /// Returns a reference to the underlying Metal device.
    pub fn as_raw(&self) -> &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice> {
        &self.0
    }
}

/// A Metal kernel source with an associated function name.
pub struct Kernel {
    pub code: String,
    pub name: String,
}

impl Kernel {
    /// Creates a kernel from source code and function name.
    pub fn new(code: impl Into<String>, function_name: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            name: function_name.into(),
        }
    }

    /// Creates a kernel from a file path and function name.
    pub fn from_file(filepath: &Path, function_name: &str) -> Result<Self, Error> {
        let code = std::fs::read_to_string(filepath)?;
        Ok(Self::new(code, function_name))
    }

    /// Compiles this kernel into a function on the given device.
    pub fn compile<'a>(&self, device: &'a Device) -> Result<Function<'a>, Error> {
        let source = objc2_foundation::NSString::from_str(&self.code);
        let library = device
            .0
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|e| Error::LibraryCompilation(e.localizedDescription().to_string()))?;

        let name = objc2_foundation::NSString::from_str(&self.name);
        let inner = library
            .newFunctionWithName(&name)
            .ok_or_else(|| Error::FunctionNotFound(self.name.clone()))?;

        Ok(Function { device, inner })
    }
}

/// A compiled Metal shader function.
pub struct Function<'a> {
    device: &'a Device,
    inner: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLFunction>>,
}

impl<'a> Function<'a> {
    /// Creates a compute pipeline from this function.
    pub fn to_pipeline(&self) -> Result<Pipeline, Error> {
        let inner = self
            .device
            .0
            .newComputePipelineStateWithFunction_error(&self.inner)
            .map_err(|e| Error::PipelineCreation(e.localizedDescription().to_string()))?;

        Ok(Pipeline { inner })
    }

    /// Returns a reference to the underlying Metal function.
    pub fn as_raw(&self) -> &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLFunction> {
        &self.inner
    }
}

/// A compiled compute pipeline ready for execution.
pub struct Pipeline {
    inner: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    >,
}

impl Pipeline {
    /// Returns a reference to the underlying Metal pipeline state.
    pub fn as_raw(
        &self,
    ) -> &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState> {
        &self.inner
    }
}

/// A GPU buffer containing elements of type `T`.
///
/// The `Copy` bound ensures the type can be safely transferred to/from GPU memory.
pub struct Buffer<T: Copy> {
    inner: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: Copy> Buffer<T> {
    /// Creates a buffer initialized with the contents of a slice.
    pub fn from_slice(device: &Device, data: &[T]) -> Result<Self, Error> {
        let len = data.len();
        let byte_len = len * std::mem::size_of::<T>();

        let inner = if byte_len == 0 {
            // Metal doesn't support zero-length buffers
            device
                .0
                .newBufferWithLength_options(
                    std::mem::size_of::<T>().max(1),
                    objc2_metal::MTLResourceOptions::StorageModeShared,
                )
                .ok_or(Error::BufferCreation)?
        } else {
            // SAFETY: data.as_ptr() is valid for `byte_len` bytes, and T: Copy ensures
            // the memory can be safely bitwise copied to GPU memory.
            unsafe {
                device
                    .0
                    .newBufferWithBytes_length_options(
                        std::ptr::NonNull::new_unchecked(data.as_ptr() as *mut std::ffi::c_void),
                        byte_len,
                        objc2_metal::MTLResourceOptions::StorageModeShared,
                    )
                    .ok_or(Error::BufferCreation)?
            }
        };

        Ok(Self {
            inner,
            len,
            _marker: PhantomData,
        })
    }

    /// Creates a buffer with space for `len` elements.
    ///
    /// Use this for output buffers that will be written by the GPU.
    pub fn with_len(device: &Device, len: usize) -> Result<Self, Error> {
        let byte_len = (len * std::mem::size_of::<T>()).max(1);

        let inner = device
            .0
            .newBufferWithLength_options(
                byte_len,
                objc2_metal::MTLResourceOptions::StorageModeShared,
            )
            .ok_or(Error::BufferCreation)?;

        Ok(Self {
            inner,
            len,
            _marker: PhantomData,
        })
    }

    /// Returns the number of elements in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the buffer contents as a slice.
    ///
    /// For output buffers, call this after the GPU work has completed.
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: The buffer was created with `len` elements of type T, and
        // StorageModeShared ensures CPU can read the memory. T: Copy ensures
        // reading the bytes as T is safe.
        let ptr = self.inner.contents().as_ptr() as *const T;
        unsafe { std::slice::from_raw_parts(ptr, self.len) }
    }

    /// Returns a reference to the underlying Metal buffer.
    pub fn as_raw(&self) -> &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer> {
        &self.inner
    }
}

/// A command queue for submitting work to the GPU.
pub struct CommandQueue {
    inner: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandQueue>>,
}

impl CommandQueue {
    /// Creates a new compute pass for encoding GPU commands.
    pub fn new_compute_pass(&self, pipeline: &Pipeline) -> Result<ComputePass, Error> {
        let command_buffer = self
            .inner
            .commandBuffer()
            .ok_or(Error::CommandBufferCreation)?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(Error::EncoderCreation)?;

        encoder.setComputePipelineState(pipeline.as_raw());

        Ok(ComputePass {
            command_buffer,
            encoder,
        })
    }

    /// Returns a reference to the underlying Metal command queue.
    pub fn as_raw(&self) -> &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandQueue> {
        &self.inner
    }
}

/// A compute pass for encoding and dispatching GPU work.
///
/// Use `bind()` to attach buffers, then `dispatch()` to specify the work size,
/// and finally `submit_and_wait()` to execute.
pub struct ComputePass {
    command_buffer:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>,
    encoder: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    >,
}

impl ComputePass {
    /// Binds a buffer to the specified index.
    pub fn bind<T: Copy>(&self, index: usize, buffer: &Buffer<T>) {
        // SAFETY: setBuffer_offset_atIndex requires a valid buffer and index.
        // Our Buffer type ensures validity, and the index is checked by Metal.
        unsafe {
            self.encoder
                .setBuffer_offset_atIndex(Some(buffer.as_raw()), 0, index);
        }
    }

    /// Dispatches a 1D compute grid with the specified number of threads.
    pub fn dispatch_1d(&self, threads: usize) {
        let grid_size = objc2_metal::MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        };
        let threadgroup_size = objc2_metal::MTLSize {
            width: threads.min(256), // Common max threadgroup size
            height: 1,
            depth: 1,
        };
        self.encoder
            .dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    }

    /// Dispatches a 2D compute grid.
    pub fn dispatch_2d(&self, width: usize, height: usize) {
        let grid_size = objc2_metal::MTLSize {
            width,
            height,
            depth: 1,
        };
        let threadgroup_size = objc2_metal::MTLSize {
            width: width.min(16),
            height: height.min(16),
            depth: 1,
        };
        self.encoder
            .dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    }

    /// Dispatches a 3D compute grid.
    pub fn dispatch_3d(&self, width: usize, height: usize, depth: usize) {
        let grid_size = objc2_metal::MTLSize {
            width,
            height,
            depth,
        };
        let threadgroup_size = objc2_metal::MTLSize {
            width: width.min(8),
            height: height.min(8),
            depth: depth.min(8),
        };
        self.encoder
            .dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    }

    /// Submits the compute pass and waits for completion.
    pub fn submit_and_wait(self) {
        self.encoder.endEncoding();
        self.command_buffer.commit();
        self.command_buffer.waitUntilCompleted();
    }

    /// Submits the compute pass without waiting.
    ///
    /// Returns a `Submission` that can be used to wait later.
    pub fn submit(self) -> Submission {
        self.encoder.endEncoding();
        self.command_buffer.commit();
        Submission {
            command_buffer: self.command_buffer,
        }
    }
}

/// A submitted GPU command that may still be executing.
pub struct Submission {
    command_buffer:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>,
}

impl Submission {
    /// Blocks until the GPU work has completed.
    pub fn wait(self) {
        self.command_buffer.waitUntilCompleted();
    }
}

//! Compute pass encoding and command submission.

use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder};

use super::buffers::Buffer;

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
    /// Creates a new compute pass from a command buffer and encoder.
    pub(crate) fn new(
        command_buffer: objc2::rc::Retained<
            objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        >,
        encoder: objc2::rc::Retained<
            objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
        >,
    ) -> Self {
        Self {
            command_buffer,
            encoder,
        }
    }

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

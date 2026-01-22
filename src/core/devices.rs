//! GPU device management.

use objc2_metal::{MTLCommandBuffer, MTLCommandQueue, MTLComputeCommandEncoder, MTLDevice};

use super::compute::ComputePass;
use super::kernels::Pipeline;
use metalogue_traits::error::Error;

/// A handle to a GPU device.
pub struct Device(objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>);

impl Device {
    /// Acquires a handle to the system's default GPU.
    pub fn acquire() -> Result<Self, Error> {
        let device = objc2_metal::MTLCreateSystemDefaultDevice().ok_or(Error::DeviceNotFound)?;
        Ok(Self(device))
    }

    /// Creates a new compute pass for encoding GPU commands.
    pub fn new_compute_pass(&self, pipeline: &Pipeline) -> Result<ComputePass, Error> {
        let command_queue = self.0.newCommandQueue().ok_or(Error::QueueCreation)?;
        let command_buffer = command_queue
            .commandBuffer()
            .ok_or(Error::CommandBufferCreation)?;
        let encoder = command_buffer
            .computeCommandEncoder()
            .ok_or(Error::EncoderCreation)?;

        encoder.setComputePipelineState(pipeline.as_raw());

        Ok(ComputePass::new(command_buffer, encoder))
    }

    /// Returns a reference to the underlying Metal device.
    pub fn as_raw(&self) -> &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice> {
        &self.0
    }
}

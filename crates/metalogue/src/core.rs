//! Core Metal compute utilities for GPU programming.

mod buffers;
mod compute;
mod devices;
mod kernels;

pub use buffers::Buffer;
pub use compute::{ComputePass, Submission};
pub use devices::Device;
pub use kernels::{Kernel, Pipeline};

// This is not a module! This source code is included
// directly in `lib.rs`.

pub mod error;

/// Trait for representing programs to be executed on devices.
/// For example: a kernel for executing on a GPU.
pub trait Kernel {
    /// Return a reference to the underlying code.
    fn code(&self) -> &str;
}

/// All supported device types.
pub enum DeviceType {
    /// Largely-serial processors.
    CPU,
    /// Largely-parallel processors.
    GPU,
}

/// Trait for devices: CPU, GPU, and maybe more one day.
pub trait Device: Default {
    /// Acquire a new device handle.
    fn acquire(kind: DeviceType) -> Result<Self, error::Error>;
    /// Return the kind of device held by an instance.
    fn kind(&self) -> DeviceType;
    /// Allocate memory on the device.
    fn allocate(&self, size: usize) -> Result<std::sync::Arc<[u8]>, error::Error>;
    /// Compile a kernel and return a callable.
    fn compile<Args>(
        &self,
        kernel: dyn Kernel,
    ) -> Result<std::sync::Arc<dyn Fn(Args)>, error::Error>;
}

///
/// Trait for dispatching pre-allocated buffers and a pre-compiled kernel to the GPU
/// with threadgroup and thread size.
pub trait Dispatch {
    /// Dispatch a kernel with the given buffers, threadgroup size, and thread count.
    fn dispatch(
        &self,
        kernel: &dyn Kernel,
        buffers: &[&[u8]],
        threadgroup_size: (usize, usize, usize),
        thread_count: (usize, usize, usize),
    ) -> Result<(), error::Error>;
}

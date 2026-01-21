//! Metal compute utilities for GPU programming with Python bindings.

// The core library only works on macOS
#![cfg(target_os = "macos")]

mod core;
mod error;

// Re-export core types for Rust usage
pub use core::*;
pub use error::Error;
pub use objc2;
pub use objc2_foundation;
pub use objc2_metal;

// Python bindings
mod python {
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::prelude::*;
    use std::path::PathBuf;

    use crate::{Error, core};

    /// Convert a Rust Error to a Python exception
    impl From<Error> for PyErr {
        fn from(err: Error) -> Self {
            PyRuntimeError::new_err(err.to_string())
        }
    }

    /// A handle to a GPU device.
    #[pyclass]
    pub struct Device {
        inner: core::Device,
    }

    #[pymethods]
    impl Device {
        /// Acquires a handle to the system's default GPU.
        #[staticmethod]
        fn acquire() -> PyResult<Self> {
            let inner = core::Device::acquire()?;
            Ok(Self { inner })
        }

        /// Creates a new command queue for submitting work to this device.
        fn create_queue(&self) -> PyResult<CommandQueue> {
            let inner = self.inner.create_queue()?;
            Ok(CommandQueue { inner })
        }

        fn __repr__(&self) -> String {
            "Device()".to_string()
        }
    }

    /// A Metal kernel source with an associated function name.
    #[pyclass]
    pub struct Kernel {
        inner: core::Kernel,
    }

    #[pymethods]
    impl Kernel {
        /// Creates a kernel from source code and function name.
        #[new]
        fn new(code: String, function_name: String) -> Self {
            Self {
                inner: core::Kernel::new(code, function_name),
            }
        }

        /// Creates a kernel from a file path and function name.
        #[staticmethod]
        fn from_file(filepath: String, function_name: String) -> PyResult<Self> {
            let path = PathBuf::from(filepath);
            let inner = core::Kernel::from_file(&path, &function_name)?;
            Ok(Self { inner })
        }

        /// Compiles this kernel into a pipeline on the given device.
        fn compile(&self, device: &Device) -> PyResult<Pipeline> {
            let function = self.inner.compile(&device.inner)?;
            let inner = function.to_pipeline()?;
            Ok(Pipeline { inner })
        }

        fn __repr__(&self) -> String {
            format!("Kernel(name='{}')", self.inner.name)
        }
    }

    /// A compiled compute pipeline ready for execution.
    #[pyclass]
    pub struct Pipeline {
        inner: core::Pipeline,
    }

    #[pymethods]
    impl Pipeline {
        fn __repr__(&self) -> String {
            "Pipeline()".to_string()
        }
    }

    /// A GPU buffer containing floating point numbers.
    #[pyclass(unsendable)]
    pub struct BufferF32 {
        inner: core::Buffer<f32>,
    }

    #[pymethods]
    impl BufferF32 {
        /// Creates a buffer initialized with the contents of a list.
        #[staticmethod]
        fn from_list(device: &Device, data: Vec<f32>) -> PyResult<Self> {
            let inner = core::Buffer::from_slice(&device.inner, &data)?;
            Ok(Self { inner })
        }

        /// Creates a buffer with space for `len` elements.
        #[staticmethod]
        fn with_len(device: &Device, len: usize) -> PyResult<Self> {
            let inner = core::Buffer::with_len(&device.inner, len)?;
            Ok(Self { inner })
        }

        /// Returns the number of elements in the buffer.
        fn len(&self) -> usize {
            self.inner.len()
        }

        /// Returns true if the buffer is empty.
        fn is_empty(&self) -> bool {
            self.inner.is_empty()
        }

        /// Returns the buffer contents as a list.
        fn to_list(&self) -> Vec<f32> {
            self.inner.as_slice().to_vec()
        }

        fn __repr__(&self) -> String {
            format!("BufferF32(len={})", self.inner.len())
        }

        fn __len__(&self) -> usize {
            self.inner.len()
        }
    }

    /// A GPU buffer containing 32-bit integers.
    #[pyclass(unsendable)]
    pub struct BufferI32 {
        inner: core::Buffer<i32>,
    }

    #[pymethods]
    impl BufferI32 {
        /// Creates a buffer initialized with the contents of a list.
        #[staticmethod]
        fn from_list(device: &Device, data: Vec<i32>) -> PyResult<Self> {
            let inner = core::Buffer::from_slice(&device.inner, &data)?;
            Ok(Self { inner })
        }

        /// Creates a buffer with space for `len` elements.
        #[staticmethod]
        fn with_len(device: &Device, len: usize) -> PyResult<Self> {
            let inner = core::Buffer::with_len(&device.inner, len)?;
            Ok(Self { inner })
        }

        /// Returns the number of elements in the buffer.
        fn len(&self) -> usize {
            self.inner.len()
        }

        /// Returns true if the buffer is empty.
        fn is_empty(&self) -> bool {
            self.inner.is_empty()
        }

        /// Returns the buffer contents as a list.
        fn to_list(&self) -> Vec<i32> {
            self.inner.as_slice().to_vec()
        }

        fn __repr__(&self) -> String {
            format!("BufferI32(len={})", self.inner.len())
        }

        fn __len__(&self) -> usize {
            self.inner.len()
        }
    }

    /// A command queue for submitting work to the GPU.
    #[pyclass]
    pub struct CommandQueue {
        inner: core::CommandQueue,
    }

    #[pymethods]
    impl CommandQueue {
        /// Creates a new compute pass for encoding GPU commands.
        fn new_compute_pass(&self, pipeline: &Pipeline) -> PyResult<ComputePass> {
            let inner = self.inner.new_compute_pass(&pipeline.inner)?;
            Ok(ComputePass { inner: Some(inner) })
        }

        fn __repr__(&self) -> String {
            "CommandQueue()".to_string()
        }
    }

    /// A compute pass for encoding and dispatching GPU work.
    #[pyclass(unsendable)]
    pub struct ComputePass {
        inner: Option<core::ComputePass>,
    }

    #[pymethods]
    impl ComputePass {
        /// Binds a float32 buffer to the specified index.
        fn bind_f32(&self, index: usize, buffer: &BufferF32) {
            if let Some(inner) = &self.inner {
                inner.bind(index, &buffer.inner);
            }
        }

        /// Binds an int32 buffer to the specified index.
        fn bind_i32(&self, index: usize, buffer: &BufferI32) {
            if let Some(inner) = &self.inner {
                inner.bind(index, &buffer.inner);
            }
        }

        /// Dispatches a 1D compute grid with the specified number of threads.
        fn dispatch_1d(&self, threads: usize) {
            if let Some(inner) = &self.inner {
                inner.dispatch_1d(threads);
            }
        }

        /// Dispatches a 2D compute grid.
        fn dispatch_2d(&self, width: usize, height: usize) {
            if let Some(inner) = &self.inner {
                inner.dispatch_2d(width, height);
            }
        }

        /// Dispatches a 3D compute grid.
        fn dispatch_3d(&self, width: usize, height: usize, depth: usize) {
            if let Some(inner) = &self.inner {
                inner.dispatch_3d(width, height, depth);
            }
        }

        /// Submits the compute pass and waits for completion.
        fn submit_and_wait(mut slf: PyRefMut<'_, Self>) -> PyResult<()> {
            if let Some(inner) = slf.inner.take() {
                inner.submit_and_wait();
                Ok(())
            } else {
                Err(PyRuntimeError::new_err("ComputePass already submitted"))
            }
        }

        /// Submits the compute pass without waiting.
        fn submit(mut slf: PyRefMut<'_, Self>) -> PyResult<Submission> {
            if let Some(inner) = slf.inner.take() {
                let submitted = inner.submit();
                Ok(Submission {
                    inner: Some(submitted),
                })
            } else {
                Err(PyRuntimeError::new_err("ComputePass already submitted"))
            }
        }

        fn __repr__(&self) -> String {
            "ComputePass()".to_string()
        }
    }

    /// A submitted GPU command that may still be executing.
    #[pyclass(unsendable)]
    pub struct Submission {
        inner: Option<core::Submission>,
    }

    #[pymethods]
    impl Submission {
        /// Blocks until the GPU work has completed.
        fn wait(mut slf: PyRefMut<'_, Self>) -> PyResult<()> {
            if let Some(inner) = slf.inner.take() {
                inner.wait();
                Ok(())
            } else {
                Err(PyRuntimeError::new_err("Submission already waited on"))
            }
        }

        fn __repr__(&self) -> String {
            "Submission()".to_string()
        }
    }

    /// A Python module implemented in Rust.
    #[pymodule]
    fn metalogue(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<Device>()?;
        m.add_class::<Kernel>()?;
        m.add_class::<Pipeline>()?;
        m.add_class::<BufferF32>()?;
        m.add_class::<BufferI32>()?;
        m.add_class::<CommandQueue>()?;
        m.add_class::<ComputePass>()?;
        m.add_class::<Submission>()?;
        Ok(())
    }
}

//! Python bindings to `metalogue` via `pyo3`.
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
    ///
    /// This represents the Metal GPU on your system. Use `Device.acquire()` to get
    /// a handle to the default GPU.
    ///
    /// # Example
    /// ```python
    /// device = Device.acquire()
    /// ```
    #[pyclass]
    pub struct Device {
        inner: core::Device,
    }

    #[pymethods]
    impl Device {
        /// Acquires a handle to the system's default GPU.
        ///
        /// Returns:
        ///     Device: A handle to the default Metal GPU device.
        ///
        /// Raises:
        ///     RuntimeError: If no GPU device is found.
        ///
        /// # Example
        /// ```python
        /// device = Device.acquire()
        /// ```
        #[staticmethod]
        fn acquire() -> PyResult<Self> {
            let inner = core::Device::acquire()?;
            Ok(Self { inner })
        }

        /// Creates a new compute pass for encoding GPU commands.
        ///
        /// A compute pass is used to bind buffers, dispatch work, and submit commands
        /// to the GPU.
        ///
        /// Args:
        ///     pipeline (Pipeline): The compiled compute pipeline to execute.
        ///
        /// Returns:
        ///     ComputePass: A new compute pass ready for encoding commands.
        ///
        /// Raises:
        ///     RuntimeError: If the compute pass cannot be created.
        ///
        /// # Example
        /// ```python
        /// pipeline = kernel.to_pipeline(device)
        /// pass = device.new_compute_pass(pipeline)
        /// ```
        fn new_compute_pass(&self, pipeline: &Pipeline) -> PyResult<ComputePass> {
            let inner = self.inner.new_compute_pass(&pipeline.inner)?;
            Ok(ComputePass { inner: Some(inner) })
        }

        fn __repr__(&self) -> String {
            "Device()".to_string()
        }
    }

    /// A Metal kernel source with an associated function name.
    ///
    /// Represents Metal shader source code that can be compiled into a pipeline.
    ///
    /// # Example
    /// ```python
    /// kernel = Kernel(code, "my_function")
    /// # or
    /// kernel = Kernel.from_file("shader.metal", "my_function")
    /// ```
    #[pyclass]
    pub struct Kernel {
        inner: core::Kernel,
    }

    #[pymethods]
    impl Kernel {
        /// Creates a kernel from source code and function name.
        ///
        /// Args:
        ///     code (str): The Metal shader source code.
        ///     function_name (str): The name of the compute function to use.
        ///
        /// Returns:
        ///     Kernel: A new kernel instance.
        ///
        /// # Example
        /// ```python
        /// code = '''
        /// kernel void add(device float* a [[buffer(0)]],
        ///                 device float* b [[buffer(1)]],
        ///                 device float* result [[buffer(2)]],
        ///                 uint id [[thread_position_in_grid]]) {
        ///     result[id] = a[id] + b[id];
        /// }
        /// '''
        /// kernel = Kernel(code, "add")
        /// ```
        #[new]
        fn new(code: String, function_name: String) -> Self {
            Self {
                inner: core::Kernel::new(code, function_name),
            }
        }

        /// Creates a kernel from a file path and function name.
        ///
        /// Args:
        ///     filepath (str): Path to the .metal shader file.
        ///     function_name (str): The name of the compute function to use.
        ///
        /// Returns:
        ///     Kernel: A new kernel instance.
        ///
        /// Raises:
        ///     RuntimeError: If the file cannot be read.
        ///
        /// # Example
        /// ```python
        /// kernel = Kernel.from_file("shaders/add.metal", "add_arrays")
        /// ```
        #[staticmethod]
        fn from_file(filepath: String, function_name: String) -> PyResult<Self> {
            let path = PathBuf::from(filepath);
            let inner = core::Kernel::from_file(&path, &function_name)?;
            Ok(Self { inner })
        }

        /// Compiles this kernel into a pipeline on the given device.
        ///
        /// Args:
        ///     device (Device): The GPU device to compile the kernel for.
        ///
        /// Returns:
        ///     Pipeline: A compiled compute pipeline ready for execution.
        ///
        /// Raises:
        ///     RuntimeError: If compilation fails.
        ///
        /// # Example
        /// ```python
        /// device = Device.acquire()
        /// pipeline = kernel.to_pipeline(device)
        /// ```
        fn to_pipeline(&self, device: &Device) -> PyResult<Pipeline> {
            let inner = self.inner.to_pipeline(&device.inner)?;
            Ok(Pipeline { inner })
        }

        fn __repr__(&self) -> String {
            format!("Kernel(name='{}')", self.inner.name)
        }
    }

    /// A compiled compute pipeline ready for execution.
    ///
    /// This is created by compiling a Kernel and is used to create compute passes.
    ///
    /// # Example
    /// ```python
    /// pipeline = kernel.to_pipeline(device)
    /// pass = device.new_compute_pass(pipeline)
    /// ```
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

    /// A GPU buffer containing 32-bit floating point numbers.
    ///
    /// This buffer stores data in GPU memory and can be used as input or output
    /// for compute shaders.
    ///
    /// # Example
    /// ```python
    /// # Create input buffer from data
    /// input = BufferF32.from_list(device, [1.0, 2.0, 3.0])
    ///
    /// # Create output buffer with specific size
    /// output = BufferF32.with_len(device, 3)
    /// ```
    #[pyclass(unsendable)]
    pub struct BufferF32 {
        inner: core::Buffer<f32>,
    }

    #[pymethods]
    impl BufferF32 {
        /// Creates a buffer initialized with the contents of a list.
        ///
        /// Args:
        ///     device (Device): The GPU device to create the buffer on.
        ///     data (list[float]): The initial data for the buffer.
        ///
        /// Returns:
        ///     BufferF32: A new buffer containing the data.
        ///
        /// Raises:
        ///     RuntimeError: If buffer creation fails.
        ///
        /// # Example
        /// ```python
        /// device = Device.acquire()
        /// buffer = BufferF32.from_list(device, [1.0, 2.0, 3.0, 4.0])
        /// ```
        #[staticmethod]
        fn from_list(device: &Device, data: Vec<f32>) -> PyResult<Self> {
            let inner = core::Buffer::from_slice(&device.inner, &data)?;
            Ok(Self { inner })
        }

        /// Creates a buffer with space for `len` elements.
        ///
        /// Use this for output buffers that will be written by the GPU.
        ///
        /// Args:
        ///     device (Device): The GPU device to create the buffer on.
        ///     len (int): The number of float32 elements to allocate.
        ///
        /// Returns:
        ///     BufferF32: A new uninitialized buffer.
        ///
        /// Raises:
        ///     RuntimeError: If buffer creation fails.
        ///
        /// # Example
        /// ```python
        /// device = Device.acquire()
        /// output = BufferF32.with_len(device, 100)
        /// ```
        #[staticmethod]
        fn with_len(device: &Device, len: usize) -> PyResult<Self> {
            let inner = core::Buffer::with_len(&device.inner, len)?;
            Ok(Self { inner })
        }

        /// Returns the number of elements in the buffer.
        ///
        /// Returns:
        ///     int: The number of float32 elements.
        fn len(&self) -> usize {
            self.inner.len()
        }

        /// Returns true if the buffer is empty.
        ///
        /// Returns:
        ///     bool: True if the buffer contains no elements.
        fn is_empty(&self) -> bool {
            self.inner.is_empty()
        }

        /// Returns the buffer contents as a list.
        ///
        /// Read the GPU buffer data back to Python. Call this after GPU work
        /// has completed to retrieve results.
        ///
        /// Returns:
        ///     list[float]: The buffer contents as a Python list.
        ///
        /// # Example
        /// ```python
        /// pass.submit_and_wait()
        /// results = output_buffer.to_list()
        /// ```
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

    /// A GPU buffer containing 32-bit signed integers.
    ///
    /// This buffer stores integer data in GPU memory and can be used as input or
    /// output for compute shaders.
    ///
    /// # Example
    /// ```python
    /// # Create input buffer from data
    /// input = BufferI32.from_list(device, [1, 2, 3])
    ///
    /// # Create output buffer with specific size
    /// output = BufferI32.with_len(device, 3)
    /// ```
    #[pyclass(unsendable)]
    pub struct BufferI32 {
        inner: core::Buffer<i32>,
    }

    #[pymethods]
    impl BufferI32 {
        /// Creates a buffer initialized with the contents of a list.
        ///
        /// Args:
        ///     device (Device): The GPU device to create the buffer on.
        ///     data (list[int]): The initial data for the buffer.
        ///
        /// Returns:
        ///     BufferI32: A new buffer containing the data.
        ///
        /// Raises:
        ///     RuntimeError: If buffer creation fails.
        ///
        /// # Example
        /// ```python
        /// device = Device.acquire()
        /// buffer = BufferI32.from_list(device, [1, 2, 3, 4])
        /// ```
        #[staticmethod]
        fn from_list(device: &Device, data: Vec<i32>) -> PyResult<Self> {
            let inner = core::Buffer::from_slice(&device.inner, &data)?;
            Ok(Self { inner })
        }

        /// Creates a buffer with space for `len` elements.
        ///
        /// Use this for output buffers that will be written by the GPU.
        ///
        /// Args:
        ///     device (Device): The GPU device to create the buffer on.
        ///     len (int): The number of int32 elements to allocate.
        ///
        /// Returns:
        ///     BufferI32: A new uninitialized buffer.
        ///
        /// Raises:
        ///     RuntimeError: If buffer creation fails.
        ///
        /// # Example
        /// ```python
        /// device = Device.acquire()
        /// output = BufferI32.with_len(device, 100)
        /// ```
        #[staticmethod]
        fn with_len(device: &Device, len: usize) -> PyResult<Self> {
            let inner = core::Buffer::with_len(&device.inner, len)?;
            Ok(Self { inner })
        }

        /// Returns the number of elements in the buffer.
        ///
        /// Returns:
        ///     int: The number of int32 elements.
        fn len(&self) -> usize {
            self.inner.len()
        }

        /// Returns true if the buffer is empty.
        ///
        /// Returns:
        ///     bool: True if the buffer contains no elements.
        fn is_empty(&self) -> bool {
            self.inner.is_empty()
        }

        /// Returns the buffer contents as a list.
        ///
        /// Read the GPU buffer data back to Python. Call this after GPU work
        /// has completed to retrieve results.
        ///
        /// Returns:
        ///     list[int]: The buffer contents as a Python list.
        ///
        /// # Example
        /// ```python
        /// pass.submit_and_wait()
        /// results = output_buffer.to_list()
        /// ```
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

    /// A compute pass for encoding and dispatching GPU work.
    ///
    /// Use this to bind buffers, dispatch compute work, and submit commands to the GPU.
    /// A compute pass can only be submitted once.
    ///
    /// # Example
    /// ```python
    /// pass = device.new_compute_pass(pipeline)
    /// pass.bind_f32(0, input_buffer)
    /// pass.bind_f32(1, output_buffer)
    /// pass.dispatch_1d(100)
    /// pass.submit_and_wait()
    /// ```
    #[pyclass(unsendable)]
    pub struct ComputePass {
        inner: Option<core::ComputePass>,
    }

    #[pymethods]
    impl ComputePass {
        /// Binds a float32 buffer to the specified index.
        ///
        /// Buffers must be bound to match the buffer indices in your shader code.
        ///
        /// Args:
        ///     index (int): The buffer binding index (matches [[buffer(N)]] in shader).
        ///     buffer (BufferF32): The buffer to bind.
        ///
        /// # Example
        /// ```python
        /// # Shader: kernel void add(device float* a [[buffer(0)]], ...)
        /// pass.bind_f32(0, input_buffer)
        /// ```
        fn bind_f32(&self, index: usize, buffer: &BufferF32) {
            if let Some(inner) = &self.inner {
                inner.bind(index, &buffer.inner);
            }
        }

        /// Binds an int32 buffer to the specified index.
        ///
        /// Buffers must be bound to match the buffer indices in your shader code.
        ///
        /// Args:
        ///     index (int): The buffer binding index (matches [[buffer(N)]] in shader).
        ///     buffer (BufferI32): The buffer to bind.
        ///
        /// # Example
        /// ```python
        /// # Shader: kernel void process(device int* data [[buffer(0)]], ...)
        /// pass.bind_i32(0, data_buffer)
        /// ```
        fn bind_i32(&self, index: usize, buffer: &BufferI32) {
            if let Some(inner) = &self.inner {
                inner.bind(index, &buffer.inner);
            }
        }

        /// Dispatches a 1D compute grid with the specified number of threads.
        ///
        /// Args:
        ///     threads (int): The total number of threads to dispatch.
        ///
        /// # Example
        /// ```python
        /// # Process 1000 elements
        /// pass.dispatch_1d(1000)
        /// ```
        fn dispatch_1d(&self, threads: usize) {
            if let Some(inner) = &self.inner {
                inner.dispatch_1d(threads);
            }
        }

        /// Dispatches a 2D compute grid.
        ///
        /// Args:
        ///     width (int): The width of the 2D grid.
        ///     height (int): The height of the 2D grid.
        ///
        /// # Example
        /// ```python
        /// # Process a 1920x1080 image
        /// pass.dispatch_2d(1920, 1080)
        /// ```
        fn dispatch_2d(&self, width: usize, height: usize) {
            if let Some(inner) = &self.inner {
                inner.dispatch_2d(width, height);
            }
        }

        /// Dispatches a 3D compute grid.
        ///
        /// Args:
        ///     width (int): The width of the 3D grid.
        ///     height (int): The height of the 3D grid.
        ///     depth (int): The depth of the 3D grid.
        ///
        /// # Example
        /// ```python
        /// # Process a 3D volume
        /// pass.dispatch_3d(128, 128, 128)
        /// ```
        fn dispatch_3d(&self, width: usize, height: usize, depth: usize) {
            if let Some(inner) = &self.inner {
                inner.dispatch_3d(width, height, depth);
            }
        }

        /// Submits the compute pass and waits for completion.
        ///
        /// This blocks until all GPU work is complete. After this call, output buffers
        /// can be safely read.
        ///
        /// Raises:
        ///     RuntimeError: If the pass was already submitted.
        ///
        /// # Example
        /// ```python
        /// pass.bind_f32(0, input)
        /// pass.bind_f32(1, output)
        /// pass.dispatch_1d(100)
        /// pass.submit_and_wait()
        /// results = output.to_list()
        /// ```
        fn submit_and_wait(mut slf: PyRefMut<'_, Self>) -> PyResult<()> {
            if let Some(inner) = slf.inner.take() {
                inner.submit_and_wait();
                Ok(())
            } else {
                Err(PyRuntimeError::new_err("ComputePass already submitted"))
            }
        }

        /// Submits the compute pass without waiting.
        ///
        /// Returns a Submission object that can be used to wait later. This allows
        /// overlapping GPU work with CPU work.
        ///
        /// Returns:
        ///     Submission: A handle to wait on the GPU work later.
        ///
        /// Raises:
        ///     RuntimeError: If the pass was already submitted.
        ///
        /// # Example
        /// ```python
        /// pass.bind_f32(0, input)
        /// pass.bind_f32(1, output)
        /// pass.dispatch_1d(100)
        /// submission = pass.submit()
        /// # Do CPU work here...
        /// submission.wait()
        /// results = output.to_list()
        /// ```
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
    ///
    /// Created by ComputePass.submit(). Call wait() to block until GPU work completes.
    ///
    /// # Example
    /// ```python
    /// submission = pass.submit()
    /// # Do other work...
    /// submission.wait()  # Wait for GPU to finish
    /// ```
    #[pyclass(unsendable)]
    pub struct Submission {
        inner: Option<core::Submission>,
    }

    #[pymethods]
    impl Submission {
        /// Blocks until the GPU work has completed.
        ///
        /// After this call returns, all GPU work is complete and output buffers
        /// can be safely read.
        ///
        /// Raises:
        ///     RuntimeError: If wait() was already called on this submission.
        ///
        /// # Example
        /// ```python
        /// submission = pass.submit()
        /// submission.wait()
        /// results = output_buffer.to_list()
        /// ```
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

    /// Metalogue - High-level Metal compute for Python.
    ///
    /// This module provides safe, easy-to-use GPU compute on Apple Silicon devices.
    #[pymodule]
    fn metalogue(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<Device>()?;
        m.add_class::<Kernel>()?;
        m.add_class::<Pipeline>()?;
        m.add_class::<BufferF32>()?;
        m.add_class::<BufferI32>()?;
        m.add_class::<ComputePass>()?;
        m.add_class::<Submission>()?;
        Ok(())
    }
}

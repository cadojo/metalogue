"""Type stubs for metalogue Python bindings."""

from typing import List

__version__: str

class Device:
    """A handle to a GPU device.

    This represents the Metal GPU on your system. Use `Device.acquire()` to get
    a handle to the default GPU.

    Example:
        device = Device.acquire()
    """

    @staticmethod
    def acquire() -> Device:
        """Acquires a handle to the system's default GPU.

        Returns:
            Device: A handle to the default Metal GPU device.

        Raises:
            RuntimeError: If no GPU device is found.

        Example:
            device = Device.acquire()
        """
        ...

    def new_compute_pass(self, pipeline: Pipeline) -> ComputePass:
        """Creates a new compute pass for encoding GPU commands.

        A compute pass is used to bind buffers, dispatch work, and submit commands
        to the GPU.

        Args:
            pipeline: The compiled compute pipeline to execute.

        Returns:
            ComputePass: A new compute pass ready for encoding commands.

        Raises:
            RuntimeError: If the compute pass cannot be created.

        Example:
            pipeline = kernel.to_pipeline(device)
            pass_ = device.new_compute_pass(pipeline)
        """
        ...

    def __repr__(self) -> str: ...

class Kernel:
    """A Metal kernel source with an associated function name.

    Represents Metal shader source code that can be compiled into a pipeline.

    Example:
        kernel = Kernel(code, "my_function")
        # or
        kernel = Kernel.from_file("shader.metal", "my_function")
    """

    def __init__(self, code: str, function_name: str) -> None:
        """Creates a kernel from source code and function name.

        Args:
            code: The Metal shader source code.
            function_name: The name of the compute function to use.

        Returns:
            Kernel: A new kernel instance.

        Example:
            code = '''
            kernel void add(device float* a [[buffer(0)]],
                            device float* b [[buffer(1)]],
                            device float* result [[buffer(2)]],
                            uint id [[thread_position_in_grid]]) {
                result[id] = a[id] + b[id];
            }
            '''
            kernel = Kernel(code, "add")
        """
        ...

    @staticmethod
    def from_file(filepath: str, function_name: str) -> Kernel:
        """Creates a kernel from a file path and function name.

        Args:
            filepath: Path to the .metal shader file.
            function_name: The name of the compute function to use.

        Returns:
            Kernel: A new kernel instance.

        Raises:
            RuntimeError: If the file cannot be read.

        Example:
            kernel = Kernel.from_file("shaders/add.metal", "add_arrays")
        """
        ...

    def to_pipeline(self, device: Device) -> Pipeline:
        """Compiles this kernel into a compute pipeline on the given device.

        Args:
            device: The GPU device to compile the kernel for.

        Returns:
            Pipeline: A compiled compute pipeline ready for execution.

        Raises:
            RuntimeError: If compilation fails.

        Example:
            device = Device.acquire()
            pipeline = kernel.to_pipeline(device)
        """
        ...

    def __repr__(self) -> str: ...

class Pipeline:
    """A compiled compute pipeline ready for execution.

    This is created by compiling a Kernel and is used to create compute passes.

    Example:
        pipeline = kernel.to_pipeline(device)
        pass_ = device.new_compute_pass(pipeline)
    """

    def __repr__(self) -> str: ...

class BufferF32:
    """A GPU buffer containing 32-bit floating point numbers.

    This buffer stores data in GPU memory and can be used as input or output
    for compute shaders.

    Example:
        # Create input buffer from data
        input = BufferF32.from_list(device, [1.0, 2.0, 3.0])

        # Create output buffer with specific size
        output = BufferF32.with_len(device, 3)
    """

    @staticmethod
    def from_list(device: Device, data: List[float]) -> BufferF32:
        """Creates a buffer initialized with the contents of a list.

        Args:
            device: The GPU device to create the buffer on.
            data: The initial data for the buffer.

        Returns:
            BufferF32: A new buffer containing the data.

        Raises:
            RuntimeError: If buffer creation fails.

        Example:
            device = Device.acquire()
            buffer = BufferF32.from_list(device, [1.0, 2.0, 3.0, 4.0])
        """
        ...

    @staticmethod
    def with_len(device: Device, length: int) -> BufferF32:
        """Creates a buffer with space for `length` elements.

        Use this for output buffers that will be written by the GPU.

        Args:
            device: The GPU device to create the buffer on.
            length: The number of float32 elements to allocate.

        Returns:
            BufferF32: A new uninitialized buffer.

        Raises:
            RuntimeError: If buffer creation fails.

        Example:
            device = Device.acquire()
            output = BufferF32.with_len(device, 100)
        """
        ...

    def len(self) -> int:
        """Returns the number of elements in the buffer.

        Returns:
            int: The number of float32 elements.
        """
        ...

    def is_empty(self) -> bool:
        """Returns true if the buffer is empty.

        Returns:
            bool: True if the buffer contains no elements.
        """
        ...

    def to_list(self) -> List[float]:
        """Returns the buffer contents as a list.

        Read the GPU buffer data back to Python. Call this after GPU work
        has completed to retrieve results.

        Returns:
            list[float]: The buffer contents as a Python list.

        Example:
            pass_.submit_and_wait()
            results = output_buffer.to_list()
        """
        ...

    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...

class BufferI32:
    """A GPU buffer containing 32-bit signed integers.

    This buffer stores integer data in GPU memory and can be used as input or
    output for compute shaders.

    Example:
        # Create input buffer from data
        input = BufferI32.from_list(device, [1, 2, 3])

        # Create output buffer with specific size
        output = BufferI32.with_len(device, 3)
    """

    @staticmethod
    def from_list(device: Device, data: List[int]) -> BufferI32:
        """Creates a buffer initialized with the contents of a list.

        Args:
            device: The GPU device to create the buffer on.
            data: The initial data for the buffer.

        Returns:
            BufferI32: A new buffer containing the data.

        Raises:
            RuntimeError: If buffer creation fails.

        Example:
            device = Device.acquire()
            buffer = BufferI32.from_list(device, [1, 2, 3, 4])
        """
        ...

    @staticmethod
    def with_len(device: Device, length: int) -> BufferI32:
        """Creates a buffer with space for `length` elements.

        Use this for output buffers that will be written by the GPU.

        Args:
            device: The GPU device to create the buffer on.
            length: The number of int32 elements to allocate.

        Returns:
            BufferI32: A new uninitialized buffer.

        Raises:
            RuntimeError: If buffer creation fails.

        Example:
            device = Device.acquire()
            output = BufferI32.with_len(device, 100)
        """
        ...

    def len(self) -> int:
        """Returns the number of elements in the buffer.

        Returns:
            int: The number of int32 elements.
        """
        ...

    def is_empty(self) -> bool:
        """Returns true if the buffer is empty.

        Returns:
            bool: True if the buffer contains no elements.
        """
        ...

    def to_list(self) -> List[int]:
        """Returns the buffer contents as a list.

        Read the GPU buffer data back to Python. Call this after GPU work
        has completed to retrieve results.

        Returns:
            list[int]: The buffer contents as a Python list.

        Example:
            pass_.submit_and_wait()
            results = output_buffer.to_list()
        """
        ...

    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...

class ComputePass:
    """A compute pass for encoding and dispatching GPU work.

    Use this to bind buffers, dispatch compute work, and submit commands to the GPU.
    A compute pass can only be submitted once.

    Note: This object is consumed when submit() or submit_and_wait() is called.

    Example:
        pass_ = device.new_compute_pass(pipeline)
        pass_.bind_f32(0, input_buffer)
        pass_.bind_f32(1, output_buffer)
        pass_.dispatch_1d(100)
        pass_.submit_and_wait()
    """

    def bind_f32(self, index: int, buffer: BufferF32) -> None:
        """Binds a float32 buffer to the specified index.

        Buffers must be bound to match the buffer indices in your shader code.

        Args:
            index: The buffer binding index (matches [[buffer(N)]] in shader).
            buffer: The buffer to bind.

        Example:
            # Shader: kernel void add(device float* a [[buffer(0)]], ...)
            pass_.bind_f32(0, input_buffer)
        """
        ...

    def bind_i32(self, index: int, buffer: BufferI32) -> None:
        """Binds an int32 buffer to the specified index.

        Buffers must be bound to match the buffer indices in your shader code.

        Args:
            index: The buffer binding index (matches [[buffer(N)]] in shader).
            buffer: The buffer to bind.

        Example:
            # Shader: kernel void process(device int* data [[buffer(0)]], ...)
            pass_.bind_i32(0, data_buffer)
        """
        ...

    def dispatch_1d(self, threads: int) -> None:
        """Dispatches a 1D compute grid with the specified number of threads.

        Args:
            threads: The total number of threads to dispatch.

        Example:
            # Process 1000 elements
            pass_.dispatch_1d(1000)
        """
        ...

    def dispatch_2d(self, width: int, height: int) -> None:
        """Dispatches a 2D compute grid.

        Args:
            width: The width of the 2D grid.
            height: The height of the 2D grid.

        Example:
            # Process a 1920x1080 image
            pass_.dispatch_2d(1920, 1080)
        """
        ...

    def dispatch_3d(self, width: int, height: int, depth: int) -> None:
        """Dispatches a 3D compute grid.

        Args:
            width: The width of the 3D grid.
            height: The height of the 3D grid.
            depth: The depth of the 3D grid.

        Example:
            # Process a 3D volume
            pass_.dispatch_3d(128, 128, 128)
        """
        ...

    def submit_and_wait(self) -> None:
        """Submits the compute pass and waits for completion.

        This blocks until all GPU work is complete. After this call, output buffers
        can be safely read.

        This consumes the ComputePass object.

        Raises:
            RuntimeError: If the pass was already submitted.

        Example:
            pass_.bind_f32(0, input)
            pass_.bind_f32(1, output)
            pass_.dispatch_1d(100)
            pass_.submit_and_wait()
            results = output.to_list()
        """
        ...

    def submit(self) -> Submission:
        """Submits the compute pass without waiting.

        Returns a Submission object that can be used to wait later. This allows
        overlapping GPU work with CPU work.

        This consumes the ComputePass object.

        Returns:
            Submission: A handle to wait on the GPU work later.

        Raises:
            RuntimeError: If the pass was already submitted.

        Example:
            pass_.bind_f32(0, input)
            pass_.bind_f32(1, output)
            pass_.dispatch_1d(100)
            submission = pass_.submit()
            # Do CPU work here...
            submission.wait()
            results = output.to_list()
        """
        ...

    def __repr__(self) -> str: ...

class Submission:
    """A submitted GPU command that may still be executing.

    Created by ComputePass.submit(). Call wait() to block until GPU work completes.

    Note: This object is consumed when wait() is called.

    Example:
        submission = pass_.submit()
        # Do other work...
        submission.wait()  # Wait for GPU to finish
    """

    def wait(self) -> None:
        """Blocks until the GPU work has completed.

        After this call returns, all GPU work is complete and output buffers
        can be safely read.

        This consumes the Submission object.

        Raises:
            RuntimeError: If wait() was already called on this submission.

        Example:
            submission = pass_.submit()
            submission.wait()
            results = output_buffer.to_list()
        """
        ...

    def __repr__(self) -> str: ...

__all__ = [
    "Device",
    "Kernel",
    "Pipeline",
    "BufferF32",
    "BufferI32",
    "ComputePass",
    "Submission",
]

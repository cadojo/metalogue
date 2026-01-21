"""Type stubs for metalogue Python bindings."""

from typing import List

__version__: str

class Device:
    """A handle to a GPU device."""

    @staticmethod
    def acquire() -> Device:
        """Acquires a handle to the system's default GPU.

        Raises:
            RuntimeError: If no GPU device is found.
        """
        ...

    def create_queue(self) -> CommandQueue:
        """Creates a new command queue for submitting work to this device.

        Raises:
            RuntimeError: If queue creation fails.
        """
        ...

    def __repr__(self) -> str: ...

class Kernel:
    """A Metal kernel source with an associated function name."""

    def __init__(self, code: str, function_name: str) -> None:
        """Creates a kernel from source code and function name.

        Args:
            code: Metal Shading Language source code
            function_name: Name of the kernel function to execute
        """
        ...

    @staticmethod
    def from_file(filepath: str, function_name: str) -> Kernel:
        """Creates a kernel from a file path and function name.

        Args:
            filepath: Path to the Metal shader file
            function_name: Name of the kernel function to execute

        Raises:
            RuntimeError: If file cannot be read.
        """
        ...

    def compile(self, device: Device) -> Pipeline:
        """Compiles this kernel into a pipeline on the given device.

        Args:
            device: The GPU device to compile for

        Raises:
            RuntimeError: If compilation fails.
        """
        ...

    def __repr__(self) -> str: ...

class Pipeline:
    """A compiled compute pipeline ready for execution."""

    def __repr__(self) -> str: ...

class BufferF32:
    """A GPU buffer containing 32-bit floating point numbers."""

    @staticmethod
    def from_list(device: Device, data: List[float]) -> BufferF32:
        """Creates a buffer initialized with the contents of a list.

        Args:
            device: The GPU device to create the buffer on
            data: List of float values to initialize the buffer with

        Raises:
            RuntimeError: If buffer creation fails.
        """
        ...

    @staticmethod
    def with_len(device: Device, length: int) -> BufferF32:
        """Creates a buffer with space for `length` elements.

        Args:
            device: The GPU device to create the buffer on
            length: Number of float elements to allocate

        Raises:
            RuntimeError: If buffer creation fails.
        """
        ...

    def len(self) -> int:
        """Returns the number of elements in the buffer."""
        ...

    def is_empty(self) -> bool:
        """Returns True if the buffer is empty."""
        ...

    def to_list(self) -> List[float]:
        """Returns the buffer contents as a list."""
        ...

    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...

class BufferI32:
    """A GPU buffer containing 32-bit signed integers."""

    @staticmethod
    def from_list(device: Device, data: List[int]) -> BufferI32:
        """Creates a buffer initialized with the contents of a list.

        Args:
            device: The GPU device to create the buffer on
            data: List of integer values to initialize the buffer with

        Raises:
            RuntimeError: If buffer creation fails.
        """
        ...

    @staticmethod
    def with_len(device: Device, length: int) -> BufferI32:
        """Creates a buffer with space for `length` elements.

        Args:
            device: The GPU device to create the buffer on
            length: Number of integer elements to allocate

        Raises:
            RuntimeError: If buffer creation fails.
        """
        ...

    def len(self) -> int:
        """Returns the number of elements in the buffer."""
        ...

    def is_empty(self) -> bool:
        """Returns True if the buffer is empty."""
        ...

    def to_list(self) -> List[int]:
        """Returns the buffer contents as a list."""
        ...

    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...

class CommandQueue:
    """A command queue for submitting work to the GPU."""

    def new_compute_pass(self, pipeline: Pipeline) -> ComputePass:
        """Creates a new compute pass for encoding GPU commands.

        Args:
            pipeline: The compiled pipeline to use for this compute pass

        Raises:
            RuntimeError: If compute pass creation fails.
        """
        ...

    def __repr__(self) -> str: ...

class ComputePass:
    """A compute pass for encoding and dispatching GPU work.

    Note: This object is consumed when submit() or submit_and_wait() is called.
    """

    def bind_f32(self, index: int, buffer: BufferF32) -> None:
        """Binds a float32 buffer to the specified index.

        Args:
            index: Buffer binding index (matches [[buffer(N)]] in Metal shader)
            buffer: The float buffer to bind
        """
        ...

    def bind_i32(self, index: int, buffer: BufferI32) -> None:
        """Binds an int32 buffer to the specified index.

        Args:
            index: Buffer binding index (matches [[buffer(N)]] in Metal shader)
            buffer: The integer buffer to bind
        """
        ...

    def dispatch_1d(self, threads: int) -> None:
        """Dispatches a 1D compute grid with the specified number of threads.

        Args:
            threads: Total number of threads to dispatch
        """
        ...

    def dispatch_2d(self, width: int, height: int) -> None:
        """Dispatches a 2D compute grid.

        Args:
            width: Width of the 2D grid
            height: Height of the 2D grid
        """
        ...

    def dispatch_3d(self, width: int, height: int, depth: int) -> None:
        """Dispatches a 3D compute grid.

        Args:
            width: Width of the 3D grid
            height: Height of the 3D grid
            depth: Depth of the 3D grid
        """
        ...

    def submit_and_wait(self) -> None:
        """Submits the compute pass and waits for completion.

        This consumes the ComputePass object.

        Raises:
            RuntimeError: If already submitted.
        """
        ...

    def submit(self) -> Submission:
        """Submits the compute pass without waiting.

        This consumes the ComputePass object.

        Returns:
            A Submission that can be used to wait later.

        Raises:
            RuntimeError: If already submitted.
        """
        ...

    def __repr__(self) -> str: ...

class Submission:
    """A submitted GPU command that may still be executing.

    Note: This object is consumed when wait() is called.
    """

    def wait(self) -> None:
        """Blocks until the GPU work has completed.

        This consumes the Submission object.

        Raises:
            RuntimeError: If already waited on.
        """
        ...

    def __repr__(self) -> str: ...

__all__ = [
    "Device",
    "Kernel",
    "Pipeline",
    "BufferF32",
    "BufferI32",
    "CommandQueue",
    "ComputePass",
    "Submission",
]

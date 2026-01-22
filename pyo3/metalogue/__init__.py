"""Metalogue: Metal compute utilities for GPU programming.

This module provides Python bindings for Metal GPU compute operations on macOS.
It allows you to write GPU kernels in Metal Shading Language and execute them
from Python with a simple, safe API.

Example:
    >>> import metalogue
    >>>
    >>> # Acquire GPU device
    >>> device = metalogue.Device.acquire()
    >>>
    >>> # Define a Metal kernel
    >>> kernel_code = '''
    ... kernel void add_arrays(
    ...     device const float* a [[buffer(0)]],
    ...     device const float* b [[buffer(1)]],
    ...     device float* result [[buffer(2)]],
    ...     uint id [[thread_position_in_grid]])
    ... {
    ...     result[id] = a[id] + b[id];
    ... }
    ... '''
    >>>
    >>> # Compile kernel to pipeline
    >>> kernel = metalogue.Kernel(kernel_code, "add_arrays")
    >>> pipeline = kernel.to_pipeline(device)
    >>>
    >>> # Create buffers
    >>> buffer_a = metalogue.BufferF32.from_list(device, [1.0, 2.0, 3.0, 4.0])
    >>> buffer_b = metalogue.BufferF32.from_list(device, [10.0, 20.0, 30.0, 40.0])
    >>> buffer_result = metalogue.BufferF32.with_len(device, 4)
    >>>
    >>> # Execute kernel
    >>> pass_ = device.new_compute_pass(pipeline)
    >>> pass_.bind_f32(0, buffer_a)
    >>> pass_.bind_f32(1, buffer_b)
    >>> pass_.bind_f32(2, buffer_result)
    >>> pass_.dispatch_1d(4)
    >>> pass_.submit_and_wait()
    >>>
    >>> # Get results
    >>> print(buffer_result.to_list())  # [11.0, 22.0, 33.0, 44.0]
"""

from .metalogue import (
    BufferF32,
    BufferI32,
    ComputePass,
    Device,
    Kernel,
    Pipeline,
    Submission,
)

__all__ = [
    "Device",
    "Kernel",
    "Pipeline",
    "BufferF32",
    "BufferI32",
    "ComputePass",
    "Submission",
]

__version__ = "0.0.1"

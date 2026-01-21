#!/usr/bin/env python3
"""Test suite to verify metalogue Python exports work correctly."""

import metalogue
import pytest


class TestMetalogueExports:
    """Test suite for metalogue Python exports."""

    def test_version_available(self):
        """Test that version is available."""
        assert hasattr(metalogue, "__version__")
        assert metalogue.__version__ is not None

    def test_all_exports_available(self):
        """Test that all expected exports are available."""
        exports = [
            "Device",
            "Kernel",
            "Pipeline",
            "BufferF32",
            "BufferI32",
            "CommandQueue",
            "ComputePass",
            "Submission",
        ]

        for name in exports:
            assert hasattr(metalogue, name), f"{name} export not found"


@pytest.fixture(scope="module")
def device():
    """Fixture to acquire and provide a GPU device."""
    device = metalogue.Device.acquire()
    assert device is not None
    return device


@pytest.fixture(scope="module")
def kernel():
    """Fixture to create a test kernel."""
    kernel_code = """
kernel void add_arrays(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* result [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    result[id] = a[id] + b[id];
}
"""
    return metalogue.Kernel(kernel_code, "add_arrays")


@pytest.fixture(scope="module")
def pipeline(device, kernel):
    """Fixture to compile a pipeline from the kernel."""
    return kernel.compile(device)


@pytest.fixture
def test_buffers(device):
    """Fixture to create test buffers for array operations."""
    buffer_a = metalogue.BufferF32.from_list(device, [1.0, 2.0, 3.0, 4.0])
    buffer_b = metalogue.BufferF32.from_list(device, [10.0, 20.0, 30.0, 40.0])
    buffer_result = metalogue.BufferF32.with_len(device, 4)
    return buffer_a, buffer_b, buffer_result


class TestDevice:
    """Test suite for Device functionality."""

    def test_device_acquisition(self, device):
        """Test that a device can be acquired."""
        assert device is not None

    def test_device_create_queue(self, device):
        """Test that a command queue can be created."""
        queue = device.create_queue()
        assert queue is not None


class TestKernel:
    """Test suite for Kernel and Pipeline functionality."""

    def test_kernel_creation(self, kernel):
        """Test that a kernel can be created."""
        assert kernel is not None

    def test_kernel_compilation(self, pipeline):
        """Test that a kernel can be compiled into a pipeline."""
        assert pipeline is not None


class TestBuffers:
    """Test suite for Buffer functionality."""

    def test_buffer_f32_from_list(self, device):
        """Test creating a float buffer from a list."""
        buffer = metalogue.BufferF32.from_list(device, [1.0, 2.0, 3.0, 4.0])
        assert buffer is not None
        assert len(buffer) == 4

    def test_buffer_f32_with_len(self, device):
        """Test creating a float buffer with specified length."""
        buffer = metalogue.BufferF32.with_len(device, 10)
        assert buffer is not None
        assert len(buffer) == 10

    def test_buffer_f32_to_list(self, device):
        """Test converting a float buffer to a list."""
        data = [1.0, 2.0, 3.0, 4.0]
        buffer = metalogue.BufferF32.from_list(device, data)
        result = buffer.to_list()
        assert result == data

    def test_buffer_i32_from_list(self, device):
        """Test creating an integer buffer from a list."""
        buffer = metalogue.BufferI32.from_list(device, [1, 2, 3, 4, 5])
        assert buffer is not None
        assert len(buffer) == 5

    def test_buffer_i32_to_list(self, device):
        """Test converting an integer buffer to a list."""
        data = [1, 2, 3, 4, 5]
        buffer = metalogue.BufferI32.from_list(device, data)
        result = buffer.to_list()
        assert result == data


class TestKernelExecution:
    """Test suite for kernel execution."""

    def test_compute_pass_creation(self, device, pipeline):
        """Test creating a compute pass."""
        queue = device.create_queue()
        compute_pass = queue.new_compute_pass(pipeline)
        assert compute_pass is not None

    def test_buffer_binding(self, device, pipeline, test_buffers):
        """Test binding buffers to a compute pass."""
        buffer_a, buffer_b, buffer_result = test_buffers
        queue = device.create_queue()
        compute_pass = queue.new_compute_pass(pipeline)

        # Should not raise any exceptions
        compute_pass.bind_f32(0, buffer_a)
        compute_pass.bind_f32(1, buffer_b)
        compute_pass.bind_f32(2, buffer_result)

    def test_kernel_dispatch(self, device, pipeline, test_buffers):
        """Test dispatching a kernel."""
        buffer_a, buffer_b, buffer_result = test_buffers
        queue = device.create_queue()
        compute_pass = queue.new_compute_pass(pipeline)

        compute_pass.bind_f32(0, buffer_a)
        compute_pass.bind_f32(1, buffer_b)
        compute_pass.bind_f32(2, buffer_result)

        # Should not raise any exceptions
        compute_pass.dispatch_1d(4)

    def test_kernel_execution_and_results(self, device, pipeline, test_buffers):
        """Test full kernel execution and verify results."""
        buffer_a, buffer_b, buffer_result = test_buffers
        queue = device.create_queue()
        compute_pass = queue.new_compute_pass(pipeline)

        compute_pass.bind_f32(0, buffer_a)
        compute_pass.bind_f32(1, buffer_b)
        compute_pass.bind_f32(2, buffer_result)
        compute_pass.dispatch_1d(4)
        compute_pass.submit_and_wait()

        result = buffer_result.to_list()
        expected = [11.0, 22.0, 33.0, 44.0]
        assert result == expected, f"Expected {expected}, got {result}"

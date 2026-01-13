//! Integration test for GPU compute operations.

// only Apple computers with M-series chips are supported
#![cfg(target_os = "macos")]

use std::ptr::NonNull;

use metalogue::objc2_foundation::*;
use metalogue::objc2_metal::*;
use metalogue::{Buffer, Device, Kernel};

const ADDER_METAL: &str = include_str!("../kernels/adder.metal");

#[test]
fn test_add_arrays() {
    // get a single GPU device handle
    let gpu = MTLCreateSystemDefaultDevice().expect("failed to retrieve GPU handle");

    // create the command queue
    let command_queue = gpu
        .newCommandQueue()
        .expect("failed to create a command queue");

    // compile the kernel
    let library = gpu
        .newLibraryWithSource_options_error(ns_string!(ADDER_METAL), None)
        .expect("failed to create a library");

    // get the function and create pipeline
    let function = library
        .newFunctionWithName(ns_string!("add_arrays"))
        .expect("failed to get function");
    let pipeline = gpu
        .newComputePipelineStateWithFunction_error(&function)
        .expect("failed to create pipeline");

    // create input data
    const COUNT: usize = 4;
    let data_a: [f32; COUNT] = [1.0, 2.0, 3.0, 4.0];
    let data_b: [f32; COUNT] = [10.0, 20.0, 30.0, 40.0];
    let buffer_size = COUNT * size_of::<f32>();

    // create buffers
    let buffer_a = unsafe {
        gpu.newBufferWithBytes_length_options(
            NonNull::new(data_a.as_ptr() as *mut _).unwrap(),
            buffer_size,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap()
    };
    let buffer_b = unsafe {
        gpu.newBufferWithBytes_length_options(
            NonNull::new(data_b.as_ptr() as *mut _).unwrap(),
            buffer_size,
            MTLResourceOptions::StorageModeShared,
        )
        .unwrap()
    };
    let buffer_result = gpu
        .newBufferWithLength_options(buffer_size, MTLResourceOptions::StorageModeShared)
        .unwrap();

    // encode and dispatch
    let command_buffer = command_queue.commandBuffer().unwrap();
    let encoder = command_buffer.computeCommandEncoder().unwrap();
    unsafe {
        encoder.setComputePipelineState(&pipeline);
        encoder.setBuffer_offset_atIndex(Some(&buffer_a), 0, 0);
        encoder.setBuffer_offset_atIndex(Some(&buffer_b), 0, 1);
        encoder.setBuffer_offset_atIndex(Some(&buffer_result), 0, 2);
    }

    let grid_size = MTLSize {
        width: COUNT,
        height: 1,
        depth: 1,
    };
    let threadgroup_size = MTLSize {
        width: COUNT,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
    encoder.endEncoding();

    // run and wait
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    // read results
    let result_ptr = buffer_result.contents().as_ptr() as *const f32;
    let results: &[f32] = unsafe { std::slice::from_raw_parts(result_ptr, COUNT) };

    assert_eq!(results, &[11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn test_add_arrays_safe() {
    // acquire GPU device
    let device = Device::acquire().expect("failed to acquire GPU");

    // compile kernel into pipeline
    let pipeline = Kernel::new(ADDER_METAL, "add_arrays")
        .compile(&device)
        .expect("failed to compile kernel")
        .to_pipeline()
        .expect("failed to create pipeline");

    // create buffers
    let buffer_a =
        Buffer::from_slice(&device, &[1.0_f32, 2.0, 3.0, 4.0]).expect("failed to create buffer a");
    let buffer_b = Buffer::from_slice(&device, &[10.0_f32, 20.0, 30.0, 40.0])
        .expect("failed to create buffer b");
    let buffer_result: Buffer<f32> =
        Buffer::with_len(&device, 4).expect("failed to create result buffer");

    // create command queue and compute pass
    let queue = device.create_queue().expect("failed to create queue");
    let pass = queue
        .new_compute_pass(&pipeline)
        .expect("failed to create compute pass");

    // bind buffers and dispatch
    pass.bind(0, &buffer_a);
    pass.bind(1, &buffer_b);
    pass.bind(2, &buffer_result);
    pass.dispatch_1d(4);
    pass.submit_and_wait();

    // read results (no unsafe!)
    assert_eq!(buffer_result.as_slice(), &[11.0, 22.0, 33.0, 44.0]);
}

//! Exploring Apple's Metal API in Rust.
//! Tracks GPU kernel execution for heatmap visualization.

use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use std::time::Instant;

/// Represents a single kernel execution event for heatmap tracking
#[derive(Debug, Clone)]
pub struct KernelExecutionEvent {
    /// Name/identifier of the kernel
    pub kernel_name: String,
    /// Dispatch index (which dispatch call this was)
    pub dispatch_id: u32,
    /// Grid dimensions (total threads)
    pub grid_size: (u32, u32, u32),
    /// Threadgroup dimensions
    pub threadgroup_size: (u32, u32, u32),
    /// Number of threadgroups
    pub threadgroup_count: (u32, u32, u32),
    /// Timestamp when dispatch was called (relative to tracker start)
    pub dispatch_time_us: u64,
    /// Timestamp when GPU completed (relative to tracker start)
    pub completion_time_us: Option<u64>,
    /// Duration of GPU execution in microseconds
    pub gpu_duration_us: Option<u64>,
}

/// Tracks kernel executions for later heatmap visualization
#[derive(Debug)]
pub struct KernelExecutionTracker {
    /// All recorded execution events
    pub events: Vec<KernelExecutionEvent>,
    /// Start time for relative timestamps
    start_time: Instant,
    /// Current dispatch counter
    dispatch_counter: u32,
}

impl KernelExecutionTracker {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            start_time: Instant::now(),
            dispatch_counter: 0,
        }
    }

    /// Record a kernel dispatch
    pub fn record_dispatch(
        &mut self,
        kernel_name: &str,
        grid_size: (u32, u32, u32),
        threadgroup_size: (u32, u32, u32),
    ) -> usize {
        let threadgroup_count = (
            (grid_size.0 + threadgroup_size.0 - 1) / threadgroup_size.0,
            (grid_size.1 + threadgroup_size.1 - 1) / threadgroup_size.1,
            (grid_size.2 + threadgroup_size.2 - 1) / threadgroup_size.2,
        );

        let event = KernelExecutionEvent {
            kernel_name: kernel_name.to_string(),
            dispatch_id: self.dispatch_counter,
            grid_size,
            threadgroup_size,
            threadgroup_count,
            dispatch_time_us: self.start_time.elapsed().as_micros() as u64,
            completion_time_us: None,
            gpu_duration_us: None,
        };

        self.dispatch_counter += 1;
        let index = self.events.len();
        self.events.push(event);
        index
    }

    /// Mark a dispatch as completed
    pub fn record_completion(&mut self, event_index: usize) {
        if let Some(event) = self.events.get_mut(event_index) {
            let completion_time = self.start_time.elapsed().as_micros() as u64;
            event.completion_time_us = Some(completion_time);
            event.gpu_duration_us = Some(completion_time - event.dispatch_time_us);
        }
    }

    /// Get data formatted for heatmap visualization
    /// Returns (x, y, intensity) tuples where intensity is based on execution time
    pub fn get_heatmap_data(&self) -> Vec<HeatmapPoint> {
        let mut points = Vec::new();

        for event in &self.events {
            // Create heatmap points for each threadgroup
            let intensity = event.gpu_duration_us.unwrap_or(1) as f32;

            for z in 0..event.threadgroup_count.2 {
                for y in 0..event.threadgroup_count.1 {
                    for x in 0..event.threadgroup_count.0 {
                        points.push(HeatmapPoint {
                            x,
                            y,
                            z,
                            dispatch_id: event.dispatch_id,
                            kernel_name: event.kernel_name.clone(),
                            intensity,
                        });
                    }
                }
            }
        }

        points
    }

    /// Print a summary of all tracked executions
    pub fn print_summary(&self) {
        println!("\n=== Kernel Execution Summary ===");
        println!("Total dispatches: {}", self.events.len());
        println!();

        for event in &self.events {
            println!("Dispatch #{}: {}", event.dispatch_id, event.kernel_name);
            println!(
                "  Grid: {}x{}x{}",
                event.grid_size.0, event.grid_size.1, event.grid_size.2
            );
            println!(
                "  Threadgroups: {}x{}x{} (size: {}x{}x{})",
                event.threadgroup_count.0,
                event.threadgroup_count.1,
                event.threadgroup_count.2,
                event.threadgroup_size.0,
                event.threadgroup_size.1,
                event.threadgroup_size.2
            );
            println!("  Dispatch time: {} µs", event.dispatch_time_us);
            if let Some(completion) = event.completion_time_us {
                println!("  Completion time: {} µs", completion);
            }
            if let Some(duration) = event.gpu_duration_us {
                println!("  GPU duration: {} µs", duration);
            }
            println!();
        }
    }
}

/// A point for heatmap visualization
#[derive(Debug, Clone)]
pub struct HeatmapPoint {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub dispatch_id: u32,
    pub kernel_name: String,
    pub intensity: f32,
}

/// Per-thread execution data written by the GPU kernel
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct ThreadExecutionData {
    /// Global thread ID
    pub thread_id_x: u32,
    pub thread_id_y: u32,
    pub thread_id_z: u32,
    /// Threadgroup ID
    pub threadgroup_id_x: u32,
    pub threadgroup_id_y: u32,
    pub threadgroup_id_z: u32,
    /// Local thread position within threadgroup
    pub local_id_x: u32,
    pub local_id_y: u32,
    pub local_id_z: u32,
    /// Execution order (via atomic counter)
    pub execution_order: u32,
    /// Simple "work" value for verification
    pub computed_value: f32,
    pub _padding: u32,
}

/// Metal Shader Library source code
const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Per-thread execution tracking data
struct ThreadExecutionData {
    uint thread_id_x;
    uint thread_id_y;
    uint thread_id_z;
    uint threadgroup_id_x;
    uint threadgroup_id_y;
    uint threadgroup_id_z;
    uint local_id_x;
    uint local_id_y;
    uint local_id_z;
    uint execution_order;
    float computed_value;
    uint _padding;
};

// Simple compute kernel that tracks its own execution
kernel void tracked_kernel(
    device ThreadExecutionData* output [[buffer(0)]],
    device atomic_uint* counter [[buffer(1)]],
    constant uint& grid_width [[buffer(2)]],
    constant uint& grid_height [[buffer(3)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]]
) {
    // Calculate linear index
    uint idx = thread_position_in_grid.x +
               thread_position_in_grid.y * grid_width +
               thread_position_in_grid.z * grid_width * grid_height;

    // Record execution order atomically
    uint order = atomic_fetch_add_explicit(counter, 1, memory_order_relaxed);

    // Store thread execution data
    output[idx].thread_id_x = thread_position_in_grid.x;
    output[idx].thread_id_y = thread_position_in_grid.y;
    output[idx].thread_id_z = thread_position_in_grid.z;
    output[idx].threadgroup_id_x = threadgroup_position_in_grid.x;
    output[idx].threadgroup_id_y = threadgroup_position_in_grid.y;
    output[idx].threadgroup_id_z = threadgroup_position_in_grid.z;
    output[idx].local_id_x = thread_position_in_threadgroup.x;
    output[idx].local_id_y = thread_position_in_threadgroup.y;
    output[idx].local_id_z = thread_position_in_threadgroup.z;
    output[idx].execution_order = order;

    // Do some simple computation
    float x = float(thread_position_in_grid.x);
    float y = float(thread_position_in_grid.y);
    output[idx].computed_value = sin(x * 0.1) * cos(y * 0.1) + float(order) * 0.001;
}
"#;

fn main() {
    println!("=== Metal Kernel Execution Tracker ===\n");

    // Create execution tracker
    let mut tracker = KernelExecutionTracker::new();

    // Get the default Metal device
    let device = MTLCreateSystemDefaultDevice().expect("Failed to get default Metal device");

    println!("Using device: {:?}", device.name());

    // Create command queue
    let command_queue = device
        .newCommandQueue()
        .expect("Failed to create command queue");

    // Compile shader source
    let source = NSString::from_str(SHADER_SOURCE);
    let library = device
        .newLibraryWithSource_options_error(&source, None)
        .expect("Failed to compile shader library");

    // Get the kernel function
    let function_name = NSString::from_str("tracked_kernel");
    let kernel_function = library
        .newFunctionWithName(&function_name)
        .expect("Failed to get kernel function");

    // Create compute pipeline state
    let pipeline_state = device
        .newComputePipelineStateWithFunction_error(&kernel_function)
        .expect("Failed to create compute pipeline state");

    // Define grid dimensions for our test
    let grid_width: u32 = 64;
    let grid_height: u32 = 64;
    let grid_depth: u32 = 1;
    let total_threads = (grid_width * grid_height * grid_depth) as usize;

    println!(
        "Dispatching kernel with grid: {}x{}x{} ({} total threads)",
        grid_width, grid_height, grid_depth, total_threads
    );

    // Create output buffer for thread execution data
    let output_buffer_size = total_threads * std::mem::size_of::<ThreadExecutionData>();
    let output_buffer = device
        .newBufferWithLength_options(output_buffer_size, MTLResourceOptions::StorageModeShared)
        .expect("Failed to create output buffer");

    // Create atomic counter buffer
    let counter_buffer = device
        .newBufferWithLength_options(
            std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
        .expect("Failed to create counter buffer");

    // Initialize counter to 0
    unsafe {
        let counter_ptr = counter_buffer.contents().as_ptr() as *mut u32;
        *counter_ptr = 0;
    }

    // Create buffers for grid dimensions
    let width_buffer = device
        .newBufferWithLength_options(
            std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
        .expect("Failed to create width buffer");
    let height_buffer = device
        .newBufferWithLength_options(
            std::mem::size_of::<u32>(),
            MTLResourceOptions::StorageModeShared,
        )
        .expect("Failed to create height buffer");

    unsafe {
        *(width_buffer.contents().as_ptr() as *mut u32) = grid_width;
        *(height_buffer.contents().as_ptr() as *mut u32) = grid_height;
    }

    // Create command buffer
    let command_buffer = command_queue
        .commandBuffer()
        .expect("Failed to create command buffer");

    // Create compute command encoder
    let compute_encoder = command_buffer
        .computeCommandEncoder()
        .expect("Failed to create compute encoder");

    // Set pipeline state and buffers
    unsafe {
        compute_encoder.setComputePipelineState(&pipeline_state);
        compute_encoder.setBuffer_offset_atIndex(Some(&output_buffer), 0, 0);
        compute_encoder.setBuffer_offset_atIndex(Some(&counter_buffer), 0, 1);
        compute_encoder.setBuffer_offset_atIndex(Some(&width_buffer), 0, 2);
        compute_encoder.setBuffer_offset_atIndex(Some(&height_buffer), 0, 3);
    }

    // Calculate threadgroup size
    let max_threads = pipeline_state.maxTotalThreadsPerThreadgroup() as u32;
    let threadgroup_size = MTLSize {
        width: 16.min(grid_width as usize),
        height: 16.min(grid_height as usize),
        depth: 1,
    };

    let grid_size = MTLSize {
        width: grid_width as usize,
        height: grid_height as usize,
        depth: grid_depth as usize,
    };

    println!(
        "Threadgroup size: {}x{}x{}",
        threadgroup_size.width, threadgroup_size.height, threadgroup_size.depth
    );
    println!("Max threads per threadgroup: {}", max_threads);

    // Record the dispatch in our tracker
    let event_index = tracker.record_dispatch(
        "tracked_kernel",
        (grid_width, grid_height, grid_depth),
        (
            threadgroup_size.width as u32,
            threadgroup_size.height as u32,
            threadgroup_size.depth as u32,
        ),
    );

    // Dispatch the kernel
    compute_encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

    // End encoding
    compute_encoder.endEncoding();

    // Commit and wait for completion
    println!("\nExecuting kernel on GPU...");
    command_buffer.commit();
    command_buffer.waitUntilCompleted();

    // Record completion
    tracker.record_completion(event_index);

    println!("Kernel execution completed!");

    // Read back results
    let output_ptr = output_buffer.contents().as_ptr() as *const ThreadExecutionData;
    let results: Vec<ThreadExecutionData> =
        unsafe { std::slice::from_raw_parts(output_ptr, total_threads) }.to_vec();

    // Analyze execution order for heatmap data
    println!("\n=== Thread Execution Analysis ===");

    // Find min/max execution order
    let min_order = results.iter().map(|t| t.execution_order).min().unwrap_or(0);
    let max_order = results.iter().map(|t| t.execution_order).max().unwrap_or(0);
    println!(
        "Execution order range: {} to {} (spread: {})",
        min_order,
        max_order,
        max_order - min_order
    );

    // Show some sample thread data
    println!("\nSample thread execution data (first 10 by execution order):");
    let mut sorted_results = results.clone();
    sorted_results.sort_by_key(|t| t.execution_order);

    for thread_data in sorted_results.iter().take(10) {
        println!(
            "  Order {:4}: thread({:2},{:2},{}) in threadgroup({},{},{}) local({:2},{:2},{})",
            thread_data.execution_order,
            thread_data.thread_id_x,
            thread_data.thread_id_y,
            thread_data.thread_id_z,
            thread_data.threadgroup_id_x,
            thread_data.threadgroup_id_y,
            thread_data.threadgroup_id_z,
            thread_data.local_id_x,
            thread_data.local_id_y,
            thread_data.local_id_z,
        );
    }

    // Create heatmap data based on execution order
    println!("\n=== Heatmap Data (Execution Order per Threadgroup) ===");
    let mut threadgroup_avg_order: std::collections::HashMap<(u32, u32, u32), (u32, u32)> =
        std::collections::HashMap::new();

    for thread_data in &results {
        let key = (
            thread_data.threadgroup_id_x,
            thread_data.threadgroup_id_y,
            thread_data.threadgroup_id_z,
        );
        let entry = threadgroup_avg_order.entry(key).or_insert((0, 0));
        entry.0 += thread_data.execution_order;
        entry.1 += 1;
    }

    println!("\nThreadgroup average execution order (for heatmap intensity):");
    let mut threadgroup_data: Vec<_> = threadgroup_avg_order
        .iter()
        .map(|(k, v)| (*k, v.0 as f32 / v.1 as f32))
        .collect();
    threadgroup_data.sort_by_key(|(k, _)| (k.1, k.0));

    for ((x, y, z), avg_order) in &threadgroup_data {
        let normalized = (avg_order - min_order as f32) / (max_order - min_order) as f32;
        let bar_len = (normalized * 20.0) as usize;
        let bar: String = "█".repeat(bar_len);
        println!(
            "  TG({:2},{:2},{}): avg_order={:6.1} intensity={:.3} {}",
            x, y, z, avg_order, normalized, bar
        );
    }

    // Print tracker summary
    tracker.print_summary();

    // Get heatmap points
    let heatmap_points = tracker.get_heatmap_data();
    println!("Generated {} heatmap points", heatmap_points.len());

    println!("\n=== Ready for Heatmap Visualization ===");
    println!("Execution data stored in tracker.events and heatmap_points");
    println!("You can export this data to JSON/CSV for visualization tools");
}

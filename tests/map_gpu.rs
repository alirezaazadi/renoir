//! Tests for the `map_gpu` operator.
//!
//! These tests verify the correctness of GPU-accelerated map operations
//! using the `map_gpu` and `map_gpu_with_strategy` operators.
//!
//! Tests are only compiled and run when a GPU feature is enabled.
//!
//! Run with:
//! ```bash
//! cargo test --features gpu-wgpu map_gpu
//! ```

#![cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]

use bytemuck::{Pod, Zeroable};
use cubecl::prelude::*;
use serde::{Deserialize, Serialize};

use renoir::operator::gpu::{GpuBatchStrategy, GpuContext, GpuKernel};
use renoir::StreamContext;

#[cfg(feature = "gpu-wgpu")]
use cubecl::wgpu::WgpuRuntime;

// ============================================================================
// Test Data Structures
// ============================================================================

/// Simple input type for testing: a single f32 value.
#[derive(Clone, Copy, Debug, Pod, Zeroable, Serialize, Deserialize, PartialEq)]
#[repr(C)]
struct SimpleInput {
    value: f32,
}

/// Simple output type for testing: a single f32 result.
#[derive(Clone, Copy, Debug, Pod, Zeroable, Serialize, Deserialize, PartialEq)]
#[repr(C)]
struct SimpleOutput {
    result: f32,
}

// ============================================================================
// Test Kernel: Square Operation
// ============================================================================

/// CubeCL kernel that squares each input value.
///
/// This is a simple kernel used for testing the map_gpu infrastructure.
#[cube(launch_unchecked)]
fn square_kernel<F: Float>(
    inputs: &Array<Line<F>>,
    outputs: &mut Array<Line<F>>,
) {
    if ABSOLUTE_POS < inputs.len() {
        let val = inputs[ABSOLUTE_POS];
        outputs[ABSOLUTE_POS] = val * val;
    }
}

/// GPU kernel that squares input values.
#[derive(Clone)]
struct SquareKernel {
    cube_dim: u32,
}

impl Default for SquareKernel {
    fn default() -> Self {
        Self { cube_dim: 256 }
    }
}

impl GpuKernel for SquareKernel {
    type Input = SimpleInput;
    type Output = SimpleOutput;

    fn execute(&self, ctx: &GpuContext, inputs: &[SimpleInput]) -> Vec<SimpleOutput> {
        if inputs.is_empty() {
            return Vec::new();
        }

        let num_elements = inputs.len();
        let client = ctx.client();

        // Extract values from input structs
        let values: Vec<f32> = inputs.iter().map(|i| i.value).collect();

        // Pad for vectorization (4 elements per Line<f32>)
        let vectorization_factor = 4usize;
        let num_padded = num_elements
            + (vectorization_factor - (num_elements % vectorization_factor)) % vectorization_factor;

        let padded_values: Vec<f32> = values
            .into_iter()
            .chain(std::iter::repeat(0.0))
            .take(num_padded)
            .collect();

        // Allocate GPU buffers
        let input_handle = client.create(f32::as_bytes(&padded_values));
        let output_handle = client.empty(num_padded * std::mem::size_of::<f32>());

        // Launch configuration
        let cube_dim = CubeDim::new(self.cube_dim, 1, 1);
        let total_lines = num_padded / vectorization_factor;
        let num_cubes = (total_lines as u32 + cube_dim.x - 1) / cube_dim.x;
        let cube_count = CubeCount::Static(num_cubes, 1, 1);

        // Launch kernel
        #[cfg(feature = "gpu-wgpu")]
        unsafe {
            square_kernel::launch_unchecked::<f32, WgpuRuntime>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<f32>(&input_handle, num_padded, vectorization_factor as u8),
                ArrayArg::from_raw_parts::<f32>(&output_handle, num_padded, vectorization_factor as u8),
            );
        }

        // Sync and read results
        ctx.sync();
        let result_bytes = client.read_one(output_handle.clone());

        // Convert to output structs
        result_bytes
            .chunks_exact(4)
            .take(num_elements)
            .map(|b| SimpleOutput {
                result: f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            })
            .collect()
    }

    fn preferred_batch_size(&self) -> Option<usize> {
        Some(1000)
    }
}

// ============================================================================
// Test Kernel: Identity (passthrough)
// ============================================================================

/// GPU kernel that returns input unchanged (identity function).
/// Useful for testing data transfer correctness.
#[derive(Clone)]
struct IdentityKernel;

impl GpuKernel for IdentityKernel {
    type Input = SimpleInput;
    type Output = SimpleOutput;

    fn execute(&self, _ctx: &GpuContext, inputs: &[SimpleInput]) -> Vec<SimpleOutput> {
        // CPU implementation for identity - just copies values
        inputs
            .iter()
            .map(|i| SimpleOutput { result: i.value })
            .collect()
    }
}

// ============================================================================
// Tests
// ============================================================================

/// Test basic map_gpu functionality with a small dataset.
#[test]
fn test_map_gpu_basic() {
    let env = StreamContext::new_local();

    // Create input data: squares of 1 through 10
    let inputs: Vec<SimpleInput> = (1..=10)
        .map(|i| SimpleInput { value: i as f32 })
        .collect();

    let result = env
        .stream_iter(inputs.into_iter())
        .map_gpu(SquareKernel::default())
        .collect_vec();

    env.execute_blocking();

    let outputs = result.get().expect("Should have results");

    // Verify results: each output should be input^2
    let expected: Vec<f32> = (1..=10).map(|i| (i * i) as f32).collect();
    let actual: Vec<f32> = outputs.iter().map(|o| o.result).collect();

    assert_eq!(actual.len(), expected.len());
    for (actual_val, expected_val) in actual.iter().zip(expected.iter()) {
        assert!(
            (actual_val - expected_val).abs() < 0.001,
            "Expected {}, got {}",
            expected_val,
            actual_val
        );
    }
}

/// Test map_gpu with custom batch strategy.
#[test]
fn test_map_gpu_with_strategy() {
    let env = StreamContext::new_local();

    let inputs: Vec<SimpleInput> = (1..=100)
        .map(|i| SimpleInput { value: i as f32 })
        .collect();

    // Use fixed batch size of 25 to force multiple batches
    let result = env
        .stream_iter(inputs.into_iter())
        .map_gpu_with_strategy(SquareKernel::default(), GpuBatchStrategy::fixed(25))
        .collect_vec();

    env.execute_blocking();

    let outputs = result.get().expect("Should have results");

    assert_eq!(outputs.len(), 100);

    // Verify all results
    for (i, output) in outputs.iter().enumerate() {
        let expected = ((i + 1) * (i + 1)) as f32;
        assert!(
            (output.result - expected).abs() < 0.001,
            "At index {}: expected {}, got {}",
            i,
            expected,
            output.result
        );
    }
}

/// Test map_gpu with identity kernel to verify data integrity.
#[test]
fn test_map_gpu_identity() {
    let env = StreamContext::new_local();

    let inputs: Vec<SimpleInput> = (0..50)
        .map(|i| SimpleInput {
            value: i as f32 * 0.5,
        })
        .collect();
    let expected: Vec<f32> = inputs.iter().map(|i| i.value).collect();

    let result = env
        .stream_iter(inputs.into_iter())
        .map_gpu(IdentityKernel)
        .collect_vec();

    env.execute_blocking();

    let outputs = result.get().expect("Should have results");
    let actual: Vec<f32> = outputs.iter().map(|o| o.result).collect();

    assert_eq!(actual, expected);
}

/// Test map_gpu with empty input.
#[test]
fn test_map_gpu_empty() {
    let env = StreamContext::new_local();

    let inputs: Vec<SimpleInput> = vec![];

    let result = env
        .stream_iter(inputs.into_iter())
        .map_gpu(SquareKernel::default())
        .collect_vec();

    env.execute_blocking();

    let outputs = result.get().expect("Should have results");
    assert!(outputs.is_empty());
}

/// Test map_gpu with single element.
#[test]
fn test_map_gpu_single_element() {
    let env = StreamContext::new_local();

    let inputs = vec![SimpleInput { value: 5.0 }];

    let result = env
        .stream_iter(inputs.into_iter())
        .map_gpu(SquareKernel::default())
        .collect_vec();

    env.execute_blocking();

    let outputs = result.get().expect("Should have results");
    assert_eq!(outputs.len(), 1);
    assert!((outputs[0].result - 25.0).abs() < 0.001);
}

/// Test map_gpu with large dataset to verify batching works correctly.
#[test]
fn test_map_gpu_large_dataset() {
    let env = StreamContext::new_local();

    let count = 10_000;
    let inputs: Vec<SimpleInput> = (0..count)
        .map(|i| SimpleInput { value: i as f32 })
        .collect();

    let result = env
        .stream_iter(inputs.into_iter())
        .map_gpu_with_strategy(SquareKernel::default(), GpuBatchStrategy::fixed(1000))
        .collect_vec();

    env.execute_blocking();

    let outputs = result.get().expect("Should have results");
    assert_eq!(outputs.len(), count);

    // Verify a sample of results
    for i in [0, 100, 500, 1000, 5000, 9999] {
        let expected = (i * i) as f32;
        assert!(
            (outputs[i].result - expected).abs() < 0.1,
            "At index {}: expected {}, got {}",
            i,
            expected,
            outputs[i].result
        );
    }
}

/// Test that map_gpu can be chained with other operators.
#[test]
fn test_map_gpu_chained() {
    let env = StreamContext::new_local();

    let inputs: Vec<SimpleInput> = (1..=5)
        .map(|i| SimpleInput { value: i as f32 })
        .collect();

    // Chain: GPU square -> CPU filter (keep values > 5) -> CPU map (add 1)
    let result = env
        .stream_iter(inputs.into_iter())
        .map_gpu(SquareKernel::default())
        .filter(|o| o.result > 5.0)
        .map(|o| o.result + 1.0)
        .collect_vec();

    env.execute_blocking();

    let outputs = result.get().expect("Should have results");

    // 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=25
    // Filter > 5: 9, 16, 25
    // Add 1: 10, 17, 26
    assert_eq!(outputs, vec![10.0, 17.0, 26.0]);
}

/// Test map_gpu with timed batch strategy.
#[test]
fn test_map_gpu_timed_strategy() {
    use std::time::Duration;

    let env = StreamContext::new_local();

    let inputs: Vec<SimpleInput> = (1..=20)
        .map(|i| SimpleInput { value: i as f32 })
        .collect();

    let result = env
        .stream_iter(inputs.into_iter())
        .map_gpu_with_strategy(
            SquareKernel::default(),
            GpuBatchStrategy::timed(10, Duration::from_millis(100)),
        )
        .collect_vec();

    env.execute_blocking();

    let outputs = result.get().expect("Should have results");
    assert_eq!(outputs.len(), 20);

    // Verify first and last results
    assert!((outputs[0].result - 1.0).abs() < 0.001);
    assert!((outputs[19].result - 400.0).abs() < 0.001);
}

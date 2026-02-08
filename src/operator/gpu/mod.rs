//! # GPU Acceleration Module for Renoir
//!
//! This module provides GPU-accelerated operators using [CubeCL](https://github.com/tracel-ai/cubecl),
//! a Rust library for writing GPU compute kernels that compile to multiple backends.
//!
//! ## Supported Backends
//!
//! - **WGPU** (`gpu-wgpu` feature): Cross-platform GPU compute via WebGPU standard.
//!   Works on macOS (Metal), Windows (DirectX 12/Vulkan), Linux (Vulkan), and web browsers.
//! - **CUDA** (`gpu-cuda` feature): NVIDIA GPU compute for maximum performance on NVIDIA hardware.
//!
//! ## Architecture Overview
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                         Renoir Stream                                   │
//! │                                                                         │
//! │  stream_iter(data)                                                      │
//! │       │                                                                 │
//! │       ▼                                                                 │
//! │  ┌─────────────────────────────────────────────────────────────────┐    │
//! │  │                    map_gpu(kernel)                              │    │
//! │  │  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐  │    │
//! │  │  │   Buffer    │  │  GPU Kernel  │  │    Output Queue        │  │    │
//! │  │  │   (batch)   │──│  Execution   │──│    (results)           │  │    │
//! │  │  └─────────────┘  └──────────────┘  └────────────────────────┘  │    │
//! │  │        ▲                                        │               │    │
//! │  │        │         Batch Strategy                 │               │    │
//! │  │        │        (Fixed/Timed/Adaptive)          │               │    │
//! │  │        │                                        ▼               │    │
//! │  └────────┼────────────────────────────────────────┼───────────────┘    │
//! │           │                                        │                    │
//! │       (input)                                  (output)                 │
//! │           │                                        │                    │
//! │           ▼                                        ▼                    │
//! │      .collect_vec()                           results                   │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Components
//!
//! - [`GpuKernel`]: Trait for defining GPU compute kernels
//! - [`GpuContext`]: Manages GPU device and memory resources
//! - [`GpuBatchStrategy`]: Controls when buffered data is sent to GPU
//! - [`MapGpu`]: The operator that applies a GPU kernel to stream elements
//!
//! ## Usage Example
//!
//! ```ignore
//! use renoir::prelude::*;
//! use renoir::operator::gpu::{GpuKernel, GpuBatchStrategy, GpuContext};
//! use bytemuck::{Pod, Zeroable};
//! use cubecl::prelude::*;
//!
//! // 1. Define input/output types (must be Pod for GPU memory transfer)
//! #[derive(Clone, Copy, Pod, Zeroable)]
//! #[repr(C)]
//! struct MyInput { value: f32 }
//!
//! #[derive(Clone, Copy, Pod, Zeroable)]
//! #[repr(C)]
//! struct MyOutput { result: f32 }
//!
//! // 2. Define a CubeCL kernel using the #[cube] macro
//! #[cube(launch_unchecked)]
//! fn my_kernel<F: Float>(inputs: &Array<F>, outputs: &mut Array<F>) {
//!     if ABSOLUTE_POS < inputs.len() {
//!         outputs[ABSOLUTE_POS] = inputs[ABSOLUTE_POS] * inputs[ABSOLUTE_POS];
//!     }
//! }
//!
//! // 3. Implement GpuKernel trait to wrap the CubeCL kernel
//! #[derive(Clone)]
//! struct MyGpuKernel;
//!
//! impl GpuKernel for MyGpuKernel {
//!     type Input = MyInput;
//!     type Output = MyOutput;
//!
//!     fn execute(&self, ctx: &GpuContext, inputs: &[MyInput]) -> Vec<MyOutput> {
//!         // GPU kernel implementation:
//!         // 1. Copy input data to GPU
//!         // 2. Launch kernel
//!         // 3. Copy results back
//!         // See examples/black_scholes_gpu.rs for complete implementation
//!         todo!()
//!     }
//! }
//!
//! // 4. Use in a Renoir stream
//! let env = StreamContext::new_local();
//! let result = env
//!     .stream_iter(data.into_iter())
//!     .map_gpu_with_strategy(MyGpuKernel, GpuBatchStrategy::fixed(100_000))
//!     .collect_vec();
//! env.execute_blocking();
//! ```
//!
//! ## Performance Considerations
//!
//! - **Batch Size**: GPU efficiency increases with larger batches. The default is 10M items.
//! - **Data Transfer**: Minimize CPU↔GPU transfers by processing large batches.
//! - **Vectorization**: CubeCL uses `Line<F>` types for SIMD-like processing (4 f32s per thread).
//! - **Kernel Compilation**: First kernel launch compiles shaders; subsequent launches are fast.
//!
//! ## When to Use GPU Acceleration
//!
//! GPU acceleration is beneficial when:
//! - Processing large amounts of data (>100K elements)
//! - Operations are compute-intensive (math operations, not I/O bound)
//! - Data parallelism is high (same operation on many elements)
//!
//! GPU may be slower for:
//! - Small datasets (overhead exceeds computation time)
//! - Memory-bound operations with little computation
//! - Highly branching code paths

// Module declarations
mod batch_strategy;
mod context;
mod kernel;
mod map_gpu;
pub(crate) mod reduce_gpu;

// Public exports for users
pub use batch_strategy::GpuBatchStrategy;
pub use context::GpuContext;
pub use kernel::GpuKernel;
pub use map_gpu::MapGpu;
pub use reduce_gpu::{ReduceGpuBackend, ReduceGpuConfig, ReduceKernel};

// Re-export CubeCL types that users need for implementing kernels.
// This allows users to import everything from renoir::operator::gpu
// without needing to add cubecl as a direct dependency.
pub use cubecl::prelude::*;

/// WGPU runtime for cross-platform GPU compute.
///
/// Use this as the runtime type parameter when launching kernels:
/// ```ignore
/// black_scholes_kernel::launch_unchecked::<f32, WgpuRuntime>(...)
/// ```
#[cfg(feature = "gpu-wgpu")]
pub use cubecl::wgpu::WgpuRuntime;

/// CUDA runtime for NVIDIA GPU compute.
///
/// Use this as the runtime type parameter when launching kernels:
/// ```ignore
/// black_scholes_kernel::launch_unchecked::<f32, CudaRuntime>(...)
/// ```
#[cfg(feature = "gpu-cuda")]
pub use cubecl::cuda::CudaRuntime;

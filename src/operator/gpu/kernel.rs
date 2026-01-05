//! # GPU Kernel Trait Definition
//!
//! This module defines the [`GpuKernel`] trait that users implement to create
//! GPU-accelerated computations for use with Renoir's `map_gpu` operator.
//!
//! ## Design Philosophy
//!
//! The `GpuKernel` trait provides a high-level abstraction over GPU programming:
//!
//! ```text
//! User's Perspective:              Under the Hood:
//! ┌─────────────────────┐         ┌─────────────────────────────────────────┐
//! │ impl GpuKernel {    │         │ 1. Allocate GPU memory buffers          │
//! │   fn execute(...) { │    ───▶ │ 2. Copy input data CPU → GPU            │
//! │     // your code    │         │ 3. Configure kernel launch parameters   │
//! │   }                 │         │ 4. Execute GPU kernel                   │
//! │ }                   │         │ 5. Synchronize (wait for completion)    │
//! └─────────────────────┘         │ 6. Copy results GPU → CPU               │
//!                                 │ 7. Return output vector                 │
//!                                 └─────────────────────────────────────────┘
//! ```
//!
//! ## Type Requirements
//!
//! Input and output types must satisfy several constraints:
//!
//! - **`Data`** (Clone + Send + 'static): Required for Renoir stream processing
//! - **`bytemuck::Pod`**: "Plain Old Data" - can be safely cast to/from bytes
//!   for GPU memory transfer. This means:
//!   - No pointers or references
//!   - No padding bytes (use `#[repr(C)]`)
//!   - All fields are also Pod
//!
//! ## Example Implementation
//!
//! See `examples/black_scholes_gpu.rs` for a complete, real-world implementation.

use super::context::GpuContext;
use crate::operator::Data;

/// Trait for defining GPU kernels compatible with Renoir's `map_gpu` operator.
///
/// Implementors of this trait define how batches of input data are processed
/// on the GPU and produce output data. The trait abstracts the details of:
/// - GPU memory allocation and data transfer
/// - Kernel launch configuration
/// - Synchronization
///
/// # Type Parameters
///
/// The associated types `Input` and `Output` must implement:
/// - `Data`: Renoir's marker trait (Clone + Send + static)
/// - `bytemuck::Pod`: Safe for raw byte conversion to/from GPU memory
///
/// # Contract
///
/// - `execute()` must return exactly one output element for each input element
/// - The output vector length must equal the input slice length
/// - The kernel should be deterministic (same inputs → same outputs)
///
/// # Example
///
/// ```ignore
/// use renoir::operator::gpu::{GpuKernel, GpuContext};
/// use bytemuck::{Pod, Zeroable};
///
/// // Input type: must be repr(C) for consistent memory layout
/// #[derive(Clone, Copy, Pod, Zeroable)]
/// #[repr(C)]
/// struct MyInput {
///     value: f32,
/// }
///
/// // Output type: same requirements as input
/// #[derive(Clone, Copy, Pod, Zeroable)]
/// #[repr(C)]
/// struct MyOutput {
///     result: f32,
/// }
///
/// #[derive(Clone)]
/// struct SquareKernel;
///
/// impl GpuKernel for SquareKernel {
///     type Input = MyInput;
///     type Output = MyOutput;
///
///     fn execute(&self, ctx: &GpuContext, inputs: &[MyInput]) -> Vec<MyOutput> {
///         // Implementation: see examples/black_scholes_gpu.rs
///         // Basic steps:
///         // 1. ctx.client().create(data) - allocate and copy to GPU
///         // 2. launch kernel using CubeCL
///         // 3. ctx.sync() - wait for completion
///         // 4. ctx.client().read_one(handle) - read results back
///         todo!()
///     }
/// }
/// ```
///
/// # Performance Tips
///
/// 1. **Batch Processing**: Process data in large batches (100K+ elements)
///    to amortize GPU overhead.
///
/// 2. **Vectorization**: Use `Line<F>` types in CubeCL kernels to process
///    multiple values per thread (typically 4 f32s).
///
/// 3. **Memory Coalescing**: Ensure threads access consecutive memory
///    addresses for optimal bandwidth.
///
/// 4. **Minimize Transfers**: The `execute()` method is called once per batch,
///    so larger batches mean fewer CPU↔GPU transfers.
pub trait GpuKernel: Clone + Send + 'static {
    /// The input element type that this kernel processes.
    ///
    /// Must be a simple, flat struct that can be safely copied to GPU memory.
    /// Use `#[repr(C)]` to ensure a consistent memory layout across platforms.
    ///
    /// # Requirements
    /// - `Data`: Clone + Send + static (for Renoir stream processing)
    /// - `bytemuck::Pod`: Safe byte-level access (for GPU memory transfer)
    type Input: Data + bytemuck::Pod;

    /// The output element type that this kernel produces.
    ///
    /// Same requirements as `Input`. The kernel must produce exactly one
    /// output element for each input element.
    type Output: Data + bytemuck::Pod;

    /// Execute the kernel on a batch of inputs, returning outputs.
    ///
    /// **Deprecated**: Prefer implementing `push()` + `flush()` for better performance.
    /// This method is kept for backward compatibility and has a default implementation
    /// that calls `push()` for each item then `flush()`.
    ///
    /// # Arguments
    ///
    /// * `ctx` - GPU context providing access to compute resources.
    /// * `inputs` - Slice of input items to process.
    ///
    /// # Returns
    ///
    /// A vector of output items, with exactly one output per input.
    fn execute(&mut self, ctx: &GpuContext, inputs: &[Self::Input]) -> Vec<Self::Output> {
        for input in inputs {
            self.push(input.clone());
        }
        self.flush(ctx)
    }

    /// Push a single item directly to the kernel's internal buffer.
    ///
    /// This method enables streaming accumulation without intermediate copies.
    /// The kernel should accumulate items in an efficient format (e.g., SoA).
    ///
    /// # Arguments
    ///
    /// * `item` - Single input item to buffer
    fn push(&mut self, item: Self::Input);

    /// Get the current number of items in the kernel's buffer.
    ///
    /// Used by the `MapGpu` operator to determine when to flush based on
    /// the batching strategy.
    fn buffer_len(&self) -> usize;

    /// Flush the internal buffer to GPU, execute, and return results.
    ///
    /// This method:
    /// 1. Pads buffer for vectorization alignment if needed
    /// 2. Uploads buffer to GPU
    /// 3. Executes the GPU kernel
    /// 4. Reads results back
    /// 5. Clears the internal buffer
    ///
    /// # Arguments
    ///
    /// * `ctx` - GPU context for execution
    ///
    /// # Returns
    ///
    /// Vector of output items corresponding to buffered inputs
    fn flush(&mut self, ctx: &GpuContext) -> Vec<Self::Output>;

    /// Optional hint for the preferred batch size.
    ///
    /// If provided, the `MapGpu` operator may use this hint to optimize
    /// batching behavior. This is advisory only - the actual batch size
    /// depends on the batching strategy and data availability.
    ///
    /// # Returns
    ///
    /// - `Some(size)`: Suggested batch size for optimal GPU utilization
    /// - `None`: No preference, use strategy defaults
    ///
    /// # Guidelines
    ///
    /// - For memory-intensive kernels: smaller batches to avoid OOM
    /// - For compute-intensive kernels: larger batches for throughput
    /// - Typical values: 100K to 10M elements
    fn preferred_batch_size(&self) -> Option<usize> {
        None
    }

    /// Optional one-time initialization before processing begins.
    ///
    /// This method is called once when the `MapGpu` operator is set up,
    /// before any calls to `execute()`. Use it for:
    ///
    /// - Pre-compiling shaders (warm-up run)
    /// - Allocating reusable buffers
    /// - Querying GPU capabilities
    ///
    /// # Arguments
    ///
    /// * `ctx` - GPU context for initialization
    ///
    /// # Default Implementation
    ///
    /// Does nothing. Override only if you have set up work to do.
    fn setup(&mut self, _ctx: &GpuContext) {}
    
    /// Drain any pending results from async pipelining.
    ///
    /// For kernels that implement async pipelining (where flush() returns
    /// results from the previous batch), this method collects the final
    /// pending results when the stream ends.
    ///
    /// # Arguments
    ///
    /// * `ctx` - GPU context for reading pending results
    ///
    /// # Returns
    ///
    /// Vector of any pending output items from pipelined execution.
    /// Returns empty vector for non-pipelined kernels.
    fn drain(&mut self, _ctx: &GpuContext) -> Vec<Self::Output> {
        Vec::new()
    }
}

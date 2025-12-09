//! # GPU-Accelerated Map Operator
//!
//! This module implements the `MapGpu` operator, which applies a GPU kernel to
//! batches of stream elements for high-throughput parallel processing.
//!
//! ## Internal Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                           MapGpu Operator                               │
//! │                                                                         │
//! │  ┌─────────────┐      ┌───────────────────┐      ┌──────────────────┐   │
//! │  │  Upstream   │      │   Input Buffer    │      │   Output Queue   │   │
//! │  │  Operator   │─────▶│   (accumulates    │─────▶│   (results to    │   │
//! │  │  .next()    │      │    until batch    │      │    emit one by   │   │
//! │  │             │      │    is ready)      │      │    one)          │   │
//! │  └─────────────┘      └─────────┬─────────┘      └────────┬─────────┘   │
//! │                                 │                         │             │
//! │                                 │  flush_to_gpu()         │             │
//! │                                 ▼                         │             │
//! │                       ┌───────────────────┐               │             │
//! │                       │   GPU Kernel      │               │             │
//! │                       │   Execution       │───────────────┘             │
//! │                       │   (via GpuKernel  │                             │
//! │                       │    trait)         │                             │
//! │                       └───────────────────┘                             │
//! │                                                                         │
//! │  Batching Strategy:                                                     │
//! │  - Fixed: flush every N items                                           │
//! │  - Timed: flush on timeout OR max size                                  │
//! │  - Adaptive: adapt batch size based on throughput                       │
//! │                                                                         │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Data Flow
//!
//! 1. **Input**: Items arrive via `prev.next()` from the upstream operator
//! 2. **Buffering**: Items accumulate in `buffer` with associated timestamps
//! 3. **Flush Trigger**: When batch is ready (size/time threshold), flush to GPU
//! 4. **GPU Execution**: `kernel.execute()` processes the entire batch
//! 5. **Output Queue**: Results stored in `output_queue`
//! 6. **Emission**: Items emitted one at a time to downstream operators
//!
//! ## Stream Element Handling
//!
//! - `Item(x)` / `Timestamped(x, ts)`: Buffer the item
//! - `Watermark(ts)`: Track max watermark, don't trigger flush
//! - `FlushBatch`: Force immediate GPU flush
//! - `FlushAndRestart`: Flush, then signal iteration boundary
//! - `Terminate`: Flush remaining items, then terminate

use std::collections::VecDeque;
use std::fmt::Display;
use std::time::Instant;

use crate::block::{BlockStructure, OperatorStructure};
use crate::operator::{Operator, StreamElement, Timestamp};
use crate::scheduler::ExecutionMetadata;

use super::batch_strategy::{AdaptiveSizer, BatchTimer, GpuBatchStrategy};
use super::context::GpuContext;
use super::kernel::GpuKernel;

/// GPU-accelerated map operator that processes stream elements in batches on the GPU.
///
/// This operator provides GPU acceleration for compute-intensive transformations
/// by batching input elements and processing them together on the GPU.
///
/// ## Type Parameters
///
/// - `K`: The GPU kernel type implementing [`GpuKernel`]
/// - `Op`: The upstream operator providing input elements
///
/// ## Batching
///
/// Items are accumulated until a flush condition is met:
/// - **Fixed strategy**: Buffer reaches target size
/// - **Timed strategy**: Timeout expires OR buffer reaches max size
/// - **Adaptive strategy**: Batch size adjusts based on throughput
///
/// After GPU processing, results are queued and emitted one at a time
/// to maintain compatibility with Renoir's pull-based streaming model.
///
/// ## Timestamp Preservation
///
/// When processing `Timestamped` items, the operator preserves the association
/// between each input's timestamp and its corresponding output. This ensures
/// event-time semantics are maintained through GPU processing.
///
/// ## Example
///
/// ```ignore
/// let result = env
///     .stream_iter(data.into_iter())
///     .map_gpu_with_strategy(MyKernel, GpuBatchStrategy::fixed(100_000))
///     .collect_vec();
/// ```
#[derive(Derivative)]
#[derivative(Debug, Clone)]
pub struct MapGpu<K, Op>
where
    K: GpuKernel,
    Op: Operator<Out = K::Input>,
{
    /// The upstream operator that provides input elements.
    prev: Op,

    /// The GPU kernel to apply to each batch of inputs.
    #[derivative(Debug = "ignore")]
    kernel: K,

    /// Strategy controlling when to flush the buffer to GPU.
    strategy: GpuBatchStrategy,

    // ════════════════════════════════════════════════════════════════════
    // Buffering State
    // ════════════════════════════════════════════════════════════════════

    /// Buffer accumulating input items until a batch is ready.
    ///
    /// Items are added one at a time from upstream and removed all at once
    /// when flushed to the GPU.
    #[derivative(Debug = "ignore")]
    buffer: Vec<K::Input>,

    /// Timestamps associated with buffered items (parallel to `buffer`).
    ///
    /// `buffer_timestamps[i]` is `Some(ts)` if `buffer[i]` came from a
    /// `Timestamped` element, `None` if it came from a plain `Item`.
    buffer_timestamps: Vec<Option<Timestamp>>,

    /// Queue of processed outputs waiting to be emitted.
    ///
    /// After GPU processing, results are stored here and emitted one at a
    /// time via `next()`. This converts batch output to streaming output.
    #[derivative(Debug = "ignore")]
    output_queue: VecDeque<(K::Output, Option<Timestamp>)>,

    // ════════════════════════════════════════════════════════════════════
    // GPU State
    // ════════════════════════════════════════════════════════════════════

    /// GPU context for kernel execution.
    ///
    /// Initialized in `setup()`, contains the GPU device and compute client.
    /// `None` before setup is called.
    #[derivative(Debug = "ignore")]
    gpu_context: Option<GpuContext>,

    // ════════════════════════════════════════════════════════════════════
    // Strategy State
    // ════════════════════════════════════════════════════════════════════

    /// Timer for timed batching strategy.
    ///
    /// Tracks time since the last flush to trigger timeout-based flushes.
    batch_timer: Option<BatchTimer>,

    /// Adaptive sizer for adaptive batching.
    ///
    /// Adjusts batch size based on observed GPU throughput.
    adaptive_sizer: Option<AdaptiveSizer>,

    // ════════════════════════════════════════════════════════════════════
    // Stream State
    // ════════════════════════════════════════════════════════════════════

    /// Maximum watermark timestamp seen so far.
    ///
    /// Used to emit a watermark after processing timestamped data.
    max_watermark: Option<Timestamp>,

    /// Watermark to emit after draining output queue.
    pending_watermark: Option<Timestamp>,

    /// Flag indicating upstream sent `Terminate`.
    received_end: bool,

    /// Flag indicating upstream sent `FlushAndRestart`.
    received_flush_restart: bool,
}

impl<K, Op> MapGpu<K, Op>
where
    K: GpuKernel,
    Op: Operator<Out = K::Input>,
{
    /// Create a new GPU map operator with the default batching strategy.
    ///
    /// The default strategy is `Fixed(10_000_000)` - flush every 10 million items.
    ///
    /// # Arguments
    ///
    /// * `prev` - Upstream operator providing input elements
    /// * `kernel` - GPU kernel to apply to batches
    pub fn new(prev: Op, kernel: K) -> Self {
        Self::with_strategy(prev, kernel, GpuBatchStrategy::default())
    }

    /// Create a new GPU map operator with a custom batching strategy.
    ///
    /// # Arguments
    ///
    /// * `prev` - Upstream operator providing input elements
    /// * `kernel` - GPU kernel to apply to batches
    /// * `strategy` - Batching strategy controlling flush behavior
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Flush every 100,000 items
    /// MapGpu::with_strategy(prev, kernel, GpuBatchStrategy::fixed(100_000))
    ///
    /// // Flush on timeout or max size
    /// MapGpu::with_strategy(prev, kernel, GpuBatchStrategy::timed(50_000, Duration::from_millis(100)))
    /// ```
    pub fn with_strategy(prev: Op, kernel: K, strategy: GpuBatchStrategy) -> Self {
        // Initialize strategy-specific state
        let batch_timer = match &strategy {
            GpuBatchStrategy::Timed { .. } => Some(BatchTimer::new(&strategy)),
            _ => None,
        };
        let adaptive_sizer = match &strategy {
            GpuBatchStrategy::Adaptive { .. } => Some(AdaptiveSizer::new(&strategy)),
            _ => None,
        };

        Self {
            prev,
            kernel,
            strategy,
            buffer: Vec::with_capacity(1024),
            buffer_timestamps: Vec::with_capacity(1024),
            output_queue: VecDeque::new(),
            gpu_context: None,
            batch_timer,
            adaptive_sizer,
            max_watermark: None,
            pending_watermark: None,
            received_end: false,
            received_flush_restart: false,
        }
    }

    /// Determine if the buffer should be flushed to GPU.
    ///
    /// Returns `true` if:
    /// - Buffer size >= target size (from strategy or adaptive sizer)
    /// - Timed strategy timeout has expired
    ///
    /// Returns `false` if the buffer is empty.
    fn should_flush(&self) -> bool {
        if self.buffer.is_empty() {
            return false;
        }

        // Get target size from adaptive sizer or strategy
        let target_size = if let Some(ref sizer) = self.adaptive_sizer {
            sizer.current_size()
        } else {
            self.strategy.target_size()
        };

        // Check the size threshold
        if self.buffer.len() >= target_size {
            return true;
        }

        // Check time threshold (for timed strategy)
        if let Some(ref timer) = self.batch_timer {
            if timer.should_flush() {
                return true;
            }
        }

        false
    }

    /// Flush the input buffer to GPU and populate the output queue.
    ///
    /// This method:
    /// 1. Drains buffered inputs and timestamps (preserving capacity)
    /// 2. Calls `kernel.execute()` to be processed on GPU
    /// 3. Updates adaptive sizer with throughput data (if applicable)
    /// 4. Resets batch timer (if applicable)
    /// 5. Pairs outputs with timestamps and adds to the output queue
    ///
    /// # Panics
    ///
    /// - If GPU context is not initialized (setup not called)
    /// - If kernel returns a different number of outputs than inputs
    fn flush_to_gpu(&mut self) {
        if self.buffer.is_empty() {
            return;
        }

        let ctx = self
            .gpu_context
            .as_ref()
            .expect("GPU context not initialized - was setup() called?");

        // Time the execution for adaptive sizing feedback
        let start = Instant::now();

        // Drain buffer contents (preserves allocated capacity for reuse)
        let inputs: Vec<K::Input> = self.buffer.drain(..).collect();
        let timestamps: Vec<Option<Timestamp>> = self.buffer_timestamps.drain(..).collect();
        let num_items = inputs.len();

        // Execute GPU kernel on the batch
        let outputs = self.kernel.execute(ctx, &inputs);

        // Update adaptive sizer with throughput measurement
        if let Some(ref mut sizer) = self.adaptive_sizer {
            sizer.update(num_items, start.elapsed());
        }

        // Reset timer for timed strategy
        if let Some(ref mut timer) = self.batch_timer {
            timer.record_flush();
        }

        // Validate output count matches input count
        assert_eq!(
            outputs.len(),
            timestamps.len(),
            "Kernel must produce exactly one output per input. Got {} outputs for {} inputs.",
            outputs.len(),
            timestamps.len()
        );

        // Pair each output with its corresponding timestamp and enqueue
        for (output, ts) in outputs.into_iter().zip(timestamps.into_iter()) {
            self.output_queue.push_back((output, ts));
        }
    }

    /// Buffer an item for later GPU processing.
    ///
    /// Adds the item to the buffer and triggers a flush if the batching
    /// strategy indicates the buffer is ready.
    ///
    /// # Arguments
    ///
    /// * `item` - Input item to buffer
    /// * `timestamp` - Optional timestamp if the item came from `Timestamped`
    fn buffer_item(&mut self, item: K::Input, timestamp: Option<Timestamp>) {
        self.buffer.push(item);
        self.buffer_timestamps.push(timestamp);

        // Check if we should flush after adding this item
        if self.should_flush() {
            self.flush_to_gpu();
        }
    }
}

impl<K, Op> Display for MapGpu<K, Op>
where
    K: GpuKernel,
    Op: Operator<Out = K::Input>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} -> MapGpu<{} -> {}>",
            self.prev,
            std::any::type_name::<K::Input>(),
            std::any::type_name::<K::Output>()
        )
    }
}

impl<K, Op> Operator for MapGpu<K, Op>
where
    K: GpuKernel,
    Op: Operator<Out = K::Input>,
{
    type Out = K::Output;

    /// Initialize the operator before processing begins.
    ///
    /// This method:
    /// 1. Recursively calls setup on upstream operator
    /// 2. Creates the GPU context (initializes a GPU device)
    /// 3. Calls `kernel.setup()` for kernel-specific initialization
    /// 4. Pre-allocates buffer capacity based on strategy
    fn setup(&mut self, metadata: &mut ExecutionMetadata) {
        // Set up the upstream operator first
        self.prev.setup(metadata);

        // Initialize GPU context (may take ~100ms on the first call)
        let ctx = GpuContext::new();

        // Allow kernel to perform any setup (e.g., shader compilation)
        self.kernel.setup(&ctx);

        self.gpu_context = Some(ctx);

        // Pre-allocate buffers based on strategy
        // Use a reasonable cap of 50M to balance memory usage and performance
        let initial_capacity = self.strategy.target_size().min(50_000_000);
        self.buffer = Vec::with_capacity(initial_capacity);
        self.buffer_timestamps = Vec::with_capacity(initial_capacity);
        
        // Pre-allocate output queue as well for better performance
        self.output_queue = VecDeque::with_capacity(initial_capacity);
    }

    /// Get the next output element.
    ///
    /// This is the core of the operator's pull-based processing:
    ///
    /// 1. **Emit buffered outputs first**: If output_queue has items, return one
    /// 2. **Emit pending watermark**: If we have a watermark to send, send it
    /// 3. **Handle termination**: Return FlushAndRestart or Terminate if flagged
    /// 4. **Pull from upstream**: Get the next element and handle appropriately
    ///
    /// The loop continues until we have something to emit or reach termination.
    fn next(&mut self) -> StreamElement<K::Output> {
        loop {
            // ═══════════════════════════════════════════════════════════════
            // Priority 1: Emit buffered outputs from GPU processing
            // ═══════════════════════════════════════════════════════════════
            if let Some((output, ts)) = self.output_queue.pop_front() {
                return match ts {
                    Some(timestamp) => StreamElement::Timestamped(output, timestamp),
                    None => StreamElement::Item(output),
                };
            }

            // ═══════════════════════════════════════════════════════════════
            // Priority 2: Emit pending watermark (after flushing outputs)
            // ═══════════════════════════════════════════════════════════════
            if let Some(wm) = self.pending_watermark.take() {
                return StreamElement::Watermark(wm);
            }

            // ═══════════════════════════════════════════════════════════════
            // Priority 3: Handle end-of-stream conditions
            // ═══════════════════════════════════════════════════════════════
            if self.received_flush_restart {
                self.received_flush_restart = false;
                return StreamElement::FlushAndRestart;
            }
            if self.received_end {
                return StreamElement::Terminate;
            }

            // ═══════════════════════════════════════════════════════════════
            // Pull from upstream and handle each element type
            // ═══════════════════════════════════════════════════════════════
            match self.prev.next() {
                // Regular data items: buffer for batch processing
                StreamElement::Item(item) => {
                    self.buffer_item(item, None);
                    // Continue loop - may have triggered a flush
                }

                // Timestamped items: buffer with timestamp preserved
                StreamElement::Timestamped(item, ts) => {
                    self.buffer_item(item, Some(ts));
                    // Continue loop - may have triggered a flush
                }

                // Watermarks: track max, don't trigger flush
                StreamElement::Watermark(ts) => {
                    self.max_watermark = Some(self.max_watermark.map_or(ts, |w| w.max(ts)));
                    // Continue pulling - watermarks pass through after data
                }

                // FlushBatch: force immediate GPU processing
                StreamElement::FlushBatch => {
                    self.flush_to_gpu();
                    // Continue the loop to emit any buffered outputs
                }

                // FlushAndRestart: end of iteration, flush and signal
                StreamElement::FlushAndRestart => {
                    // Flush any remaining buffered items
                    self.flush_to_gpu();

                    // Schedule watermark to be emitted after outputs
                    if let Some(wm) = self.max_watermark.take() {
                        self.pending_watermark = Some(wm);
                    }

                    self.received_flush_restart = true;
                    // Continue the loop to drain outputs before signaling
                }

                // Terminate: end of stream, flush and terminate
                StreamElement::Terminate => {
                    // Flush any remaining buffered items
                    self.flush_to_gpu();

                    // Schedule watermark to be emitted after outputs
                    if let Some(wm) = self.max_watermark.take() {
                        self.pending_watermark = Some(wm);
                    }

                    self.received_end = true;
                    // Continue loop to drain outputs before terminating
                }
            }
        }
    }

    /// Return the operator structure for debugging and visualization.
    fn structure(&self) -> BlockStructure {
        self.prev
            .structure()
            .add_operator(OperatorStructure::new::<K::Output, _>("MapGpu"))
    }
}

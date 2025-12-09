//! GPU batching strategies for controlling when data is flushed to the GPU.
//!
//! Batching is crucial for GPU performance - processing item one at a time
//! would be extremely inefficient due to kernel launch and data transfer overhead.

use std::time::{Duration, Instant};

/// Strategy for batching items before sending to GPU for processing.
///
/// GPU processing is most efficient when operating on large batches of data.
/// This enum defines different strategies for deciding when to flush the
/// accumulated buffer to the GPU.
#[derive(Debug, Clone)]
pub enum GpuBatchStrategy {
    /// Flush every N items.
    ///
    /// This is the simplest strategy - accumulate exactly N items before
    /// processing. Good for predictable workloads where you know the
    /// optimal batch size.
    ///
    /// Default: 10,000,000 items (10M)
    Fixed(usize),

    /// Flush after a time interval OR when the buffer reaches max_size.
    ///
    /// This strategy balances latency and throughput:
    /// - Ensures data doesn't wait too long (bounded latency)
    /// - Still batches efficiently when data arrives quickly
    ///
    /// Useful for streaming scenarios where the data arrival rate varies.
    Timed {
        /// Maximum number of items before forcing a flush
        max_size: usize,
        /// Maximum time to wait before flushing
        interval: Duration,
    },

    /// Adaptively adjust batch size based on GPU throughput.
    ///
    /// Starts with `min_size` and increases towards `max_size` as long as
    /// throughput improves. Can adapt to different GPU capabilities and
    /// workload characteristics.
    Adaptive {
        /// Minimum batch size
        min_size: usize,
        /// Maximum batch size
        max_size: usize,
    },
}

impl Default for GpuBatchStrategy {
    fn default() -> Self {
        // 10 million items is a good default for most GPU workloads
        // This balances kernel launch overhead with memory efficiency
        GpuBatchStrategy::Fixed(10_000_000)
    }
}

impl GpuBatchStrategy {
    /// Create a fixed-size batching strategy.
    pub fn fixed(size: usize) -> Self {
        assert!(size > 0, "Batch size must be positive");
        GpuBatchStrategy::Fixed(size)
    }

    /// Create a time-based batching strategy.
    pub fn timed(max_size: usize, interval: Duration) -> Self {
        assert!(max_size > 0, "Max batch size must be positive");
        GpuBatchStrategy::Timed { max_size, interval }
    }

    /// Create an adaptive batching strategy.
    ///
    /// This strategy automatically adjusts batch size based on observed
    /// GPU throughput, starting at `min_size` and growing towards `max_size`.
    pub fn adaptive(min_size: usize, max_size: usize) -> Self {
        assert!(min_size > 0, "Min batch size must be positive");
        assert!(max_size >= min_size, "Max size must be >= min size");
        GpuBatchStrategy::Adaptive { min_size, max_size }
    }

    /// Get the maximum batch size for this strategy.
    pub fn max_size(&self) -> usize {
        match self {
            GpuBatchStrategy::Fixed(size) => *size,
            GpuBatchStrategy::Timed { max_size, .. } => *max_size,
            GpuBatchStrategy::Adaptive { max_size, .. } => *max_size,
        }
    }

    /// Get the initial/target batch size for this strategy.
    pub fn target_size(&self) -> usize {
        match self {
            GpuBatchStrategy::Fixed(size) => *size,
            GpuBatchStrategy::Timed { max_size, .. } => *max_size,
            GpuBatchStrategy::Adaptive { min_size, .. } => *min_size,
        }
    }
}

/// Internal state tracker for batch timing.
#[derive(Debug, Clone)]
pub(crate) struct BatchTimer {
    last_flush: Instant,
    interval: Option<Duration>,
}

impl BatchTimer {
    pub fn new(strategy: &GpuBatchStrategy) -> Self {
        let interval = match strategy {
            GpuBatchStrategy::Timed { interval, .. } => Some(*interval),
            _ => None,
        };
        Self {
            last_flush: Instant::now(),
            interval,
        }
    }

    /// Check if a timed flush should occur.
    pub fn should_flush(&self) -> bool {
        if let Some(interval) = self.interval {
            self.last_flush.elapsed() >= interval
        } else {
            false
        }
    }

    /// Record that a flush occurred.
    pub fn record_flush(&mut self) {
        self.last_flush = Instant::now();
    }
}

/// Internal state tracker for adaptive batch sizing.
///
/// Monitors GPU throughput and adjusts batch size to optimize performance.
#[derive(Debug, Clone)]
pub(crate) struct AdaptiveSizer {
    min_size: usize,
    max_size: usize,
    current_size: usize,
    last_throughput: Option<f64>,
}

impl AdaptiveSizer {
    pub fn new(strategy: &GpuBatchStrategy) -> Self {
        match strategy {
            GpuBatchStrategy::Adaptive { min_size, max_size } => Self {
                min_size: *min_size,
                max_size: *max_size,
                current_size: *min_size,
                last_throughput: None,
            },
            GpuBatchStrategy::Fixed(size) => Self {
                min_size: *size,
                max_size: *size,
                current_size: *size,
                last_throughput: None,
            },
            GpuBatchStrategy::Timed { max_size, .. } => Self {
                min_size: *max_size,
                max_size: *max_size,
                current_size: *max_size,
                last_throughput: None,
            },
        }
    }

    /// Get the current target batch size.
    pub fn current_size(&self) -> usize {
        self.current_size
    }

    /// Update sizing based on observed throughput (items/second).
    pub fn update(&mut self, items_processed: usize, duration: Duration) {
        let throughput = items_processed as f64 / duration.as_secs_f64();

        if let Some(last) = self.last_throughput {
            // If throughput improved with a larger batch, try increasing
            if throughput > last * 1.05 && self.current_size < self.max_size {
                self.current_size = (self.current_size * 2).min(self.max_size);
            }
            // If throughput degraded significantly, try decreasing
            else if throughput < last * 0.8 && self.current_size > self.min_size {
                self.current_size = (self.current_size / 2).max(self.min_size);
            }
        }

        self.last_throughput = Some(throughput);
    }
}


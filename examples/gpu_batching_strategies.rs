//! # GPU Batching Strategies Example
//!
//! This example demonstrates the three batching strategies available for the `map_gpu` operator:
//!
//! 1. **Fixed**: Flush every N items (best for predictable workloads)
//! 2. **Timed**: Flush on timeout or max size (best for variable data rates)
//! 3. **Adaptive**: Auto-tune batch size based on throughput (best for unknown hardware)
//!
//! ## Running
//!
//! ```bash
//! cargo run --example gpu_batching_strategies --features gpu-wgpu --release
//! ```
//!
//! ## Key Concepts
//!
//! - **Batch size** affects GPU utilization: larger batches = better throughput
//! - **Timed strategy** bounds latency while maintaining throughput
//! - **Adaptive strategy** automatically finds optimal batch size

#[path = "kernels/black_scholes.rs"]
mod black_scholes_kernel;

use std::time::{Duration, Instant};

use renoir::operator::gpu::GpuBatchStrategy;
use renoir::prelude::*;
use renoir::utils::{create_banner, TableBuilder};

use black_scholes_kernel::*;

// ============================================================================
// Strategy Result
// ============================================================================

struct StrategyResult {
    name: String,
    duration: Duration,
    throughput: f64,
}

impl StrategyResult {
    fn new(name: &str, duration: Duration, items: usize) -> Self {
        let throughput = items as f64 / duration.as_secs_f64();
        Self {
            name: name.to_string(),
            duration,
            throughput,
        }
    }
}

// ============================================================================
// Benchmark Helper
// ============================================================================

fn run_with_strategy(
    options: &[BlackScholesInput],
    strategy: GpuBatchStrategy,
    strategy_name: &str,
) -> StrategyResult {
    let env = StreamContext::new_local();
    let start = Instant::now();

    let result = env
        .stream_iter(options.to_vec().into_iter())
        .map_gpu_with_strategy(BlackScholesKernel::default(), strategy)
        .collect_vec();

    env.execute_blocking();

    let duration = start.elapsed();
    let results = result.get().unwrap();

    StrategyResult::new(strategy_name, duration, results.len())
}

// ============================================================================
// Main Example
// ============================================================================

fn main() {
    print!("{}", create_banner("GPU Batching Strategies Comparison", &[]));

    // Test with different input sizes
    let test_sizes = vec![100_000, 500_000, 1_000_000];

    for num_options in test_sizes {
        println!("═══════════════════════════════════════════════════════════════");
        println!("Testing with {} options", num_options);
        println!("═══════════════════════════════════════════════════════════════\n");

        // Generate test data
        let options = generate_options(num_options);

        // ═══════════════════════════════════════════════════════════════════
        // Strategy 1: Fixed (Small Batches - 10K items)
        // ═══════════════════════════════════════════════════════════════════
        println!("1. Fixed Strategy (10K batch size)");
        println!("   - Flushes every 10,000 items");
        println!("   - Good for: Low memory usage, frequent intermediate results");

        let fixed_small = run_with_strategy(&options, GpuBatchStrategy::fixed(10_000), "Fixed (10K)");

        println!(
            "   Result: {:.3}s, {:.2}M items/sec\n",
            fixed_small.duration.as_secs_f64(),
            fixed_small.throughput / 1_000_000.0
        );

        // ═══════════════════════════════════════════════════════════════════
        // Strategy 2: Fixed (Large Batches - 100K items)
        // ═══════════════════════════════════════════════════════════════════
        println!("2. Fixed Strategy (100K batch size)");
        println!("   - Flushes every 100,000 items");
        println!("   - Good for: Better GPU utilization, higher throughput");

        let fixed_large =
            run_with_strategy(&options, GpuBatchStrategy::fixed(100_000), "Fixed (100K)");

        println!(
            "   Result: {:.3}s, {:.2}M items/sec\n",
            fixed_large.duration.as_secs_f64(),
            fixed_large.throughput / 1_000_000.0
        );

        // ═══════════════════════════════════════════════════════════════════
        // Strategy 3: Fixed (Full Input as Single Batch)
        // ═══════════════════════════════════════════════════════════════════
        println!("3. Fixed Strategy (Full input)");
        println!("   - Single batch = single GPU kernel launch");
        println!("   - Good for: Maximum throughput when input size is known");

        let fixed_full =
            run_with_strategy(&options, GpuBatchStrategy::fixed(num_options), "Fixed (Full)");

        println!(
            "   Result: {:.3}s, {:.2}M items/sec\n",
            fixed_full.duration.as_secs_f64(),
            fixed_full.throughput / 1_000_000.0
        );

        // ═══════════════════════════════════════════════════════════════════
        // Strategy 4: Timed (Max 100K items OR 100ms timeout)
        // ═══════════════════════════════════════════════════════════════════
        println!("4. Timed Strategy (100K max, 100ms timeout)");
        println!("   - Flushes when batch reaches 100K OR 100ms passes");
        println!("   - Good for: Streaming with latency requirements");

        let timed = run_with_strategy(
            &options,
            GpuBatchStrategy::timed(100_000, Duration::from_millis(100)),
            "Timed (100K,100ms)",
        );

        println!(
            "   Result: {:.3}s, {:.2}M items/sec\n",
            timed.duration.as_secs_f64(),
            timed.throughput / 1_000_000.0
        );

        // ═══════════════════════════════════════════════════════════════════
        // Strategy 5: Adaptive (10K min, 10M max)
        // ═══════════════════════════════════════════════════════════════════
        println!("5. Adaptive Strategy (10K-10M range)");
        println!("   - Auto-tunes batch size based on throughput");
        println!("   - Good for: Unknown hardware, variable workloads");

        let adaptive = run_with_strategy(
            &options,
            GpuBatchStrategy::adaptive(10_000, 10_000_000),
            "Adaptive (10K-10M)",
        );

        println!(
            "   Result: {:.3}s, {:.2}M items/sec\n",
            adaptive.duration.as_secs_f64(),
            adaptive.throughput / 1_000_000.0
        );

        // ═══════════════════════════════════════════════════════════════════
        // Summary Table
        // ═══════════════════════════════════════════════════════════════════
        println!("Summary for {} options:", num_options);
        let table = TableBuilder::new(&[
            (" Strategy              ", 24),
            (" Time (s)      ", 16),
            (" Throughput (M/s)  ", 20),
        ]);
        print!("{}", table.header());

        let results = vec![fixed_small, fixed_large, fixed_full, timed, adaptive];
        for r in &results {
            print!(
                "{}",
                table.row(&[
                    &format!(" {:22}", r.name),
                    &format!(" {:>14.4}", r.duration.as_secs_f64()),
                    &format!(" {:>18.2}", r.throughput / 1_000_000.0),
                ])
            );
        }

        print!("{}", table.footer());

        // Find best strategy
        let best = results
            .iter()
            .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap())
            .unwrap();
        println!(
            "Best strategy: {} ({:.2}M items/sec)\n",
            best.name,
            best.throughput / 1_000_000.0
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Recommendations
    // ═══════════════════════════════════════════════════════════════════════
    println!("═══════════════════════════════════════════════════════════════");
    println!("RECOMMENDATIONS:");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("1. For batch processing (known input size):");
    println!("   → Use Fixed strategy with batch_size = input_size");
    println!("   → Minimizes kernel launch overhead\n");

    println!("2. For streaming with latency requirements:");
    println!("   → Use Timed strategy with appropriate interval");
    println!("   → Example: GpuBatchStrategy::timed(100_000, Duration::from_millis(100))\n");

    println!("3. For unknown hardware or variable workloads:");
    println!("   → Use Adaptive strategy");
    println!("   → Example: GpuBatchStrategy::adaptive(10_000, 10_000_000)\n");

    println!("4. Default (good general-purpose choice):");
    println!("   → GpuBatchStrategy::default() uses Fixed(10_000_000)");
    println!("   → Good balance of throughput and memory usage");
}

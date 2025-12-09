//! # GPU Batching Strategy Comparison Benchmark
//!
//! This benchmark compares the performance of different batching strategies
//! for the `map_gpu` operator across various input sizes.
//!
//! ## Running the Benchmark
//!
//! ```bash
//! # Run with default settings
//! cargo bench --bench gpu_black_scholes_batching_comparison --features gpu-wgpu
//!
//! # Run with custom maximum input size
//! MAX_OPTIONS=10000000 cargo bench --bench gpu_black_scholes_batching_comparison --features gpu-wgpu
//! ```
//!
//! ## Output
//!
//! Results are saved to `benches/results/comparison/{date}/comparison_benchmark_{timestamp}.json`

#[path = "common.rs"]
mod common;

#[path = "../../examples/kernels/black_scholes.rs"]
mod black_scholes_kernel;

use std::env;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use serde::{Deserialize, Serialize};

use black_scholes_kernel::*;
use common::*;
use renoir::operator::gpu::GpuBatchStrategy;
use renoir::prelude::*;
use renoir::utils::{create_banner, TableBuilder};

use renoir::operator::gpu::GpuContext;

// ============================================================================
// Benchmark-specific Data Structures
// ============================================================================

/// Single benchmark result for a strategy/size combination.
#[derive(Serialize, Deserialize, Clone)]
struct BatchingResult {
    test_id: usize,
    timestamp: DateTime<Utc>,
    strategy_name: String,
    strategy_type: String,
    batch_config: String,
    items_count: usize,
    total_time_s: f64,
    throughput: f64,
    time_per_item_us: f64,
}

/// Container for all benchmark results.
#[derive(Serialize, Deserialize, Clone)]
struct BatchingReport {
    benchmark_type: String,
    start_time: DateTime<Utc>,
    platform: String,
    total_tests: usize,
    results: Vec<BatchingResult>,
}

// ============================================================================
// ============================================================================
// Strategy Definitions
// ============================================================================

/// Strategy configuration for benchmarking.
struct StrategyConfig {
    name: String,
    strategy_type: String,
    batch_config: String,
    strategy: GpuBatchStrategy,
}

impl StrategyConfig {
    fn fixed(batch_size: usize) -> Self {
        let size_str = if batch_size >= 1_000_000 {
            format!("{}M", batch_size / 1_000_000)
        } else {
            format!("{}K", batch_size / 1_000)
        };
        Self {
            name: format!("Fixed({})", size_str),
            strategy_type: "fixed".to_string(),
            batch_config: format!("{}", batch_size),
            strategy: GpuBatchStrategy::fixed(batch_size),
        }
    }

    fn timed(max_size: usize, interval_ms: u64) -> Self {
        let size_str = if max_size >= 1_000_000 {
            format!("{}M", max_size / 1_000_000)
        } else {
            format!("{}K", max_size / 1_000)
        };
        Self {
            name: format!("Timed({},{}ms)", size_str, interval_ms),
            strategy_type: "timed".to_string(),
            batch_config: format!("max={},interval={}ms", max_size, interval_ms),
            strategy: GpuBatchStrategy::timed(max_size, Duration::from_millis(interval_ms)),
        }
    }

    fn adaptive(min_size: usize, max_size: usize) -> Self {
        let min_str = if min_size >= 1_000_000 {
            format!("{}M", min_size / 1_000_000)
        } else {
            format!("{}K", min_size / 1_000)
        };
        let max_str = if max_size >= 1_000_000 {
            format!("{}M", max_size / 1_000_000)
        } else {
            format!("{}K", max_size / 1_000)
        };
        Self {
            name: format!("Adaptive({}-{})", min_str, max_str),
            strategy_type: "adaptive".to_string(),
            batch_config: format!("min={},max={}", min_size, max_size),
            strategy: GpuBatchStrategy::adaptive(min_size, max_size),
        }
    }

    fn fixed_full(input_size: usize) -> Self {
        Self {
            name: "Fixed(Full)".to_string(),
            strategy_type: "fixed".to_string(),
            batch_config: format!("{} (input size)", input_size),
            strategy: GpuBatchStrategy::fixed(input_size),
        }
    }
}

// ============================================================================
// Benchmark Runner
// ============================================================================

fn run_strategy_benchmark(
    test_id: usize,
    options: &[BlackScholesInput],
    config: &StrategyConfig,
) -> BatchingResult {
    let timestamp = Utc::now();
    let num_options = options.len();

    let env = StreamContext::new_local();
    let start = Instant::now();

    let result = env
        .stream_iter(options.to_vec().into_iter())
        .map_gpu_with_strategy(BlackScholesKernel::default(), config.strategy.clone())
        .collect_vec();

    env.execute_blocking();

    let duration = start.elapsed();
    let results = result.get().unwrap();

    let total_time = duration.as_secs_f64();
    let throughput = results.len() as f64 / total_time;
    let time_per_item_us = total_time * 1_000_000.0 / results.len() as f64;

    BatchingResult {
        test_id,
        timestamp,
        strategy_name: config.name.clone(),
        strategy_type: config.strategy_type.clone(),
        batch_config: config.batch_config.clone(),
        items_count: num_options,
        total_time_s: total_time,
        throughput,
        time_per_item_us,
    }
}

// ============================================================================
// Main Benchmark
// ============================================================================

fn batching_comparison_bench(c: &mut Criterion) {
    // Configuration
    let max_options = get_max_options();
    let test_sizes = get_benchmark_test_sizes();
    let start_time = Utc::now();

    let platform = get_platform_info();
    let filename = get_benchmark_filepath(BenchmarkType::Comparison, &start_time);

    print!(
        "{}",
        create_banner(
            &format!("GPU {} Benchmark", BenchmarkType::Comparison.display_name()),
            &[
                &format!("Platform:    {}", platform),
                &format!("Max options: {} ({})", format_number(max_options), format_number_short(max_options)),
                &format!("Test sizes:  {}", test_sizes.len()),
                &format!("Output:      {}", filename.display()),
            ]
        )
    );

    // Initialize GPU with warmup
    println!("Initializing GPU and warming up...");
    let _gpu_ctx = GpuContext::new();
    let mut kernel = BlackScholesKernel::default();
    kernel.setup(&_gpu_ctx);
    println!("GPU ready.\n");


    let mut results = Vec::new();
    let mut test_id = 0;

    // ═══════════════════════════════════════════════════════════════════════
    // Run custom benchmarks with JSON output
    // ═══════════════════════════════════════════════════════════════════════

    for &num_options in &test_sizes {
        print!(
            "{}",
            create_banner(&format!("Testing {} options", format_number(num_options)), &[])
        );

        let options = generate_options(num_options);

        // Define strategies to test
        let strategies = vec![
            StrategyConfig::fixed(100_000),          // Fixed (100K)
            StrategyConfig::fixed(1_000_000),        // Fixed (1M)
            StrategyConfig::fixed(10_000_000),       // Fixed (10M)
            StrategyConfig::fixed_full(num_options), // Fixed (Full input)
            StrategyConfig::timed(1_000_000, 100),   // Timed (1M, 100ms)
            StrategyConfig::adaptive(100_000, 10_000_000), // Adaptive (100K-10M)
        ];

        let table = TableBuilder::new(&[
            (" Strategy                  ", 28),
            (" Time (s)      ", 16),
            (" Throughput (M/s)  ", 20),
        ]);
        print!("{}", table.header());

        let mut size_results = Vec::new();

        for config in &strategies {
            test_id += 1;
            let result = run_strategy_benchmark(test_id, &options, config);

            print!(
                "{}",
                table.row(&[
                    &format!(" {:26}", result.strategy_name),
                    &format!(" {:>14.4}", result.total_time_s),
                    &format!(" {:>18.2}", result.throughput / 1_000_000.0),
                ])
            );

            size_results.push(result.clone());
            results.push(result);
        }

        print!("{}", table.footer());

        // Find best for this size
        let best = size_results
            .iter()
            .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap())
            .unwrap();
        println!(
            "Best for {} options: {} ({:.2}M items/sec)\n",
            num_options,
            best.strategy_name,
            best.throughput / 1_000_000.0
        );

        // Save incrementally
        let report = BatchingReport {
            benchmark_type: BenchmarkType::Comparison.folder_name().to_string(),
            start_time,
            platform: platform.clone(),
            total_tests: test_id,
            results: results.clone(),
        };
        save_json(&filename, &report);
    }

    // Summary statistics
    let total_tests = results.len();
    let avg_throughput = results.iter().map(|r| r.throughput).sum::<f64>() / total_tests as f64;
    let max_throughput = results.iter().map(|r| r.throughput).fold(0.0f64, f64::max);

    // Find best strategy overall
    let best_overall = results
        .iter()
        .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap())
        .unwrap();

    print!(
        "{}",
        create_banner(
            "BATCHING COMPARISON SUMMARY",
            &[
                &format!("Total tests:           {:>6}", total_tests),
                &format!("Test sizes:            {:>6}", test_sizes.len()),
                &format!("Strategies tested:     {:>6}", 6),
                &format!("Avg throughput:        {:>6.2}M/s", avg_throughput / 1_000_000.0),
                &format!("Max throughput:        {:>6.2}M/s", max_throughput / 1_000_000.0),
                &format!("Best strategy:         {}", best_overall.strategy_name),
            ]
        )
    );

    println!("Results saved to: {}", filename.display());

    // Generate visualization
    run_plotter(&filename, BenchmarkType::Comparison, &start_time);

    // ═══════════════════════════════════════════════════════════════════════
    // Criterion benchmarks
    // ═══════════════════════════════════════════════════════════════════════

    let mut group = c.benchmark_group("batching-strategies");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    // Only benchmark smaller sizes with criterion
    let criterion_sizes: Vec<usize> = test_sizes
        .iter()
        .filter(|&&s| s <= 1_000_000)
        .copied()
        .collect();

    for &num_options in &criterion_sizes {
        let options = generate_options(num_options);
        group.throughput(Throughput::Elements(num_options as u64));

        // Fixed (100K)
        group.bench_with_input(
            BenchmarkId::new("fixed-100k", num_options),
            &options,
            |b, options| {
                b.iter(|| {
                    let env = StreamContext::new_local();
                    let result = env
                        .stream_iter(options.clone().into_iter())
                        .map_gpu_with_strategy(
                            BlackScholesKernel::default(),
                            GpuBatchStrategy::fixed(100_000),
                        )
                        .collect_vec();
                    env.execute_blocking();
                    std::hint::black_box(result.get())
                })
            },
        );

        // Fixed (1M)
        group.bench_with_input(
            BenchmarkId::new("fixed-1m", num_options),
            &options,
            |b, options| {
                b.iter(|| {
                    let env = StreamContext::new_local();
                    let result = env
                        .stream_iter(options.clone().into_iter())
                        .map_gpu_with_strategy(
                            BlackScholesKernel::default(),
                            GpuBatchStrategy::fixed(1_000_000),
                        )
                        .collect_vec();
                    env.execute_blocking();
                    std::hint::black_box(result.get())
                })
            },
        );

        // Fixed (Full input)
        group.bench_with_input(
            BenchmarkId::new("fixed-full", num_options),
            &options,
            |b, options| {
                b.iter(|| {
                    let env = StreamContext::new_local();
                    let result = env
                        .stream_iter(options.clone().into_iter())
                        .map_gpu_with_strategy(
                            BlackScholesKernel::default(),
                            GpuBatchStrategy::fixed(options.len()),
                        )
                        .collect_vec();
                    env.execute_blocking();
                    std::hint::black_box(result.get())
                })
            },
        );

        // Timed (1M, 100ms)
        group.bench_with_input(
            BenchmarkId::new("timed-1m-100ms", num_options),
            &options,
            |b, options| {
                b.iter(|| {
                    let env = StreamContext::new_local();
                    let result = env
                        .stream_iter(options.clone().into_iter())
                        .map_gpu_with_strategy(
                            BlackScholesKernel::default(),
                            GpuBatchStrategy::timed(1_000_000, Duration::from_millis(100)),
                        )
                        .collect_vec();
                    env.execute_blocking();
                    std::hint::black_box(result.get())
                })
            },
        );

        // Adaptive (100K-10M)
        group.bench_with_input(
            BenchmarkId::new("adaptive-100k-10m", num_options),
            &options,
            |b, options| {
                b.iter(|| {
                    let env = StreamContext::new_local();
                    let result = env
                        .stream_iter(options.clone().into_iter())
                        .map_gpu_with_strategy(
                            BlackScholesKernel::default(),
                            GpuBatchStrategy::adaptive(100_000, 10_000_000),
                        )
                        .collect_vec();
                    env.execute_blocking();
                    std::hint::black_box(result.get())
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, batching_comparison_bench);
criterion_main!(benches);


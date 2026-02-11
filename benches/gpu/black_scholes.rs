//! # Black-Scholes Benchmark
//!
//! Compares CPU and GPU performance for Black-Scholes option pricing using
//! Renoir's streaming operators (`map` and `map_gpu_with_strategy`).
//!
//! ## Strategies Benchmarked
//!
//! 1. **CPU Sequential**: Single worker with `map(black_scholes_cpu)`
//! 2. **CPU Parallel**: Multiple workers with `map(black_scholes_cpu)` + `shuffle()`
//! 3. **GPU**: Single worker with `map_gpu_with_strategy` using `BlackScholesKernel`
//!
//! ## Usage
//!
//! ```bash
//! # Run with WGPU backend (cross-platform: Metal, Vulkan, DirectX 12)
//! cargo bench --bench gpu_black_scholes --features gpu-wgpu
//!
//! # Run with CUDA backend (NVIDIA GPUs only)
//! cargo bench --bench gpu_black_scholes --features gpu-cuda
//!
//! # Run with custom max options
//! MAX_OPTIONS=100000000 cargo bench --bench gpu_black_scholes --features gpu-wgpu
//!
//! # Run with specific parallelism
//! RENOIR_WORKERS=4 cargo bench --bench gpu_black_scholes --features gpu-wgpu
//! ```
//!
//! ## Output
//!
//! Results are saved to `benches/results/black_scholes/{date}/` as JSON files
//! and automatically plotted using `benches/tools/plot_black_scholes.py`.

use std::time::Instant;

use chrono::Utc;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;

mod common;
use common::{
    compute_stats, format_duration_with_stddev, format_number, format_number_short,
    format_speedup, get_benchmark_filepath, get_benchmark_test_sizes, get_platform_info,
    parse_num_runs_env, run_plotter, save_json, BenchmarkType, BenchmarkReport, TestResult,
    WARMUP_RUNS,
};

// Import only the black_scholes kernel from examples (not the entire kernels' module)
#[path = "../../examples/kernels/black_scholes.rs"]
mod black_scholes_kernel;
use black_scholes_kernel::{
    black_scholes_cpu, BlackScholesInput, BlackScholesKernel,
    DEFAULT_RISK_FREE_RATE, DEFAULT_VOLATILITY,
};

use renoir::operator::gpu::GpuBatchStrategy;
use renoir::prelude::*;
use renoir::utils::{create_banner, Table};

// ============================================================================
// Configuration
// ============================================================================

/// Fixed GPU batch size (5 million items)
const GPU_BATCH_SIZE: usize = 5_000_000;

/// GPU vectorization factor (elements per thread)
const GPU_VECTORIZATION_FACTOR: usize = 16;

/// Benchmark configuration
const BENCHMARK_SEED: u64 = 42;

/// FLOPS per Black-Scholes calculation (approximate)
const FLOPS_PER_OPTION: f64 = 40.0;

// ============================================================================
// Result Structures - Using shared structs from common.rs
// ============================================================================

// TestResult and BenchmarkReport are imported from common

// ============================================================================
// Internal Benchmark Result
// ============================================================================

struct TimingResult {
    duration_s: f64,
    output_count: usize,
    results: Vec<black_scholes_kernel::BlackScholesOutput>,  // Actual outputs for validation
}

/// Generate random Black-Scholes options for benchmarking (parallelized with rayon)
/// 
/// Uses per-element parallel generation - each element gets a deterministic RNG
/// seeded by (base_seed + index). This is the fastest approach because SmallRng
/// initialization is extremely cheap and rayon's work-stealing is very efficient.
fn generate_options(num_options: usize, seed: u64) -> Vec<BlackScholesInput> {
    use rayon::prelude::*;
    
    const MIN_PRICE: f32 = 10.0;
    const MAX_PRICE: f32 = 100.0;
    const MIN_TIME: f32 = 0.5;
    const MAX_TIME: f32 = 2.0;
    
    (0..num_options)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(i as u64));
            BlackScholesInput {
                stock_price: rng.random_range(MIN_PRICE..MAX_PRICE),
                strike_price: rng.random_range(MIN_PRICE..MAX_PRICE),
                time_to_expiry: rng.random_range(MIN_TIME..MAX_TIME),
                risk_free_rate: DEFAULT_RISK_FREE_RATE,
                volatility: DEFAULT_VOLATILITY,
            }
        })
        .collect()
}

// ============================================================================
// Benchmark Functions
// ============================================================================

/// Run CPU sequential benchmark using Renoir's `map` operator
fn benchmark_cpu_sequential(data: &[BlackScholesInput]) -> TimingResult {
    let config = RuntimeConfig::local(1).unwrap();
    let env = StreamContext::new(config);
    let input = data.to_vec();
    
    let start = Instant::now();
    let output = env.stream_iter(input.into_iter()).map(black_scholes_cpu).collect_vec();
    env.execute_blocking();
    let total_time = start.elapsed();
    let results = output.get().unwrap();
    let output_count = results.len();
    
    TimingResult {
        duration_s: total_time.as_secs_f64(),
        output_count,
        results,
    }
}

/// Run CPU parallel benchmark using Renoir's `map` operator with multiple workers
fn benchmark_cpu_parallel(data: &[BlackScholesInput], num_workers: usize) -> TimingResult {
    let config = RuntimeConfig::local(num_workers as u64).unwrap();
    let env = StreamContext::new(config);
    let input = data.to_vec();
    
    let start = Instant::now();
    let output = env
        .stream_iter(input.into_iter())
        .shuffle()
        .map(black_scholes_cpu)
        .collect_vec();
    env.execute_blocking();
    let total_time = start.elapsed();
    let results = output.get().unwrap();
    let output_count = results.len();
    
    TimingResult {
        duration_s: total_time.as_secs_f64(),
        output_count,
        results,
    }
}

/// Run GPU benchmark.
#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]
fn benchmark_gpu(data: &[BlackScholesInput]) -> TimingResult {
    let config = RuntimeConfig::local(1).unwrap();
    let env = StreamContext::new(config);
    let input = data.to_vec();
    
    let start = Instant::now();
    let output = env
        .stream_iter(input.into_iter())
        .map_gpu_with_strategy(
            BlackScholesKernel::default().with_vectorization(GPU_VECTORIZATION_FACTOR),
            GpuBatchStrategy::fixed(GPU_BATCH_SIZE),
        )
        .collect_vec();
    env.execute_blocking();
    let total_time = start.elapsed();
    let results = output.get().unwrap();
    let output_count = results.len();
    
    TimingResult {
        duration_s: total_time.as_secs_f64(),
        output_count,
        results,
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Calculate GFLOPS for a given number of options and time
fn calculate_gflops(num_options: usize, time_s: f64) -> f64 {
    if time_s <= 0.0 {
        return 0.0;
    }
    (num_options as f64 * FLOPS_PER_OPTION) / time_s / 1e9
}

/// Calculate data size in GB (5 f32 inputs per option)
fn calculate_data_size_gb(num_options: usize) -> f64 {
    (num_options * 5 * size_of::<f32>()) as f64 / (1024.0 * 1024.0 * 1024.0)
}

// ============================================================================
// Main Benchmark Runner
// ============================================================================

fn main() {
    let timestamp = Utc::now();
    let test_sizes = get_benchmark_test_sizes();
    let num_workers = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    let num_runs = parse_num_runs_env();

    // Print header banner
    print!(
        "{}",
        create_banner(
            "Black-Scholes Benchmark: CPU vs GPU Comparison",
            &[
                &format!("Platform:       {}", get_platform_info()),
                &format!("Workers:        {}", num_workers),
                &format!("GPU Batch Size: {}", format_number(GPU_BATCH_SIZE)),
                &format!("Runs/Test:      {} (+ {} warmup)", num_runs, WARMUP_RUNS),
                &format!(
                    "Test Sizes:     {} sizes from {} to {}",
                    test_sizes.len(),
                    format_number_short(*test_sizes.first().unwrap_or(&0)),
                    format_number_short(*test_sizes.last().unwrap_or(&0))
                ),
            ]
        )
    );

    let mut all_results = Vec::new();
    
    // Setup for incremental saving (crash-resistant)
    let kernel = BlackScholesKernel::default();
    let system_config = common::SystemConfig::detect(kernel.vectorization_factor, GPU_BATCH_SIZE);
    let json_path = get_benchmark_filepath(BenchmarkType::BlackScholes, &timestamp);

    // Create table with timing (mean±stddev) and speedup columns
    let mut table = Table::new(&[
        ("Test", 6),
        ("Items", 14),
        ("Runs", 6),
        ("Seq (mean±σ)", 16),
        ("Par (mean±σ)", 16),
        ("GPU (mean±σ)", 16),
        ("Seq/GPU", 8),
        ("Par/GPU", 8),
        ("Valid", 9),
    ]);

    // Run benchmarks for each test size
    for (i, &num_options) in test_sizes.iter().enumerate() {
        let test_id = i + 1;
        let test_timestamp = Utc::now();

        // Start a new row
        table.start_row();
        table.set_cell(0, &format!("{:>4}", test_id));
        table.set_cell(1, &format!("{:>12}", format_number(num_options)));
        table.set_cell(2, &format!("{:>4}", num_runs));

        // Generate data once (shared across all runs)
        let data = generate_options(num_options, BENCHMARK_SEED);

        // --- Multi-run CPU Sequential ---
        let mut seq_durations = Vec::with_capacity(num_runs);
        let mut last_cpu_seq = benchmark_cpu_sequential(&data); // first run = warmup
        for _ in 0..WARMUP_RUNS.saturating_sub(1) {
            last_cpu_seq = benchmark_cpu_sequential(&data);
        }
        for _ in 0..num_runs {
            let result = benchmark_cpu_sequential(&data);
            seq_durations.push(result.duration_s);
            last_cpu_seq = result;
        }
        let seq_stats = compute_stats(&seq_durations);
        table.set_cell(3, &format_duration_with_stddev(seq_stats.mean, seq_stats.stddev));

        // --- Multi-run CPU Parallel ---
        let mut par_durations = Vec::with_capacity(num_runs);
        let mut last_cpu_par = benchmark_cpu_parallel(&data, num_workers);
        for _ in 0..WARMUP_RUNS.saturating_sub(1) {
            last_cpu_par = benchmark_cpu_parallel(&data, num_workers);
        }
        for _ in 0..num_runs {
            let result = benchmark_cpu_parallel(&data, num_workers);
            par_durations.push(result.duration_s);
            last_cpu_par = result;
        }
        let par_stats = compute_stats(&par_durations);
        table.set_cell(4, &format_duration_with_stddev(par_stats.mean, par_stats.stddev));

        // --- Multi-run GPU ---
        #[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]
        let (gpu_stats, last_gpu) = {
            let mut gpu_durations = Vec::with_capacity(num_runs);
            let mut last = benchmark_gpu(&data); // warmup
            for _ in 0..WARMUP_RUNS.saturating_sub(1) {
                last = benchmark_gpu(&data);
            }
            for _ in 0..num_runs {
                let result = benchmark_gpu(&data);
                gpu_durations.push(result.duration_s);
                last = result;
            }
            (compute_stats(&gpu_durations), last)
        };

        #[cfg(not(any(feature = "gpu-wgpu", feature = "gpu-cuda")))]
        let (gpu_stats, last_gpu) = (
            compute_stats(&[f64::MAX]),
            TimingResult { duration_s: f64::MAX, output_count: 0, results: Vec::new() },
        );
        table.set_cell(5, &format_duration_with_stddev(gpu_stats.mean, gpu_stats.stddev));

        // Calculate speedups from mean times
        let speedup = seq_stats.mean / gpu_stats.mean;
        let speedup_parallel = par_stats.mean / gpu_stats.mean;
        table.set_cell(6, &format_speedup(speedup));
        table.set_cell(7, &format_speedup(speedup_parallel));

        // Calculate GFLOPS from mean times
        let cpu_gflops = calculate_gflops(num_options, seq_stats.mean);
        let renoir_gflops = calculate_gflops(num_options, par_stats.mean);
        let gpu_gflops = calculate_gflops(num_options, gpu_stats.mean);

        // Compute validation metrics using outputs from last run
        let validation_sample_size = last_cpu_seq.results.len().min(last_gpu.results.len()).min(10_000);
        let (validation_numerical_ok, validation_max_error, validation_avg_error) = if validation_sample_size > 0 {
            let cpu_subset: Vec<_> = last_cpu_seq.results.iter().take(validation_sample_size).cloned().collect();
            let gpu_subset: Vec<_> = last_gpu.results.iter().take(validation_sample_size).cloned().collect();
            
            let tolerance = 1e-4f32;
            let mut max_error = 0.0f32;
            let mut total_error = 0.0f32;
            let mut all_ok = true;
            
            for (cpu_out, gpu_out) in cpu_subset.iter().zip(gpu_subset.iter()) {
                let call_error = (cpu_out.call_price - gpu_out.call_price).abs();
                let put_error = (cpu_out.put_price - gpu_out.put_price).abs();
                max_error = max_error.max(call_error).max(put_error);
                total_error += call_error + put_error;
                if max_error > tolerance {
                    all_ok = false;
                }
            }
            
            let avg_error = if validation_sample_size > 0 {
                total_error / (validation_sample_size * 2) as f32
            } else {
                0.0
            };
            
            (all_ok, max_error, avg_error)
        } else {
            (false, f32::MAX, f32::MAX)
        };
        
        let validation_passed = {
            let count_ok = last_cpu_seq.output_count == num_options 
                && last_cpu_par.output_count == num_options 
                && last_gpu.output_count == num_options;
            count_ok && validation_numerical_ok
        };
        
        let valid_str = if validation_passed { "    ✓    " } else { "    ✗    " };
        table.set_cell(8, valid_str);
        
        // Finalize the row
        table.flush_row();
        
        // Create the result with statistics
        let result = TestResult {
            test_id,
            timestamp: test_timestamp.to_rfc3339(),
            items_count: num_options,
            data_size_gb: calculate_data_size_gb(num_options),
            
            renoir_seq_total_time_s: seq_stats.mean,
            renoir_par_total_time_s: par_stats.mean,
            gpu_total_time_s: gpu_stats.mean,
            
            renoir_seq_stats: Some(seq_stats),
            renoir_par_stats: Some(par_stats),
            gpu_stats: Some(gpu_stats),
            
            cpu_seq_compute_s: None,
            cpu_par_compute_s: None,
            gpu_compute_s: None,
            
            speedup,
            speedup_parallel,
            speedup_seq_compute: None,
            speedup_par_compute: None,
            
            renoir_seq_gflops: cpu_gflops,
            renoir_par_gflops: renoir_gflops,
            gpu_gflops,
            
            cpu_workers: num_workers,
            batch_size: GPU_BATCH_SIZE,
            num_runs,
            
            validation_passed,
            validation_max_error: validation_max_error as f64,
            validation_avg_error: validation_avg_error as f64,
            validation_sample_size,
        };

        all_results.push(result);
        
        // Incremental save: update JSON after each test (crash-resistant)
        let report = BenchmarkReport {
            benchmark_type: "black_scholes".to_string(),
            start_time: timestamp.to_rfc3339(),
            platform: get_platform_info(),
            system_config: system_config.clone(),
            total_tests: test_sizes.len(),
            cpu_workers: num_workers,
            batch_size: GPU_BATCH_SIZE,
            num_runs,
            warmup_runs: WARMUP_RUNS,
            results: all_results.clone(),
            monte_carlo_config: None,
        };
        save_json(&json_path, &report);
    }

    table.finish();

    let report = BenchmarkReport {
        benchmark_type: "black_scholes".to_string(),
        start_time: timestamp.to_rfc3339(),
        platform: get_platform_info(),
        system_config,
        total_tests: all_results.len(),
        cpu_workers: num_workers,
        batch_size: GPU_BATCH_SIZE,
        num_runs,
        warmup_runs: WARMUP_RUNS,
        results: all_results.clone(),
        monte_carlo_config: None,
    };

    // Save results
    let json_path = get_benchmark_filepath(BenchmarkType::BlackScholes, &timestamp);
    save_json(&json_path, &report);
    println!("\nResults saved to: {}", json_path.display());

    // Generate plots
    run_plotter(&json_path, BenchmarkType::BlackScholes, &timestamp);

    // Print summary banner
    print_summary(&all_results);
}

/// Print a summary of the benchmark results
fn print_summary(results: &[TestResult]) {
    let mut summary_lines: Vec<String> = Vec::new();

    for result in results {
        // Find best strategy
        let strategies = [
            ("Renoir Seq", result.renoir_seq_total_time_s),
            ("Renoir Par", result.renoir_par_total_time_s),
            ("GPU", result.gpu_total_time_s),
        ];

        let (best_name, best_time) = strategies
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        let throughput_mops = result.items_count as f64 / best_time / 1_000_000.0;

        summary_lines.push(format!(
            "{:>10}: {} ({:.2} M opts/s)",
            format_number_short(result.items_count),
            best_name,
            throughput_mops
        ));
    }

    let summary_refs: Vec<&str> = summary_lines.iter().map(|s| s.as_str()).collect();
    print!("{}", create_banner("Best Strategy by Problem Size", &summary_refs));
}

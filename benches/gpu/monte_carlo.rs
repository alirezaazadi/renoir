//! # Monte Carlo Benchmark
//!
//! Compares CPU and GPU performance for Monte Carlo option pricing using
//! Renoir's streaming operators (`map` and `map_gpu_with_strategy`).
//!
//! ## Strategies Benchmarked
//!
//! 1. **CPU Sequential**: Single worker with `map(monte_carlo_cpu)`
//! 2. **CPU Parallel**: Multiple workers with `map(monte_carlo_cpu)` + `shuffle()`
//! 3. **GPU**: Single worker with `map_gpu_with_strategy` using `MonteCarloKernel`
//!
//! ## Usage
//!
//! ```bash
//! # Run with WGPU backend (cross-platform: Metal, Vulkan, DirectX 12)
//! cargo bench --bench gpu_monte_carlo --features gpu-wgpu
//!
//! # Run with CUDA backend (NVIDIA GPUs only)
//! cargo bench --bench gpu_monte_carlo --features gpu-cuda
//!
//! # Run with custom max options (default is 300K due to high compute per option)
//! MAX_OPTIONS=100000 cargo bench --bench gpu_monte_carlo --features gpu-wgpu
//! ```
//!
//! ## Output
//!
//! Results are saved to `benches/results/monte_carlo/{date}/` as JSON files
//! and automatically plotted using `benches/tools/plot_monte_carlo.py`.
//!
//! Features result validation, intermediate JSON saves, and chart generation support.

use std::time::Instant;
use chrono::Utc;

mod common;
use common::{
    compute_stats, format_duration_with_stddev, format_number, format_number_short,
    format_speedup, get_benchmark_filepath, get_benchmark_test_sizes, get_platform_info,
    parse_num_runs_env, run_plotter, save_json, BenchmarkReport, BenchmarkType, MonteCarloConfig,
    SystemConfig, TestResult, WARMUP_RUNS,
};


// Import only the monte_carlo kernel from examples (not the entire kernels' module)
#[path = "../../examples/kernels/monte_carlo.rs"]
mod monte_carlo_kernel;
use monte_carlo_kernel::{
    monte_carlo_cpu, MonteCarloInput, MonteCarloKernel, MonteCarloOutput,
    MC_NUM_PATHS, MC_TIME_STEPS,
};

use renoir::operator::gpu::GpuBatchStrategy;
use renoir::prelude::*;
use renoir::utils::{create_banner, Table};

// ============================================================================
// Configuration
// ============================================================================

/// GPU batch size for streaming processing.
const GPU_BATCH_SIZE: usize = 100_000;

/// Seed for generating benchmark INPUT DATA (stock prices, strikes, etc.).
/// 
/// This ensures reproducible benchmark inputs across runs, making performance
/// comparisons fair and repeatable.
/// 
/// **Important**: This is NOT the Monte Carlo simulation seed! There are two
/// levels of seeding in this benchmark:
/// 
/// 1. **Input Generation Seed** (`BENCHMARK_SEED`): Creates the test options
///    (stock prices, strikes, etc.). Same seed = same test inputs every run.
/// 
/// 2. **Per-Option Simulation Seed** (`deterministic_seed(&input)`): Each option
///    gets a unique, reproducible seed derived from its parameters. This ensures
///    CPU and GPU produce identical Monte Carlo results for the same input.
/// 
/// This separation allows fair performance benchmarking (reproducible inputs)
/// while maintaining CPU/GPU validation (deterministic simulation per option).
const BENCHMARK_SEED: u64 = 42;

const VALIDATION_SAMPLE_SIZE: usize = 1000;
/// Tolerance for Monte Carlo validation (not used - validation disabled due to inherent variance)
const VALIDATION_TOLERANCE: f32 = 0.5;
// ~1M FLOPs per option (1000 paths * 50 steps * 20 ops)
const FLOPS_PER_OPTION: f64 = (MC_NUM_PATHS as f64) * (MC_TIME_STEPS as f64) * 20.0;


// ============================================================================
// Data Structures - Using shared structs from common.rs
// ============================================================================

// TestResult, BenchmarkReport, and MonteCarloConfig are imported from common


// ============================================================================
// Validation
// ============================================================================

/// Validate GPU results against CPU results.
///
/// Since both CPU and GPU now use deterministic seeding via `deterministic_seed(&input)`,
/// they produce identical results for the same inputs. We can do exact comparison.
fn validate_results(
    cpu_results: &[MonteCarloOutput],
    gpu_results: &[MonteCarloOutput],
    sample_size: usize,
) -> (bool, f64, f64, usize) {
    let n = cpu_results.len().min(gpu_results.len()).min(sample_size);
    if n == 0 {
        return (true, 0.0, 0.0, 0);
    }
    
    let mut max_error = 0.0f64;
    let mut total_error = 0.0f64;
    let mut all_ok = true;
    
    // Tolerance for floating-point comparison
    // Allow small relative error due to f32 precision differences
    const TOLERANCE: f64 = 0.01; // 1% relative error
    
    for i in 0..n {
        let cpu = &cpu_results[i];
        let gpu = &gpu_results[i];
        
        let call_err = (cpu.call_price - gpu.call_price).abs() as f64;
        let put_err = (cpu.put_price - gpu.put_price).abs() as f64;
        
        max_error = max_error.max(call_err).max(put_err);
        total_error += call_err + put_err;
        
        // Check relative error for non-zero prices
        if cpu.call_price.abs() > 0.01 {
            let rel_err = call_err / cpu.call_price.abs() as f64;
            if rel_err > TOLERANCE {
                all_ok = false;
            }
        }
    }
    
    let avg_error = total_error / (n * 2) as f64;
    (all_ok, max_error, avg_error, n)
}

// ============================================================================
// Data Generation
// ============================================================================

/// Generate random Monte Carlo inputs for benchmarking (parallelized with rayon)
/// 
/// Uses per-element parallel generation - each element gets a deterministic RNG
/// seeded by (base_seed + index). This is the fastest approach because SmallRng
/// initialization is extremely cheap and rayon's work-stealing is very efficient.
fn generate_data(count: usize) -> Vec<MonteCarloInput> {
    use rayon::prelude::*;
    use rand::prelude::*;
    use rand::rngs::SmallRng;
    
    (0..count)
        .into_par_iter()
        .map(|i| {
            let mut rng = SmallRng::seed_from_u64(BENCHMARK_SEED.wrapping_add(i as u64));
            MonteCarloInput {
                stock_price: rng.random_range(50.0..150.0),
                strike_price: rng.random_range(50.0..150.0),
                time_to_expiry: rng.random_range(0.5..2.0),
                risk_free_rate: 0.05,
                volatility: 0.2,
            }
        })
        .collect()
}

// ============================================================================
// Benchmarks
// ============================================================================

struct BenchResult {
    duration_s: f64,
    results: Vec<MonteCarloOutput>,
}

fn benchmark_cpu_seq(data: &[MonteCarloInput]) -> BenchResult {
    let config = RuntimeConfig::local(1).unwrap();
    let env = StreamContext::new(config);
    let input = data.to_vec();
    
    let start = Instant::now();
    let output = env.stream_iter(input.into_iter())
        .map(monte_carlo_cpu)
        .collect_vec();
    env.execute_blocking();
    let total_time = start.elapsed();
    let results = output.get().unwrap();
    
    BenchResult {
        duration_s: total_time.as_secs_f64(),
        results,
    }
}

fn benchmark_cpu_par(data: &[MonteCarloInput], workers: usize) -> BenchResult {
    let config = RuntimeConfig::local(workers as u64).unwrap();
    let env = StreamContext::new(config);
    let input = data.to_vec();
    
    let start = Instant::now();
    let output = env.stream_iter(input.into_iter())
        .shuffle()
        .map(monte_carlo_cpu)
        .collect_vec();
    env.execute_blocking();
    let total_time = start.elapsed();
    let results = output.get().unwrap();
    
    BenchResult {
        duration_s: total_time.as_secs_f64(),
        results,
    }
}

#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]
fn benchmark_gpu(data: &[MonteCarloInput]) -> BenchResult {
    let config = RuntimeConfig::local(1).unwrap();
    let env = StreamContext::new(config);
    let input = data.to_vec();
    
    let start = Instant::now();
    let output = env.stream_iter(input.into_iter())
        .map_gpu_with_strategy(
            MonteCarloKernel::default(),
            GpuBatchStrategy::fixed(GPU_BATCH_SIZE),
        )
        .collect_vec();
    env.execute_blocking();
    let total_time = start.elapsed();
    let results = output.get().unwrap();
    
    BenchResult {
        duration_s: total_time.as_secs_f64(),
        results,
    }
}

// ============================================================================
// Main
// ============================================================================

/// Default max for Monte Carlo (much smaller than Black-Scholes due to ~1000x higher compute per option)
const DEFAULT_MC_MAX_OPTIONS: usize = 300_000;

fn main() {
    let timestamp = Utc::now();
    
    // Use common test sizes, but with smaller default max for Monte Carlo
    // (Monte Carlo is ~1000x more expensive per option: 1000 paths × 50 steps)
    let max_options = std::env::var("MAX_OPTIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MC_MAX_OPTIONS);
    let sizes: Vec<usize> = get_benchmark_test_sizes()
        .into_iter()
        .filter(|&s| s <= max_options)
        .collect();
    
    let workers = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);
    let num_runs = parse_num_runs_env();

    print!("{}", create_banner("Monte Carlo Benchmark", &[
        &format!("Paths/Option: {}", format_number(MC_NUM_PATHS as usize)),
        &format!("Steps/Path:   {}", format_number(MC_TIME_STEPS as usize)),
        &format!("Workers:      {}", workers),
        &format!("Runs/Test:    {} (+ {} warmup)", num_runs, WARMUP_RUNS),
        &format!("Validation:   {} samples, {:.0}% tol", VALIDATION_SAMPLE_SIZE, VALIDATION_TOLERANCE * 100.0),
    ]));

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

    let mut results = Vec::new();
    let json_path = get_benchmark_filepath(BenchmarkType::MonteCarlo, &timestamp);
    
    // Ensure parent directory exists
    if let Some(parent) = json_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    for (idx, &size) in sizes.iter().enumerate() {
        // Start a new row
        table.start_row();
        table.set_cell(0, &format!("{:>4}", idx + 1));
        table.set_cell(1, &format!("{:>12}", format_number(size)));
        table.set_cell(2, &format!("{:>4}", num_runs));
        
        // Generate data once (shared across all runs)
        let data = generate_data(size);
        let data_size_gb = (size * size_of::<MonteCarloInput>()) as f64 / (1024.0 * 1024.0 * 1024.0);
        
        // --- Multi-run CPU Sequential ---
        let mut seq_durations = Vec::with_capacity(num_runs);
        let mut last_cpu_seq = benchmark_cpu_seq(&data); // warmup
        for _ in 0..WARMUP_RUNS.saturating_sub(1) {
            last_cpu_seq = benchmark_cpu_seq(&data);
        }
        for _ in 0..num_runs {
            let result = benchmark_cpu_seq(&data);
            seq_durations.push(result.duration_s);
            last_cpu_seq = result;
        }
        let seq_stats = compute_stats(&seq_durations);
        table.set_cell(3, &format_duration_with_stddev(seq_stats.mean, seq_stats.stddev));
        
        // --- Multi-run CPU Parallel ---
        let mut par_durations = Vec::with_capacity(num_runs);
        let _ = benchmark_cpu_par(&data, workers); // warmup
        for _ in 0..WARMUP_RUNS.saturating_sub(1) {
            let _ = benchmark_cpu_par(&data, workers);
        }
        for _ in 0..num_runs {
            let result = benchmark_cpu_par(&data, workers);
            par_durations.push(result.duration_s);
        }
        let par_stats = compute_stats(&par_durations);
        table.set_cell(4, &format_duration_with_stddev(par_stats.mean, par_stats.stddev));
        
        // --- Multi-run GPU ---
        #[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]
        let (gpu_stats_computed, last_gpu) = {
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
        let (gpu_stats_computed, last_gpu) = (
            compute_stats(&[999.9]),
            BenchResult { duration_s: 999.9, results: vec![] },
        );
        table.set_cell(5, &format_duration_with_stddev(gpu_stats_computed.mean, gpu_stats_computed.stddev));

        // Validate GPU results against CPU (using last run's outputs)
        let (valid_passed, max_err, avg_err, valid_n) = validate_results(
            &last_cpu_seq.results, &last_gpu.results, VALIDATION_SAMPLE_SIZE
        );

        // Calculate speedups from mean times
        let speedup_seq = seq_stats.mean / gpu_stats_computed.mean;
        let speedup_par = par_stats.mean / gpu_stats_computed.mean;
        table.set_cell(6, &format_speedup(speedup_seq));
        table.set_cell(7, &format_speedup(speedup_par));
        
        let valid_str = if valid_passed { "    ✓    " } else { "    ✗    " };
        table.set_cell(8, valid_str);
        
        // Finalize the row
        table.flush_row();
        
        // Print validation details
        if !valid_passed {
            println!("  ⚠ Validation: max_err={:.2}%, avg_err={:.2}%", max_err * 100.0, avg_err * 100.0);
        }

        results.push(TestResult {
            test_id: idx + 1,
            timestamp: Utc::now().to_rfc3339(),
            items_count: size,
            data_size_gb,
            renoir_seq_total_time_s: seq_stats.mean,
            renoir_par_total_time_s: par_stats.mean,
            gpu_total_time_s: gpu_stats_computed.mean,
            renoir_seq_gflops: (size as f64 * FLOPS_PER_OPTION) / seq_stats.mean / 1e9,
            renoir_par_gflops: (size as f64 * FLOPS_PER_OPTION) / par_stats.mean / 1e9,
            gpu_gflops: (size as f64 * FLOPS_PER_OPTION) / gpu_stats_computed.mean / 1e9,
            renoir_seq_stats: Some(seq_stats),
            renoir_par_stats: Some(par_stats),
            gpu_stats: Some(gpu_stats_computed),
            cpu_seq_compute_s: None,
            cpu_par_compute_s: None,
            gpu_compute_s: None,
            speedup: speedup_seq,
            speedup_parallel: speedup_par,
            speedup_seq_compute: None,
            speedup_par_compute: None,
            cpu_workers: workers,
            batch_size: GPU_BATCH_SIZE,
            num_runs,
            validation_passed: valid_passed,
            validation_max_error: max_err,
            validation_avg_error: avg_err,
            validation_sample_size: valid_n,
        });
        
        // Save intermediate results after each test
        let intermediate_report = BenchmarkReport {
            benchmark_type: "monte_carlo".to_string(),
            start_time: timestamp.to_rfc3339(),
            platform: get_platform_info(),
            system_config: SystemConfig::detect(1, GPU_BATCH_SIZE),
            monte_carlo_config: Some(MonteCarloConfig {
                num_paths: MC_NUM_PATHS,
                time_steps: MC_TIME_STEPS,
                flops_per_option: FLOPS_PER_OPTION,
            }),
            total_tests: sizes.len(),
            cpu_workers: workers,
            batch_size: GPU_BATCH_SIZE,
            num_runs,
            warmup_runs: WARMUP_RUNS,
            results: results.clone(),
        };
        save_json(&json_path, &intermediate_report);
    }
    
    table.finish();
    
    // Final save with all results
    let final_report = BenchmarkReport {
        benchmark_type: "monte_carlo".to_string(),
        start_time: timestamp.to_rfc3339(),
        platform: get_platform_info(),
        system_config: SystemConfig::detect(1, GPU_BATCH_SIZE),
        monte_carlo_config: Some(MonteCarloConfig {
            num_paths: MC_NUM_PATHS,
            time_steps: MC_TIME_STEPS,
            flops_per_option: FLOPS_PER_OPTION,
        }),
        total_tests: sizes.len(),
        cpu_workers: workers,
        batch_size: GPU_BATCH_SIZE,
        num_runs,
        warmup_runs: WARMUP_RUNS,
        results,
    };
    save_json(&json_path, &final_report);
    println!("\nResults saved to: {}", json_path.display());

    // Generate plots
    run_plotter(&json_path, BenchmarkType::MonteCarlo, &timestamp);

    // Print summary banner
    print_summary(&final_report.results);
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

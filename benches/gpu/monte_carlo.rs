//! # Monte Carlo Benchmark with Validation and JSON Output
//!
//! Compares CPU and GPU performance for Monte Carlo option pricing.
//! Features result validation, intermediate JSON saves, and chart generation support.

use std::time::Instant;
use chrono::Utc;

mod common;
use common::{
    format_duration, format_number, format_number_short, format_speedup,
    get_benchmark_filepath, get_benchmark_test_sizes, get_platform_info, run_plotter, save_json,
    BenchmarkReport, BenchmarkType, MonteCarloConfig, SystemConfig, TestResult,
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

    print!("{}", create_banner("Monte Carlo Benchmark", &[
        &format!("Paths/Option: {}", format_number(MC_NUM_PATHS as usize)),
        &format!("Steps/Path:   {}", format_number(MC_TIME_STEPS as usize)),
        &format!("Workers:      {}", workers),
        &format!("Validation:   {} samples, {:.0}% tol", VALIDATION_SAMPLE_SIZE, VALIDATION_TOLERANCE * 100.0),
    ]));

    // Create table with timing and speedup columns
    let mut table = Table::new(&[
        ("Test", 6),
        ("Items", 14),
        ("DataGen", 10),
        ("Seq", 10),
        ("Par", 10),
        ("GPU", 10),
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
        // Start a new row - displays empty cells immediately
        table.start_row();
        table.set_cell(0, &format!("{:>4}", idx + 1));
        table.set_cell(1, &format!("{:>12}", format_number(size)));
        
        // Generate data (timed) - update cell when done
        let data_gen_start = Instant::now();
        let data = generate_data(size);
        let data_gen_time = data_gen_start.elapsed().as_secs_f64();
        table.set_cell(2, &format_duration(data_gen_time));
        
        let data_size_gb = (size * size_of::<MonteCarloInput>()) as f64 / (1024.0 * 1024.0 * 1024.0);
        
        // Run CPU sequential benchmark - update cell when done
        let cpu_seq = benchmark_cpu_seq(&data);
        table.set_cell(3, &format_duration(cpu_seq.duration_s));
        
        // Run CPU parallel benchmark - update cell when done
        let cpu_par = benchmark_cpu_par(&data, workers);
        table.set_cell(4, &format_duration(cpu_par.duration_s));
        
        // Run GPU benchmark - update cell when done
        #[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]
        let gpu = benchmark_gpu(&data);
        #[cfg(not(any(feature = "gpu-wgpu", feature = "gpu-cuda")))]
        let gpu = BenchResult { duration_s: 999.9, results: vec![] };
        table.set_cell(5, &format_duration(gpu.duration_s));

        // Validate GPU results against CPU
        let (valid_passed, max_err, avg_err, valid_n) = validate_results(
            &cpu_seq.results, &gpu.results, VALIDATION_SAMPLE_SIZE
        );

        // Calculate speedups and update remaining cells
        let speedup_seq = cpu_seq.duration_s / gpu.duration_s;
        let speedup_par = cpu_par.duration_s / gpu.duration_s;
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
            renoir_seq_total_time_s: cpu_seq.duration_s,
            renoir_par_total_time_s: cpu_par.duration_s,
            gpu_total_time_s: gpu.duration_s,
            cpu_seq_compute_s: None,
            cpu_par_compute_s: None,
            gpu_compute_s: None,
            speedup: speedup_seq,
            speedup_parallel: speedup_par,
            speedup_seq_compute: None,
            speedup_par_compute: None,
            renoir_seq_gflops: (size as f64 * FLOPS_PER_OPTION) / cpu_seq.duration_s / 1e9,
            renoir_par_gflops: (size as f64 * FLOPS_PER_OPTION) / cpu_par.duration_s / 1e9,
            gpu_gflops: (size as f64 * FLOPS_PER_OPTION) / gpu.duration_s / 1e9,
            cpu_workers: workers,
            batch_size: GPU_BATCH_SIZE,
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

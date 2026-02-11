//! # GPU Reduce Benchmark
//!
//! Compares CPU and GPU performance for reduction operations (Sum, Product,
//! Min, Max) using Renoir's streaming operators.
//!
//! ## Strategies Benchmarked
//!
//! 1. **Renoir Sequential**: Single worker with `reduce_assoc`
//! 2. **Renoir Parallel**: Multiple workers with `reduce_assoc` + `shuffle()`
//! 3. **GPU**: Single worker with `reduce_gpu_with` using CubeCL
//!
//! ## Usage
//!
//! ```bash
//! # Run with WGPU backend (cross-platform: Metal, Vulkan, DirectX 12)
//! cargo bench --bench gpu_reduce --features gpu-wgpu
//!
//! # Run with CUDA backend (NVIDIA GPUs only)
//! cargo bench --bench gpu_reduce --features gpu-cuda
//!
//! # Run with custom max items
//! MAX_OPTIONS=500000000 cargo bench --bench gpu_reduce --features gpu-wgpu
//!
//! # Run with specific parallelism
//! RENOIR_WORKERS=4 cargo bench --bench gpu_reduce --features gpu-wgpu
//! ```
//!
//! ## Output
//!
//! Results are saved to `benches/results/reduce/{date}/` as JSON files
//! and automatically plotted using `benches/tools/plot_reduce_benchmark.py`.

use std::io::Write;
use std::time::Instant;

use chrono::Utc;
use rand::prelude::*;
use rand::rngs::SmallRng;
use serde::{Deserialize, Serialize};

mod common;
use common::{
    compute_stats, format_duration_with_stddev, format_number, format_number_short,
    format_speedup, get_benchmark_filepath, get_benchmark_test_sizes, get_platform_info,
    parse_num_runs_env, run_plotter, save_json, BenchmarkType, RunStats, SystemConfig,
    WARMUP_RUNS,
};

#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]
use renoir::operator::{ReduceGpuBackend, ReduceGpuConfig, ReduceKernel};
use renoir::prelude::*;
use renoir::RuntimeConfig;

// ============================================================================
// Configuration
// ============================================================================

/// Default GPU batch size for the reduce operator.
const GPU_BATCH_SIZE: usize = 4_000_000;
/// Default GPU tile size for the reduce kernel.
const GPU_TILE_SIZE: usize = 1 << 18; // 262_144
/// GPU threads estimate (for metadata reporting).
const GPU_THREADS_ESTIMATE: usize = 256 * 64;

/// FLOPS per element for a simple reduce (1 operation per element).
const FLOPS_PER_ELEMENT: f64 = 1.0;

// On WGPU we use f32; on CUDA we use f64.
#[cfg(all(feature = "gpu-wgpu", not(feature = "gpu-cuda")))]
type SampleFloat = f32;
#[cfg(all(feature = "gpu-cuda", not(feature = "gpu-wgpu")))]
type SampleFloat = f64;
#[cfg(not(any(feature = "gpu-wgpu", feature = "gpu-cuda")))]
type SampleFloat = f64;

const SAMPLE_FLOAT_SIZE: usize = std::mem::size_of::<SampleFloat>();

// ============================================================================
// Reduce Operator Enum
// ============================================================================

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum ReduceOp {
    Sum,
    Product,
    Min,
    Max,
}

impl std::fmt::Display for ReduceOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReduceOp::Sum => write!(f, "sum"),
            ReduceOp::Product => write!(f, "product"),
            ReduceOp::Min => write!(f, "min"),
            ReduceOp::Max => write!(f, "max"),
        }
    }
}

// ============================================================================
// Result Structures
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReduceTestResult {
    pub test_id: usize,
    pub timestamp: String,
    pub operator: ReduceOp,
    pub items_count: usize,
    pub data_size_gb: f64,
    // Timing (mean across runs)
    pub renoir_seq_total_time_s: f64,
    pub renoir_par_total_time_s: f64,
    pub gpu_total_time_s: f64,
    // Multi-run statistics
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub renoir_seq_stats: Option<RunStats>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub renoir_par_stats: Option<RunStats>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_stats: Option<RunStats>,
    // Speedups (based on mean times)
    pub speedup_seq: f64,
    pub speedup_par: f64,
    // GFLOPS (based on mean times)
    pub renoir_seq_gflops: f64,
    pub renoir_par_gflops: f64,
    pub gpu_gflops: f64,
    // System / config
    pub cpu_workers: usize,
    pub gpu_threads: usize,
    pub batch_size: usize,
    pub tile_size: usize,
    #[serde(default = "default_num_runs")]
    pub num_runs: usize,
    // Values (from last run)
    pub cpu_result: f64,
    pub renoir_result: f64,
    pub gpu_result: f64,
    // Validation
    pub validation_passed: bool,
    pub error_margin: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ReduceBenchmarkReport {
    pub benchmark_type: String,
    pub start_time: String,
    pub platform: String,
    pub system_config: SystemConfig,
    pub gpu_enabled: bool,
    pub total_tests: usize,
    pub cpu_workers: usize,
    pub batch_size: usize,
    pub tile_size: usize,
    #[serde(default = "default_num_runs")]
    pub num_runs: usize,
    #[serde(default)]
    pub warmup_runs: usize,
    pub results: Vec<ReduceTestResult>,
}

fn default_num_runs() -> usize { 1 }

// ============================================================================
// Data Generation
// ============================================================================

fn generate_data(size: usize, op: ReduceOp) -> Vec<SampleFloat> {
    let mut rng = SmallRng::seed_from_u64(42);
    match op {
        // Products overflow quickly so keep values near 1.0.
        ReduceOp::Product => (0..size)
            .map(|_| rng.random_range(0.99f64..1.01) as SampleFloat)
            .collect(),
        _ => (0..size)
            .map(|_| rng.random_range(1.0f64..100.0) as SampleFloat)
            .collect(),
    }
}

/// Calculate expected result in f64 precision.
fn calculate_expected(data: &[SampleFloat], op: ReduceOp) -> f64 {
    match op {
        ReduceOp::Sum => data.iter().fold(0.0, |acc, &v| acc + v as f64),
        ReduceOp::Product => data.iter().fold(1.0, |acc, &v| acc * v as f64),
        ReduceOp::Min => data
            .iter()
            .fold(f64::INFINITY, |acc, &v| acc.min(v as f64)),
        ReduceOp::Max => data
            .iter()
            .fold(f64::NEG_INFINITY, |acc, &v| acc.max(v as f64)),
    }
}

fn validate_result(expected: f64, actual: f64, op: ReduceOp) -> (bool, f64) {
    let error_margin = match op {
        ReduceOp::Sum => (expected - actual).abs() / expected.abs().max(1.0),
        ReduceOp::Product => {
            if expected.abs() < 1e-10 {
                (expected - actual).abs()
            } else {
                (expected - actual).abs() / expected.abs()
            }
        }
        ReduceOp::Min | ReduceOp::Max => (expected - actual).abs(),
    };
    let threshold = match op {
        ReduceOp::Sum => 0.001,
        ReduceOp::Product => 0.01,
        ReduceOp::Min | ReduceOp::Max => 0.0001,
    };
    (error_margin <= threshold, error_margin)
}

// ============================================================================
// Benchmark Functions
// ============================================================================

/// CPU sequential reduce using a simple fold (single-threaded, no Renoir).
fn benchmark_cpu_sequential(data: &[SampleFloat], op: ReduceOp) -> (f64, f64) {
    let start = Instant::now();
    let result = match op {
        ReduceOp::Sum => data.iter().fold(0.0, |acc, &v| acc + v as f64),
        ReduceOp::Product => data.iter().fold(1.0, |acc, &v| acc * v as f64),
        ReduceOp::Min => data
            .iter()
            .fold(f64::INFINITY, |acc, &v| acc.min(v as f64)),
        ReduceOp::Max => data
            .iter()
            .fold(f64::NEG_INFINITY, |acc, &v| acc.max(v as f64)),
    };
    (result, start.elapsed().as_secs_f64())
}

/// Renoir parallel reduce using `reduce_assoc` with multiple workers.
fn benchmark_renoir_parallel(
    data: &[SampleFloat],
    op: ReduceOp,
    num_workers: usize,
) -> (f64, f64) {
    let config = RuntimeConfig::local(num_workers as u64).unwrap();
    let env = StreamContext::new(config);
    let d = std::sync::Arc::new(data.to_vec());
    let start = Instant::now();
    let output = env
        .stream_par_iter(0..d.len())
        .batch_mode(BatchMode::fixed(4096))
        .map(move |idx| d[idx] as f64)
        .reduce_assoc(move |a, b| match op {
            ReduceOp::Sum => a + b,
            ReduceOp::Product => a * b,
            ReduceOp::Min => a.min(b),
            ReduceOp::Max => a.max(b),
        })
        .collect_vec();
    env.execute_blocking();
    let elapsed = start.elapsed().as_secs_f64();
    let result = output.get().unwrap()[0];
    (result, elapsed)
}

/// GPU reduce using `reduce_gpu_with`.
#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]
fn benchmark_gpu_reduce(
    data: &[SampleFloat],
    op: ReduceOp,
    batch_size: usize,
    tile_size: usize,
) -> (f64, f64) {
    let env = StreamContext::new_local();
    let config = ReduceGpuConfig::default()
        .with_batch_size(batch_size)
        .with_tile_size(tile_size)
        .with_backend(ReduceGpuBackend::Auto);
    let kernel = match op {
        ReduceOp::Sum => ReduceKernel::Sum,
        ReduceOp::Product => ReduceKernel::Product,
        ReduceOp::Min => ReduceKernel::Min,
        ReduceOp::Max => ReduceKernel::Max,
    };
    let input = data.to_vec();

    let start = Instant::now();
    let output = env
        .stream_iter(input.into_iter())
        .reduce_gpu_with(kernel, config)
        .collect_vec();
    env.execute_blocking();
    let elapsed = start.elapsed().as_secs_f64();
    let result = output.get().unwrap()[0];
    (result, elapsed)
}

#[cfg(not(any(feature = "gpu-wgpu", feature = "gpu-cuda")))]
fn benchmark_gpu_reduce(
    data: &[SampleFloat],
    op: ReduceOp,
    _batch_size: usize,
    _tile_size: usize,
) -> (f64, f64) {
    benchmark_cpu_sequential(data, op)
}

// ============================================================================
// Helpers
// ============================================================================

fn calculate_gflops(items: usize, time_s: f64) -> f64 {
    if time_s <= 0.0 {
        return 0.0;
    }
    (items.saturating_sub(1) as f64 * FLOPS_PER_ELEMENT) / time_s / 1e9
}

fn calculate_data_size_gb(items: usize) -> f64 {
    (items * SAMPLE_FLOAT_SIZE) as f64 / (1024.0 * 1024.0 * 1024.0)
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

    let operators = [
        ReduceOp::Sum,
        ReduceOp::Product,
        ReduceOp::Min,
        ReduceOp::Max,
    ];

    let gpu_enabled = cfg!(any(feature = "gpu-wgpu", feature = "gpu-cuda"));

    // ── Header ──────────────────────────────────────────────────────────────────
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║         GPU Reduce Benchmark: CPU vs GPU Comparison          ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
    println!("Configuration:");
    println!("  Platform:       {}", get_platform_info());
    println!("  CPU Workers:    {}", num_workers);
    println!("  GPU Enabled:    {}", gpu_enabled);
    println!("  GPU Batch Size: {}", format_number(GPU_BATCH_SIZE));
    println!("  GPU Tile Size:  {}", format_number(GPU_TILE_SIZE));
    println!("  Runs/Test:      {} (+ {} warmup)", num_runs, WARMUP_RUNS);
    println!(
        "  Test Sizes:     {} sizes from {} to {}",
        test_sizes.len(),
        format_number_short(*test_sizes.first().unwrap_or(&0)),
        format_number_short(*test_sizes.last().unwrap_or(&0))
    );
    println!(
        "  Operators:      {}",
        operators
            .iter()
            .map(|o| format!("{}", o))
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("  Float Type:     {} ({} bytes)", std::any::type_name::<SampleFloat>(), SAMPLE_FLOAT_SIZE);
    println!();

    let system_config = SystemConfig::detect(1, GPU_BATCH_SIZE);
    let json_path = get_benchmark_filepath(BenchmarkType::Reduce, &timestamp);
    let mut all_results: Vec<ReduceTestResult> = Vec::new();
    let mut test_id: usize = 0;

    // ── Table header ────────────────────────────────────────────────────────────
    println!(
        "{:>5} {:>5} {:>14} {:>8} {:>4} {:>16} {:>16} {:>16} {:>8} {:>8} {:>7}",
        "Test", "Op", "Items", "DataGB", "Runs", "Seq (mean±σ)", "Par (mean±σ)", "GPU (mean±σ)", "Seq/GPU", "Par/GPU", "Valid"
    );
    println!("{}", "-".repeat(130));

    for &num_items in &test_sizes {
        for &op in &operators {
            test_id += 1;
            let test_timestamp = Utc::now();

            let data = generate_data(num_items, op);
            let data_size_gb = calculate_data_size_gb(num_items);
            let expected = calculate_expected(&data, op);

            // --- Multi-run CPU Sequential ---
            let mut seq_durations = Vec::with_capacity(num_runs);
            let mut last_cpu_result = 0.0f64;
            // Warmup
            for _ in 0..WARMUP_RUNS {
                let (r, _) = benchmark_cpu_sequential(&data, op);
                last_cpu_result = r;
            }
            for _ in 0..num_runs {
                let (r, t) = benchmark_cpu_sequential(&data, op);
                seq_durations.push(t);
                last_cpu_result = r;
            }
            let seq_stats = compute_stats(&seq_durations);

            // --- Multi-run Renoir Parallel ---
            let mut par_durations = Vec::with_capacity(num_runs);
            let mut last_renoir_result = 0.0f64;
            // Warmup
            for _ in 0..WARMUP_RUNS {
                let (r, _) = benchmark_renoir_parallel(&data, op, num_workers);
                last_renoir_result = r;
            }
            for _ in 0..num_runs {
                let (r, t) = benchmark_renoir_parallel(&data, op, num_workers);
                par_durations.push(t);
                last_renoir_result = r;
            }
            let par_stats = compute_stats(&par_durations);

            // --- Multi-run GPU ---
            let mut gpu_durations = Vec::with_capacity(num_runs);
            let mut last_gpu_result = 0.0f64;
            // Warmup
            for _ in 0..WARMUP_RUNS {
                let (r, _) = benchmark_gpu_reduce(&data, op, GPU_BATCH_SIZE, GPU_TILE_SIZE);
                last_gpu_result = r;
            }
            for _ in 0..num_runs {
                let (r, t) = benchmark_gpu_reduce(&data, op, GPU_BATCH_SIZE, GPU_TILE_SIZE);
                gpu_durations.push(t);
                last_gpu_result = r;
            }
            let gpu_stats_computed = compute_stats(&gpu_durations);

            // Speedups from mean times
            let speedup_seq = if gpu_stats_computed.mean > 0.0 {
                seq_stats.mean / gpu_stats_computed.mean
            } else {
                1.0
            };
            let speedup_par = if gpu_stats_computed.mean > 0.0 {
                par_stats.mean / gpu_stats_computed.mean
            } else {
                1.0
            };

            // Validation (against GPU result when GPU is enabled)
            let (validation_passed, error_margin) = if gpu_enabled {
                validate_result(expected, last_gpu_result, op)
            } else {
                (true, 0.0)
            };

            let valid_str = if validation_passed { "  ✓  " } else { "  ✗  " };

            println!(
                "{:>5} {:>5} {:>14} {:>8.3} {:>4} {:>16} {:>16} {:>16} {:>8} {:>8} {:>7}",
                test_id,
                op,
                format_number(num_items),
                data_size_gb,
                num_runs,
                format_duration_with_stddev(seq_stats.mean, seq_stats.stddev),
                format_duration_with_stddev(par_stats.mean, par_stats.stddev),
                format_duration_with_stddev(gpu_stats_computed.mean, gpu_stats_computed.stddev),
                format_speedup(speedup_seq),
                format_speedup(speedup_par),
                valid_str
            );
            std::io::stdout().flush().ok();

            let result = ReduceTestResult {
                test_id,
                timestamp: test_timestamp.to_rfc3339(),
                operator: op,
                items_count: num_items,
                data_size_gb,
                renoir_seq_total_time_s: seq_stats.mean,
                renoir_par_total_time_s: par_stats.mean,
                gpu_total_time_s: gpu_stats_computed.mean,
                renoir_seq_gflops: calculate_gflops(num_items, seq_stats.mean),
                renoir_par_gflops: calculate_gflops(num_items, par_stats.mean),
                gpu_gflops: calculate_gflops(num_items, gpu_stats_computed.mean),
                renoir_seq_stats: Some(seq_stats),
                renoir_par_stats: Some(par_stats),
                gpu_stats: Some(gpu_stats_computed),
                speedup_seq,
                speedup_par,
                cpu_workers: num_workers,
                gpu_threads: GPU_THREADS_ESTIMATE,
                batch_size: GPU_BATCH_SIZE,
                tile_size: GPU_TILE_SIZE,
                num_runs,
                cpu_result: last_cpu_result,
                renoir_result: last_renoir_result,
                gpu_result: last_gpu_result,
                validation_passed,
                error_margin,
            };

            all_results.push(result);

            // Incremental save
            let report = ReduceBenchmarkReport {
                benchmark_type: "reduce".to_string(),
                start_time: timestamp.to_rfc3339(),
                platform: get_platform_info(),
                system_config: system_config.clone(),
                gpu_enabled,
                total_tests: test_sizes.len() * operators.len(),
                cpu_workers: num_workers,
                batch_size: GPU_BATCH_SIZE,
                tile_size: GPU_TILE_SIZE,
                num_runs,
                warmup_runs: WARMUP_RUNS,
                results: all_results.clone(),
            };
            save_json(&json_path, &report);
        }
    }

    println!("{}", "-".repeat(130));

    // ── Final save ──────────────────────────────────────────────────────────────
    let report = ReduceBenchmarkReport {
        benchmark_type: "reduce".to_string(),
        start_time: timestamp.to_rfc3339(),
        platform: get_platform_info(),
        system_config,
        gpu_enabled,
        total_tests: all_results.len(),
        cpu_workers: num_workers,
        batch_size: GPU_BATCH_SIZE,
        tile_size: GPU_TILE_SIZE,
        num_runs,
        warmup_runs: WARMUP_RUNS,
        results: all_results.clone(),
    };
    save_json(&json_path, &report);
    println!("\nResults saved to: {}", json_path.display());

    // ── Generate plots ──────────────────────────────────────────────────────────
    run_plotter(&json_path, BenchmarkType::Reduce, &timestamp);

    // ── Print summary ───────────────────────────────────────────────────────────
    print_summary(&all_results, num_workers);
}

fn print_summary(results: &[ReduceTestResult], num_workers: usize) {
    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                      BENCHMARK SUMMARY                       ║");
    println!("╚════════════════════════════════════════════════════════════════╝");

    for op in [ReduceOp::Sum, ReduceOp::Product, ReduceOp::Min, ReduceOp::Max] {
        let op_results: Vec<_> = results.iter().filter(|r| matches!((&r.operator, &op),
            (ReduceOp::Sum, ReduceOp::Sum) |
            (ReduceOp::Product, ReduceOp::Product) |
            (ReduceOp::Min, ReduceOp::Min) |
            (ReduceOp::Max, ReduceOp::Max)
        )).collect();

        if op_results.is_empty() {
            continue;
        }

        let avg_speedup_seq: f64 =
            op_results.iter().map(|r| r.speedup_seq).sum::<f64>() / op_results.len() as f64;
        let avg_speedup_par: f64 =
            op_results.iter().map(|r| r.speedup_par).sum::<f64>() / op_results.len() as f64;
        let max_speedup_seq = op_results
            .iter()
            .map(|r| r.speedup_seq)
            .fold(f64::NEG_INFINITY, f64::max);

        let gpu_wins_seq = op_results.iter().filter(|r| r.speedup_seq > 1.0).count();
        let gpu_wins_par = op_results.iter().filter(|r| r.speedup_par > 1.0).count();
        let validation_ok = op_results.iter().filter(|r| r.validation_passed).count();

        println!("\n  ── {} ──", op);
        println!("    Tests:            {}", op_results.len());
        println!("    Avg Seq/GPU:      {:.2}x", avg_speedup_seq);
        println!("    Avg Par({}w)/GPU: {:.2}x", num_workers, avg_speedup_par);
        println!("    Max Seq/GPU:      {:.2}x", max_speedup_seq);
        println!(
            "    GPU wins (seq):   {}/{} ({:.0}%)",
            gpu_wins_seq,
            op_results.len(),
            100.0 * gpu_wins_seq as f64 / op_results.len() as f64
        );
        println!(
            "    GPU wins (par):   {}/{} ({:.0}%)",
            gpu_wins_par,
            op_results.len(),
            100.0 * gpu_wins_par as f64 / op_results.len() as f64
        );
        println!(
            "    Validation OK:    {}/{}",
            validation_ok,
            op_results.len()
        );
    }

    println!("\n{}", "=".repeat(66));
}

//! # Black-Scholes GPU Optimized Benchmark
//!
//! This benchmark uses optimized methodology for fair CPU vs GPU comparison:
//! - SoA (Structure-of-Arrays) data layout (no AoS→SoA conversion overhead)
//! - Kernel-only timing (excluding data transfer)
//! - Full-size warmup runs
//! - Rayon for CPU parallel instead of Renoir
//!
//! ## Running the Benchmark
//!
//! ```bash
//! # Run with default test sizes (up to 10M options)
//! cargo bench --bench gpu_black_scholes_optimized --features gpu-wgpu
//!
//! # Run with custom maximum problem size
//! MAX_OPTIONS=100000000 cargo bench --bench gpu_black_scholes_optimized --features gpu-wgpu
//! ```
//!
//! ## Output
//!
//! Results are saved to `benches/results/optimized/{date}/optimized_benchmark_{timestamp}.json`

#[path = "common.rs"]
mod common;

#[path = "../../examples/kernels/black_scholes.rs"]
mod black_scholes_kernel;

use std::env;
use std::time::Instant;

use chrono::Utc;
use criterion::{criterion_group, criterion_main, Criterion};
use rayon::prelude::*;

use black_scholes_kernel::*;
use common::*;
use renoir::operator::gpu::GpuContext;
use renoir::utils::{create_banner, TableBuilder};

// ============================================================================
// Constants
// ============================================================================

/// Maximum batch size for GPU processing to avoid memory issues.
/// Datasets larger than this will be processed in chunks.
const GPU_MAX_BATCH_SIZE: usize = 50_000_000;

// ============================================================================
// Optimized Benchmark Implementation
// ============================================================================

/// Run an optimized benchmark with SoA data and kernel-only timing.
fn run_optimized_benchmark(
    test_id: usize,
    num_options: usize,
    gpu_ctx: &GpuContext,
) -> BenchmarkResult {
    let timestamp = Utc::now();

    // Generate SoA data (no AoS overhead)
    let mut soa_data = BlackScholesSoA::generate(num_options);
    soa_data.pad(4);

    // CPU Sequential
    let cpu_start = Instant::now();
    let mut cpu_calls = Vec::with_capacity(num_options);
    for i in 0..num_options {
        let s = soa_data.stocks[i];
        let k = soa_data.strikes[i];
        let t = soa_data.times[i];
        let r = soa_data.rates[i];
        let v = soa_data.vols[i];

        let sqrt_t = t.sqrt();
        let v_sqrt_t = v * sqrt_t;
        let d1 = ((s / k).ln() + (r + 0.5 * v * v) * t) / v_sqrt_t;
        let d2 = d1 - v_sqrt_t;
        let cnd_d1 = 0.5 * (1.0 + libm::erff(d1 / std::f32::consts::SQRT_2));
        let cnd_d2 = 0.5 * (1.0 + libm::erff(d2 / std::f32::consts::SQRT_2));
        let exp_rt = (-r * t).exp();
        let call = s * cnd_d1 - k * exp_rt * cnd_d2;
        cpu_calls.push(call);
    }
    let cpu_time = cpu_start.elapsed().as_secs_f64();

    // CPU Parallel (Rayon)
    let par_start = Instant::now();
    let _par_calls: Vec<f32> = (0..num_options)
        .into_par_iter()
        .map(|i| {
            let s = soa_data.stocks[i];
            let k = soa_data.strikes[i];
            let t = soa_data.times[i];
            let r = soa_data.rates[i];
            let v = soa_data.vols[i];

            let sqrt_t = t.sqrt();
            let v_sqrt_t = v * sqrt_t;
            let d1 = ((s / k).ln() + (r + 0.5 * v * v) * t) / v_sqrt_t;
            let d2 = d1 - v_sqrt_t;
            let cnd_d1 = 0.5 * (1.0 + libm::erff(d1 / std::f32::consts::SQRT_2));
            let cnd_d2 = 0.5 * (1.0 + libm::erff(d2 / std::f32::consts::SQRT_2));
            let exp_rt = (-r * t).exp();
            s * cnd_d1 - k * exp_rt * cnd_d2
        })
        .collect();
    let par_time = par_start.elapsed().as_secs_f64();

    // Optimized GPU (kernel-only timing) - use batching for large datasets
    let (kernel_time, gpu_calls, gpu_threads) = if num_options > GPU_MAX_BATCH_SIZE {
        // Use batched processing for large datasets
        let (calls, _puts, pipeline_result, threads) =
            process_pipelined_soa(gpu_ctx, &soa_data, num_options, GPU_MAX_BATCH_SIZE);
        (pipeline_result.kernel_time_s, calls, threads)
    } else {
        // Single batch for smaller datasets
        let (kernel_time, _full_time, calls, threads) =
            run_optimized_gpu_benchmark(gpu_ctx, &soa_data, num_options);
        (kernel_time, calls, threads)
    };

    let gpu_time = kernel_time;

    // Metrics
    let data_size_gb =
        (num_options * 5 * std::mem::size_of::<f32>()) as f64 / (1024.0 * 1024.0 * 1024.0);
    let speedup = cpu_time / gpu_time;

    let flops_per_option = 40.0;
    let total_flops = num_options as f64 * flops_per_option;
    let cpu_gflops = total_flops / cpu_time / 1e9;
    let gpu_gflops = total_flops / gpu_time / 1e9;
    let renoir_gflops = total_flops / par_time / 1e9;

    // Validation
    let cpu_sample = cpu_calls.first().copied().unwrap_or(0.0) as f64;
    let gpu_sample = gpu_calls.first().copied().unwrap_or(0.0) as f64;
    let error_margin = (cpu_sample - gpu_sample).abs();
    let validation_passed =
        error_margin < 0.001 || (cpu_sample.abs() > 0.0 && error_margin / cpu_sample.abs() < 0.001);

    BenchmarkResult {
        test_id,
        timestamp,
        operator: "black_scholes_optimized".to_string(),
        items_count: num_options,
        data_size_gb,
        cpu_total_time_s: cpu_time,
        gpu_total_time_s: gpu_time,
        renoir_total_time_s: par_time,
        speedup,
        cpu_gflops,
        gpu_gflops,
        renoir_gflops,
        cpu_workers: rayon::current_num_threads(),
        gpu_threads,
        batch_size: 256,
        cpu_result: cpu_sample,
        gpu_result: gpu_sample,
        validation_passed,
        error_margin,
    }
}

// ============================================================================
// Main Benchmark
// ============================================================================

fn black_scholes_optimized_bench(c: &mut Criterion) {
    let max_options = get_max_options();
    let test_sizes = get_benchmark_test_sizes();
    let total_tests = test_sizes.len();
    let start_time = Utc::now();

    let platform = get_platform_info();
    let filename = get_benchmark_filepath(BenchmarkType::Optimized, &start_time);

    print!(
        "{}",
        create_banner(
            &format!("Black-Scholes {} Benchmark", BenchmarkType::Optimized.display_name()),
            &[
                &format!("Platform:    {}", platform),
                &format!("Mode:        Kernel-only timing, SoA data layout"),
                &format!("Max options: {}", max_options),
                &format!("Test sizes:  {}", total_tests),
                &format!("Output:      {}", filename.display()),
            ]
        )
    );

    println!("Methodology:");
    println!("  - Data: Structure-of-Arrays (no AoS→SoA conversion)");
    println!("  - Timing: Kernel execution only (excludes data transfer)");
    println!("  - Warmup: Full-size warmup with same data");
    println!("  - CPU Parallel: Rayon\n");

    // Initialize GPU
    println!("Initializing GPU...");
    let gpu_ctx = GpuContext::new();
    println!("GPU ready.\n");


    let mut results = Vec::with_capacity(total_tests);

    // Table for results (includes Rayon parallel for comparison)
    let table = TableBuilder::new(&[
        ("Test", 6),
        ("     Options     ", 18),
        (" CPU Seq ", 11),
        (" Rayon Par", 12),
        (" GPU Kernel", 12),
        ("GPU vs Seq", 12),
        ("GPU vs Par", 12),
        ("Valid", 7),
    ]);
    print!("{}", table.header());

    for (idx, &num_options) in test_sizes.iter().enumerate() {
        let test_id = idx + 1;

        let result = run_optimized_benchmark(test_id, num_options, &gpu_ctx);

        let valid_str = if result.validation_passed {
            "Yes"
        } else {
            "NO!"
        };

        // GPU vs Rayon parallel speedup
        let gpu_vs_par = result.renoir_total_time_s / result.gpu_total_time_s;

        print!(
            "{}",
            table.row(&[
                &format!(" {:>3} ", test_id),
                &format!(" {:>16} ", format_number(num_options)),
                &format!(" {:>7.4}s", result.cpu_total_time_s),
                &format!("  {:>7.4}s ", result.renoir_total_time_s),
                &format!("  {:>7.4}s ", result.gpu_total_time_s),
                &format!("   {:>6.1}x ", result.speedup),
                &format!("   {:>6.1}x ", gpu_vs_par),
                &format!(" {:>3} ", valid_str),
            ])
        );

        results.push(result);

        // Incremental save
        let report = BenchmarkReport {
            benchmark_type: BenchmarkType::Optimized.folder_name().to_string(),
            start_time,
            platform: platform.clone(),
            gpu_enabled: true,
            total_tests,
            results: results.clone(),
            is_streaming: false,
            streaming_config: None,
        };
        save_json(&filename, &report);
    }

    print!("{}", table.footer());

    // Summary
    let speedups_vs_seq: Vec<f64> = results.iter().map(|r| r.speedup).collect();
    let speedups_vs_par: Vec<f64> = results.iter().map(|r| r.renoir_total_time_s / r.gpu_total_time_s).collect();
    
    let avg_speedup_seq = speedups_vs_seq.iter().sum::<f64>() / speedups_vs_seq.len() as f64;
    let max_speedup_seq = speedups_vs_seq.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let gpu_wins_vs_seq = speedups_vs_seq.iter().filter(|&&s| s > 1.0).count();
    
    let avg_speedup_par = speedups_vs_par.iter().sum::<f64>() / speedups_vs_par.len() as f64;
    let max_speedup_par = speedups_vs_par.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let gpu_wins_vs_par = speedups_vs_par.iter().filter(|&&s| s > 1.0).count();

    print!(
        "{}",
        create_banner(
            "OPTIMIZED BENCHMARK SUMMARY",
            &[
                &format!("Total tests:                {:>6}", total_tests),
                &format!(""),
                &format!("GPU vs CPU Sequential:"),
                &format!("  Wins:                     {:>6} ({:.1}%)", gpu_wins_vs_seq, 100.0 * gpu_wins_vs_seq as f64 / total_tests as f64),
                &format!("  Avg speedup:              {:>6.1}x", avg_speedup_seq),
                &format!("  Max speedup:              {:>6.1}x", max_speedup_seq),
                &format!(""),
                &format!("GPU vs Rayon Parallel ({} workers):", results.first().map(|r| r.cpu_workers).unwrap_or(0)),
                &format!("  Wins:                     {:>6} ({:.1}%)", gpu_wins_vs_par, 100.0 * gpu_wins_vs_par as f64 / total_tests as f64),
                &format!("  Avg speedup:              {:>6.1}x", avg_speedup_par),
                &format!("  Max speedup:              {:>6.1}x", max_speedup_par),
            ]
        )
    );

    println!("Results saved to: {}", filename.display());

    // Generate visualization
    run_plotter(&filename, BenchmarkType::Optimized, &start_time);

    // Minimal criterion benchmark to satisfy harness
    let mut group = c.benchmark_group("optimized-placeholder");
    group.sample_size(10);
    group.bench_function("optimized-complete", |b| {
        b.iter(|| std::hint::black_box(42))
    });
    group.finish();
}

criterion_group!(benches, black_scholes_optimized_bench);
criterion_main!(benches);


//! # Black-Scholes GPU vs CPU Standard Benchmark
//!
//! This benchmark compares the performance of Black-Scholes option pricing across:
//! - **Renoir CPU Sequential**: Single-threaded processing using the `map` operator
//! - **Renoir CPU Parallel**: Multi-threaded processing with shuffled workload distribution
//! - **Renoir GPU**: GPU-accelerated processing using the `map_gpu` operator
//!
//! ## Running the Benchmark
//!
//! ```bash
//! # Run with default test sizes (up to 10M options)
//! cargo bench --bench gpu_black_scholes_standard --features gpu-wgpu
//!
//! # Run with custom maximum problem size (e.g., 100M options)
//! MAX_OPTIONS=100000000 cargo bench --bench gpu_black_scholes_standard --features gpu-wgpu
//! ```
//!
//! ## Output
//!
//! Results are saved to `benches/results/standard/{date}/standard_benchmark_{timestamp}.json`

#[path = "common.rs"]
mod common;

#[path = "../../examples/kernels/black_scholes.rs"]
mod black_scholes_kernel;

use std::env;
use std::time::{Duration, Instant};

use chrono::Utc;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use black_scholes_kernel::*;
use common::*;
use renoir::config::RuntimeConfig;
use renoir::operator::gpu::GpuContext;
use renoir::prelude::*;
use renoir::utils::{create_banner, TableBuilder};

// ============================================================================
// Benchmark Implementation
// ============================================================================

/// Run a single benchmark test for a given problem size.
fn run_single_benchmark(
    test_id: usize,
    num_options: usize,
    options: Vec<BlackScholesInput>,
    gpu_ctx: &GpuContext,
    kernel: &BlackScholesKernel,
    cpu_workers: usize,
) -> BenchmarkResult {
    let timestamp = Utc::now();

    // CPU Sequential
    let cpu_start = Instant::now();
    let env = StreamContext::new_local();
    let cpu_result = env
        .stream_iter(options.clone().into_iter())
        .map(black_scholes_cpu)
        .collect_vec();
    env.execute_blocking();
    let cpu_time = cpu_start.elapsed().as_secs_f64();
    let cpu_results = cpu_result.get().unwrap();

    // CPU Parallel (dynamic workers)
    let par_start = Instant::now();
    let config = RuntimeConfig::local(cpu_workers as u64).unwrap();
    let env = StreamContext::new(config);
    let _par_result = env
        .stream_iter(options.clone().into_iter())
        .shuffle()
        .map(black_scholes_cpu)
        .collect_vec();
    env.execute_blocking();
    let par_time = par_start.elapsed().as_secs_f64();

    // GPU - use pipelining for large datasets (>50M options)
    let gpu_start = Instant::now();
    let (gpu_results, _pipeline_info) = if num_options > PIPELINE_CHUNK_SIZE {
        // Use pipelined processing for large datasets
        process_pipelined(gpu_ctx, kernel, &options, PIPELINE_CHUNK_SIZE)
    } else {
        // Single batch for smaller datasets
        let results = kernel.execute(gpu_ctx, &options);
        let time = gpu_start.elapsed().as_secs_f64();
        (
            results,
            PipelineResult {
                total_time_s: time,
                kernel_time_s: time,
                items_processed: num_options,
                chunks_processed: 1,
            },
        )
    };
    let gpu_time = gpu_start.elapsed().as_secs_f64();

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
    let cpu_sample = cpu_results
        .first()
        .map(|r| r.call_price as f64)
        .unwrap_or(0.0);
    let gpu_sample = gpu_results
        .first()
        .map(|r| r.call_price as f64)
        .unwrap_or(0.0);
    let error_margin = (cpu_sample - gpu_sample).abs();
    let validation_passed = error_margin < 0.001 || error_margin / cpu_sample.abs() < 0.001;

    // GPU threads
    let cube_dim = 256u32;
    let vectorization_factor = 4usize;
    let num_elements_padded = num_options
        + (vectorization_factor - (num_options % vectorization_factor)) % vectorization_factor;
    let total_lines = num_elements_padded / vectorization_factor;
    let num_cubes = (total_lines as u32 + cube_dim - 1) / cube_dim;
    let gpu_threads = num_cubes * cube_dim;

    BenchmarkResult {
        test_id,
        timestamp,
        operator: "black_scholes".to_string(),
        items_count: num_options,
        data_size_gb,
        cpu_total_time_s: cpu_time,
        gpu_total_time_s: gpu_time,
        renoir_total_time_s: par_time,
        speedup,
        cpu_gflops,
        gpu_gflops,
        renoir_gflops,
        cpu_workers,
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

fn black_scholes_standard_bench(c: &mut Criterion) {
    let max_options = get_max_options();
    let test_sizes = get_benchmark_test_sizes();

    // Use all available cores by default for a fair comparison
    let cpu_workers: usize = env::var("CPU_WORKERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(8)
        });
    let total_tests = test_sizes.len();
    let start_time = Utc::now();

    let platform = get_platform_info();
    let filename = get_benchmark_filepath(BenchmarkType::Standard, &start_time);

    print!(
        "{}",
        create_banner(
            &format!("Black-Scholes {} Benchmark: CPU vs GPU", BenchmarkType::Standard.display_name()),
            &[
                &format!("Platform:    {}", platform),
                &format!("Max options: {} ({})", format_number(max_options), format_number_short(max_options)),
                &format!("CPU workers: {}", cpu_workers),
                &format!("Test sizes:  {}", total_tests),
                &format!("Output:      {}", filename.display()),
            ]
        )
    );

    // Initialize GPU
    println!("Initializing GPU and warming up...");
    let gpu_ctx = GpuContext::new();
    let mut kernel = BlackScholesKernel::default();
    kernel.setup(&gpu_ctx);
    println!("GPU ready.\n");


    let mut results = Vec::with_capacity(total_tests);

    // Table for results (includes CPU parallel for comparison)
    let table = TableBuilder::new(&[
        ("Test", 6),
        ("     Options     ", 18),
        (" CPU Seq ", 11),
        (" CPU Par ", 11),
        ("   GPU   ", 11),
        ("GPU vs Seq", 12),
        ("GPU vs Par", 12),
        ("Valid", 7),
    ]);
    print!("{}", table.header());

    for (idx, &num_options) in test_sizes.iter().enumerate() {
        let test_id = idx + 1;
        let options = generate_options(num_options);
        let result = run_single_benchmark(test_id, num_options, options, &gpu_ctx, &kernel, cpu_workers);

        let valid_str = if result.validation_passed {
            "Yes"
        } else {
            "NO!"
        };

        // Calculate GPU vs CPU parallel speedup
        let gpu_vs_par = result.renoir_total_time_s / result.gpu_total_time_s;

        print!(
            "{}",
            table.row(&[
                &format!(" {:>3} ", test_id),
                &format!(" {:>16} ", format_number(num_options)),
                &format!(" {:>7.4}s", result.cpu_total_time_s),
                &format!(" {:>7.4}s", result.renoir_total_time_s),
                &format!(" {:>7.4}s", result.gpu_total_time_s),
                &format!("   {:>6.2}x ", result.speedup),
                &format!("   {:>6.2}x ", gpu_vs_par),
                &format!(" {:>3} ", valid_str),
            ])
        );

        results.push(result);

        // Incremental save
        let report = BenchmarkReport {
            benchmark_type: BenchmarkType::Standard.folder_name().to_string(),
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
            "STANDARD BENCHMARK SUMMARY",
            &[
                &format!("Total tests:                {:>6}", total_tests),
                &format!(""),
                &format!("GPU vs CPU Sequential:"),
                &format!("  Wins:                     {:>6} ({:.1}%)", gpu_wins_vs_seq, 100.0 * gpu_wins_vs_seq as f64 / total_tests as f64),
                &format!("  Avg speedup:              {:>6.1}x", avg_speedup_seq),
                &format!("  Max speedup:              {:>6.1}x", max_speedup_seq),
                &format!(""),
                &format!("GPU vs CPU Parallel ({} workers):", cpu_workers),
                &format!("  Wins:                     {:>6} ({:.1}%)", gpu_wins_vs_par, 100.0 * gpu_wins_vs_par as f64 / total_tests as f64),
                &format!("  Avg speedup:              {:>6.1}x", avg_speedup_par),
                &format!("  Max speedup:              {:>6.1}x", max_speedup_par),
            ]
        )
    );

    println!("Results saved to: {}", filename.display());

    // Generate visualization
    run_plotter(&filename, BenchmarkType::Standard, &start_time);

    // Criterion benchmarks
    let mut group = c.benchmark_group("black-scholes");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    let criterion_sizes: Vec<usize> = test_sizes
        .iter()
        .filter(|&&s| s <= 1_000_000)
        .copied()
        .collect();

    for &num_options in &criterion_sizes {
        let options = generate_options(num_options);
        group.throughput(Throughput::Elements(num_options as u64));

        group.bench_with_input(
            BenchmarkId::new("renoir-cpu-sequential", num_options),
            &options,
            |b, options| {
                b.iter(|| {
                    let env = StreamContext::new_local();
                    let result = env
                        .stream_iter(options.clone().into_iter())
                        .map(black_scholes_cpu)
                        .collect_vec();
                    env.execute_blocking();
                    std::hint::black_box(result.get())
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("renoir-cpu-parallel-{}", cpu_workers), num_options),
            &options,
            |b, options| {
                b.iter(|| {
                    let config = RuntimeConfig::local(cpu_workers as u64).unwrap();
                    let env = StreamContext::new(config);
                    let result = env
                        .stream_iter(options.clone().into_iter())
                        .shuffle()
                        .map(black_scholes_cpu)
                        .collect_vec();
                    env.execute_blocking();
                    std::hint::black_box(result.get())
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("direct-gpu", num_options),
            &options,
            |b, options| b.iter(|| std::hint::black_box(kernel.execute(&gpu_ctx, options))),
        );
    }

    group.finish();
}

criterion_group!(benches, black_scholes_standard_bench);
criterion_main!(benches);

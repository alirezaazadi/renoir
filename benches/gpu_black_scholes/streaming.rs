//! # Black-Scholes GPU Streaming Benchmark
//!
//! This benchmark compares 4 processing strategies:
//! - **CPU Sequential**: Single-threaded CPU baseline
//! - **GPU Sequential**: Single producer with double-buffered GPU pipeline
//! - **CPU Parallel**: Multi-threaded CPU with Rayon
//! - **GPU Parallel**: Multi-worker producers with double-buffered GPU pipeline
//!
//! ## Key Optimizations
//!
//! ### Fixed Batch Sizing (10M)
//! The benchmark uses a fixed batch size of 10 million items, which provides
//! excellent GPU utilization while staying well below the 50M memory cliff.
//!
//! ### Double-Buffered Pipeline
//! Data generation is overlapped with GPU execution using a bounded channel:
//! - Producer threads generate batches ahead of time
//! - GPU processes batches from the channel while new ones are being prepared
//! - This hides data generation latency behind GPU compute time
//!
//! ## Running the Benchmark
//!
//! ```bash
//! # Run with default stream size (1B items)
//! cargo bench --bench gpu_black_scholes_streaming --features gpu-wgpu
//!
//! # Run with custom stream size
//! MAX_OPTIONS=100000000 cargo bench --bench gpu_black_scholes_streaming --features gpu-wgpu
//!
//! # Customize CPU workers and batch size
//! CPU_WORKERS=8 BATCH_SIZE=20000000 cargo bench --bench gpu_black_scholes_streaming --features gpu-wgpu
//! ```
//!
//! ## Output
//!
//! Results are saved to `benches/results/streaming/{date}/streaming_benchmark_{timestamp}.json`

#[path = "common.rs"]
mod common;

#[path = "../../examples/kernels/black_scholes.rs"]
mod black_scholes_kernel;

use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use criterion::{criterion_group, criterion_main, Criterion};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use black_scholes_kernel::*;
use common::*;
use renoir::operator::gpu::GpuContext;
use renoir::utils::{create_banner, TableBuilder};

// ============================================================================
// Configuration Constants
// ============================================================================

/// Maximum GPU batch size to avoid memory issues (50M cliff).
const GPU_MAX_BATCH: usize = 50_000_000;

/// Default fixed batch size for GPU processing.
const DEFAULT_BATCH_SIZE: usize = 10_000_000;

/// Number of batches to buffer ahead in the pipeline.
const BUFFER_AHEAD: usize = 2;

// ============================================================================
// Extended Benchmark Result for Streaming
// ============================================================================

/// Extended result for streaming benchmark with all 4 strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingResult {
    pub test_id: usize,
    pub timestamp: DateTime<Utc>,
    pub items_count: usize,
    pub data_size_gb: f64,
    
    // Times for all 4 strategies
    pub cpu_seq_time_s: f64,
    pub gpu_seq_time_s: f64,  // GPU with double-buffer, single producer
    pub cpu_par_time_s: f64,
    pub gpu_par_time_s: f64,  // GPU with double-buffer, multi-worker producers
    
    // Speedups
    pub gpu_seq_vs_cpu_seq: f64,
    pub gpu_par_vs_cpu_par: f64,
    pub gpu_par_vs_cpu_seq: f64,
    
    // Throughput (GFLOPS)
    pub cpu_seq_gflops: f64,
    pub gpu_seq_gflops: f64,
    pub cpu_par_gflops: f64,
    pub gpu_par_gflops: f64,
    
    pub cpu_workers: usize,
    pub batch_size: usize,
    
    pub validation_passed: bool,
}

/// Report container for streaming benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingReport {
    pub benchmark_type: String,
    pub start_time: DateTime<Utc>,
    pub platform: String,
    pub total_tests: usize,
    pub cpu_workers: usize,
    pub batch_size: usize,
    pub buffer_ahead: usize,
    pub results: Vec<StreamingResult>,
}

// ============================================================================
// Strategy 1: CPU Sequential
// ============================================================================

/// Run CPU sequential benchmark - single-threaded baseline.
fn run_cpu_sequential(total_items: usize) -> (Duration, usize) {
    let start = Instant::now();
    
    let options = generate_options(total_items);
    let results: Vec<_> = options.iter().map(|opt| black_scholes_cpu(*opt)).collect();
    
    (start.elapsed(), results.len())
}

// ============================================================================
// Strategy 2: GPU Sequential (Double-Buffered)
// ============================================================================

/// Run GPU sequential with double-buffered pipeline.
/// Single producer thread generates batches while GPU processes.
fn run_gpu_sequential(total_items: usize, batch_size: usize) -> (Duration, usize) {
    let effective_batch = batch_size.min(GPU_MAX_BATCH);
    
    let start = Instant::now();
    
    // Channel for pre-generated batches
    let (tx, rx) = mpsc::sync_channel::<BlackScholesSoA>(BUFFER_AHEAD);
    
    // Producer thread - generates batches ahead of time
    let producer = {
        let total = total_items;
        let batch = effective_batch;
        thread::spawn(move || {
            let mut remaining = total;
            let mut offset = 0usize;
            
            while remaining > 0 {
                let size = remaining.min(batch);
                let soa = generate_soa_with_seed(size, offset as u64);
                
                if tx.send(soa).is_err() {
                    break;
                }
                remaining -= size;
                offset += size;
            }
        })
    };
    
    // Consumer - processes on GPU
    let ctx = GpuContext::new();
    let mut total_processed = 0;
    
    while let Ok(batch_soa) = rx.recv() {
        let batch_len = batch_soa.stocks.len();
        let (_, _, results, _) = run_optimized_gpu_benchmark(&ctx, &batch_soa, batch_len);
        total_processed += results.len();
    }
    
    producer.join().unwrap();
    
    (start.elapsed(), total_processed)
}

// ============================================================================
// Strategy 3: CPU Parallel
// ============================================================================

/// Run CPU parallel benchmark - multi-threaded with Rayon.
fn run_cpu_parallel(total_items: usize) -> (Duration, usize) {
    let start = Instant::now();
    
    let options = generate_options(total_items);
    let results: Vec<_> = options.par_iter().map(|opt| black_scholes_cpu(*opt)).collect();
    
    (start.elapsed(), results.len())
}

// ============================================================================
// Strategy 4: GPU Parallel (Multi-Worker + Double-Buffered)
// ============================================================================

/// Run GPU parallel with multi-worker producers and double-buffered pipeline.
/// Multiple producer threads generate batches in parallel while GPU processes.
fn run_gpu_parallel(
    total_items: usize,
    batch_size: usize,
    num_workers: usize,
) -> (Duration, usize) {
    let effective_batch = batch_size.min(GPU_MAX_BATCH);
    let num_batches = (total_items + effective_batch - 1) / effective_batch;
    
    let start = Instant::now();
    
    // Bounded channel - buffer enough batches to keep GPU fed
    let buffer_size = (num_workers + 1).min(BUFFER_AHEAD * 2);
    let (tx, rx) = mpsc::sync_channel::<BlackScholesSoA>(buffer_size);
    
    // Atomic counter for batch distribution among workers
    let batch_counter = Arc::new(AtomicUsize::new(0));
    
    // Spawn producer threads
    let producers: Vec<_> = (0..num_workers)
        .map(|worker_id| {
            let tx = tx.clone();
            let counter = Arc::clone(&batch_counter);
            let total = total_items;
            let b_size = effective_batch;
            let n_batches = num_batches;
            
            thread::spawn(move || {
                loop {
                    // Atomically claim the next batch
                    let batch_idx = counter.fetch_add(1, Ordering::SeqCst);
                    if batch_idx >= n_batches {
                        break;
                    }
                    
                    // Calculate batch boundaries
                    let start_idx = batch_idx * b_size;
                    let end_idx = (start_idx + b_size).min(total);
                    let size = end_idx - start_idx;
                    
                    if size == 0 {
                        break;
                    }
                    
                    // Generate batch with unique seed
                    let seed = (worker_id as u64 * 1_000_000 + batch_idx as u64) ^ 0xDEADBEEF;
                    let soa = generate_soa_with_seed(size, seed);
                    
                    // Send to channel (blocks if buffer full)
                    if tx.send(soa).is_err() {
                        break;
                    }
                }
            })
        })
        .collect();
    
    // Drop the original sender so channel closes when all producers finish
    drop(tx);
    
    // Consumer - processes on GPU
    let ctx = GpuContext::new();
    let mut total_processed = 0;
    
    while let Ok(batch_soa) = rx.recv() {
        let batch_len = batch_soa.stocks.len();
        let (_, _, results, _) = run_optimized_gpu_benchmark(&ctx, &batch_soa, batch_len);
        total_processed += results.len();
    }
    
    // Wait for all producers to finish
    for producer in producers {
        let _ = producer.join();
    }
    
    (start.elapsed(), total_processed)
}

// ============================================================================
// Single Benchmark Run
// ============================================================================

/// Run a single benchmark test with all 4 strategies.
fn run_single_benchmark(
    test_id: usize,
    num_items: usize,
    cpu_workers: usize,
    batch_size: usize,
) -> StreamingResult {
    let timestamp = Utc::now();
    
    // Strategy 1: CPU Sequential
    let (cpu_seq_duration, cpu_seq_processed) = run_cpu_sequential(num_items);
    let cpu_seq_time = cpu_seq_duration.as_secs_f64();
    
    // Strategy 2: GPU Sequential (Double-Buffered)
    let (gpu_seq_duration, gpu_seq_processed) = run_gpu_sequential(num_items, batch_size);
    let gpu_seq_time = gpu_seq_duration.as_secs_f64();
    
    // Strategy 3: CPU Parallel
    let (cpu_par_duration, cpu_par_processed) = run_cpu_parallel(num_items);
    let cpu_par_time = cpu_par_duration.as_secs_f64();
    
    // Strategy 4: GPU Parallel (Multi-Worker + Double-Buffered)
    let (gpu_par_duration, gpu_par_processed) = run_gpu_parallel(num_items, batch_size, cpu_workers);
    let gpu_par_time = gpu_par_duration.as_secs_f64();
    
    // Calculate metrics
    let data_size_gb = (num_items * 5 * std::mem::size_of::<f32>()) as f64 / (1024.0 * 1024.0 * 1024.0);
    let flops_per_option = 40.0;
    
    let validation_passed = cpu_seq_processed == num_items
        && gpu_seq_processed == num_items
        && cpu_par_processed == num_items
        && gpu_par_processed == num_items;
    
    StreamingResult {
        test_id,
        timestamp,
        items_count: num_items,
        data_size_gb,
        
        cpu_seq_time_s: cpu_seq_time,
        gpu_seq_time_s: gpu_seq_time,
        cpu_par_time_s: cpu_par_time,
        gpu_par_time_s: gpu_par_time,
        
        gpu_seq_vs_cpu_seq: cpu_seq_time / gpu_seq_time,
        gpu_par_vs_cpu_par: cpu_par_time / gpu_par_time,
        gpu_par_vs_cpu_seq: cpu_seq_time / gpu_par_time,
        
        cpu_seq_gflops: num_items as f64 * flops_per_option / cpu_seq_time / 1e9,
        gpu_seq_gflops: num_items as f64 * flops_per_option / gpu_seq_time / 1e9,
        cpu_par_gflops: num_items as f64 * flops_per_option / cpu_par_time / 1e9,
        gpu_par_gflops: num_items as f64 * flops_per_option / gpu_par_time / 1e9,
        
        cpu_workers,
        batch_size,
        validation_passed,
    }
}

// ============================================================================
// Main Benchmark
// ============================================================================

fn black_scholes_streaming_bench(c: &mut Criterion) {
    let max_items = get_max_options();
    let test_sizes = get_benchmark_test_sizes();
    
    // Get CPU workers from env or use all available cores
    let cpu_workers: usize = env::var("CPU_WORKERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(8)
        });
    
    // Get batch size from env or use default (10M)
    let batch_size: usize = env::var("BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_BATCH_SIZE);
    
    let total_tests = test_sizes.len();
    let start_time = Utc::now();
    
    let platform = get_platform_info();
    let filename = get_benchmark_filepath(BenchmarkType::Streaming, &start_time);
    
    print!(
        "{}",
        create_banner(
            "Black-Scholes Streaming Benchmark",
            &[
                &format!("Platform:        {}", platform),
                &format!("Max items:       {} ({})", format_number(max_items), format_number_short(max_items)),
                &format!("Batch size:      {} ({})", format_number(batch_size), format_number_short(batch_size)),
                &format!("Buffer ahead:    {} batches", BUFFER_AHEAD),
                &format!("CPU workers:     {}", cpu_workers),
                &format!("Test sizes:      {}", total_tests),
                &format!("Output:          {}", filename.display()),
            ]
        )
    );
    
    println!("\nStrategies being compared:");
    println!("  1. CPU Sequential:     Single-threaded CPU baseline");
    println!("  2. GPU Seq+DblBuf:     Single producer, double-buffered GPU");
    println!("  3. CPU Parallel:       Multi-threaded CPU with Rayon ({} threads)", cpu_workers);
    println!("  4. GPU Par+DblBuf:     Multi-worker producers, double-buffered GPU\n");
    
    let mut results: Vec<StreamingResult> = Vec::with_capacity(total_tests);
    
    // Table for results - match standard benchmark format
    let table = TableBuilder::new(&[
        ("Test", 6),
        ("     Options     ", 18),
        (" CPU Seq ", 11),
        (" GPU Seq ", 11),
        (" CPU Par ", 11),
        (" GPU Par ", 11),
        ("Seq Spd", 9),
        ("Par Spd", 9),
        ("Valid", 7),
    ]);
    print!("{}", table.header());
    
    for (idx, &num_items) in test_sizes.iter().enumerate() {
        let test_id = idx + 1;
        
        let result = run_single_benchmark(test_id, num_items, cpu_workers, batch_size);
        
        let valid_str = if result.validation_passed { "Yes" } else { "NO!" };
        
        print!(
            "{}",
            table.row(&[
                &format!(" {:>3} ", test_id),
                &format!(" {:>16} ", format_number(num_items)),
                &format!(" {:>7.4}s", result.cpu_seq_time_s),
                &format!(" {:>7.4}s", result.gpu_seq_time_s),
                &format!(" {:>7.4}s", result.cpu_par_time_s),
                &format!(" {:>7.4}s", result.gpu_par_time_s),
                &format!("  {:>5.2}x", result.gpu_seq_vs_cpu_seq),
                &format!("  {:>5.2}x", result.gpu_par_vs_cpu_par),
                &format!(" {:>3} ", valid_str),
            ])
        );
        
        results.push(result.clone());
        
        // Incremental save
        save_json(
            &filename,
            &StreamingReport {
                benchmark_type: "streaming".to_string(),
                start_time,
                platform: platform.clone(),
                total_tests,
                cpu_workers,
                batch_size,
                buffer_ahead: BUFFER_AHEAD,
                results: results.clone(),
            },
        );
    }
    
    print!("{}", table.footer());
    
    // Summary statistics
    let gpu_seq_wins = results.iter().filter(|r| r.gpu_seq_vs_cpu_seq > 1.0).count();
    let gpu_par_wins = results.iter().filter(|r| r.gpu_par_vs_cpu_par > 1.0).count();
    
    let avg_gpu_seq_speedup: f64 = results.iter().map(|r| r.gpu_seq_vs_cpu_seq).sum::<f64>() / total_tests as f64;
    let avg_gpu_par_speedup: f64 = results.iter().map(|r| r.gpu_par_vs_cpu_par).sum::<f64>() / total_tests as f64;
    
    let max_gpu_seq_speedup = results.iter().map(|r| r.gpu_seq_vs_cpu_seq).fold(f64::NEG_INFINITY, f64::max);
    let max_gpu_par_speedup = results.iter().map(|r| r.gpu_par_vs_cpu_par).fold(f64::NEG_INFINITY, f64::max);
    
    print!(
        "{}",
        create_banner(
            "STREAMING BENCHMARK SUMMARY",
            &[
                &format!("Total tests:                  {:>6}", total_tests),
                &format!("Batch size:                   {}", format_number_short(batch_size)),
                &format!("Buffer ahead:                 {} batches", BUFFER_AHEAD),
                &format!(""),
                &format!("GPU Sequential vs CPU Sequential:"),
                &format!("  Wins:                       {:>6} ({:.1}%)", gpu_seq_wins, 100.0 * gpu_seq_wins as f64 / total_tests as f64),
                &format!("  Avg speedup:                {:>6.2}x", avg_gpu_seq_speedup),
                &format!("  Max speedup:                {:>6.2}x", max_gpu_seq_speedup),
                &format!(""),
                &format!("GPU Parallel vs CPU Parallel:"),
                &format!("  Wins:                       {:>6} ({:.1}%)", gpu_par_wins, 100.0 * gpu_par_wins as f64 / total_tests as f64),
                &format!("  Avg speedup:                {:>6.2}x", avg_gpu_par_speedup),
                &format!("  Max speedup:                {:>6.2}x", max_gpu_par_speedup),
                &format!(""),
                &format!("Recommendations:"),
                &format!("  Small inputs (<1M):         CPU Parallel"),
                &format!("  Large inputs (>10M):        GPU Par+DblBuf"),
            ]
        )
    );
    
    println!("Results saved to: {}", filename.display());
    
    // Generate visualization
    run_plotter(&filename, BenchmarkType::Streaming, &start_time);
    
    // Minimal criterion benchmark to satisfy harness
    let mut group = c.benchmark_group("streaming-placeholder");
    group.sample_size(10);
    group.bench_function("streaming-complete", |b| {
        b.iter(|| std::hint::black_box(42))
    });
    group.finish();
}

criterion_group!(benches, black_scholes_streaming_bench);
criterion_main!(benches);
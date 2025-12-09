//! # Black-Scholes GPU Streaming Example
//!
//! This example demonstrates advanced GPU streaming techniques for high-throughput
//! data processing, combining:
//!
//! - **Double-Buffered Pipeline**: Overlaps data generation with GPU execution
//! - **Multi-Worker Data Generation**: Parallel CPU threads feed the GPU
//! - **Fixed Batch Sizing**: Optimal 10M batch size for GPU utilization
//! - **GPU Context Caching**: Reuses GPU context across all batches
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │                                                                             │
//! │ Worker 0: [Gen Batch 0] [Gen Batch 4] [Gen Batch 8]  ...                    │
//! │ Worker 1: [Gen Batch 1] [Gen Batch 5] [Gen Batch 9]  ...                    │
//! │ Worker 2: [Gen Batch 2] [Gen Batch 6] [Gen Batch 10] ...                    │
//! │ Worker 3: [Gen Batch 3] [Gen Batch 7] [Gen Batch 11] ...                    │
//! │           ↓             ↓             ↓              ↓                      │
//! │           └─────────────┴─────────────┴──────────────┘                      │
//! │                                 ↓                                           │
//! │                    ┌──────────────────────────┐                             │
//! │                    │   Bounded Channel (n+1)  │  ← Double-buffer            │
//! │                    └──────────────────────────┘                             │
//! │                                 ↓                                           │
//! │                    ┌──────────────────────────┐                             │
//! │                    │     GPU Consumer         │  ← Single GPU thread        │
//! │                    │   (Cached GPU Context)   │                             │
//! │                    └──────────────────────────┘                             │
//! │                                                                             │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Running the Example
//!
//! ```bash
//! # Default: 100M options, 10M batch size, auto-detect CPU workers
//! cargo run --example black_scholes_gpu_streaming --release --features gpu-wgpu
//!
//! # Custom configuration
//! TOTAL_OPTIONS=500000000 BATCH_SIZE=20000000 CPU_WORKERS=8 \
//!     cargo run --example black_scholes_gpu_streaming --release --features gpu-wgpu
//! ```

#[path = "kernels/black_scholes.rs"]
mod black_scholes;

use std::env;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use black_scholes::*;
use renoir::operator::gpu::GpuContext;
use renoir::utils::{create_banner, TableBuilder};

// ============================================================================
// Configuration
// ============================================================================

/// Maximum GPU batch size to avoid memory issues (50M cliff).
const GPU_MAX_BATCH: usize = 50_000_000;

/// Default fixed batch size for GPU processing.
const DEFAULT_BATCH_SIZE: usize = 10_000_000;

/// Default total options to process.
const DEFAULT_TOTAL_OPTIONS: usize = 100_000_000;

/// Number of batches to buffer ahead in the pipeline.
const BUFFER_AHEAD: usize = 2;

// ============================================================================
// Strategy 1: Simple Sequential (Baseline)
// ============================================================================

/// Simple sequential GPU processing without pipelining.
/// This is the baseline to compare against.
fn run_sequential(total_items: usize, batch_size: usize) -> (std::time::Duration, usize) {
    let effective_batch = batch_size.min(GPU_MAX_BATCH);
    let start = Instant::now();

    let ctx = GpuContext::new();
    let mut total_processed = 0;
    let mut remaining = total_items;
    let mut offset = 0;

    while remaining > 0 {
        let size = remaining.min(effective_batch);

        // Generate data (blocking)
        let mut soa = BlackScholesSoA::generate_with_seed(size, offset as u64);
        soa.pad(4); // Pad for GPU vectorization

        // Process on GPU (blocking) - using the optimized benchmark function
        let (_kernel_time, _full_time, calls, _threads) = 
            run_optimized_gpu_benchmark(&ctx, &soa, size);
        total_processed += calls.len();

        remaining -= size;
        offset += size;
    }

    (start.elapsed(), total_processed)
}

// ============================================================================
// Strategy 2: Double-Buffered Pipeline
// ============================================================================

/// Double-buffered GPU pipeline with single producer.
/// Overlaps data generation with GPU execution.
fn run_double_buffered(total_items: usize, batch_size: usize) -> (std::time::Duration, usize) {
    let effective_batch = batch_size.min(GPU_MAX_BATCH);
    let start = Instant::now();

    // Bounded channel - blocks producer when buffer is full
    let (tx, rx) = mpsc::sync_channel::<BlackScholesSoA>(BUFFER_AHEAD);

    // Producer thread - generates batches ahead of time
    let producer = thread::spawn(move || {
        let mut remaining = total_items;
        let mut offset = 0;

        while remaining > 0 {
            let size = remaining.min(effective_batch);
            let mut soa = BlackScholesSoA::generate_with_seed(size, offset as u64);
            soa.pad(4); // Pad for GPU vectorization

            if tx.send(soa).is_err() {
                break; // Consumer dropped, stop producing
            }

            remaining -= size;
            offset += size;
        }
    });

    // Consumer - processes on GPU while producer prepares next batch
    let ctx = GpuContext::new();
    let mut total_processed = 0;

    while let Ok(batch_soa) = rx.recv() {
        let batch_len = batch_soa.stocks.len();
        let (_kernel_time, _full_time, calls, _threads) = 
            run_optimized_gpu_benchmark(&ctx, &batch_soa, batch_len);
        total_processed += calls.len();
    }

    producer.join().unwrap();
    (start.elapsed(), total_processed)
}

// ============================================================================
// Strategy 3: Multi-Worker + Double-Buffered Pipeline
// ============================================================================

/// Multi-worker double-buffered GPU pipeline.
/// Multiple producer threads with atomic work distribution.
fn run_parallel_double_buffered(
    total_items: usize,
    batch_size: usize,
    num_workers: usize,
) -> (std::time::Duration, usize) {
    let effective_batch = batch_size.min(GPU_MAX_BATCH);
    let start = Instant::now();

    // Bounded channel for double-buffering
    let buffer_size = (num_workers + 1).min(BUFFER_AHEAD * 2);
    let (tx, rx) = mpsc::sync_channel::<BlackScholesSoA>(buffer_size);

    // Atomic counter for work distribution among producers
    let items_claimed = Arc::new(AtomicUsize::new(0));

    // Spawn multiple producer threads
    let producers: Vec<_> = (0..num_workers)
        .map(|worker_id| {
            let tx = tx.clone();
            let counter = Arc::clone(&items_claimed);

            thread::spawn(move || {
                let mut batches_produced = 0;

                loop {
                    // Atomically claim next batch
                    let start_idx = counter.fetch_add(effective_batch, Ordering::SeqCst);
                    if start_idx >= total_items {
                        break;
                    }

                    let size = (total_items - start_idx).min(effective_batch);

                    // Use deterministic seed based on position
                    let seed = (worker_id as u64 * 1_000_000 + start_idx as u64) ^ 0xDEADBEEF;
                    let mut soa = BlackScholesSoA::generate_with_seed(size, seed);
                    soa.pad(4); // Pad for GPU vectorization

                    if tx.send(soa).is_err() {
                        break;
                    }

                    batches_produced += 1;
                }

                batches_produced
            })
        })
        .collect();

    drop(tx); // Close sender so receiver knows when to stop

    // Single GPU consumer
    let ctx = GpuContext::new();
    let mut total_processed = 0;
    let mut batches_consumed = 0;

    while let Ok(batch_soa) = rx.recv() {
        let batch_len = batch_soa.stocks.len();
        let (_kernel_time, _full_time, calls, _threads) = 
            run_optimized_gpu_benchmark(&ctx, &batch_soa, batch_len);
        total_processed += calls.len();
        batches_consumed += 1;
    }

    // Wait for all producers to finish
    let total_batches_produced: usize = producers
        .into_iter()
        .map(|p| p.join().unwrap())
        .sum();

    println!(
        "    Producers created {} batches, consumer processed {} batches",
        total_batches_produced, batches_consumed
    );

    (start.elapsed(), total_processed)
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let data_size_gb = |n: usize| (n * 5 * std::mem::size_of::<f32>()) as f64 / (1024.0 * 1024.0 * 1024.0);
    
    // Get configuration from environment
    let total_options: usize = env::var("TOTAL_OPTIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TOTAL_OPTIONS);

    let batch_size: usize = env::var("BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_BATCH_SIZE);

    let cpu_workers: usize = env::var("CPU_WORKERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4)
        });

    print!(
        "{}",
        create_banner(
            "Black-Scholes GPU Streaming Example",
            &[
                &format!("Total options:   {} ({:.1}M)", total_options, total_options as f64 / 1e6),
                &format!("Batch size:      {} ({:.1}M)", batch_size, batch_size as f64 / 1e6),
                &format!("CPU workers:     {}", cpu_workers),
                &format!("Buffer ahead:    {} batches", BUFFER_AHEAD),
                &format!("Data size:       {:.2} GB (5 floats per option)", data_size_gb(total_options)),
            ]
        )
    );

    // ========================================================================
    // Strategy 1: Sequential (Baseline)
    // ========================================================================
    print!(
        "{}",
        create_banner(
            "Strategy 1: Sequential GPU (Baseline)",
            &[
                "Generate batch → Process on GPU → Repeat",
                "GPU sits idle during data generation",
            ]
        )
    );

    let (seq_duration, seq_processed) = run_sequential(total_options, batch_size);
    let seq_throughput = seq_processed as f64 / seq_duration.as_secs_f64() / 1e6;

    println!("  Result: {} options in {:.3}s", seq_processed, seq_duration.as_secs_f64());
    println!("  Throughput: {:.2} M options/sec\n", seq_throughput);

    // ========================================================================
    // Strategy 2: Double-Buffered Pipeline
    // ========================================================================
    print!(
        "{}",
        create_banner(
            "Strategy 2: Double-Buffered Pipeline",
            &[
                "Single producer thread generates batches ahead",
                "GPU processes while next batch is being prepared",
                &format!("Buffer size: {} batches", BUFFER_AHEAD),
            ]
        )
    );

    let (dbl_duration, dbl_processed) = run_double_buffered(total_options, batch_size);
    let dbl_throughput = dbl_processed as f64 / dbl_duration.as_secs_f64() / 1e6;
    let dbl_speedup = seq_duration.as_secs_f64() / dbl_duration.as_secs_f64();

    println!("  Result: {} options in {:.3}s", dbl_processed, dbl_duration.as_secs_f64());
    println!("  Throughput: {:.2} M options/sec", dbl_throughput);
    println!("  Speedup vs Sequential: {:.2}x\n", dbl_speedup);

    // ========================================================================
    // Strategy 3: Multi-Worker + Double-Buffered
    // ========================================================================
    print!(
        "{}",
        create_banner(
            "Strategy 3: Multi-Worker + Double-Buffered Pipeline",
            &[
                &format!("{} producer threads with atomic work distribution", cpu_workers),
                "Batches generated in parallel, consumed by single GPU",
                "Maximizes both CPU and GPU utilization",
            ]
        )
    );

    let (par_duration, par_processed) = run_parallel_double_buffered(total_options, batch_size, cpu_workers);
    let par_throughput = par_processed as f64 / par_duration.as_secs_f64() / 1e6;
    let par_speedup_seq = seq_duration.as_secs_f64() / par_duration.as_secs_f64();
    let par_speedup_dbl = dbl_duration.as_secs_f64() / par_duration.as_secs_f64();

    println!("  Result: {} options in {:.3}s", par_processed, par_duration.as_secs_f64());
    println!("  Throughput: {:.2} M options/sec", par_throughput);
    println!("  Speedup vs Sequential: {:.2}x", par_speedup_seq);
    println!("  Speedup vs Double-Buffered: {:.2}x\n", par_speedup_dbl);

    // ========================================================================
    // Summary Table
    // ========================================================================
    print!("{}", create_banner("SUMMARY", &[]));
    
    let table = TableBuilder::new(&[
        ("    Strategy             ", 25),
        ("   Time   ", 10),
        (" Throughput ", 12),
        (" Speedup ", 9),
    ]);
    print!("{}", table.header());
    
    print!(
        "{}",
        table.row(&[
            " Sequential (baseline)  ",
            &format!(" {:>6.3}s ", seq_duration.as_secs_f64()),
            &format!(" {:>6.2} M/s", seq_throughput),
            "   1.00x ",
        ])
    );
    print!(
        "{}",
        table.row(&[
            " Double-Buffered        ",
            &format!(" {:>6.3}s ", dbl_duration.as_secs_f64()),
            &format!(" {:>6.2} M/s", dbl_throughput),
            &format!("   {:.2}x ", dbl_speedup),
        ])
    );
    print!(
        "{}",
        table.row(&[
            " Multi-Worker + DblBuf  ",
            &format!(" {:>6.3}s ", par_duration.as_secs_f64()),
            &format!(" {:>6.2} M/s", par_throughput),
            &format!("   {:.2}x ", par_speedup_seq),
        ])
    );
    print!("{}", table.footer());

    // Validation
    let all_valid = seq_processed == total_options
        && dbl_processed == total_options
        && par_processed == total_options;

    print!(
        "{}",
        create_banner(
            "VALIDATION",
            &[
                &format!(
                    "Status: {}",
                    if all_valid { "PASSED ✓" } else { "FAILED ✗" }
                ),
                &format!("Expected:        {}", total_options),
                &format!("Sequential:      {}", seq_processed),
                &format!("Double-Buffered: {}", dbl_processed),
                &format!("Multi-Worker:    {}", par_processed),
            ]
        )
    );
}

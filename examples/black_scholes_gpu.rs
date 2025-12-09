//! # Black-Scholes Option Pricing with GPU Acceleration
//!
//! This example demonstrates how to use the `map_gpu` operator to perform
//! GPU-accelerated Black-Scholes option pricing calculations using Renoir.
//!
//! ## Running
//!
//! ```bash
//! cargo run --example black_scholes_gpu --features gpu-wgpu --release
//! ```

#[path = "kernels/black_scholes.rs"]
mod black_scholes_kernel;

use std::time::Instant;

use renoir::operator::gpu::GpuBatchStrategy;
use renoir::prelude::*;
use renoir::utils::{create_banner, TableBuilder};

use black_scholes_kernel::*;

// ============================================================================
// Main Example
// ============================================================================

fn main() {
    print!("{}", create_banner("Black-Scholes GPU Example with Renoir", &[]));

    let num_options = 1_000_000;

    // =========================================================================
    // APPROACH 1: Standard AoS (Array-of-Structures) - Renoir Streaming
    // =========================================================================
    print!("{}", create_banner("APPROACH 1: Standard Renoir Streaming (AoS)", &[
        &format!("Generating {} options in AoS format...", num_options),
        "Processing with Renoir + GPU (streaming)...",
    ]));

    let options = generate_options(num_options);

    let env = StreamContext::new_local();
    let start = Instant::now();

    let result = env
        .stream_iter(options.clone().into_iter())
        .map_gpu_with_strategy(
            BlackScholesKernel::default(),
            GpuBatchStrategy::fixed(100_000),
        )
        .collect_vec();

    env.execute_blocking();

    let gpu_duration = start.elapsed();
    let gpu_results = result.get().unwrap();

    print!("{}", create_banner("Streaming Results", &[
        &format!("Total options processed: {}", gpu_results.len()),
        &format!("Time: {:.3}s", gpu_duration.as_secs_f64()),
        &format!("Throughput: {:.2}M options/sec", num_options as f64 / gpu_duration.as_secs_f64() / 1_000_000.0),
    ]));

    // =========================================================================
    // APPROACH 2: Full Batch Processing
    // =========================================================================
    print!("{}", create_banner("APPROACH 2: Full Batch Processing", &[
        "Processing entire input as single GPU batch...",
    ]));

    let env = StreamContext::new_local();
    let start = Instant::now();

    let result = env
        .stream_iter(options.clone().into_iter())
        .map_gpu_with_strategy(
            BlackScholesKernel::default(),
            GpuBatchStrategy::fixed(num_options),
        )
        .collect_vec();

    env.execute_blocking();

    let batch_duration = start.elapsed();
    let batch_results = result.get().unwrap();

    print!("{}", create_banner("Full Batch Results", &[
        &format!("Total options processed: {}", batch_results.len()),
        &format!("Time: {:.3}s", batch_duration.as_secs_f64()),
        &format!("Throughput: {:.2}M options/sec", num_options as f64 / batch_duration.as_secs_f64() / 1_000_000.0),
    ]));

    // =========================================================================
    // Validation: Compare GPU vs CPU
    // =========================================================================
    let (passed, max_error, avg_error) = validate_results(&options, &gpu_results, 0.001);

    print!("{}", create_banner("VALIDATION", &[
        &format!("Status: {}", if passed { "PASSED ✓" } else { "FAILED ✗" }),
        &format!("Max error: {:.6}", max_error),
        &format!("Avg error: {:.9}", avg_error),
    ]));

    // Show sample results using TableBuilder
    println!("Sample results (first 5 options):");
    let table = TableBuilder::new(&[
        ("   Stock   ", 11),
        ("  Strike   ", 11),
        ("   Time    ", 11),
        ("    Call     ", 13),
        ("    Put      ", 13),
    ]);
    print!("{}", table.header());

    for (input, output) in options.iter().zip(gpu_results.iter()).take(5) {
        print!("{}", table.row(&[
            &format!(" {:>8.2} ", input.stock_price),
            &format!(" {:>8.2} ", input.strike_price),
            &format!(" {:>8.2} ", input.time_to_expiry),
            &format!(" {:>10.4} ", output.call_price),
            &format!(" {:>10.4} ", output.put_price),
        ]));
    }
    print!("{}", table.footer());

    // =========================================================================
    // Summary
    // =========================================================================
    print!(
        "{}",
        create_banner(
            "SUMMARY",
            &[
                &format!("Options processed: {}", num_options),
                &format!("Streaming time:    {:.3}s", gpu_duration.as_secs_f64()),
                &format!("Batch time:        {:.3}s", batch_duration.as_secs_f64()),
                &format!(
                    "Best throughput:   {:.2}M/s",
                    num_options as f64 / batch_duration.as_secs_f64().min(gpu_duration.as_secs_f64()) / 1_000_000.0
                ),
                &format!("Validation:        {}", if passed { "PASSED" } else { "FAILED" }),
            ]
        )
    );
}

//! # Monte Carlo GPU vs CPU Comparison Example
//!
//! This example demonstrates the Monte Carlo option pricing algorithm
//! running on both GPU and CPU, comparing their outputs side-by-side.
//!
//! ## Features
//! - Shows identical results between GPU and CPU (deterministic seeding)
//! - Displays pricing for multiple option scenarios
//! - Measures and compares execution time
//!
//! ## Run with:
//! ```bash
//! cargo run --example monte_carlo_comparison --features gpu-wgpu
//! ```

// Import monte_carlo kernel directly (not via kernels/mod.rs) to avoid
// compiling unused black_scholes module
#[path = "kernels/monte_carlo.rs"]
mod monte_carlo;

use monte_carlo::{
    monte_carlo_cpu, MonteCarloInput, MonteCarloKernel, MonteCarloOutput,
    MC_NUM_PATHS, MC_TIME_STEPS,
};
use renoir::operator::gpu::GpuBatchStrategy;
use renoir::prelude::*;
use renoir::utils::{create_banner, Table};
use std::time::Instant;

fn main() {
    // Print title banner with configuration info
    let config_line1 = format!("Configuration: {} paths × {} time steps per option", MC_NUM_PATHS, MC_TIME_STEPS);
    let config_line2 = format!("FLOPs per option: ~{}", MC_NUM_PATHS * MC_TIME_STEPS * 20);
    print!("{}", create_banner(
        "Monte Carlo Option Pricing: GPU vs CPU Comparison",
        &[&config_line1, &config_line2],
    ));

    // Define test scenarios
    let scenarios = vec![
        ("ATM Call (S=K)", MonteCarloInput {
            stock_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        }),
        ("Deep ITM Call (S>>K)", MonteCarloInput {
            stock_price: 150.0,
            strike_price: 100.0,
            time_to_expiry: 0.5,
            risk_free_rate: 0.05,
            volatility: 0.20,
        }),
        ("Deep OTM Call (S<<K)", MonteCarloInput {
            stock_price: 50.0,
            strike_price: 100.0,
            time_to_expiry: 0.5,
            risk_free_rate: 0.05,
            volatility: 0.20,
        }),
        ("High Volatility", MonteCarloInput {
            stock_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 0.5,
            risk_free_rate: 0.05,
            volatility: 0.60,
        }),
        ("Long Expiry (2Y)", MonteCarloInput {
            stock_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 2.0,
            risk_free_rate: 0.03,
            volatility: 0.25,
        }),
        ("Big Number ($10K)", MonteCarloInput {
            stock_price: 10000.0,
            strike_price: 10000.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        }),
        ("Short Expiry (1M)", MonteCarloInput {
            stock_price: 100.0,
            strike_price: 105.0,
            time_to_expiry: 0.083, // ~1 month
            risk_free_rate: 0.05,
            volatility: 0.30,
        }),
        ("Zero Interest Rate", MonteCarloInput {
            stock_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.0,
            volatility: 0.20,
        }),
    ];

    let inputs: Vec<MonteCarloInput> = scenarios.iter().map(|(_, input)| *input).collect();
    let names: Vec<&str> = scenarios.iter().map(|(name, _)| *name).collect();

    // ========================================================================
    // CPU Execution
    // ========================================================================
    print!("{}", create_banner("CPU Execution (Sequential)", &[]));
    
    let cpu_start = Instant::now();
    let cpu_results: Vec<MonteCarloOutput> = inputs.iter().map(|input| monte_carlo_cpu(*input)).collect();
    let cpu_time = cpu_start.elapsed();
    
    println!("  Time: {:?}\n", cpu_time);

    // ========================================================================
    // GPU Execution
    // ========================================================================
    print!("{}", create_banner("GPU Execution (Parallel)", &[]));
    
    let gpu_start = Instant::now();
    let gpu_results = run_gpu(inputs.clone());
    let gpu_time = gpu_start.elapsed();
    
    println!("  Time: {:?}\n", gpu_time);

    // ========================================================================
    // Side-by-Side Comparison Table
    // ========================================================================
    let mut comparison_table = Table::new(&[
        ("Scenario", 22),
        ("CPU Call", 12),
        ("CPU Put", 12),
        ("GPU Call", 12),
        ("GPU Put", 12),
        ("Match", 6),
    ]);

    let mut all_match = true;
    for (i, name) in names.iter().enumerate() {
        let cpu = &cpu_results[i];
        let gpu = &gpu_results[i];
        
        // Check if results match (within floating point tolerance)
        let call_match = (cpu.call_price - gpu.call_price).abs() < 0.001;
        let put_match = (cpu.put_price - gpu.put_price).abs() < 0.001;
        let matches = call_match && put_match;
        all_match &= matches;
        
        let match_symbol = if matches { "✓" } else { "✗" };
        
        comparison_table.add_row(&[
            name,
            &format!("{:.4}", cpu.call_price),
            &format!("{:.4}", cpu.put_price),
            &format!("{:.4}", gpu.call_price),
            &format!("{:.4}", gpu.put_price),
            match_symbol,
        ]);
    }
    comparison_table.finish();
    println!();

    // ========================================================================
    // Summary
    // ========================================================================
    let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
    let match_status = if all_match {
        "✓ All GPU results match CPU results (deterministic seeding works!)"
    } else {
        "✗ Some results differ - check RNG synchronization"
    };
    
    let cpu_time_str = format!("  CPU time: {:>10.2?}", cpu_time);
    let gpu_time_str = format!("  GPU time: {:>10.2?}", gpu_time);
    let speedup_str = format!("  Speedup:  {:>10.2}x", speedup);
    print!("{}", create_banner(
        "SUMMARY",
        &[match_status, "", "Performance:", &cpu_time_str, &gpu_time_str, &speedup_str],
    ));

    // ========================================================================
    // Detailed Results Table
    // ========================================================================
    print!("{}", create_banner("DETAILED RESULTS", &[]));
    
    let mut details_table = Table::new(&[
        ("#", 3),
        ("Scenario", 22),
        ("Stock", 10),
        ("Strike", 10),
        ("Expiry", 8),
        ("Rate", 6),
        ("Vol", 6),
        ("Call Δ", 12),
        ("Put Δ", 12),
    ]);
    
    for (i, name) in names.iter().enumerate() {
        let input = &inputs[i];
        let cpu = &cpu_results[i];
        let gpu = &gpu_results[i];
        let call_diff = (gpu.call_price - cpu.call_price).abs();
        let put_diff = (gpu.put_price - cpu.put_price).abs();
        
        details_table.add_row(&[
            &format!("{}", i + 1),
            name,
            &format!("${:.2}", input.stock_price),
            &format!("${:.2}", input.strike_price),
            &format!("{:.2}Y", input.time_to_expiry),
            &format!("{:.1}%", input.risk_free_rate * 100.0),
            &format!("{:.0}%", input.volatility * 100.0),
            &format!("${:.6}", call_diff),
            &format!("${:.6}", put_diff),
        ]);
    }
    details_table.finish();
}

/// Run Monte Carlo on GPU using Renoir streaming
fn run_gpu(inputs: Vec<MonteCarloInput>) -> Vec<MonteCarloOutput> {
    let config = RuntimeConfig::local(1).unwrap();
    let env = StreamContext::new(config);
    
    let output = env.stream_iter(inputs.into_iter())
        .map_gpu_with_strategy(
            MonteCarloKernel::default(),
            GpuBatchStrategy::fixed(10000),
        )
        .collect_vec();
    
    env.execute_blocking();
    output.get().unwrap()
}

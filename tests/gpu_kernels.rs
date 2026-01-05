//! # GPU Kernel Integration Tests
//!
//! Integration tests for Black-Scholes and Monte Carlo GPU kernels.
//! Compares GPU results with CPU results for correctness validation.
//!
//! Note: Unit tests for CPU functions are in the respective kernel files:
//! - `examples/kernels/black_scholes.rs`
//! - `examples/kernels/monte_carlo.rs`

#[path = "../examples/kernels/mod.rs"]
mod kernels;

use kernels::black_scholes::{black_scholes_cpu, BlackScholesInput, BlackScholesOutput};
use kernels::monte_carlo::{monte_carlo_cpu, MonteCarloInput, MonteCarloOutput};
use rand::prelude::*;
use rand::rngs::StdRng;

// ============================================================================
// Test Case Generators - 1000 Cases Including Big Numbers
// ============================================================================

const NUM_TEST_CASES: usize = 1000;
const TEST_SEED: u64 = 12345;

/// Generate 1000 Black-Scholes test inputs covering normal, edge, and big number cases.
fn generate_black_scholes_inputs() -> Vec<BlackScholesInput> {
    let mut rng = StdRng::seed_from_u64(TEST_SEED);
    let mut inputs = Vec::with_capacity(NUM_TEST_CASES);
    
    // First 20: Hand-picked edge cases including big numbers
    let edge_cases = vec![
        // Standard cases
        BlackScholesInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.20 },
        BlackScholesInput { stock_price: 150.0, strike_price: 100.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.20 },
        BlackScholesInput { stock_price: 50.0, strike_price: 100.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.20 },
        BlackScholesInput { stock_price: 100.0, strike_price: 105.0, time_to_expiry: 0.01, risk_free_rate: 0.05, volatility: 0.30 },
        BlackScholesInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 3.0, risk_free_rate: 0.03, volatility: 0.25 },
        // Big numbers - stock prices up to 100,000
        BlackScholesInput { stock_price: 10000.0, strike_price: 10000.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.20 },
        BlackScholesInput { stock_price: 50000.0, strike_price: 50000.0, time_to_expiry: 0.5, risk_free_rate: 0.04, volatility: 0.15 },
        BlackScholesInput { stock_price: 100000.0, strike_price: 100000.0, time_to_expiry: 1.0, risk_free_rate: 0.03, volatility: 0.25 },
        BlackScholesInput { stock_price: 75000.0, strike_price: 50000.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.20 }, // ITM big
        BlackScholesInput { stock_price: 25000.0, strike_price: 50000.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.20 }, // OTM big
        // Extreme volatility with big numbers
        BlackScholesInput { stock_price: 10000.0, strike_price: 10000.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.80 },
        BlackScholesInput { stock_price: 10000.0, strike_price: 10000.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.05 },
        // Very small prices
        BlackScholesInput { stock_price: 0.01, strike_price: 0.01, time_to_expiry: 0.25, risk_free_rate: 0.05, volatility: 0.40 },
        BlackScholesInput { stock_price: 0.001, strike_price: 0.001, time_to_expiry: 0.5, risk_free_rate: 0.03, volatility: 0.50 },
        // Zero interest rate
        BlackScholesInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 1.0, risk_free_rate: 0.0, volatility: 0.20 },
        BlackScholesInput { stock_price: 10000.0, strike_price: 10000.0, time_to_expiry: 1.0, risk_free_rate: 0.0, volatility: 0.20 },
        // Very long expiry
        BlackScholesInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 10.0, risk_free_rate: 0.02, volatility: 0.20 },
        BlackScholesInput { stock_price: 5000.0, strike_price: 5000.0, time_to_expiry: 5.0, risk_free_rate: 0.03, volatility: 0.25 },
        // Very short expiry
        BlackScholesInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 0.001, risk_free_rate: 0.05, volatility: 0.20 },
        BlackScholesInput { stock_price: 10000.0, strike_price: 10000.0, time_to_expiry: 0.001, risk_free_rate: 0.05, volatility: 0.20 },
    ];
    inputs.extend(edge_cases);
    
    // Remaining: Random test cases with varying scales
    while inputs.len() < NUM_TEST_CASES {
        // Choose scale: 10% small (0.01-10), 60% normal (10-1000), 30% big (1000-100000)
        let scale_choice: f32 = rng.random();
        let (min_price, max_price) = if scale_choice < 0.1 {
            (0.01f32, 10.0f32)
        } else if scale_choice < 0.7 {
            (10.0f32, 1000.0f32)
        } else {
            (1000.0f32, 100000.0f32)
        };
        
        let stock_price = rng.random_range(min_price..max_price);
        let strike_ratio: f32 = rng.random_range(0.5..1.5);
        let strike_price = stock_price * strike_ratio;
        
        inputs.push(BlackScholesInput {
            stock_price,
            strike_price,
            time_to_expiry: rng.random_range(0.01..5.0),
            risk_free_rate: rng.random_range(0.0..0.15),
            volatility: rng.random_range(0.05..1.0),
        });
    }
    
    inputs
}

/// Generate 1000 Monte Carlo test inputs covering normal, edge, and big number cases.
fn generate_monte_carlo_inputs() -> Vec<MonteCarloInput> {
    let mut rng = StdRng::seed_from_u64(TEST_SEED + 1);
    let mut inputs = Vec::with_capacity(NUM_TEST_CASES);
    
    // First 20: Hand-picked edge cases including big numbers
    let edge_cases = vec![
        MonteCarloInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.20 },
        MonteCarloInput { stock_price: 150.0, strike_price: 100.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.20 },
        MonteCarloInput { stock_price: 50.0, strike_price: 100.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.20 },
        // Big numbers
        MonteCarloInput { stock_price: 10000.0, strike_price: 10000.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.20 },
        MonteCarloInput { stock_price: 50000.0, strike_price: 50000.0, time_to_expiry: 0.5, risk_free_rate: 0.04, volatility: 0.15 },
        MonteCarloInput { stock_price: 100000.0, strike_price: 100000.0, time_to_expiry: 1.0, risk_free_rate: 0.03, volatility: 0.25 },
        MonteCarloInput { stock_price: 75000.0, strike_price: 50000.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.20 },
        MonteCarloInput { stock_price: 25000.0, strike_price: 50000.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.20 },
        // Small prices
        MonteCarloInput { stock_price: 1.0, strike_price: 1.0, time_to_expiry: 0.25, risk_free_rate: 0.05, volatility: 0.35 },
        MonteCarloInput { stock_price: 10.0, strike_price: 10.0, time_to_expiry: 0.5, risk_free_rate: 0.03, volatility: 0.40 },
    ];
    inputs.extend(edge_cases);
    
    // Remaining: Random test cases
    while inputs.len() < NUM_TEST_CASES {
        let scale_choice: f32 = rng.random();
        let (min_price, max_price) = if scale_choice < 0.1 {
            (1.0f32, 10.0f32)
        } else if scale_choice < 0.7 {
            (10.0f32, 1000.0f32)
        } else {
            (1000.0f32, 100000.0f32)
        };
        
        let stock_price = rng.random_range(min_price..max_price);
        let strike_ratio: f32 = rng.random_range(0.5..1.5);
        let strike_price = stock_price * strike_ratio;
        
        inputs.push(MonteCarloInput {
            stock_price,
            strike_price,
            time_to_expiry: rng.random_range(0.1..3.0),
            risk_free_rate: rng.random_range(0.0..0.10),
            volatility: rng.random_range(0.10..0.80),
        });
    }
    
    inputs
}

// ============================================================================
// Edge Case Test Sets
// ============================================================================

/// Returns 10 edge case test inputs for Black-Scholes option pricing.
fn black_scholes_edge_cases() -> Vec<(BlackScholesInput, &'static str)> {
    vec![
        (BlackScholesInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.20 }, "ATM option"),
        (BlackScholesInput { stock_price: 150.0, strike_price: 100.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.20 }, "Deep ITM call"),
        (BlackScholesInput { stock_price: 50.0, strike_price: 100.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.20 }, "Deep OTM call"),
        (BlackScholesInput { stock_price: 100.0, strike_price: 105.0, time_to_expiry: 0.01, risk_free_rate: 0.05, volatility: 0.30 }, "Short expiry"),
        (BlackScholesInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 3.0, risk_free_rate: 0.03, volatility: 0.25 }, "Long expiry (LEAPS)"),
        (BlackScholesInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.80 }, "High volatility"),
        (BlackScholesInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.05 }, "Low volatility"),
        (BlackScholesInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 1.0, risk_free_rate: 0.0, volatility: 0.20 }, "Zero interest rate"),
        (BlackScholesInput { stock_price: 1.0, strike_price: 1.0, time_to_expiry: 0.25, risk_free_rate: 0.05, volatility: 0.40 }, "Penny stock"),
        (BlackScholesInput { stock_price: 5000.0, strike_price: 5000.0, time_to_expiry: 0.5, risk_free_rate: 0.04, volatility: 0.15 }, "High-priced stock"),
    ]
}

/// Returns 10 edge case test inputs for Monte Carlo option pricing.
fn monte_carlo_edge_cases() -> Vec<(MonteCarloInput, &'static str)> {
    vec![
        (MonteCarloInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.20 }, "ATM option"),
        (MonteCarloInput { stock_price: 150.0, strike_price: 100.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.20 }, "Deep ITM call"),
        (MonteCarloInput { stock_price: 50.0, strike_price: 100.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.20 }, "Deep OTM call"),
        (MonteCarloInput { stock_price: 100.0, strike_price: 105.0, time_to_expiry: 0.1, risk_free_rate: 0.05, volatility: 0.30 }, "Short expiry"),
        (MonteCarloInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 2.0, risk_free_rate: 0.03, volatility: 0.25 }, "Long expiry"),
        (MonteCarloInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.60 }, "High volatility"),
        (MonteCarloInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.10 }, "Low volatility"),
        (MonteCarloInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 1.0, risk_free_rate: 0.0, volatility: 0.20 }, "Zero interest rate"),
        (MonteCarloInput { stock_price: 10.0, strike_price: 10.0, time_to_expiry: 0.25, risk_free_rate: 0.05, volatility: 0.35 }, "Low-priced stock"),
        (MonteCarloInput { stock_price: 1000.0, strike_price: 1000.0, time_to_expiry: 0.5, risk_free_rate: 0.04, volatility: 0.20 }, "High-priced stock"),
    ]
}

// ============================================================================
// Cross-Validation: Black-Scholes vs Monte Carlo
// ============================================================================

#[test]
fn test_black_scholes_vs_monte_carlo_consistency() {
    // Compare Black-Scholes (analytical) with Monte Carlo (simulation)
    // They should produce similar results for European options
    
    let bs_input = BlackScholesInput {
        stock_price: 100.0,
        strike_price: 100.0,
        time_to_expiry: 1.0,
        risk_free_rate: 0.05,
        volatility: 0.20,
    };
    
    let mc_input = MonteCarloInput {
        stock_price: 100.0,
        strike_price: 100.0,
        time_to_expiry: 1.0,
        risk_free_rate: 0.05,
        volatility: 0.20,
    };
    
    let bs_output = black_scholes_cpu(bs_input);
    let mc_output = monte_carlo_cpu(mc_input);
    
    // Monte Carlo should be within 20% of Black-Scholes (due to variance with limited paths)
    let call_error = (bs_output.call_price - mc_output.call_price).abs() / bs_output.call_price;
    assert!(call_error < 0.20, "MC call differs from BS by {:.1}%", call_error * 100.0);
}

// ============================================================================
// GPU Integration Tests
// ============================================================================

#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]
mod gpu_integration_tests {
    use super::*;
    use kernels::black_scholes::BlackScholesKernel;
    use kernels::monte_carlo::MonteCarloKernel;
    use renoir::operator::gpu::GpuBatchStrategy;
    use renoir::prelude::*;

    /// Run Black-Scholes on GPU and return results
    fn run_black_scholes_gpu(inputs: Vec<BlackScholesInput>) -> Vec<BlackScholesOutput> {
        let config = RuntimeConfig::local(1).unwrap();
        let env = StreamContext::new(config);
        
        let output = env.stream_iter(inputs.into_iter())
            .map_gpu_with_strategy(
                BlackScholesKernel::default(),
                GpuBatchStrategy::fixed(1000),
            )
            .collect_vec();
        env.execute_blocking();
        
        output.get().unwrap()
    }

    /// Run Black-Scholes on CPU via Renoir streaming
    fn run_black_scholes_cpu(inputs: Vec<BlackScholesInput>) -> Vec<BlackScholesOutput> {
        let config = RuntimeConfig::local(1).unwrap();
        let env = StreamContext::new(config);
        
        let output = env.stream_iter(inputs.into_iter())
            .map(black_scholes_cpu)
            .collect_vec();
        env.execute_blocking();
        
        output.get().unwrap()
    }

    /// Run Monte Carlo on GPU and return results
    fn run_monte_carlo_gpu(inputs: Vec<MonteCarloInput>) -> Vec<MonteCarloOutput> {
        let config = RuntimeConfig::local(1).unwrap();
        let env = StreamContext::new(config);
        
        let output = env.stream_iter(inputs.into_iter())
            .map_gpu_with_strategy(
                MonteCarloKernel::default(),
                GpuBatchStrategy::fixed(1000),
            )
            .collect_vec();
        env.execute_blocking();
        
        output.get().unwrap()
    }

    /// Run Monte Carlo on CPU via Renoir streaming
    fn run_monte_carlo_cpu(inputs: Vec<MonteCarloInput>) -> Vec<MonteCarloOutput> {
        let config = RuntimeConfig::local(1).unwrap();
        let env = StreamContext::new(config);
        
        let output = env.stream_iter(inputs.into_iter())
            .map(monte_carlo_cpu)
            .collect_vec();
        env.execute_blocking();
        
        output.get().unwrap()
    }

    // ========================================================================
    // Black-Scholes GPU vs CPU Tests
    // ========================================================================

    #[test]
    fn test_black_scholes_gpu_vs_cpu_edge_cases() {
        let cases = black_scholes_edge_cases();
        let inputs: Vec<BlackScholesInput> = cases.iter().map(|(input, _)| *input).collect();
        let names: Vec<&str> = cases.iter().map(|(_, name)| *name).collect();
        
        let cpu_results = run_black_scholes_cpu(inputs.clone());
        let gpu_results = run_black_scholes_gpu(inputs);
        
        assert_eq!(cpu_results.len(), gpu_results.len(), "Result count mismatch");
        
        for (i, (cpu, gpu)) in cpu_results.iter().zip(gpu_results.iter()).enumerate() {
            let name = names[i];
            
            // Check GPU output is valid
            assert!(!gpu.call_price.is_nan(), "{}: GPU call price is NaN", name);
            assert!(!gpu.put_price.is_nan(), "{}: GPU put price is NaN", name);
            assert!(gpu.call_price >= 0.0, "{}: GPU call price is negative", name);
            assert!(gpu.put_price >= 0.0, "{}: GPU put price is negative", name);
            
            // Compare GPU with CPU - allow 1% relative error for floating point differences
            let call_tolerance = cpu.call_price.abs().max(0.01) * 0.01;
            let put_tolerance = cpu.put_price.abs().max(0.01) * 0.01;
            
            let call_error = (cpu.call_price - gpu.call_price).abs();
            let put_error = (cpu.put_price - gpu.put_price).abs();
            
            assert!(call_error < call_tolerance, 
                "{}: GPU call price {:.6} differs from CPU {:.6} by {:.6}", 
                name, gpu.call_price, cpu.call_price, call_error);
            assert!(put_error < put_tolerance, 
                "{}: GPU put price {:.6} differs from CPU {:.6} by {:.6}", 
                name, gpu.put_price, cpu.put_price, put_error);
        }
    }

    #[test]
    fn test_black_scholes_gpu_batch_processing() {
        // Test that GPU correctly processes a batch of 100 options
        let inputs: Vec<BlackScholesInput> = (0..100).map(|i| {
            BlackScholesInput {
                stock_price: 50.0 + i as f32,
                strike_price: 100.0,
                time_to_expiry: 0.5,
                risk_free_rate: 0.05,
                volatility: 0.20,
            }
        }).collect();
        
        let cpu_results = run_black_scholes_cpu(inputs.clone());
        let gpu_results = run_black_scholes_gpu(inputs);
        
        assert_eq!(cpu_results.len(), 100);
        assert_eq!(gpu_results.len(), 100);
        
        let mut max_error = 0.0f32;
        for (cpu, gpu) in cpu_results.iter().zip(gpu_results.iter()) {
            let error = (cpu.call_price - gpu.call_price).abs();
            if error > max_error {
                max_error = error;
            }
        }
        
        // Max error should be tiny (floating point precision)
        assert!(max_error < 0.001, "Max GPU vs CPU error too high: {:.6}", max_error);
    }

    #[test]
    fn test_black_scholes_gpu_vs_cpu_1000_cases() {
        // Test 1000 cases including big numbers (up to 100,000 stock price)
        let inputs = generate_black_scholes_inputs();
        assert_eq!(inputs.len(), NUM_TEST_CASES, "Should have {} inputs", NUM_TEST_CASES);
        
        let cpu_results = run_black_scholes_cpu(inputs.clone());
        let gpu_results = run_black_scholes_gpu(inputs.clone());
        
        assert_eq!(cpu_results.len(), NUM_TEST_CASES);
        assert_eq!(gpu_results.len(), NUM_TEST_CASES);
        
        let mut max_rel_error = 0.0f64;
        let mut max_abs_error = 0.0f32;
        let mut failed_count = 0usize;
        
        for (i, (cpu, gpu)) in cpu_results.iter().zip(gpu_results.iter()).enumerate() {
            let input = &inputs[i];
            
            // Check GPU output is valid
            assert!(!gpu.call_price.is_nan(), "Test {}: GPU call is NaN (stock={:.2})", i, input.stock_price);
            assert!(!gpu.put_price.is_nan(), "Test {}: GPU put is NaN (stock={:.2})", i, input.stock_price);
            // Allow tiny negative values due to floating point precision
            assert!(gpu.call_price >= -0.001, "Test {}: GPU call is significantly negative: {:.6}", i, gpu.call_price);
            
            let call_error = (cpu.call_price - gpu.call_price).abs();
            
            // Track max absolute error
            if call_error > max_abs_error {
                max_abs_error = call_error;
            }
            
            // Calculate relative error (handle near-zero prices)
            if cpu.call_price.abs() > 0.001 {
                let rel_error = (call_error / cpu.call_price.abs()) as f64;
                if rel_error > max_rel_error {
                    max_rel_error = rel_error;
                }
                
                // For big numbers, allow slightly larger relative tolerance
                let tolerance = if input.stock_price > 10000.0 { 0.02 } else { 0.01 };
                if rel_error > tolerance {
                    failed_count += 1;
                    if failed_count <= 5 {
                        eprintln!("MISMATCH Test {}: stock={:.2} strike={:.2} cpu_call={:.6} gpu_call={:.6} err={:.4}%",
                            i, input.stock_price, input.strike_price, cpu.call_price, gpu.call_price, rel_error * 100.0);
                    }
                }
            }
        }
        
        eprintln!("Black-Scholes 1000 tests: max_rel_err={:.4}%, max_abs_err={:.6}, failures={}",
            max_rel_error * 100.0, max_abs_error, failed_count);
        
        assert_eq!(failed_count, 0, "{} out of {} tests failed tolerance check", failed_count, NUM_TEST_CASES);
    }

    #[test]
    fn test_black_scholes_big_numbers_extreme() {
        // Huge number edge cases
        let inputs = vec![
            BlackScholesInput { stock_price: 100_000.0, strike_price: 100_000.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.20 },
            BlackScholesInput { stock_price: 500_000.0, strike_price: 500_000.0, time_to_expiry: 0.5, risk_free_rate: 0.04, volatility: 0.15 },
            BlackScholesInput { stock_price: 1_000_000.0, strike_price: 1_000_000.0, time_to_expiry: 1.0, risk_free_rate: 0.03, volatility: 0.25 },
            BlackScholesInput { stock_price: 750_000.0, strike_price: 500_000.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.20 }, // ITM
            BlackScholesInput { stock_price: 250_000.0, strike_price: 500_000.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.20 }, // OTM
        ];
        
        let cpu_results = run_black_scholes_cpu(inputs.clone());
        let gpu_results = run_black_scholes_gpu(inputs.clone());
        
        for (i, (cpu, gpu)) in cpu_results.iter().zip(gpu_results.iter()).enumerate() {
            let input = &inputs[i];
            
            assert!(!gpu.call_price.is_nan(), "Extreme test {}: GPU call is NaN", i);
            assert!(gpu.call_price >= 0.0, "Extreme test {}: GPU call is negative", i);
            
            // For million-scale prices, allow 5% tolerance due to float precision
            let tolerance = cpu.call_price.abs().max(1.0) * 0.05;
            let error = (cpu.call_price - gpu.call_price).abs();
            
            assert!(error < tolerance,
                "Extreme test {} (stock={:.0}): GPU={:.2} CPU={:.2} err={:.2}",
                i, input.stock_price, gpu.call_price, cpu.call_price, error);
        }
    }

    // ========================================================================
    // Monte Carlo GPU vs CPU Tests
    // ========================================================================

    #[test]
    fn test_monte_carlo_gpu_sanity_check() {
        // Monte Carlo GPU vs CPU - verify GPU produces reasonable values
        let cases = monte_carlo_edge_cases();
        let inputs: Vec<MonteCarloInput> = cases.iter().map(|(input, _)| *input).collect();
        let names: Vec<&str> = cases.iter().map(|(_, name)| *name).collect();
        
        let gpu_results = run_monte_carlo_gpu(inputs.clone());
        
        assert_eq!(gpu_results.len(), 10, "Should have 10 results");
        
        for (i, gpu) in gpu_results.iter().enumerate() {
            let name = names[i];
            let input = &inputs[i];
            
            // Check GPU output is valid (no NaN, non-negative)
            assert!(!gpu.call_price.is_nan(), "{}: GPU call price is NaN", name);
            assert!(!gpu.put_price.is_nan(), "{}: GPU put price is NaN", name);
            assert!(gpu.call_price >= 0.0, "{}: GPU call price is negative: {}", name, gpu.call_price);
            
            // Sanity check: deep ITM call should be > intrinsic value - some tolerance
            if input.stock_price > input.strike_price * 1.3 {
                let intrinsic = input.stock_price - input.strike_price;
                assert!(gpu.call_price > intrinsic * 0.5, 
                    "{}: Deep ITM GPU call {:.2} too low (intrinsic {:.2})", 
                    name, gpu.call_price, intrinsic);
            }
            
            // Sanity check: deep OTM call should be small
            if input.stock_price < input.strike_price * 0.6 {
                assert!(gpu.call_price < input.stock_price * 0.5, 
                    "{}: Deep OTM GPU call {:.2} too high", name, gpu.call_price);
            }
        }
    }

    #[test]
    fn test_monte_carlo_gpu_batch_processing() {
        // Verify GPU can process a batch of Monte Carlo options
        let inputs: Vec<MonteCarloInput> = (0..50).map(|i| {
            MonteCarloInput {
                stock_price: 80.0 + i as f32,
                strike_price: 100.0,
                time_to_expiry: 0.5,
                risk_free_rate: 0.05,
                volatility: 0.25,
            }
        }).collect();
        
        let gpu_results = run_monte_carlo_gpu(inputs);
        
        assert_eq!(gpu_results.len(), 50, "Should have 50 results");
        
        // Verify all results are valid
        for (i, gpu) in gpu_results.iter().enumerate() {
            assert!(!gpu.call_price.is_nan(), "Result {}: call is NaN", i);
            assert!(gpu.call_price >= 0.0, "Result {}: call is negative", i);
        }
        
        // Results should show a trend: higher stock price = higher call price
        // Check that last result (stock=129) has higher call than first (stock=80)
        assert!(gpu_results[49].call_price > gpu_results[0].call_price,
            "Higher stock price should yield higher call price");
    }

    #[test]
    fn test_monte_carlo_gpu_vs_cpu_1000_cases() {
        // Now that CPU and GPU use same deterministic seeds and identical RNG algorithms,
        // we can do exact comparison like Black-Scholes
        let inputs = generate_monte_carlo_inputs();
        assert_eq!(inputs.len(), NUM_TEST_CASES, "Should have {} inputs", NUM_TEST_CASES);
        
        let cpu_results = run_monte_carlo_cpu(inputs.clone());
        let gpu_results = run_monte_carlo_gpu(inputs.clone());
        
        assert_eq!(cpu_results.len(), NUM_TEST_CASES);
        assert_eq!(gpu_results.len(), NUM_TEST_CASES);
        
        let mut max_rel_error = 0.0f64;
        let mut max_abs_error = 0.0f32;
        let mut failed_count = 0usize;
        
        for (i, (cpu, gpu)) in cpu_results.iter().zip(gpu_results.iter()).enumerate() {
            let input = &inputs[i];
            
            // Check GPU output is valid
            assert!(!gpu.call_price.is_nan(), "MC Test {}: GPU call is NaN (stock={:.2})", i, input.stock_price);
            assert!(!gpu.put_price.is_nan(), "MC Test {}: GPU put is NaN (stock={:.2})", i, input.stock_price);
            
            let call_error = (cpu.call_price - gpu.call_price).abs();
            
            if call_error > max_abs_error {
                max_abs_error = call_error;
            }
            
            // Calculate relative error
            if cpu.call_price.abs() > 0.01 {
                let rel_error = (call_error / cpu.call_price.abs()) as f64;
                if rel_error > max_rel_error {
                    max_rel_error = rel_error;
                }
                
                // Allow 1% relative error for floating point differences
                if rel_error > 0.01 {
                    failed_count += 1;
                    if failed_count <= 5 {
                        eprintln!("MC MISMATCH Test {}: stock={:.2} cpu_call={:.6} gpu_call={:.6} err={:.4}%",
                            i, input.stock_price, cpu.call_price, gpu.call_price, rel_error * 100.0);
                    }
                }
            }
        }
        
        eprintln!("Monte Carlo 1000 GPU vs CPU: max_rel_err={:.4}%, max_abs_err={:.6}, failures={}",
            max_rel_error * 100.0, max_abs_error, failed_count);
        
        assert_eq!(failed_count, 0, "{} out of {} Monte Carlo tests failed tolerance check", failed_count, NUM_TEST_CASES);
    }

    #[test]
    fn test_monte_carlo_big_numbers_extreme() {
        // Huge number edge cases for Monte Carlo
        let inputs = vec![
            MonteCarloInput { stock_price: 100_000.0, strike_price: 100_000.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.20 },
            MonteCarloInput { stock_price: 500_000.0, strike_price: 500_000.0, time_to_expiry: 0.5, risk_free_rate: 0.04, volatility: 0.15 },
            MonteCarloInput { stock_price: 1_000_000.0, strike_price: 1_000_000.0, time_to_expiry: 1.0, risk_free_rate: 0.03, volatility: 0.25 },
            MonteCarloInput { stock_price: 750_000.0, strike_price: 500_000.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.20 },
            MonteCarloInput { stock_price: 250_000.0, strike_price: 500_000.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.20 },
        ];
        
        let cpu_results = run_monte_carlo_cpu(inputs.clone());
        let gpu_results = run_monte_carlo_gpu(inputs.clone());
        
        for (i, (cpu, gpu)) in cpu_results.iter().zip(gpu_results.iter()).enumerate() {
            let input = &inputs[i];
            
            assert!(!gpu.call_price.is_nan(), "MC Extreme test {}: GPU call is NaN", i);
            assert!(gpu.call_price >= -0.001, "MC Extreme test {}: GPU call is negative: {}", i, gpu.call_price);
            
            // For million-scale prices, allow 5% tolerance
            let tolerance = cpu.call_price.abs().max(1.0) * 0.05;
            let error = (cpu.call_price - gpu.call_price).abs();
            
            assert!(error < tolerance,
                "MC Extreme test {} (stock={:.0}): GPU={:.2} CPU={:.2} err={:.2}",
                i, input.stock_price, gpu.call_price, cpu.call_price, error);
        }
    }

    // ========================================================================
    // Double-Buffering Correctness Test
    // ========================================================================

    #[test]
    fn test_double_buffering_correctness() {
        // Test double-buffering correctness with multiple batches
        let test_inputs = [
            MonteCarloInput { stock_price: 100.0, strike_price: 105.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.2 },
            MonteCarloInput { stock_price: 110.0, strike_price: 100.0, time_to_expiry: 0.5, risk_free_rate: 0.03, volatility: 0.25 },
            MonteCarloInput { stock_price: 50.0, strike_price: 55.0, time_to_expiry: 2.0, risk_free_rate: 0.04, volatility: 0.3 },
        ];
        
        // Compute expected CPU results
        let cpu_results: Vec<MonteCarloOutput> = test_inputs
            .iter()
            .map(|input| monte_carlo_cpu(*input))
            .collect();
        
        // Run through GPU with double-buffering
        let gpu_results = run_monte_carlo_gpu(test_inputs.to_vec());
        
        // Verify results match
        assert_eq!(cpu_results.len(), gpu_results.len(), "Result count mismatch");
        
        let tolerance = 1e-3;
        for (i, (cpu, gpu)) in cpu_results.iter().zip(gpu_results.iter()).enumerate() {
            let call_diff = (cpu.call_price - gpu.call_price).abs();
            let put_diff = (cpu.put_price - gpu.put_price).abs();
            
            assert!(call_diff < tolerance && put_diff < tolerance,
                "Double-buffering test {} failed: call_diff={:.6}, put_diff={:.6}",
                i, call_diff, put_diff);
        }
    }

    // ========================================================================
    // Cross-Validation: GPU Black-Scholes vs CPU Monte Carlo
    // ========================================================================

    #[test]
    fn test_gpu_black_scholes_vs_cpu_monte_carlo() {
        // Compare analytical GPU Black-Scholes with stochastic CPU Monte Carlo
        // They should produce similar results for European options
        let bs_input = BlackScholesInput {
            stock_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        };
        
        let mc_input = MonteCarloInput {
            stock_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        };
        
        let bs_results = run_black_scholes_gpu(vec![bs_input]);
        let mc_result = monte_carlo_cpu(mc_input);
        
        // Monte Carlo with 1000 paths should be within 20% of Black-Scholes
        let call_error = (bs_results[0].call_price - mc_result.call_price).abs() / bs_results[0].call_price;
        assert!(call_error < 0.20, 
            "GPU BS call {:.4} vs CPU MC call {:.4}, error={:.2}%",
            bs_results[0].call_price, mc_result.call_price, call_error * 100.0);
    }

    #[test]
    fn test_gpu_monte_carlo_vs_cpu_black_scholes() {
        // Reverse comparison: GPU Monte Carlo vs CPU Black-Scholes
        let mc_input = MonteCarloInput {
            stock_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        };
        
        let bs_input = BlackScholesInput {
            stock_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        };
        
        let mc_results = run_monte_carlo_gpu(vec![mc_input]);
        let bs_result = black_scholes_cpu(bs_input);
        
        let call_error = (mc_results[0].call_price - bs_result.call_price).abs() / bs_result.call_price;
        assert!(call_error < 0.20,
            "GPU MC call {:.4} vs CPU BS call {:.4}, error={:.2}%",
            mc_results[0].call_price, bs_result.call_price, call_error * 100.0);
    }

    #[test]
    fn test_cross_validation_multiple_scenarios() {
        // Cross-validate both methods on multiple scenarios
        let scenarios = vec![
            // ATM
            (100.0, 100.0, 1.0, 0.05, 0.20),
            // ITM
            (120.0, 100.0, 0.5, 0.05, 0.20),
            // OTM
            (80.0, 100.0, 0.5, 0.05, 0.20),
            // High vol
            (100.0, 100.0, 1.0, 0.05, 0.50),
            // Low vol
            (100.0, 100.0, 1.0, 0.05, 0.10),
        ];
        
        for (s, k, t, r, v) in scenarios {
            let bs_input = BlackScholesInput {
                stock_price: s, strike_price: k, time_to_expiry: t,
                risk_free_rate: r, volatility: v,
            };
            let mc_input = MonteCarloInput {
                stock_price: s, strike_price: k, time_to_expiry: t,
                risk_free_rate: r, volatility: v,
            };
            
            let bs_result = run_black_scholes_gpu(vec![bs_input]);
            let mc_result = run_monte_carlo_gpu(vec![mc_input]);
            
            // Allow larger tolerance for Monte Carlo variance
            let call_diff = (bs_result[0].call_price - mc_result[0].call_price).abs();
            let tolerance = bs_result[0].call_price.max(1.0) * 0.25;
            
            assert!(call_diff < tolerance,
                "S={}, K={}: BS={:.4}, MC={:.4}, diff={:.4} > tol={:.4}",
                s, k, bs_result[0].call_price, mc_result[0].call_price, call_diff, tolerance);
        }
    }

    // ========================================================================
    // Boundary and Stress Tests
    // ========================================================================

    #[test]
    fn test_gpu_boundary_values() {
        // Test with boundary values that might cause numerical issues
        let cases = vec![
            // Very small values
            BlackScholesInput { stock_price: 0.01, strike_price: 0.01, time_to_expiry: 0.01, risk_free_rate: 0.001, volatility: 0.05 },
            // Very large values
            BlackScholesInput { stock_price: 100000.0, strike_price: 100000.0, time_to_expiry: 5.0, risk_free_rate: 0.10, volatility: 1.0 },
            // Extreme moneyness
            BlackScholesInput { stock_price: 1000.0, strike_price: 1.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.20 },
            // Negative rates
            BlackScholesInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 1.0, risk_free_rate: -0.02, volatility: 0.20 },
            // Very short expiry
            BlackScholesInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 0.0001, risk_free_rate: 0.05, volatility: 0.20 },
        ];
        
        let gpu_results = run_black_scholes_gpu(cases.clone());
        let cpu_results = run_black_scholes_cpu(cases);
        
        for (i, (gpu, cpu)) in gpu_results.iter().zip(cpu_results.iter()).enumerate() {
            assert!(!gpu.call_price.is_nan(), "Boundary case {}: GPU call is NaN", i);
            assert!(!gpu.call_price.is_infinite(), "Boundary case {}: GPU call is infinite", i);
            
            // GPU should match CPU
            let tolerance = cpu.call_price.abs().max(0.01) * 0.02;
            let error = (gpu.call_price - cpu.call_price).abs();
            assert!(error < tolerance,
                "Boundary case {}: GPU={:.6} CPU={:.6} error={:.6}",
                i, gpu.call_price, cpu.call_price, error);
        }
    }

    #[test]
    fn test_monte_carlo_boundary_values() {
        let cases = vec![
            MonteCarloInput { stock_price: 0.1, strike_price: 0.1, time_to_expiry: 0.01, risk_free_rate: 0.001, volatility: 0.10 },
            MonteCarloInput { stock_price: 100000.0, strike_price: 100000.0, time_to_expiry: 5.0, risk_free_rate: 0.10, volatility: 1.0 },
            MonteCarloInput { stock_price: 500.0, strike_price: 100.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.20 },
            MonteCarloInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 1.0, risk_free_rate: -0.02, volatility: 0.20 },
        ];
        
        let gpu_results = run_monte_carlo_gpu(cases.clone());
        let cpu_results = run_monte_carlo_cpu(cases);
        
        for (i, (gpu, cpu)) in gpu_results.iter().zip(cpu_results.iter()).enumerate() {
            assert!(!gpu.call_price.is_nan(), "MC boundary case {}: GPU call is NaN", i);
            assert!(!gpu.call_price.is_infinite(), "MC boundary case {}: GPU call is infinite", i);
            
            let tolerance = cpu.call_price.abs().max(0.1) * 0.02;
            let error = (gpu.call_price - cpu.call_price).abs();
            assert!(error < tolerance,
                "MC boundary case {}: GPU={:.4} CPU={:.4} error={:.4}",
                i, gpu.call_price, cpu.call_price, error);
        }
    }

    // ========================================================================
    // Comprehensive Integration Scenarios
    // ========================================================================

    #[test]
    fn test_end_to_end_streaming_black_scholes() {
        // Test end-to-end streaming with realistic data patterns
        let mut inputs = Vec::new();
        
        // Generate a realistic portfolio of options
        for stock_price in [50.0, 100.0, 150.0, 200.0] {
            for strike_delta in [-20.0, -10.0, 0.0, 10.0, 20.0] {
                for expiry in [0.25, 0.5, 1.0, 2.0] {
                    inputs.push(BlackScholesInput {
                        stock_price,
                        strike_price: stock_price + strike_delta,
                        time_to_expiry: expiry,
                        risk_free_rate: 0.05,
                        volatility: 0.20,
                    });
                }
            }
        }
        
        let gpu_results = run_black_scholes_gpu(inputs.clone());
        let cpu_results = run_black_scholes_cpu(inputs);
        
        assert_eq!(gpu_results.len(), cpu_results.len());
        
        let mut max_error = 0.0f32;
        for (gpu, cpu) in gpu_results.iter().zip(cpu_results.iter()) {
            let error = (gpu.call_price - cpu.call_price).abs();
            if error > max_error {
                max_error = error;
            }
        }
        
        assert!(max_error < 0.01, "Max error in streaming test: {:.6}", max_error);
    }

    #[test]
    fn test_end_to_end_streaming_monte_carlo() {
        // Test Monte Carlo with realistic portfolio
        let mut inputs = Vec::new();
        
        for stock_price in [80.0, 100.0, 120.0] {
            for strike in [90.0, 100.0, 110.0] {
                for vol in [0.15, 0.25, 0.35] {
                    inputs.push(MonteCarloInput {
                        stock_price,
                        strike_price: strike,
                        time_to_expiry: 1.0,
                        risk_free_rate: 0.05,
                        volatility: vol,
                    });
                }
            }
        }
        
        let gpu_results = run_monte_carlo_gpu(inputs.clone());
        let cpu_results = run_monte_carlo_cpu(inputs);
        
        assert_eq!(gpu_results.len(), cpu_results.len());
        
        let mut max_error = 0.0f32;
        for (gpu, cpu) in gpu_results.iter().zip(cpu_results.iter()) {
            let error = (gpu.call_price - cpu.call_price).abs();
            if error > max_error {
                max_error = error;
            }
        }
        
        assert!(max_error < 0.01, "Max MC streaming error: {:.6}", max_error);
    }

    #[test]
    fn test_mixed_batch_sizes() {
        // Test with varying batch sizes in sequence
        let sizes = vec![1, 10, 100, 1000, 10, 1];
        
        for size in sizes {
            let inputs: Vec<BlackScholesInput> = (0..size).map(|i| BlackScholesInput {
                stock_price: 90.0 + i as f32,
                strike_price: 100.0,
                time_to_expiry: 0.5,
                risk_free_rate: 0.05,
                volatility: 0.20,
            }).collect();
            
            let gpu_results = run_black_scholes_gpu(inputs.clone());
            let cpu_results = run_black_scholes_cpu(inputs);
            
            assert_eq!(gpu_results.len(), size);
            
            for (gpu, cpu) in gpu_results.iter().zip(cpu_results.iter()) {
                let error = (gpu.call_price - cpu.call_price).abs();
                assert!(error < 0.001, "Mixed batch size {}: error={}", size, error);
            }
        }
    }

    #[test]
    fn test_consistency_across_multiple_runs() {
        // Verify determinism: same inputs should always produce same outputs
        let inputs: Vec<BlackScholesInput> = (0..100).map(|i| BlackScholesInput {
            stock_price: 90.0 + i as f32,
            strike_price: 100.0,
            time_to_expiry: 0.5,
            risk_free_rate: 0.05,
            volatility: 0.20,
        }).collect();
        
        let results1 = run_black_scholes_gpu(inputs.clone());
        let results2 = run_black_scholes_gpu(inputs.clone());
        let results3 = run_black_scholes_gpu(inputs);
        
        for i in 0..100 {
            assert_eq!(results1[i].call_price, results2[i].call_price,
                "Run 1 vs 2 differ at index {}", i);
            assert_eq!(results2[i].call_price, results3[i].call_price,
                "Run 2 vs 3 differ at index {}", i);
        }
    }

    #[test]
    fn test_monte_carlo_consistency() {
        // Monte Carlo should also be deterministic with same seeds
        let inputs: Vec<MonteCarloInput> = (0..50).map(|i| MonteCarloInput {
            stock_price: 90.0 + i as f32,
            strike_price: 100.0,
            time_to_expiry: 0.5,
            risk_free_rate: 0.05,
            volatility: 0.25,
        }).collect();
        
        let results1 = run_monte_carlo_gpu(inputs.clone());
        let results2 = run_monte_carlo_gpu(inputs);
        
        for i in 0..50 {
            assert_eq!(results1[i].call_price, results2[i].call_price,
                "MC runs differ at index {}", i);
        }
    }

     #[test]
    fn test_mixed_pipeline_execution() {
        // Test a pipeline mixing CPU and GPU operators: Source -> Map(CPU) -> Map(GPU) -> Sink
        // CPU pre-processes input, GPU computes, CPU post-processes.
        let config = RuntimeConfig::local(1).unwrap();
        let env = StreamContext::new(config);
        
        let inputs = vec![
            BlackScholesInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.20 },
            BlackScholesInput { stock_price: 120.0, strike_price: 100.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.20 },
        ];
        
        let results = env.stream_iter(inputs.into_iter())
            .map(|mut input| {
                // CPU pre-processing: bump stock price by 10%
                input.stock_price *= 1.1;
                input
            })
            .map_gpu_with_strategy(
                BlackScholesKernel::default(),
                GpuBatchStrategy::fixed(100),
            )
            .map(|output| {
                // CPU post-processing: convert call price to string for fun, then parse back or just return price
                output.call_price
            })
            .collect_vec();
            
        env.execute_blocking();
        
        let final_results = results.get().unwrap();
        assert_eq!(final_results.len(), 2);
        
        // Validation
        // First input: 100 * 1.1 = 110. Black Scholes at 110.
        let expected_input = BlackScholesInput { stock_price: 110.0, strike_price: 100.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.20 };
        let expected_output = black_scholes_cpu(expected_input);
        assert!((final_results[0] - expected_output.call_price).abs() < 0.01);
    }
}

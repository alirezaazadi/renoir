//! Black-Scholes Kernel Unit Tests
//!
//! Tests for both CPU and GPU implementations.

use super::super::black_scholes::*;

// ============================================================================
// CPU Unit Tests
// ============================================================================

#[test]
fn test_cnd_at_zero() {
    // N(0) = 0.5 (symmetric distribution)
    let result = cnd_cpu(0.0);
    assert!((result - 0.5).abs() < 1e-6, "N(0) should be 0.5, got {}", result);
}

#[test]
fn test_cnd_symmetry() {
    // N(-x) = 1 - N(x)
    for x in [0.5, 1.0, 2.0, 3.0] {
        let left = cnd_cpu(-x);
        let right = 1.0 - cnd_cpu(x);
        assert!((left - right).abs() < 1e-6, 
            "N(-{}) = {} should equal 1 - N({}) = {}", x, left, x, right);
    }
}

#[test]
fn test_cnd_extreme_values() {
    // N(-∞) → 0, N(+∞) → 1
    assert!(cnd_cpu(-10.0) < 1e-6, "N(-10) should be near 0");
    assert!((cnd_cpu(10.0) - 1.0).abs() < 1e-6, "N(10) should be near 1");
}

#[test]
fn test_black_scholes_atm_option() {
    let input = BlackScholesInput {
        stock_price: 100.0,
        strike_price: 100.0,
        time_to_expiry: 1.0,
        risk_free_rate: 0.05,
        volatility: 0.20,
    };
    
    let output = black_scholes_cpu(input);
    
    // ATM call should be around $10.45 (verified against reference)
    assert!((output.call_price - 10.45).abs() < 0.5, 
        "ATM call price unexpected: {}", output.call_price);
    assert!((output.put_price - 5.57).abs() < 0.5, 
        "ATM put price unexpected: {}", output.put_price);
}

#[test]
fn test_black_scholes_deep_itm_call() {
    let input = BlackScholesInput {
        stock_price: 150.0,
        strike_price: 100.0,
        time_to_expiry: 0.5,
        risk_free_rate: 0.05,
        volatility: 0.20,
    };
    
    let output = black_scholes_cpu(input);
    
    // Deep ITM call should be close to intrinsic value (S - K)
    let intrinsic = input.stock_price - input.strike_price;
    assert!(output.call_price >= intrinsic, 
        "ITM call {} should exceed intrinsic {}", output.call_price, intrinsic);
    assert!(output.call_price < intrinsic + 10.0, 
        "ITM call should be close to intrinsic");
}

#[test]
fn test_black_scholes_deep_otm_call() {
    let input = BlackScholesInput {
        stock_price: 50.0,
        strike_price: 100.0,
        time_to_expiry: 0.5,
        risk_free_rate: 0.05,
        volatility: 0.20,
    };
    
    let output = black_scholes_cpu(input);
    
    // Deep OTM call should have small value close to 0
    assert!(output.call_price < 1.0, 
        "OTM call should be near zero: {}", output.call_price);
    assert!(output.call_price >= 0.0, "Call price should be non-negative");
}

#[test]
fn test_black_scholes_put_call_parity() {
    // Put-call parity: C - P = S - K*e^(-rT)
    let cases = [
        BlackScholesInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.20 },
        BlackScholesInput { stock_price: 150.0, strike_price: 100.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.20 },
        BlackScholesInput { stock_price: 50.0, strike_price: 100.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.30 },
        BlackScholesInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 0.01, risk_free_rate: 0.05, volatility: 0.30 },
        BlackScholesInput { stock_price: 10000.0, strike_price: 10000.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.20 },
    ];
    
    for input in cases {
        let output = black_scholes_cpu(input);
        let df = (-input.risk_free_rate * input.time_to_expiry).exp();
        let parity_lhs = output.call_price - output.put_price;
        let parity_rhs = input.stock_price - input.strike_price * df;
        let parity_error = (parity_lhs - parity_rhs).abs();
        
        assert!(parity_error < 0.01, 
            "Put-call parity violated for S={}, K={}: error={:.6}", 
            input.stock_price, input.strike_price, parity_error);
    }
}

#[test]
fn test_black_scholes_edge_cases() {
    let cases = vec![
        ("Short expiry", BlackScholesInput { 
            stock_price: 100.0, strike_price: 105.0, time_to_expiry: 0.01, 
            risk_free_rate: 0.05, volatility: 0.30 
        }),
        ("Long expiry", BlackScholesInput { 
            stock_price: 100.0, strike_price: 100.0, time_to_expiry: 3.0, 
            risk_free_rate: 0.03, volatility: 0.25 
        }),
        ("High volatility", BlackScholesInput { 
            stock_price: 100.0, strike_price: 100.0, time_to_expiry: 0.5, 
            risk_free_rate: 0.05, volatility: 0.80 
        }),
        ("Low volatility", BlackScholesInput { 
            stock_price: 100.0, strike_price: 100.0, time_to_expiry: 0.5, 
            risk_free_rate: 0.05, volatility: 0.05 
        }),
        ("Zero interest rate", BlackScholesInput { 
            stock_price: 100.0, strike_price: 100.0, time_to_expiry: 1.0, 
            risk_free_rate: 0.0, volatility: 0.20 
        }),
        ("Big numbers", BlackScholesInput { 
            stock_price: 100000.0, strike_price: 100000.0, time_to_expiry: 1.0, 
            risk_free_rate: 0.05, volatility: 0.20 
        }),
    ];
    
    for (name, input) in cases {
        let output = black_scholes_cpu(input);
        
        // Basic sanity checks
        assert!(output.call_price >= 0.0, "{}: call should be non-negative", name);
        assert!(output.put_price >= 0.0, "{}: put should be non-negative", name);
        assert!(!output.call_price.is_nan(), "{}: call should not be NaN", name);
        assert!(!output.put_price.is_nan(), "{}: put should not be NaN", name);
    }
}

#[test]
fn test_black_scholes_output_non_negative() {
    // Prices should never be negative
    let input = BlackScholesInput {
        stock_price: 1.0,
        strike_price: 1.0,
        time_to_expiry: 0.25,
        risk_free_rate: 0.05,
        volatility: 0.40,
    };
    
    let output = black_scholes_cpu(input);
    assert!(output.call_price >= 0.0);
    assert!(output.put_price >= 0.0);
}

// ============================================================================
// Boundary and Invalid Input Tests
// ============================================================================

#[test]
fn test_black_scholes_very_small_values() {
    // Test with very small but valid values
    let input = BlackScholesInput {
        stock_price: 0.01,
        strike_price: 0.01,
        time_to_expiry: 0.001,
        risk_free_rate: 0.001,
        volatility: 0.01,
    };
    
    let output = black_scholes_cpu(input);
    assert!(!output.call_price.is_nan(), "Call should not be NaN for small values");
    assert!(!output.put_price.is_nan(), "Put should not be NaN for small values");
    assert!(output.call_price >= 0.0, "Call should be non-negative");
}

#[test]
fn test_black_scholes_very_large_values() {
    // Test with very large values to check for overflow
    let input = BlackScholesInput {
        stock_price: 1_000_000.0,
        strike_price: 1_000_000.0,
        time_to_expiry: 10.0,
        risk_free_rate: 0.20,
        volatility: 2.0,
    };
    
    let output = black_scholes_cpu(input);
    assert!(!output.call_price.is_nan(), "Call should not be NaN for large values");
    assert!(!output.put_price.is_nan(), "Put should not be NaN for large values");
    assert!(!output.call_price.is_infinite(), "Call should not be infinite");
}

#[test]
fn test_black_scholes_extreme_moneyness() {
    // Very deep ITM: S >> K
    let itm_input = BlackScholesInput {
        stock_price: 1000.0,
        strike_price: 1.0,
        time_to_expiry: 1.0,
        risk_free_rate: 0.05,
        volatility: 0.20,
    };
    
    let itm_output = black_scholes_cpu(itm_input);
    let intrinsic = itm_input.stock_price - itm_input.strike_price;
    assert!(itm_output.call_price >= intrinsic * 0.99, 
        "Deep ITM call should be close to intrinsic value");
    assert!(itm_output.put_price < 1.0, "Deep ITM put should be near zero");
    
    // Very deep OTM: S << K
    let otm_input = BlackScholesInput {
        stock_price: 1.0,
        strike_price: 1000.0,
        time_to_expiry: 1.0,
        risk_free_rate: 0.05,
        volatility: 0.20,
    };
    
    let otm_output = black_scholes_cpu(otm_input);
    assert!(otm_output.call_price < 1.0, "Deep OTM call should be near zero");
    assert!(otm_output.put_price > 900.0, "Deep OTM put should be near intrinsic");
}

#[test]
fn test_black_scholes_negative_interest_rate() {
    // Negative interest rates are valid in some markets (e.g., Europe/Japan)
    let input = BlackScholesInput {
        stock_price: 100.0,
        strike_price: 100.0,
        time_to_expiry: 1.0,
        risk_free_rate: -0.01,
        volatility: 0.20,
    };
    
    let output = black_scholes_cpu(input);
    assert!(!output.call_price.is_nan(), "Should handle negative rates");
    assert!(!output.put_price.is_nan(), "Should handle negative rates");
    assert!(output.call_price >= 0.0);
    assert!(output.put_price >= 0.0);
    
    // Put-call parity should still hold
    let df = (-input.risk_free_rate * input.time_to_expiry).exp();
    let parity_error = ((output.call_price - output.put_price) - 
                        (input.stock_price - input.strike_price * df)).abs();
    assert!(parity_error < 0.01, "Put-call parity should hold with negative rates");
}

#[test]
fn test_black_scholes_very_short_expiry() {
    // Test with very short time to expiry (e.g., end of trading day)
    let input = BlackScholesInput {
        stock_price: 100.0,
        strike_price: 100.0,
        time_to_expiry: 0.0001, // ~1 hour
        risk_free_rate: 0.05,
        volatility: 0.20,
    };
    
    let output = black_scholes_cpu(input);
    assert!(!output.call_price.is_nan());
    // Very short expiry ATM option should have very small value
    assert!(output.call_price < 1.0);
}

#[test]
fn test_black_scholes_very_long_expiry() {
    // Test with very long time to expiry (e.g., 30 years)
    let input = BlackScholesInput {
        stock_price: 100.0,
        strike_price: 100.0,
        time_to_expiry: 30.0,
        risk_free_rate: 0.05,
        volatility: 0.20,
    };
    
    let output = black_scholes_cpu(input);
    assert!(!output.call_price.is_nan());
    assert!(!output.call_price.is_infinite());
    // Long expiry option should have significant value
    assert!(output.call_price > 10.0);
}

#[test]
fn test_black_scholes_extreme_volatility() {
    // Very low volatility (nearly deterministic)
    let low_vol = BlackScholesInput {
        stock_price: 100.0,
        strike_price: 100.0,
        time_to_expiry: 1.0,
        risk_free_rate: 0.05,
        volatility: 0.001,
    };
    
    let low_vol_output = black_scholes_cpu(low_vol);
    assert!(!low_vol_output.call_price.is_nan());
    
    // Very high volatility (very uncertain)
    let high_vol = BlackScholesInput {
        stock_price: 100.0,
        strike_price: 100.0,
        time_to_expiry: 1.0,
        risk_free_rate: 0.05,
        volatility: 5.0,
    };
    
    let high_vol_output = black_scholes_cpu(high_vol);
    assert!(!high_vol_output.call_price.is_nan());
    assert!(high_vol_output.call_price > low_vol_output.call_price, 
        "Higher volatility should increase option value");
}

#[test]
fn test_black_scholes_precision_loss_scenarios() {
    // Test scenario where S and K are very close but not equal
    let input = BlackScholesInput {
        stock_price: 100.0000,
        strike_price: 100.0001,
        time_to_expiry: 1.0,
        risk_free_rate: 0.05,
        volatility: 0.20,
    };
    
    let output = black_scholes_cpu(input);
    assert!(!output.call_price.is_nan());
    assert!(output.call_price >= 0.0);
    
    // Test with very disparate values
    let disparate = BlackScholesInput {
        stock_price: 0.001,
        strike_price: 1000000.0,
        time_to_expiry: 1.0,
        risk_free_rate: 0.05,
        volatility: 0.20,
    };
    
    let disparate_output = black_scholes_cpu(disparate);
    assert!(!disparate_output.call_price.is_nan());
    // Call should be essentially zero
    assert!(disparate_output.call_price < 0.001);
}

#[test]
fn test_cnd_boundary_values() {
    // Test CND at specific known points
    let test_cases = vec![
        (-3.0, 0.00135),  // 3 sigma
        (-2.0, 0.02275),  // 2 sigma
        (-1.0, 0.15866),  // 1 sigma
        (0.0, 0.5),       // mean
        (1.0, 0.84134),   // 1 sigma
        (2.0, 0.97725),   // 2 sigma
        (3.0, 0.99865),   // 3 sigma
    ];
    
    for (x, expected) in test_cases {
        let result = cnd_cpu(x);
        let error = (result - expected).abs();
        assert!(error < 0.001, "CND({}) = {}, expected {}, error = {}", 
            x, result, expected, error);
    }
}

#[test]
fn test_black_scholes_denormalized_numbers() {
    // Test with values that might produce denormalized floating point numbers
    let input = BlackScholesInput {
        stock_price: 1e-30,
        strike_price: 1e-30,
        time_to_expiry: 0.1,
        risk_free_rate: 0.05,
        volatility: 0.20,
    };
    
    let output = black_scholes_cpu(input);
    // Should produce valid (possibly very small) results
    assert!(!output.call_price.is_nan());
    assert!(output.call_price.is_finite());
}

#[test]
fn test_black_scholes_invalid_inputs() {
    // NaN inputs should propagate NaN (or handle gracefully if we wanted, but typically propagate)
    let input_nan = BlackScholesInput {
        stock_price: f32::NAN,
        strike_price: 100.0,
        time_to_expiry: 1.0,
        risk_free_rate: 0.05,
        volatility: 0.20,
    };
    let output = black_scholes_cpu(input_nan);
    assert!(output.call_price.is_nan(), "NaN input should produce NaN output");

    // Infinite inputs
    let input_inf = BlackScholesInput {
        stock_price: f32::INFINITY,
        strike_price: 100.0,
        time_to_expiry: 1.0,
        risk_free_rate: 0.05,
        volatility: 0.20,
    };
    let output_inf = black_scholes_cpu(input_inf);
    // With infinite stock, call should be infinite (or very large), put should be 0
    assert!(output_inf.call_price.is_infinite() || output_inf.call_price > 1e30);
}

// ============================================================================
// GPU Unit Tests
// ============================================================================

#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]
mod gpu_tests {
    use super::*;
    use renoir::operator::gpu::{GpuBatchStrategy, GpuContext, GpuKernel};
    use renoir::prelude::*;

    // ========================================================================
    // Helper Functions
    // ========================================================================

    fn run_gpu(inputs: Vec<BlackScholesInput>) -> Vec<BlackScholesOutput> {
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

    fn run_cpu(inputs: Vec<BlackScholesInput>) -> Vec<BlackScholesOutput> {
        let config = RuntimeConfig::local(1).unwrap();
        let env = StreamContext::new(config);
        
        let output = env.stream_iter(inputs.into_iter())
            .map(black_scholes_cpu)
            .collect_vec();
        env.execute_blocking();
        
        output.get().unwrap()
    }

    // ========================================================================
    // GPU Kernel Method Tests (push, flush, drain)
    // ========================================================================

    #[test]
    fn test_kernel_push_increments_buffer() {
        let mut kernel = BlackScholesKernel::default();
        
        assert_eq!(kernel.buffer_len(), 0, "Buffer should start empty");
        
        let input = BlackScholesInput {
            stock_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        };
        
        kernel.push(input);
        assert_eq!(kernel.buffer_len(), 1, "Buffer should have 1 item after push");
        
        kernel.push(input);
        kernel.push(input);
        assert_eq!(kernel.buffer_len(), 3, "Buffer should have 3 items after 3 pushes");
    }

    #[test]
    fn test_kernel_flush_clears_buffer() {
        let ctx = GpuContext::new();
        let mut kernel = BlackScholesKernel::default();
        
        let input = BlackScholesInput {
            stock_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        };
        
        kernel.push(input);
        kernel.push(input);
        assert_eq!(kernel.buffer_len(), 2);
        
        let _results = kernel.flush(&ctx);
        assert_eq!(kernel.buffer_len(), 0, "Buffer should be cleared after flush");
    }

    #[test]
    fn test_kernel_flush_returns_previous_batch() {
        let ctx = GpuContext::new();
        let mut kernel = BlackScholesKernel::default();
        
        let input1 = BlackScholesInput {
            stock_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        };
        
        // First flush: push 2 items, flush returns empty (no previous batch)
        kernel.push(input1);
        kernel.push(input1);
        let results1 = kernel.flush(&ctx);
        assert_eq!(results1.len(), 0, "First flush should return empty (double-buffering)");
        
        // Second flush: push 1 item, flush returns 2 items from previous batch
        let input2 = BlackScholesInput {
            stock_price: 120.0,
            strike_price: 100.0,
            time_to_expiry: 0.5,
            risk_free_rate: 0.05,
            volatility: 0.25,
        };
        kernel.push(input2);
        let results2 = kernel.flush(&ctx);
        assert_eq!(results2.len(), 2, "Second flush should return 2 items from first batch");
    }

    #[test]
    fn test_kernel_drain_returns_pending() {
        let ctx = GpuContext::new();
        let mut kernel = BlackScholesKernel::default();
        
        let input = BlackScholesInput {
            stock_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        };
        
        // Push and flush to create pending batch
        kernel.push(input);
        kernel.push(input);
        kernel.push(input);
        let _results = kernel.flush(&ctx);  // Returns empty, creates pending
        
        // Drain should return the pending batch
        let drained = kernel.drain(&ctx);
        assert_eq!(drained.len(), 3, "Drain should return 3 items from pending batch");
    }

    #[test]
    fn test_kernel_drain_empty_when_no_pending() {
        let ctx = GpuContext::new();
        let mut kernel = BlackScholesKernel::default();
        
        // Drain with nothing pending
        let drained = kernel.drain(&ctx);
        assert_eq!(drained.len(), 0, "Drain should return empty when nothing pending");
    }

    #[test]
    fn test_kernel_full_workflow() {
        let ctx = GpuContext::new();
        let mut kernel = BlackScholesKernel::default();
        
        let inputs: Vec<BlackScholesInput> = (0..5).map(|i| BlackScholesInput {
            stock_price: 100.0 + i as f32 * 10.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        }).collect();
        
        // Push all inputs
        for input in &inputs {
            kernel.push(*input);
        }
        assert_eq!(kernel.buffer_len(), 5);
        
        // Flush sends to GPU
        let results1 = kernel.flush(&ctx);
        assert_eq!(results1.len(), 0, "First flush returns empty");
        assert_eq!(kernel.buffer_len(), 0, "Buffer cleared");
        
        // Drain gets results
        let final_results = kernel.drain(&ctx);
        assert_eq!(final_results.len(), 5, "Drain returns 5 results");
        
        // Verify results match CPU
        for (i, (result, input)) in final_results.iter().zip(inputs.iter()).enumerate() {
            let cpu = black_scholes_cpu(*input);
            let error = (result.call_price - cpu.call_price).abs();
            assert!(error < 0.01, "Result {}: GPU={} CPU={}", i, result.call_price, cpu.call_price);
        }
    }

    // ========================================================================
    // GPU CND/erf Tests (validated via output correctness)
    // ========================================================================

    #[test]
    fn test_gpu_cnd_via_atm_pricing() {
        // ATM option tests GPU's CND implementation at d1, d2 values near 0
        let input = BlackScholesInput {
            stock_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        };
        
        let gpu_results = run_gpu(vec![input]);
        let cpu_result = black_scholes_cpu(input);
        
        // If GPU CND is correct, prices should match
        let error = (gpu_results[0].call_price - cpu_result.call_price).abs();
        assert!(error < 0.01, "GPU CND should match CPU: error={}", error);
    }

    #[test]
    fn test_gpu_cnd_via_extreme_itm_otm() {
        // Deep ITM/OTM tests GPU's CND at extreme values
        let inputs = vec![
            BlackScholesInput { stock_price: 200.0, strike_price: 100.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.20 }, // Very ITM
            BlackScholesInput { stock_price: 50.0, strike_price: 100.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.20 },  // Very OTM
        ];
        
        let gpu_results = run_gpu(inputs.clone());
        
        // ITM: call should be very close to intrinsic
        let intrinsic = inputs[0].stock_price - inputs[0].strike_price;
        assert!(gpu_results[0].call_price >= intrinsic * 0.99, 
            "GPU ITM call {} should be >= intrinsic {}", gpu_results[0].call_price, intrinsic);
        
        // OTM: call should be very small
        assert!(gpu_results[1].call_price < 0.1, 
            "GPU OTM call {} should be near zero", gpu_results[1].call_price);
    }

    // ========================================================================
    // GPU vs CPU Comparison Tests
    // ========================================================================

    #[test]
    fn test_gpu_single_option() {
        let input = BlackScholesInput {
            stock_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        };
        
        let gpu_result = run_gpu(vec![input]);
        let cpu_result = black_scholes_cpu(input);
        
        assert_eq!(gpu_result.len(), 1);
        assert!((gpu_result[0].call_price - cpu_result.call_price).abs() < 0.001,
            "GPU call {} differs from CPU {}", gpu_result[0].call_price, cpu_result.call_price);
    }

    #[test]
    fn test_gpu_vectorization_correctness() {
        // Test that vectorization produces correct results
        // Create inputs that test padding and alignment
        let inputs: Vec<BlackScholesInput> = (0..17).map(|i| BlackScholesInput {
            stock_price: 100.0 + i as f32,
            strike_price: 100.0,
            time_to_expiry: 0.5,
            risk_free_rate: 0.05,
            volatility: 0.20,
        }).collect();
        
        let gpu_results = run_gpu(inputs.clone());
        let cpu_results = run_cpu(inputs);
        
        assert_eq!(gpu_results.len(), cpu_results.len());
        
        for (i, (gpu, cpu)) in gpu_results.iter().zip(cpu_results.iter()).enumerate() {
            let error = (gpu.call_price - cpu.call_price).abs();
            assert!(error < 0.001, "Option {}: GPU={} CPU={} error={}", 
                i, gpu.call_price, cpu.call_price, error);
        }
    }

    #[test]
    fn test_gpu_output_valid() {
        let inputs = vec![
            BlackScholesInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.20 },
            BlackScholesInput { stock_price: 50.0, strike_price: 100.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.30 },
            BlackScholesInput { stock_price: 150.0, strike_price: 100.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.20 },
        ];
        
        let results = run_gpu(inputs);
        
        for (i, result) in results.iter().enumerate() {
            assert!(!result.call_price.is_nan(), "Option {}: call is NaN", i);
            assert!(!result.put_price.is_nan(), "Option {}: put is NaN", i);
            assert!(result.call_price >= 0.0, "Option {}: call is negative", i);
            assert!(result.put_price >= 0.0, "Option {}: put is negative", i);
        }
    }

    #[test]
    fn test_gpu_big_numbers() {
        let inputs = vec![
            BlackScholesInput { stock_price: 100_000.0, strike_price: 100_000.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.20 },
            BlackScholesInput { stock_price: 1_000_000.0, strike_price: 1_000_000.0, time_to_expiry: 0.5, risk_free_rate: 0.03, volatility: 0.25 },
        ];
        
        let gpu_results = run_gpu(inputs.clone());
        let cpu_results = run_cpu(inputs);
        
        for (i, (gpu, cpu)) in gpu_results.iter().zip(cpu_results.iter()).enumerate() {
            let tolerance = cpu.call_price.abs() * 0.05;
            let error = (gpu.call_price - cpu.call_price).abs();
            assert!(error < tolerance, "Big number test {}: error {} > tol {}", i, error, tolerance);
        }
    }

    #[test]
    fn test_gpu_put_call_parity() {
        let input = BlackScholesInput {
            stock_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        };
        
        let results = run_gpu(vec![input]);
        let result = &results[0];
        
        // Verify put-call parity holds for GPU output
        let df = (-input.risk_free_rate * input.time_to_expiry).exp();
        let parity_lhs = result.call_price - result.put_price;
        let parity_rhs = input.stock_price - input.strike_price * df;
        let error = (parity_lhs - parity_rhs).abs();
        
        assert!(error < 0.01, "GPU put-call parity error: {}", error);
    }

    // ========================================================================
    // GPU Batch Size and Alignment Tests
    // ========================================================================

    #[test]
    fn test_gpu_various_batch_sizes() {
        // Test different batch sizes including edge cases around vectorization boundaries
        let batch_sizes = vec![1, 2, 3, 4, 5, 7, 15, 16, 17, 63, 64, 65, 
                               127, 128, 129, 255, 256, 257, 511, 512, 513, 1000];
        
        for batch_size in batch_sizes {
            let inputs: Vec<BlackScholesInput> = (0..batch_size).map(|i| BlackScholesInput {
                stock_price: 90.0 + i as f32,
                strike_price: 100.0,
                time_to_expiry: 0.5,
                risk_free_rate: 0.05,
                volatility: 0.20,
            }).collect();
            
            let gpu_results = run_gpu(inputs.clone());
            let cpu_results = run_cpu(inputs);
            
            assert_eq!(gpu_results.len(), batch_size, "Batch size {}: incorrect result count", batch_size);
            
            for (i, (gpu, cpu)) in gpu_results.iter().zip(cpu_results.iter()).enumerate() {
                let error = (gpu.call_price - cpu.call_price).abs();
                assert!(error < 0.001, 
                    "Batch size {}, item {}: GPU={} CPU={} error={}", 
                    batch_size, i, gpu.call_price, cpu.call_price, error);
            }
        }
    }

    #[test]
    fn test_gpu_large_batch() {
        // Test large batch to stress-test GPU grid dispatch
        let batch_size = 100_000;
        let inputs: Vec<BlackScholesInput> = (0..batch_size).map(|i| BlackScholesInput {
            stock_price: 50.0 + (i % 100) as f32,
            strike_price: 100.0,
            time_to_expiry: 0.5,
            risk_free_rate: 0.05,
            volatility: 0.20,
        }).collect();
        
        let gpu_results = run_gpu(inputs);
        
        assert_eq!(gpu_results.len(), batch_size);
        
        // Verify all results are valid
        for (i, result) in gpu_results.iter().enumerate() {
            assert!(!result.call_price.is_nan(), "Large batch item {}: call is NaN", i);
            assert!(!result.put_price.is_nan(), "Large batch item {}: put is NaN", i);
            assert!(result.call_price >= 0.0, "Large batch item {}: call is negative", i);
        }
    }

    #[test]
    fn test_gpu_unaligned_vectorization() {
        // Test sizes that require padding for vectorization (not multiples of 4)
        let sizes = vec![1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15];
        
        for size in sizes {
            let inputs: Vec<BlackScholesInput> = (0..size).map(|_| BlackScholesInput {
                stock_price: 100.0,
                strike_price: 100.0,
                time_to_expiry: 1.0,
                risk_free_rate: 0.05,
                volatility: 0.20,
            }).collect();
            
            let gpu_results = run_gpu(inputs.clone());
            let cpu_results = run_cpu(inputs);
            
            assert_eq!(gpu_results.len(), size);
            
            for (gpu, cpu) in gpu_results.iter().zip(cpu_results.iter()) {
                let error = (gpu.call_price - cpu.call_price).abs();
                assert!(error < 0.001, "Unaligned size {}: error={}", size, error);
            }
        }
    }

    // ========================================================================
    // GPU Double-Buffering Edge Cases
    // ========================================================================

    #[test]
    fn test_gpu_multiple_flushes_without_push() {
        let ctx = GpuContext::new();
        let mut kernel = BlackScholesKernel::default();
        
        let input = BlackScholesInput {
            stock_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        };
        
        // Push and flush first batch
        kernel.push(input);
        let results1 = kernel.flush(&ctx);
        assert_eq!(results1.len(), 0, "First flush returns empty");
        
        // Flush again without pushing (should return previous batch)
        let results2 = kernel.flush(&ctx);
        assert_eq!(results2.len(), 1, "Second flush returns first batch");
        
        // Flush again without pushing (should return empty)
        let results3 = kernel.flush(&ctx);
        assert_eq!(results3.len(), 0, "Third flush returns empty");
    }

    #[test]
    fn test_gpu_drain_without_flush() {
        let ctx = GpuContext::new();
        let mut kernel = BlackScholesKernel::default();
        
        // Drain without any flush
        let results = kernel.drain(&ctx);
        assert_eq!(results.len(), 0, "Drain without flush should return empty");
    }

    #[test]
    fn test_gpu_multiple_drains() {
        let ctx = GpuContext::new();
        let mut kernel = BlackScholesKernel::default();
        
        let input = BlackScholesInput {
            stock_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        };
        
        kernel.push(input);
        kernel.flush(&ctx);
        
        // First drain gets results
        let results1 = kernel.drain(&ctx);
        assert_eq!(results1.len(), 1);
        
        // Second drain should be empty
        let results2 = kernel.drain(&ctx);
        assert_eq!(results2.len(), 0, "Multiple drains should return empty after first");
    }

    #[test]
    fn test_gpu_interleaved_push_flush() {
        let ctx = GpuContext::new();
        let mut kernel = BlackScholesKernel::default();
        
        let inputs: Vec<BlackScholesInput> = (0..3).map(|i| BlackScholesInput {
            stock_price: 100.0 + i as f32 * 10.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        }).collect();
        
        // Batch 1: 1 item
        kernel.push(inputs[0]);
        let r1 = kernel.flush(&ctx);
        assert_eq!(r1.len(), 0);
        
        // Batch 2: 2 items
        kernel.push(inputs[1]);
        kernel.push(inputs[2]);
        let r2 = kernel.flush(&ctx);
        assert_eq!(r2.len(), 1, "Should return 1 item from batch 1");
        
        // Drain final batch
        let r3 = kernel.drain(&ctx);
        assert_eq!(r3.len(), 2, "Should return 2 items from batch 2");
    }

    // ========================================================================
    // GPU Numerical Accuracy with Proven Reference Values
    // ========================================================================

    #[test]
    fn test_gpu_reference_values() {
        // Known reference values from Black-Scholes implementation
        // We verify GPU matches CPU, and CPU produces reasonable values
        struct ReferenceCase {
            input: BlackScholesInput,
            expected_call: f32,
            name: &'static str,
        }
        
        let cases = vec![
            ReferenceCase {
                input: BlackScholesInput {
                    stock_price: 100.0,
                    strike_price: 100.0,
                    time_to_expiry: 1.0,
                    risk_free_rate: 0.05,
                    volatility: 0.20,
                },
                expected_call: 10.4506,
                name: "ATM 1Y",
            },
            ReferenceCase {
                input: BlackScholesInput {
                    stock_price: 100.0,
                    strike_price: 95.0,
                    time_to_expiry: 0.25,
                    risk_free_rate: 0.10,
                    volatility: 0.50,
                },
                expected_call: 13.70, // Approximate reference
                name: "ITM 3M high vol",
            },
        ];
        
        for case in cases {
            let cpu_result = black_scholes_cpu(case.input);
            let gpu_result = run_gpu(vec![case.input]);
            
            // CPU should match reference within a reasonable tolerance
            let cpu_call_err = (cpu_result.call_price - case.expected_call).abs();
            assert!(cpu_call_err < 0.5, "{}: CPU call error={}", case.name, cpu_call_err);
            
            // GPU should match CPU within 0.001
            let gpu_call_err = (gpu_result[0].call_price - cpu_result.call_price).abs();
            let gpu_put_err = (gpu_result[0].put_price - cpu_result.put_price).abs();
            assert!(gpu_call_err < 0.001, "{}: GPU call error={}", case.name, gpu_call_err);
            assert!(gpu_put_err < 0.001, "{}: GPU put error={}", case.name, gpu_put_err);
        }
    }

     #[test]
    fn test_kernel_preferred_batch_size() {
        let kernel = BlackScholesKernel::default();
        assert!(kernel.preferred_batch_size().is_some(), "Should report preferred batch size");
        assert!(kernel.preferred_batch_size().unwrap() > 0, "Batch size should be positive");
    }

    #[test]
    fn test_gpu_nan_handling() {
        let input = BlackScholesInput {
            stock_price: f32::NAN,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        };
        // GPU should handle this without crashing, output is likely NaN
        let results = run_gpu(vec![input]);
        assert!(results[0].call_price.is_nan(), "GPU should propagate NaN");
    }

    // ========================================================================
    // GPU Stress and Performance Tests
    // ========================================================================

    #[test]
    fn test_gpu_repeated_execution() {
        // Test for memory leaks and resource cleanup
        let input = BlackScholesInput {
            stock_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        };
        
        let expected = black_scholes_cpu(input);
        
        // Run 100 times to detect memory issues
        for i in 0..100 {
            let result = run_gpu(vec![input]);
            assert_eq!(result.len(), 1, "Iteration {}: wrong result count", i);
            
            let error = (result[0].call_price - expected.call_price).abs();
            assert!(error < 0.001, "Iteration {}: error={}", i, error);
        }
    }
}

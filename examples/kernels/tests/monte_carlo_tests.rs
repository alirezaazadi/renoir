//! Monte Carlo Kernel Unit Tests
//!
//! Tests for both CPU and GPU implementations.

use super::super::monte_carlo::*;

// ============================================================================
// Deterministic Seed Tests
// ============================================================================

#[test]
fn test_deterministic_seed_reproducible() {
    // Same inputs should produce same seed
    let input = MonteCarloInput {
        stock_price: 100.0,
        strike_price: 105.0,
        time_to_expiry: 1.0,
        risk_free_rate: 0.05,
        volatility: 0.2,
    };
    
    let seed1 = deterministic_seed(&input);
    let seed2 = deterministic_seed(&input);
    
    assert_eq!(seed1, seed2, "Same inputs should produce same seed");
}

#[test]
fn test_deterministic_seed_unique() {
    // Different inputs should produce different seeds
    let input1 = MonteCarloInput {
        stock_price: 100.0,
        strike_price: 105.0,
        time_to_expiry: 1.0,
        risk_free_rate: 0.05,
        volatility: 0.2,
    };
    
    let input2 = MonteCarloInput {
        stock_price: 100.0,
        strike_price: 106.0, // Different strike
        time_to_expiry: 1.0,
        risk_free_rate: 0.05,
        volatility: 0.2,
    };
    
    let seed1 = deterministic_seed(&input1);
    let seed2 = deterministic_seed(&input2);
    
    assert_ne!(seed1, seed2, "Different inputs should produce different seeds");
}

#[test]
fn test_deterministic_seed_non_zero() {
    // Seed should never be zero (xorshift breaks on zero)
    let inputs = vec![
        MonteCarloInput { stock_price: 0.0, strike_price: 0.0, time_to_expiry: 0.0, risk_free_rate: 0.0, volatility: 0.0 },
        MonteCarloInput { stock_price: 1.0, strike_price: 1.0, time_to_expiry: 1.0, risk_free_rate: 0.0, volatility: 0.0 },
        MonteCarloInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.2 },
    ];
    
    for input in inputs {
        let seed = deterministic_seed(&input);
        assert_ne!(seed, 0, "Seed should never be zero");
    }
}

// ============================================================================
// RNG Tests (testing internal functions via monte_carlo_cpu behavior)
// ============================================================================

#[test]
fn test_monte_carlo_reproducible() {
    // Same input should always produce same output (deterministic seeding)
    let input = MonteCarloInput {
        stock_price: 100.0,
        strike_price: 105.0,
        time_to_expiry: 1.0,
        risk_free_rate: 0.05,
        volatility: 0.2,
    };
    
    let output1 = monte_carlo_cpu(input);
    let output2 = monte_carlo_cpu(input);
    
    assert_eq!(output1.call_price, output2.call_price, 
        "Monte Carlo should be reproducible");
    assert_eq!(output1.put_price, output2.put_price, 
        "Monte Carlo should be reproducible");
}

// ============================================================================
// Monte Carlo Pricing Tests
// ============================================================================

#[test]
fn test_monte_carlo_atm_option() {
    let input = MonteCarloInput {
        stock_price: 100.0,
        strike_price: 100.0,
        time_to_expiry: 1.0,
        risk_free_rate: 0.05,
        volatility: 0.20,
    };
    
    let output = monte_carlo_cpu(input);
    
    // ATM call should be roughly similar to Black-Scholes (~10.45)
    // Allow for Monte Carlo variance (wider range)
    assert!(output.call_price > 8.0, "ATM call too low: {}", output.call_price);
    assert!(output.call_price < 13.0, "ATM call too high: {}", output.call_price);
}

#[test]
fn test_monte_carlo_deep_itm_call() {
    let input = MonteCarloInput {
        stock_price: 150.0,
        strike_price: 100.0,
        time_to_expiry: 0.5,
        risk_free_rate: 0.05,
        volatility: 0.20,
    };
    
    let output = monte_carlo_cpu(input);
    
    // Deep ITM call should be close to intrinsic value
    let intrinsic = input.stock_price - input.strike_price;
    assert!(output.call_price >= intrinsic * 0.9, 
        "ITM call {} should be near intrinsic {}", output.call_price, intrinsic);
}

#[test]
fn test_monte_carlo_deep_otm_call() {
    let input = MonteCarloInput {
        stock_price: 50.0,
        strike_price: 100.0,
        time_to_expiry: 0.5,
        risk_free_rate: 0.05,
        volatility: 0.20,
    };
    
    let output = monte_carlo_cpu(input);
    
    // Deep OTM call should have small value
    assert!(output.call_price < 5.0, "OTM call should be small: {}", output.call_price);
    assert!(output.call_price >= 0.0, "Call price should be non-negative");
}

#[test]
fn test_monte_carlo_put_call_parity() {
    // Put-call parity: C - P = S - K*e^(-rT)
    // Allow larger tolerance for Monte Carlo variance
    let cases = [
        MonteCarloInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.20 },
        MonteCarloInput { stock_price: 150.0, strike_price: 100.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.20 },
        MonteCarloInput { stock_price: 10000.0, strike_price: 10000.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.20 },
    ];
    
    for input in cases {
        let output = monte_carlo_cpu(input);
        let df = (-input.risk_free_rate * input.time_to_expiry).exp();
        let parity_lhs = output.call_price - output.put_price;
        let parity_rhs = input.stock_price - input.strike_price * df;
        let parity_error = (parity_lhs - parity_rhs).abs();
        
        // Monte Carlo has higher variance, allow 5% of stock price as tolerance
        let tolerance = input.stock_price * 0.05;
        assert!(parity_error < tolerance, 
            "Put-call parity violated for S={}: error={:.6} > tol={:.6}", 
            input.stock_price, parity_error, tolerance);
    }
}

#[test]
fn test_monte_carlo_edge_cases() {
    let cases = vec![
        ("Short expiry", MonteCarloInput { 
            stock_price: 100.0, strike_price: 105.0, time_to_expiry: 0.1, 
            risk_free_rate: 0.05, volatility: 0.30 
        }),
        ("Long expiry", MonteCarloInput { 
            stock_price: 100.0, strike_price: 100.0, time_to_expiry: 2.0, 
            risk_free_rate: 0.03, volatility: 0.25 
        }),
        ("High volatility", MonteCarloInput { 
            stock_price: 100.0, strike_price: 100.0, time_to_expiry: 0.5, 
            risk_free_rate: 0.05, volatility: 0.60 
        }),
        ("Low volatility", MonteCarloInput { 
            stock_price: 100.0, strike_price: 100.0, time_to_expiry: 0.5, 
            risk_free_rate: 0.05, volatility: 0.10 
        }),
        ("Zero interest rate", MonteCarloInput { 
            stock_price: 100.0, strike_price: 100.0, time_to_expiry: 1.0, 
            risk_free_rate: 0.0, volatility: 0.20 
        }),
        ("Big numbers", MonteCarloInput { 
            stock_price: 100000.0, strike_price: 100000.0, time_to_expiry: 1.0, 
            risk_free_rate: 0.05, volatility: 0.20 
        }),
    ];
    
    for (name, input) in cases {
        let output = monte_carlo_cpu(input);
        
        // Basic sanity checks
        assert!(output.call_price >= 0.0, "{}: call should be non-negative", name);
        assert!(!output.call_price.is_nan(), "{}: call should not be NaN", name);
        assert!(!output.put_price.is_nan(), "{}: put should not be NaN", name);
    }
}

#[test]
fn test_monte_carlo_invalid_inputs() {
    // NaN input test - Monte Carlo may or may not propagate NaN
    // depending on how the seed and RNG handle it
    let input = MonteCarloInput {
        stock_price: f32::NAN,
        strike_price: 100.0,
        time_to_expiry: 1.0,
        risk_free_rate: 0.05,
        volatility: 0.20,
    };
    let output = monte_carlo_cpu(input);
    // Either NaN or some numeric result is acceptable
    // (depends on how deterministic_seed handles NaN bits)
    let _ = output.call_price; // Just verify it doesn't panic
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

    fn run_gpu(inputs: Vec<MonteCarloInput>) -> Vec<MonteCarloOutput> {
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

    fn run_cpu(inputs: Vec<MonteCarloInput>) -> Vec<MonteCarloOutput> {
        let config = RuntimeConfig::local(1).unwrap();
        let env = StreamContext::new(config);
        
        let output = env.stream_iter(inputs.into_iter())
            .map(monte_carlo_cpu)
            .collect_vec();
        env.execute_blocking();
        
        output.get().unwrap()
    }

    // ========================================================================
    // GPU Kernel Method Tests (push, flush, drain)
    // ========================================================================

    #[test]
    fn test_kernel_push_increments_buffer() {
        let mut kernel = MonteCarloKernel::default();
        
        assert_eq!(kernel.buffer_len(), 0, "Buffer should start empty");
        
        let input = MonteCarloInput {
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
        let mut kernel = MonteCarloKernel::default();
        
        let input = MonteCarloInput {
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
        let mut kernel = MonteCarloKernel::default();
        
        let input1 = MonteCarloInput {
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
        let input2 = MonteCarloInput {
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
        let mut kernel = MonteCarloKernel::default();
        
        let input = MonteCarloInput {
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
        let mut kernel = MonteCarloKernel::default();
        
        // Drain with nothing pending
        let drained = kernel.drain(&ctx);
        assert_eq!(drained.len(), 0, "Drain should return empty when nothing pending");
    }

    #[test]
    fn test_kernel_full_workflow() {
        let ctx = GpuContext::new();
        let mut kernel = MonteCarloKernel::default();
        
        let inputs: Vec<MonteCarloInput> = (0..5).map(|i| MonteCarloInput {
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
        
        // Verify results are valid
        for result in &final_results {
            assert!(!result.call_price.is_nan());
            assert!(result.call_price >= 0.0);
        }
    }

    // ========================================================================
    // GPU RNG Tests (validated via deterministic output matching CPU)
    // ========================================================================

    #[test]
    fn test_gpu_rng_deterministic() {
        // GPU uses same xorshift + box_muller as CPU
        // Same input should produce identical results
        let input = MonteCarloInput {
            stock_price: 100.0,
            strike_price: 105.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.2,
        };
        
        let cpu_result = monte_carlo_cpu(input);
        let gpu_result = run_gpu(vec![input]);
        
        assert_eq!(gpu_result.len(), 1);
        
        // GPU and CPU should match exactly due to same deterministic RNG
        let error = (cpu_result.call_price - gpu_result[0].call_price).abs();
        assert!(error < 0.001, 
            "GPU RNG should match CPU: gpu={} cpu={} error={}", 
            gpu_result[0].call_price, cpu_result.call_price, error);
    }

    #[test]
    fn test_gpu_rng_different_seeds() {
        // Different inputs should produce different results (different seeds)
        let input1 = MonteCarloInput {
            stock_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        };
        
        let input2 = MonteCarloInput {
            stock_price: 100.0,
            strike_price: 101.0, // Different strike = different seed
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        };
        
        let results = run_gpu(vec![input1, input2]);
        
        assert_eq!(results.len(), 2);
        assert_ne!(results[0].call_price, results[1].call_price,
            "Different inputs should produce different results due to different RNG seeds");
    }

    #[test]
    fn test_gpu_rng_produces_valid_prices() {
        // GPU RNG should produce reasonable option prices
        let inputs = vec![
            MonteCarloInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.20 },
            MonteCarloInput { stock_price: 50.0, strike_price: 100.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.30 },
            MonteCarloInput { stock_price: 150.0, strike_price: 100.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.20 },
        ];
        
        let results = run_gpu(inputs.clone());
        
        for (i, (input, result)) in inputs.iter().zip(results.iter()).enumerate() {
            // Prices should be non-negative and not NaN
            assert!(!result.call_price.is_nan(), "Option {}: call is NaN", i);
            assert!(result.call_price >= 0.0, "Option {}: call is negative", i);
            
            // ITM call should be near intrinsic value
            if input.stock_price > input.strike_price * 1.2 {
                let intrinsic = input.stock_price - input.strike_price;
                assert!(result.call_price >= intrinsic * 0.8,
                    "Option {}: ITM call {} too low (intrinsic {})", i, result.call_price, intrinsic);
            }
        }
    }

    // ========================================================================
    // GPU vs CPU Comparison Tests
    // ========================================================================

    #[test]
    fn test_gpu_single_option() {
        let input = MonteCarloInput {
            stock_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        };
        
        let gpu_result = run_gpu(vec![input]);
        let cpu_result = monte_carlo_cpu(input);
        
        assert_eq!(gpu_result.len(), 1);
        assert!((gpu_result[0].call_price - cpu_result.call_price).abs() < 0.01,
            "GPU call {} differs from CPU {}", gpu_result[0].call_price, cpu_result.call_price);
    }

    #[test]
    fn test_gpu_deterministic() {
        let input = MonteCarloInput {
            stock_price: 100.0,
            strike_price: 105.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.2,
        };
        
        let result1 = run_gpu(vec![input]);
        let result2 = run_gpu(vec![input]);
        
        assert_eq!(result1[0].call_price, result2[0].call_price, 
            "GPU should be deterministic");
    }

    #[test]
    fn test_gpu_batch_processing() {
        let inputs: Vec<MonteCarloInput> = (0..20).map(|i| MonteCarloInput {
            stock_price: 80.0 + i as f32 * 3.0,
            strike_price: 100.0,
            time_to_expiry: 0.5,
            risk_free_rate: 0.05,
            volatility: 0.25,
        }).collect();
        
        let gpu_results = run_gpu(inputs.clone());
        let cpu_results = run_cpu(inputs);
        
        assert_eq!(gpu_results.len(), cpu_results.len());
        
        for (i, (gpu, cpu)) in gpu_results.iter().zip(cpu_results.iter()).enumerate() {
            let error = (gpu.call_price - cpu.call_price).abs();
            assert!(error < 0.01, "Option {}: GPU={} CPU={} error={}", 
                i, gpu.call_price, cpu.call_price, error);
        }
    }

    #[test]
    fn test_gpu_output_valid() {
        let inputs = vec![
            MonteCarloInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.20 },
            MonteCarloInput { stock_price: 50.0, strike_price: 100.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.30 },
            MonteCarloInput { stock_price: 150.0, strike_price: 100.0, time_to_expiry: 0.5, risk_free_rate: 0.05, volatility: 0.20 },
        ];
        
        let results = run_gpu(inputs);
        
        for (i, result) in results.iter().enumerate() {
            assert!(!result.call_price.is_nan(), "Option {}: call is NaN", i);
            assert!(!result.put_price.is_nan(), "Option {}: put is NaN", i);
            assert!(result.call_price >= 0.0, "Option {}: call is negative", i);
        }
    }

    #[test]
    fn test_gpu_big_numbers() {
        let inputs = vec![
            MonteCarloInput { stock_price: 100_000.0, strike_price: 100_000.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.20 },
            MonteCarloInput { stock_price: 500_000.0, strike_price: 500_000.0, time_to_expiry: 0.5, risk_free_rate: 0.03, volatility: 0.15 },
        ];
        
        let gpu_results = run_gpu(inputs.clone());
        let cpu_results = run_cpu(inputs);
        
        for (i, (gpu, cpu)) in gpu_results.iter().zip(cpu_results.iter()).enumerate() {
            let tolerance = cpu.call_price.abs().max(1.0) * 0.05;
            let error = (gpu.call_price - cpu.call_price).abs();
            assert!(error < tolerance, "Big number test {}: error {} > tol {}", i, error, tolerance);
        }
    }

    #[test]
    fn test_gpu_itm_otm_sanity() {
        let itm_input = MonteCarloInput {
            stock_price: 120.0,
            strike_price: 100.0,
            time_to_expiry: 0.5,
            risk_free_rate: 0.05,
            volatility: 0.20,
        };
        
        let otm_input = MonteCarloInput {
            stock_price: 80.0,
            strike_price: 100.0,
            time_to_expiry: 0.5,
            risk_free_rate: 0.05,
            volatility: 0.20,
        };
        
        let results = run_gpu(vec![itm_input, otm_input]);
        
        assert!(results[0].call_price > results[1].call_price,
            "ITM call {} should be greater than OTM call {}", 
            results[0].call_price, results[1].call_price);
    }

    // ========================================================================
    // GPU Batch Size and Alignment Tests
    // ========================================================================

    #[test]
    fn test_gpu_various_batch_sizes() {
        let batch_sizes = vec![1, 2, 3, 7, 15, 16, 17, 63, 64, 65, 
                               127, 128, 129, 255, 256, 257, 500];
        
        for batch_size in batch_sizes {
            let inputs: Vec<MonteCarloInput> = (0..batch_size).map(|i| MonteCarloInput {
                stock_price: 90.0 + i as f32,
                strike_price: 100.0,
                time_to_expiry: 0.5,
                risk_free_rate: 0.05,
                volatility: 0.25,
            }).collect();
            
            let gpu_results = run_gpu(inputs.clone());
            let cpu_results = run_cpu(inputs);
            
            assert_eq!(gpu_results.len(), batch_size);
            
            for (i, (gpu, cpu)) in gpu_results.iter().zip(cpu_results.iter()).enumerate() {
                let error = (gpu.call_price - cpu.call_price).abs();
                assert!(error < 0.01, 
                    "Batch size {}, item {}: GPU={} CPU={} error={}", 
                    batch_size, i, gpu.call_price, cpu.call_price, error);
            }
        }
    }

    #[test]
    fn test_gpu_large_batch_monte_carlo() {
        let batch_size = 50_000;
        let inputs: Vec<MonteCarloInput> = (0..batch_size).map(|i| MonteCarloInput {
            stock_price: 80.0 + (i % 50) as f32,
            strike_price: 100.0,
            time_to_expiry: 0.5,
            risk_free_rate: 0.05,
            volatility: 0.25,
        }).collect();
        
        let gpu_results = run_gpu(inputs);
        
        assert_eq!(gpu_results.len(), batch_size);
        
        for (i, result) in gpu_results.iter().enumerate() {
            assert!(!result.call_price.is_nan(), "Item {}: call is NaN", i);
            assert!(result.call_price >= 0.0, "Item {}: call is negative", i);
        }
    }

    // ========================================================================
    // GPU Double-Buffering Edge Cases
    // ========================================================================

    #[test]
    fn test_gpu_multiple_flushes_mc() {
        let ctx = GpuContext::new();
        let mut kernel = MonteCarloKernel::default();
        
        let input = MonteCarloInput {
            stock_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        };
        
        kernel.push(input);
        let r1 = kernel.flush(&ctx);
        assert_eq!(r1.len(), 0);
        
        let r2 = kernel.flush(&ctx);
        assert_eq!(r2.len(), 1);
        
        let r3 = kernel.flush(&ctx);
        assert_eq!(r3.len(), 0);
    }

    #[test]
    fn test_gpu_drain_multiple_times_mc() {
        let ctx = GpuContext::new();
        let mut kernel = MonteCarloKernel::default();
        
        let input = MonteCarloInput {
            stock_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        };
        
        kernel.push(input);
        kernel.flush(&ctx);
        
        let r1 = kernel.drain(&ctx);
        assert_eq!(r1.len(), 1);
        
        let r2 = kernel.drain(&ctx);
        assert_eq!(r2.len(), 0);
    }

    // ========================================================================
    // GPU Stress Tests
    // ========================================================================

    #[test]
    fn test_gpu_repeated_execution_mc() {
        let input = MonteCarloInput {
            stock_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        };
        
        let expected = monte_carlo_cpu(input);
        
        for i in 0..50 {
            let result = run_gpu(vec![input]);
            assert_eq!(result.len(), 1, "Iteration {}", i);
            
            let error = (result[0].call_price - expected.call_price).abs();
            assert!(error < 0.01, "Iteration {}: error={}", i, error);
        }
    }

    #[test]
    fn test_gpu_nan_inputs() {
        // NaN input test - GPU may or may not propagate NaN
        // The seed generation uses bit manipulation which may produce valid seeds even from NaN
        let input = MonteCarloInput {
            stock_price: f32::NAN,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        };
        let results = run_gpu(vec![input]);
        // Just verify execution completes without crashing
        // NaN propagation depends on seed generation implementation
        let _ = results[0].call_price;
    }
}

// ============================================================================
// Boundary and Invalid Input Tests (CPU)
// ============================================================================

#[test]
fn test_monte_carlo_very_small_values() {
    let input = MonteCarloInput {
        stock_price: 0.1,
        strike_price: 0.1,
        time_to_expiry: 0.01,
        risk_free_rate: 0.001,
        volatility: 0.05,
    };
    
    let output = monte_carlo_cpu(input);
    assert!(!output.call_price.is_nan());
    assert!(!output.put_price.is_nan());
    assert!(output.call_price >= 0.0);
}

#[test]
fn test_monte_carlo_very_large_values() {
    let input = MonteCarloInput {
        stock_price: 1_000_000.0,
        strike_price: 1_000_000.0,
        time_to_expiry: 5.0,
        risk_free_rate: 0.10,
        volatility: 1.0,
    };
    
    let output = monte_carlo_cpu(input);
    assert!(!output.call_price.is_nan());
    assert!(!output.call_price.is_infinite());
}

#[test]
fn test_monte_carlo_negative_interest_rate() {
    let input = MonteCarloInput {
        stock_price: 100.0,
        strike_price: 100.0,
        time_to_expiry: 1.0,
        risk_free_rate: -0.02,
        volatility: 0.20,
    };
    
    let output = monte_carlo_cpu(input);
    assert!(!output.call_price.is_nan());
    assert!(output.call_price >= 0.0);
    
    // Put-call parity should still hold
    let df = (-input.risk_free_rate * input.time_to_expiry).exp();
    let parity_error = ((output.call_price - output.put_price) - 
                        (input.stock_price - input.strike_price * df)).abs();
    let tolerance = input.stock_price * 0.05;
    assert!(parity_error < tolerance);
}

#[test]
fn test_monte_carlo_extreme_moneyness() {
    // Very deep ITM
    let itm = MonteCarloInput {
        stock_price: 500.0,
        strike_price: 100.0,
        time_to_expiry: 1.0,
        risk_free_rate: 0.05,
        volatility: 0.20,
    };
    
    let itm_output = monte_carlo_cpu(itm);
    let intrinsic = itm.stock_price - itm.strike_price;
    assert!(itm_output.call_price >= intrinsic * 0.8);
    
    // Very deep OTM
    let otm = MonteCarloInput {
        stock_price: 10.0,
        strike_price: 500.0,
        time_to_expiry: 1.0,
        risk_free_rate: 0.05,
        volatility: 0.20,
    };
    
    let otm_output = monte_carlo_cpu(otm);
    assert!(otm_output.call_price < 10.0);
}

#[test]
fn test_monte_carlo_extreme_volatility() {
    // Very low volatility
    let low_vol = MonteCarloInput {
        stock_price: 100.0,
        strike_price: 100.0,
        time_to_expiry: 1.0,
        risk_free_rate: 0.05,
        volatility: 0.01,
    };
    
    let low_vol_output = monte_carlo_cpu(low_vol);
    assert!(!low_vol_output.call_price.is_nan());
    
    // Very high volatility
    let high_vol = MonteCarloInput {
        stock_price: 100.0,
        strike_price: 100.0,
        time_to_expiry: 1.0,
        risk_free_rate: 0.05,
        volatility: 3.0,
    };
    
    let high_vol_output = monte_carlo_cpu(high_vol);
    assert!(!high_vol_output.call_price.is_nan());
    assert!(high_vol_output.call_price > low_vol_output.call_price);
}

// ============================================================================
// RNG Quality Tests
// ============================================================================

#[test]
fn test_rng_uniformity() {
    // Test that xorshift produces uniform distribution
    // We'll use a simple chi-square-like test
    let mut seed = 12345u32;
    let n_samples = 10000;
    let n_bins = 10;
    let mut bins = vec![0u32; n_bins];
    
    for _ in 0..n_samples {
        // Simulate xorshift
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let uniform = (seed >> 9) as f32 * 1.1920929e-7;
        
        let bin = ((uniform * n_bins as f32).floor() as usize).min(n_bins - 1);
        bins[bin] += 1;
    }
    
    // Each bin should have roughly n_samples / n_bins = 1000
    let expected = n_samples / n_bins as u32;
    let tolerance = (expected as f32 * 0.15) as u32; // 15% tolerance
    
    for (i, &count) in bins.iter().enumerate() {
        let diff = if count > expected { count - expected } else { expected - count };
        assert!(diff < tolerance, 
            "Bin {} has {} samples (expected {} ± {})", i, count, expected, tolerance);
    }
}

#[test]
fn test_rng_seed_diversity() {
    // Test that different inputs produce sufficiently different seeds
    let inputs = vec![
        MonteCarloInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.20 },
        MonteCarloInput { stock_price: 100.1, strike_price: 100.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.20 },
        MonteCarloInput { stock_price: 100.0, strike_price: 100.1, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.20 },
        MonteCarloInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 1.1, risk_free_rate: 0.05, volatility: 0.20 },
        MonteCarloInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 1.0, risk_free_rate: 0.06, volatility: 0.20 },
        MonteCarloInput { stock_price: 100.0, strike_price: 100.0, time_to_expiry: 1.0, risk_free_rate: 0.05, volatility: 0.21 },
    ];
    
    let seeds: Vec<u32> = inputs.iter().map(deterministic_seed).collect();
    
    // All seeds should be unique
    for i in 0..seeds.len() {
        for j in (i+1)..seeds.len() {
            assert_ne!(seeds[i], seeds[j], 
                "Inputs {} and {} produced same seed", i, j);
        }
    }
}

#[test]
fn test_monte_carlo_convergence() {
    // Monte Carlo should converge to true value with more paths
    // We'll compare against Black-Scholes as the "true" value
    use super::super::black_scholes::{black_scholes_cpu, BlackScholesInput};
    
    let input = MonteCarloInput {
        stock_price: 100.0,
        strike_price: 100.0,
        time_to_expiry: 1.0,
        risk_free_rate: 0.05,
        volatility: 0.20,
    };
    
    let bs_input = BlackScholesInput {
        stock_price: input.stock_price,
        strike_price: input.strike_price,
        time_to_expiry: input.time_to_expiry,
        risk_free_rate: input.risk_free_rate,
        volatility: input.volatility,
    };
    
    let bs_result = black_scholes_cpu(bs_input);
    let mc_result = monte_carlo_cpu(input);
    
    // With 1000 paths, should be within 20% of Black-Scholes
    let error = (mc_result.call_price - bs_result.call_price).abs();
    let relative_error = error / bs_result.call_price;
    assert!(relative_error < 0.20, 
        "MC call {} should be within 20% of BS call {}, error={}%", 
        mc_result.call_price, bs_result.call_price, relative_error * 100.0);
}

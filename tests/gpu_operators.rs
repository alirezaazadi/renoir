//! # GPU Operator Integration Tests
//!
//! Tests for the `map_gpu` and `reduce_gpu` streaming operators.
//! These tests validate that the operators work correctly within Renoir's
//! dataflow pipeline -- batching, streaming, collecting, and producing
//! correct results end-to-end.
//!
//! Run with:
//!   cargo test --test gpu_operators --features gpu-wgpu
//!   cargo test --test gpu_operators --features gpu-cuda

#[path = "../examples/kernels/mod.rs"]
mod kernels;

use kernels::black_scholes::{black_scholes_cpu, BlackScholesInput, BlackScholesKernel};

// ============================================================================
// map_gpu Operator Tests
// ============================================================================

#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]
mod map_gpu_tests {
    use super::*;
    use renoir::operator::gpu::GpuBatchStrategy;
    use renoir::prelude::*;

    #[test]
    fn test_map_gpu_basic_black_scholes() {
        // Verify map_gpu produces results that match CPU for a small batch
        let inputs: Vec<BlackScholesInput> = (0..100)
            .map(|i| BlackScholesInput {
                stock_price: 80.0 + (i as f32) * 0.5,
                strike_price: 100.0,
                time_to_expiry: 1.0,
                risk_free_rate: 0.05,
                volatility: 0.20,
            })
            .collect();

        let cpu_results: Vec<_> = inputs.iter().map(|i| black_scholes_cpu(*i)).collect();

        let config = RuntimeConfig::local(1).unwrap();
        let env = StreamContext::new(config);
        let gpu_output = env
            .stream_iter(inputs.into_iter())
            .map_gpu_with_strategy(
                BlackScholesKernel::default(),
                GpuBatchStrategy::fixed(1000),
            )
            .collect_vec();
        env.execute_blocking();
        let gpu_results = gpu_output.get().unwrap();

        assert_eq!(
            gpu_results.len(),
            cpu_results.len(),
            "GPU and CPU should produce the same number of results"
        );

        for (i, (gpu, cpu)) in gpu_results.iter().zip(cpu_results.iter()).enumerate() {
            let error = (gpu.call_price - cpu.call_price).abs();
            assert!(
                error < 0.01,
                "Result {i}: GPU call_price={:.6} vs CPU call_price={:.6}, error={:.6}",
                gpu.call_price,
                cpu.call_price,
                error
            );
        }
    }

    #[test]
    fn test_map_gpu_empty_stream() {
        // map_gpu should handle an empty input stream gracefully
        let config = RuntimeConfig::local(1).unwrap();
        let env = StreamContext::new(config);
        let output = env
            .stream_iter(std::iter::empty::<BlackScholesInput>())
            .map_gpu_with_strategy(
                BlackScholesKernel::default(),
                GpuBatchStrategy::fixed(1000),
            )
            .collect_vec();
        env.execute_blocking();
        let results = output.get().unwrap();

        assert!(results.is_empty(), "Empty input should produce empty output");
    }

    #[test]
    fn test_map_gpu_single_element() {
        // map_gpu should handle a single element correctly
        let input = BlackScholesInput {
            stock_price: 100.0,
            strike_price: 100.0,
            time_to_expiry: 1.0,
            risk_free_rate: 0.05,
            volatility: 0.20,
        };
        let cpu_result = black_scholes_cpu(input);

        let config = RuntimeConfig::local(1).unwrap();
        let env = StreamContext::new(config);
        let output = env
            .stream_iter(std::iter::once(input))
            .map_gpu_with_strategy(
                BlackScholesKernel::default(),
                GpuBatchStrategy::fixed(1000),
            )
            .collect_vec();
        env.execute_blocking();
        let results = output.get().unwrap();

        assert_eq!(results.len(), 1, "Should produce exactly one result");
        let error = (results[0].call_price - cpu_result.call_price).abs();
        assert!(error < 0.01, "Single element error: {error:.6}");
    }

    #[test]
    fn test_map_gpu_multiple_batches() {
        // Use a small batch size to force multiple GPU dispatches
        let inputs: Vec<BlackScholesInput> = (0..500)
            .map(|i| BlackScholesInput {
                stock_price: 50.0 + (i as f32) * 0.4,
                strike_price: 100.0,
                time_to_expiry: 0.5,
                risk_free_rate: 0.03,
                volatility: 0.25,
            })
            .collect();

        let cpu_results: Vec<_> = inputs.iter().map(|i| black_scholes_cpu(*i)).collect();

        let config = RuntimeConfig::local(1).unwrap();
        let env = StreamContext::new(config);
        let gpu_output = env
            .stream_iter(inputs.into_iter())
            .map_gpu_with_strategy(
                BlackScholesKernel::default(),
                GpuBatchStrategy::fixed(100), // small batch → multiple dispatches
            )
            .collect_vec();
        env.execute_blocking();
        let gpu_results = gpu_output.get().unwrap();

        assert_eq!(gpu_results.len(), cpu_results.len());

        let max_error = gpu_results
            .iter()
            .zip(cpu_results.iter())
            .map(|(g, c)| (g.call_price - c.call_price).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_error < 0.01,
            "Max error across multiple batches: {max_error:.6}"
        );
    }

    #[test]
    fn test_map_gpu_preserves_order() {
        // Verify output order matches input order.
        // Use stock prices well above the strike so call prices are clearly
        // distinct and monotonically increasing (avoiding near-zero f32 noise).
        let inputs: Vec<BlackScholesInput> = (0..200)
            .map(|i| BlackScholesInput {
                stock_price: 100.0 + (i as f32) * 1.0, // 100..300, all ITM
                strike_price: 100.0,
                time_to_expiry: 1.0,
                risk_free_rate: 0.05,
                volatility: 0.20,
            })
            .collect();

        let config = RuntimeConfig::local(1).unwrap();
        let env = StreamContext::new(config);
        let gpu_output = env
            .stream_iter(inputs.into_iter())
            .map_gpu_with_strategy(
                BlackScholesKernel::default(),
                GpuBatchStrategy::fixed(50),
            )
            .collect_vec();
        env.execute_blocking();
        let gpu_results = gpu_output.get().unwrap();

        assert_eq!(gpu_results.len(), 200);

        // With monotonically increasing stock price (all ITM) and fixed strike,
        // call prices should be monotonically increasing
        for i in 1..gpu_results.len() {
            assert!(
                gpu_results[i].call_price >= gpu_results[i - 1].call_price,
                "Order violated at index {i}: {} < {}",
                gpu_results[i].call_price,
                gpu_results[i - 1].call_price
            );
        }
    }
}

// ============================================================================
// reduce_gpu Operator Tests
// ============================================================================

#[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]
mod reduce_gpu_tests {
    use renoir::operator::{ReduceGpuConfig, ReduceKernel};
    use renoir::prelude::*;

    #[test]
    fn test_reduce_gpu_sum_basic() {
        let data: Vec<f64> = (1..=1000).map(|x| x as f64).collect();
        let expected: f64 = data.iter().sum();

        let config = RuntimeConfig::local(1).unwrap();
        let env = StreamContext::new(config);
        let output = env
            .stream_iter(data.into_iter())
            .reduce_gpu(ReduceKernel::Sum)
            .collect_vec();
        env.execute_blocking();
        let results = output.get().unwrap();

        assert_eq!(results.len(), 1, "Reduce should produce exactly one result");
        let rel_error = (results[0] - expected).abs() / expected;
        assert!(
            rel_error < 0.001,
            "Sum: got {}, expected {}, relative error {:.6}",
            results[0],
            expected,
            rel_error
        );
    }

    #[test]
    fn test_reduce_gpu_product() {
        // Use values close to 1.0 to avoid overflow/underflow
        let data: Vec<f64> = (0..100).map(|i| 1.0 + (i as f64) * 0.001).collect();
        let expected: f64 = data.iter().product();

        let config = RuntimeConfig::local(1).unwrap();
        let env = StreamContext::new(config);
        let output = env
            .stream_iter(data.into_iter())
            .reduce_gpu(ReduceKernel::Product)
            .collect_vec();
        env.execute_blocking();
        let results = output.get().unwrap();

        assert_eq!(results.len(), 1);
        let rel_error = (results[0] - expected).abs() / expected.abs();
        assert!(
            rel_error < 0.01,
            "Product: got {}, expected {}, relative error {:.6}",
            results[0],
            expected,
            rel_error
        );
    }

    #[test]
    fn test_reduce_gpu_min() {
        let data: Vec<f64> = vec![42.0, 7.0, 99.0, 3.14, 1000.0, -5.5, 0.0, 88.8];
        let expected = -5.5f64;

        let config = RuntimeConfig::local(1).unwrap();
        let env = StreamContext::new(config);
        let output = env
            .stream_iter(data.into_iter())
            .reduce_gpu(ReduceKernel::Min)
            .collect_vec();
        env.execute_blocking();
        let results = output.get().unwrap();

        assert_eq!(results.len(), 1);
        assert!(
            (results[0] - expected).abs() < 0.001,
            "Min: got {}, expected {}",
            results[0],
            expected
        );
    }

    #[test]
    fn test_reduce_gpu_max() {
        let data: Vec<f64> = vec![42.0, 7.0, 99.0, 3.14, 1000.0, -5.5, 0.0, 88.8];
        let expected = 1000.0f64;

        let config = RuntimeConfig::local(1).unwrap();
        let env = StreamContext::new(config);
        let output = env
            .stream_iter(data.into_iter())
            .reduce_gpu(ReduceKernel::Max)
            .collect_vec();
        env.execute_blocking();
        let results = output.get().unwrap();

        assert_eq!(results.len(), 1);
        assert!(
            (results[0] - expected).abs() < 0.001,
            "Max: got {}, expected {}",
            results[0],
            expected
        );
    }

    #[test]
    fn test_reduce_gpu_sum_large() {
        // Test with a larger dataset that requires multiple batch flushes
        let data: Vec<f64> = (1..=100_000).map(|x| x as f64).collect();
        let expected: f64 = data.iter().sum();

        let reduce_config = ReduceGpuConfig::default().with_batch_size(10_000);

        let config = RuntimeConfig::local(1).unwrap();
        let env = StreamContext::new(config);
        let output = env
            .stream_iter(data.into_iter())
            .reduce_gpu_with(ReduceKernel::Sum, reduce_config)
            .collect_vec();
        env.execute_blocking();
        let results = output.get().unwrap();

        assert_eq!(results.len(), 1);
        let rel_error = (results[0] - expected).abs() / expected;
        assert!(
            rel_error < 0.001,
            "Large sum: got {}, expected {}, relative error {:.6}",
            results[0],
            expected,
            rel_error
        );
    }

    #[test]
    fn test_reduce_gpu_min_max_large() {
        // Verify min/max over a large range
        let data: Vec<f64> = (0..50_000).map(|i| (i as f64) - 25_000.0).collect();
        let expected_min = -25_000.0f64;
        let expected_max = 24_999.0f64;

        let config = RuntimeConfig::local(1).unwrap();

        // Min
        let env = StreamContext::new(config.clone());
        let min_output = env
            .stream_iter(data.clone().into_iter())
            .reduce_gpu(ReduceKernel::Min)
            .collect_vec();
        env.execute_blocking();
        let min_results = min_output.get().unwrap();
        assert_eq!(min_results.len(), 1);
        assert!(
            (min_results[0] - expected_min).abs() < 0.01,
            "Min: got {}, expected {}",
            min_results[0],
            expected_min
        );

        // Max
        let env = StreamContext::new(config);
        let max_output = env
            .stream_iter(data.into_iter())
            .reduce_gpu(ReduceKernel::Max)
            .collect_vec();
        env.execute_blocking();
        let max_results = max_output.get().unwrap();
        assert_eq!(max_results.len(), 1);
        assert!(
            (max_results[0] - expected_max).abs() < 0.01,
            "Max: got {}, expected {}",
            max_results[0],
            expected_max
        );
    }

    #[test]
    fn test_reduce_gpu_with_map_pipeline() {
        // Test reduce_gpu as part of a larger pipeline: map → reduce
        let data: Vec<i64> = (1..=1000).collect();
        // Sum of squares: 1^2 + 2^2 + ... + 1000^2 = n(n+1)(2n+1)/6
        let expected = 1000.0 * 1001.0 * 2001.0 / 6.0;

        let config = RuntimeConfig::local(1).unwrap();
        let env = StreamContext::new(config);
        let output = env
            .stream_iter(data.into_iter())
            .map(|x| (x * x) as f64)
            .reduce_gpu(ReduceKernel::Sum)
            .collect_vec();
        env.execute_blocking();
        let results = output.get().unwrap();

        assert_eq!(results.len(), 1);
        let rel_error = (results[0] - expected).abs() / expected;
        assert!(
            rel_error < 0.001,
            "Sum of squares: got {}, expected {}, relative error {:.6}",
            results[0],
            expected,
            rel_error
        );
    }

    #[test]
    fn test_reduce_gpu_single_element() {
        let config = RuntimeConfig::local(1).unwrap();
        let env = StreamContext::new(config);
        let output = env
            .stream_iter(std::iter::once(42.0f64))
            .reduce_gpu(ReduceKernel::Sum)
            .collect_vec();
        env.execute_blocking();
        let results = output.get().unwrap();

        assert_eq!(results.len(), 1);
        assert!(
            (results[0] - 42.0).abs() < 0.01,
            "Single element sum: got {}, expected 42.0",
            results[0]
        );
    }
}

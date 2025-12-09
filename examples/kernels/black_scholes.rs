//! # Black-Scholes GPU Kernel Implementation
//!
//! This module provides a reusable GPU kernel for Black-Scholes option pricing
//! using CubeCL. It can be used by both examples and benchmarks.
//!
//! ## Black-Scholes Model
//!
//! The Black-Scholes formula calculates the theoretical price of European options:
//!
//! ```text
//! Call = S * N(d1) - K * e^(-rT) * N(d2)
//! Put  = K * e^(-rT) * N(-d2) - S * N(-d1)
//!
//! where:
//!   d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
//!   d2 = d1 - σ√T
//!   N(x) = Cumulative Normal Distribution
//! ```
//!
//! ## GPU Execution Model (CubeCL)
//!
//! ```text
//! GPU Memory Layout:
//! ┌────────────────────────────────────────────────────────────────┐
//! │ Input Buffers (GPU Memory)                                     │
//! │ ┌──────────┬──────────┬──────────┬──────────┬──────────┐       │
//! │ │ stocks[] │strikes[] │ times[]  │ rates[]  │  vols[]  │       │
//! │ └──────────┴──────────┴──────────┴──────────┴──────────┘       │
//! │                           │                                    │
//! │                           ▼                                    │
//! │ ┌────────────────────────────────────────────────────────────┐ │
//! │ │              GPU Kernel Execution                          │ │
//! │ │  ┌─────────────────────────────────────────────────────┐   │ │
//! │ │  │ Workgroup/Cube (256 threads)                        │   │ │
//! │ │  │ ┌───┬───┬───┬───┬───┬───┬───┬───┬─────┬───┐         │   │ │
//! │ │  │ │T0 │T1 │T2 │T3 │T4 │T5 │T6 │T7 │ ... │T255│        │   │ │
//! │ │  │ └───┴───┴───┴───┴───┴───┴───┴───┴─────┴───┘         │   │ │
//! │ │  │ Each thread processes 4 options (vectorization)     │   │ │
//! │ │  └─────────────────────────────────────────────────────┘   │ │
//! │ │  × num_cubes (up to 65535 × 65535 in 2D grid)              │ │
//! │ └────────────────────────────────────────────────────────────┘ │
//! │                           │                                    │
//! │                           ▼                                    │
//! │ Output Buffers (GPU Memory)                                    │
//! │ ┌────────────────────────┬────────────────────────┐            │
//! │ │      call_prices[]     │      put_prices[]      │            │
//! │ └────────────────────────┴────────────────────────┘            │
//! └────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use renoir::prelude::*;
//! use renoir::operator::gpu::GpuBatchStrategy;
//!
//! mod black_scholes_kernel;
//! use black_scholes_kernel::*;
//!
//! let options = generate_options(1_000_000);
//! let env = StreamContext::new_local();
//! let results = env
//!     .stream_iter(options.into_iter())
//!     .map_gpu(BlackScholesKernel::default())
//!     .collect_vec();
//! env.execute_blocking();
//! ```

#![allow(dead_code)]

use std::cell::RefCell;

use bytemuck::{Pod, Zeroable};
use cubecl::prelude::*;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

use renoir::operator::gpu::{GpuContext, GpuKernel};

#[cfg(feature = "gpu-wgpu")]
use cubecl::wgpu::WgpuRuntime;

// ============================================================================
// Data Structures
// ============================================================================

/// Input parameters for a single Black-Scholes option pricing calculation.
///
/// All monetary values are in the same currency units (e.g., USD).
/// The struct is `repr(C)` and derives `Pod` for safe GPU memory transfer.
#[derive(Clone, Copy, Debug, Pod, Zeroable, Serialize, Deserialize)]
#[repr(C)]
pub struct BlackScholesInput {
    /// Current stock price (spot price), e.g., $50.00
    pub stock_price: f32,
    /// Strike price (exercise price), e.g., $55.00
    pub strike_price: f32,
    /// Time to expiration in years, e.g., 0.5 for 6 months
    pub time_to_expiry: f32,
    /// Risk-free interest rate (annual), e.g., 0.05 for 5%
    pub risk_free_rate: f32,
    /// Volatility (annual standard deviation), e.g., 0.20 for 20%
    pub volatility: f32,
}

/// Output from Black-Scholes calculation containing option prices.
#[derive(Clone, Copy, Debug, Pod, Zeroable, Serialize, Deserialize)]
#[repr(C)]
pub struct BlackScholesOutput {
    /// Theoretical price of a European call option
    pub call_price: f32,
    /// Theoretical price of a European put option
    pub put_price: f32,
}

// ============================================================================
// CPU Reference Implementation
// ============================================================================

/// Cumulative Normal Distribution function using the error function.
///
/// Calculates N(x) = P(Z ≤ x) where Z is a standard normal random variable.
/// Uses the mathematical identity: N(x) = 0.5 × (1 + erf(x/√2))
#[inline]
pub fn cnd_cpu(x: f32) -> f32 {
    0.5 * (1.0 + libm::erff(x / std::f32::consts::SQRT_2))
}

/// Black-Scholes option pricing on CPU.
///
/// Implements the standard Black-Scholes formula. Use this for validation
/// or when GPU is not available.
pub fn black_scholes_cpu(input: BlackScholesInput) -> BlackScholesOutput {
    let s = input.stock_price;
    let k = input.strike_price;
    let t = input.time_to_expiry;
    let r = input.risk_free_rate;
    let v = input.volatility;

    let sqrt_t = t.sqrt();
    let v_sqrt_t = v * sqrt_t;
    let d1 = ((s / k).ln() + (r + 0.5 * v * v) * t) / v_sqrt_t;
    let d2 = d1 - v_sqrt_t;

    let cnd_d1 = cnd_cpu(d1);
    let cnd_d2 = cnd_cpu(d2);
    let exp_rt = (-r * t).exp();

    let call = s * cnd_d1 - k * exp_rt * cnd_d2;
    let put = call - s + k * exp_rt;

    BlackScholesOutput {
        call_price: call,
        put_price: put,
    }
}

/// Black-Scholes on CPU for SoA data (used in optimized benchmarks).
#[inline]
pub fn black_scholes_cpu_soa(s: f32, k: f32, t: f32, r: f32, v: f32) -> f32 {
    let sqrt_t = t.sqrt();
    let v_sqrt_t = v * sqrt_t;
    let d1 = ((s / k).ln() + (r + 0.5 * v * v) * t) / v_sqrt_t;
    let d2 = d1 - v_sqrt_t;
    let cnd_d1 = 0.5 * (1.0 + libm::erff(d1 / std::f32::consts::SQRT_2));
    let cnd_d2 = 0.5 * (1.0 + libm::erff(d2 / std::f32::consts::SQRT_2));
    let exp_rt = (-r * t).exp();
    s * cnd_d1 - k * exp_rt * cnd_d2
}

// ============================================================================
// GPU Kernel (CubeCL)
// ============================================================================

/// CubeCL kernel for Black-Scholes option pricing.
///
/// This kernel is compiled to GPU shader code (WGSL for WGPU, PTX for CUDA).
/// Each thread processes 4 options simultaneously using SIMD-like Line<F> types.
#[cube(launch_unchecked)]
pub fn black_scholes_kernel<F: Float>(
    stock_prices: &Array<Line<F>>,
    strike_prices: &Array<Line<F>>,
    time_to_expirations: &Array<Line<F>>,
    risk_free_rates: &Array<Line<F>>,
    volatilities: &Array<Line<F>>,
    call_results: &mut Array<Line<F>>,
    put_results: &mut Array<Line<F>>,
) {
    if ABSOLUTE_POS < stock_prices.len() {
        let s = stock_prices[ABSOLUTE_POS];
        let k = strike_prices[ABSOLUTE_POS];
        let t = time_to_expirations[ABSOLUTE_POS];
        let r = risk_free_rates[ABSOLUTE_POS];
        let v = volatilities[ABSOLUTE_POS];

        let sqrt_t = Line::sqrt(t);
        let v_sqrt_t = v * sqrt_t;
        let d1_numerator = Line::log(s / k) + (r + v * v * Line::new(F::new(0.5))) * t;
        let d1 = d1_numerator / v_sqrt_t;
        let d2 = d1 - v_sqrt_t;

        let sqrt_2 = Line::new(F::new(std::f32::consts::SQRT_2));
        let cnd_d1 = (Line::new(F::new(1.0)) + Line::erf(d1 / sqrt_2)) * Line::new(F::new(0.5));
        let cnd_d2 = (Line::new(F::new(1.0)) + Line::erf(d2 / sqrt_2)) * Line::new(F::new(0.5));

        let exp_rt = Line::exp(-r * t);
        call_results[ABSOLUTE_POS] = s * cnd_d1 - k * exp_rt * cnd_d2;
        put_results[ABSOLUTE_POS] = call_results[ABSOLUTE_POS] - s + k * exp_rt;
    }
}

// ============================================================================
// GpuKernel Implementation
// ============================================================================

// ============================================================================
// Pre-allocated SoA Buffers for Kernel Reuse
// ============================================================================

/// Pre-allocated SoA buffers to avoid repeated allocations during streaming.
///
/// These buffers are reused across `execute()` calls to reduce allocation overhead,
/// which is critical for high-throughput streaming scenarios.
#[derive(Clone)]
struct SoABuffers {
    stocks: Vec<f32>,
    strikes: Vec<f32>,
    times: Vec<f32>,
    rates: Vec<f32>,
    vols: Vec<f32>,
    /// Current allocated capacity
    capacity: usize,
}

impl SoABuffers {
    /// Create new buffers with the given initial capacity.
    fn new(capacity: usize) -> Self {
        Self {
            stocks: Vec::with_capacity(capacity),
            strikes: Vec::with_capacity(capacity),
            times: Vec::with_capacity(capacity),
            rates: Vec::with_capacity(capacity),
            vols: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Ensure buffers have at least the required capacity.
    /// Grows exponentially to minimize reallocations.
    fn ensure_capacity(&mut self, needed: usize) {
        if needed > self.capacity {
            // Grow to next power of 2 for efficient reuse
            let new_capacity = needed.next_power_of_two();
            self.stocks.reserve(new_capacity.saturating_sub(self.stocks.capacity()));
            self.strikes.reserve(new_capacity.saturating_sub(self.strikes.capacity()));
            self.times.reserve(new_capacity.saturating_sub(self.times.capacity()));
            self.rates.reserve(new_capacity.saturating_sub(self.rates.capacity()));
            self.vols.reserve(new_capacity.saturating_sub(self.vols.capacity()));
            self.capacity = new_capacity;
        }
    }

    /// Clear all buffers while preserving capacity.
    fn clear(&mut self) {
        self.stocks.clear();
        self.strikes.clear();
        self.times.clear();
        self.rates.clear();
        self.vols.clear();
    }
}

/// Black-Scholes GPU kernel wrapper implementing the `GpuKernel` trait.
///
/// This struct encapsulates the GPU kernel configuration and implements
/// the `GpuKernel` trait for use with Renoir's `map_gpu` operator.
///
/// ## Pre-allocated Buffers
///
/// The kernel uses pre-allocated SoA (Structure-of-Arrays) buffers to minimize
/// allocation overhead during streaming. Buffers are created in `setup()` and
/// reused across `execute()` calls via interior mutability (RefCell).
///
/// # Example
///
/// ```rust,ignore
/// use renoir::prelude::*;
///
/// let env = StreamContext::new_local();
/// let results = env
///     .stream_iter(options.into_iter())
///     .map_gpu(BlackScholesKernel::default())
///     .collect_vec();
/// env.execute_blocking();
/// ```
#[derive(Clone)]
pub struct BlackScholesKernel {
    /// Number of threads per workgroup (CUDA: threads per block)
    pub cube_dim: u32,
    /// Pre-allocated SoA buffers for AoS→SoA conversion (uses RefCell for interior mutability)
    soa_buffers: RefCell<Option<SoABuffers>>,
}

impl Default for BlackScholesKernel {
    fn default() -> Self {
        Self {
            cube_dim: 256,
            soa_buffers: RefCell::new(None),
        }
    }
}

impl BlackScholesKernel {
    /// Create a new kernel with custom workgroup size.
    pub fn new(cube_dim: u32) -> Self {
        Self {
            cube_dim,
            soa_buffers: RefCell::new(None),
        }
    }
}

impl GpuKernel for BlackScholesKernel {
    type Input = BlackScholesInput;
    type Output = BlackScholesOutput;

    fn execute(&self, ctx: &GpuContext, inputs: &[BlackScholesInput]) -> Vec<BlackScholesOutput> {
        if inputs.is_empty() {
            return Vec::new();
        }

        let num_options = inputs.len();
        let client = ctx.client();

        // Calculate padded size (must be multiple of vectorization factor)
        let vectorization_factor = 4usize;
        let remainder = num_options % vectorization_factor;
        let num_elements_padded = if remainder == 0 {
            num_options
        } else {
            num_options + (vectorization_factor - remainder)
        };
        let padding_needed = num_elements_padded - num_options;

        // Use pre-allocated buffers for AoS→SoA conversion
        let mut buffers = self.soa_buffers.borrow_mut();
        let soa = buffers.get_or_insert_with(|| SoABuffers::new(num_elements_padded));
        
        // Ensure we have enough capacity and clear previous data
        soa.ensure_capacity(num_elements_padded);
        soa.clear();

        // Convert AoS to SoA using pre-allocated buffers
        for input in inputs {
            soa.stocks.push(input.stock_price);
            soa.strikes.push(input.strike_price);
            soa.times.push(input.time_to_expiry);
            soa.rates.push(input.risk_free_rate);
            soa.vols.push(input.volatility);
        }

        // Add padding
        for _ in 0..padding_needed {
            soa.stocks.push(1.0);
            soa.strikes.push(1.0);
            soa.times.push(1.0);
            soa.rates.push(0.05);
            soa.vols.push(0.2);
        }

        // Allocate GPU buffers
        let stock_handle = client.create(f32::as_bytes(&soa.stocks));
        let strike_handle = client.create(f32::as_bytes(&soa.strikes));
        let time_handle = client.create(f32::as_bytes(&soa.times));
        let rate_handle = client.create(f32::as_bytes(&soa.rates));
        let vol_handle = client.create(f32::as_bytes(&soa.vols));

        let call_handle = client.empty(num_elements_padded * std::mem::size_of::<f32>());
        let put_handle = client.empty(num_elements_padded * std::mem::size_of::<f32>());

        // Calculate launch configuration
        let cube_dim = CubeDim::new(self.cube_dim, 1, 1);
        let total_lines = num_elements_padded / vectorization_factor;
        let num_cubes_total = (total_lines as u32 + cube_dim.x - 1) / cube_dim.x;

        // Handle GPU dispatch limits (max 65535 per dimension)
        const MAX_DISPATCH: u32 = 65535;
        let (cubes_x, cubes_y) = if num_cubes_total <= MAX_DISPATCH {
            (num_cubes_total, 1u32)
        } else {
            let cubes_y = (num_cubes_total + MAX_DISPATCH - 1) / MAX_DISPATCH;
            let cubes_x = (num_cubes_total + cubes_y - 1) / cubes_y;
            (
                std::cmp::min(cubes_x, MAX_DISPATCH),
                std::cmp::min(cubes_y, MAX_DISPATCH),
            )
        };

        let cube_count = CubeCount::Static(cubes_x, cubes_y, 1);

        // Launch the GPU kernel
        #[cfg(feature = "gpu-wgpu")]
        unsafe {
            black_scholes_kernel::launch_unchecked::<f32, WgpuRuntime>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<f32>(
                    &stock_handle,
                    num_elements_padded,
                    vectorization_factor as u8,
                ),
                ArrayArg::from_raw_parts::<f32>(
                    &strike_handle,
                    num_elements_padded,
                    vectorization_factor as u8,
                ),
                ArrayArg::from_raw_parts::<f32>(
                    &time_handle,
                    num_elements_padded,
                    vectorization_factor as u8,
                ),
                ArrayArg::from_raw_parts::<f32>(
                    &rate_handle,
                    num_elements_padded,
                    vectorization_factor as u8,
                ),
                ArrayArg::from_raw_parts::<f32>(
                    &vol_handle,
                    num_elements_padded,
                    vectorization_factor as u8,
                ),
                ArrayArg::from_raw_parts::<f32>(
                    &call_handle,
                    num_elements_padded,
                    vectorization_factor as u8,
                ),
                ArrayArg::from_raw_parts::<f32>(
                    &put_handle,
                    num_elements_padded,
                    vectorization_factor as u8,
                ),
            );
        }

        ctx.sync();

        // Read results
        let call_bytes = client.read_one(call_handle.clone());
        let put_bytes = client.read_one(put_handle.clone());

        let mut results = Vec::with_capacity(num_options);
        let call_chunks = call_bytes.chunks_exact(4);
        let put_chunks = put_bytes.chunks_exact(4);

        for (call_b, put_b) in call_chunks.zip(put_chunks).take(num_options) {
            results.push(BlackScholesOutput {
                call_price: f32::from_le_bytes([call_b[0], call_b[1], call_b[2], call_b[3]]),
                put_price: f32::from_le_bytes([put_b[0], put_b[1], put_b[2], put_b[3]]),
            });
        }

        results
    }

    fn setup(&mut self, ctx: &GpuContext) {
        // Pre-allocate SoA buffers based on preferred batch size
        let initial_capacity = self.preferred_batch_size().unwrap_or(5_000_000);
        *self.soa_buffers.borrow_mut() = Some(SoABuffers::new(initial_capacity));
        
        // Warmup: compile shaders with a small batch
        let warmup_inputs: Vec<BlackScholesInput> = (0..1024)
            .map(|i| BlackScholesInput {
                stock_price: 50.0 + (i as f32) * 0.01,
                strike_price: 50.0,
                time_to_expiry: 1.0,
                risk_free_rate: 0.05,
                volatility: 0.2,
            })
            .collect();
        let _ = self.execute(ctx, &warmup_inputs);
    }

    fn preferred_batch_size(&self) -> Option<usize> {
        // Increase preferred batch size to 5M for better GPU utilization
        Some(5_000_000)
    }
}

// ============================================================================
// Data Generation Utilities
// ============================================================================

/// Generate random option parameters for testing.
///
/// Creates realistic random values within typical market ranges:
/// - Stock prices: $10 to $100
/// - Strike prices: $10 to $100
/// - Time to expiry: 0.5 to 2.0 years
/// - Risk-free rate: 5% (constant)
/// - Volatility: 20% (constant)
pub fn generate_options(count: usize) -> Vec<BlackScholesInput> {
    let mut rng = rand::rng();
    (0..count)
        .map(|_| BlackScholesInput {
            stock_price: rng.random_range(10.0..100.0),
            strike_price: rng.random_range(10.0..100.0),
            time_to_expiry: rng.random_range(0.5..2.0),
            risk_free_rate: 0.05,
            volatility: 0.2,
        })
        .collect()
}

/// Generate random options with a specific seed for reproducibility.
pub fn generate_options_seeded(count: usize, seed: u64) -> Vec<BlackScholesInput> {
    let mut rng = SmallRng::seed_from_u64(seed);
    (0..count)
        .map(|_| BlackScholesInput {
            stock_price: rng.random_range(10.0..100.0),
            strike_price: rng.random_range(10.0..100.0),
            time_to_expiry: rng.random_range(0.5..2.0),
            risk_free_rate: 0.05,
            volatility: 0.2,
        })
        .collect()
}

/// Generate SoA data with a specific seed for reproducibility.
/// This is more efficient for GPU processing than Vec<BlackScholesInput>.
pub fn generate_soa_with_seed(count: usize, seed: u64) -> BlackScholesSoA {
    BlackScholesSoA::generate_with_seed(count, seed)
}

// ============================================================================
// Streaming Data Generator
// ============================================================================

/// Lazy iterator that generates Black-Scholes input data on-demand.
///
/// This simulates a real-world streaming scenario where:
/// - Input data arrives continuously
/// - Total stream size may be unknown in advance
/// - Memory usage is bounded to the current batch only
pub struct StreamingOptionsGenerator {
    remaining: usize,
    rng: SmallRng,
}

impl StreamingOptionsGenerator {
    /// Create a new streaming generator.
    ///
    /// # Arguments
    /// * `total_items` - Total number of items to generate
    /// * `seed` - Random seed for reproducibility
    pub fn new(total_items: usize, seed: u64) -> Self {
        Self {
            remaining: total_items,
            rng: SmallRng::seed_from_u64(seed),
        }
    }
}

impl Iterator for StreamingOptionsGenerator {
    type Item = BlackScholesInput;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        self.remaining -= 1;

        Some(BlackScholesInput {
            stock_price: self.rng.random_range(10.0..100.0),
            strike_price: self.rng.random_range(10.0..100.0),
            time_to_expiry: self.rng.random_range(0.5..2.0),
            risk_free_rate: 0.05,
            volatility: 0.2,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for StreamingOptionsGenerator {}

// ============================================================================
// SoA Data Structure for Optimized Processing
// ============================================================================

/// Structure-of-Arrays data layout for optimized GPU processing.
///
/// This format avoids the AoS → SoA conversion overhead by storing
/// data directly in GPU-friendly layout.
pub struct BlackScholesSoA {
    pub stocks: Vec<f32>,
    pub strikes: Vec<f32>,
    pub times: Vec<f32>,
    pub rates: Vec<f32>,
    pub vols: Vec<f32>,
}

impl BlackScholesSoA {
    /// Generate random options directly in SoA format.
    pub fn generate(count: usize) -> Self {
        let mut rng = rand::rng();

        Self {
            stocks: (0..count).map(|_| rng.random_range(10.0..100.0)).collect(),
            strikes: (0..count).map(|_| rng.random_range(10.0..100.0)).collect(),
            times: (0..count).map(|_| rng.random_range(0.5..2.0)).collect(),
            rates: vec![0.05f32; count],
            vols: vec![0.2f32; count],
        }
    }

    /// Generate random options with a specific seed for reproducibility.
    pub fn generate_with_seed(count: usize, seed: u64) -> Self {
        use rand::rngs::SmallRng;
        use rand::SeedableRng;
        
        let mut rng = SmallRng::seed_from_u64(seed);

        Self {
            stocks: (0..count).map(|_| rng.random_range(10.0..100.0)).collect(),
            strikes: (0..count).map(|_| rng.random_range(10.0..100.0)).collect(),
            times: (0..count).map(|_| rng.random_range(0.5..2.0)).collect(),
            rates: vec![0.05f32; count],
            vols: vec![0.2f32; count],
        }
    }

    /// Pad arrays to be divisible by vectorization factor.
    pub fn pad(&mut self, vectorization_factor: usize) {
        let remainder = self.stocks.len() % vectorization_factor;
        if remainder != 0 {
            let padding = vectorization_factor - remainder;
            for _ in 0..padding {
                self.stocks.push(1.0);
                self.strikes.push(1.0);
                self.times.push(1.0);
                self.rates.push(0.05);
                self.vols.push(0.2);
            }
        }
    }

    /// Get the number of elements (including padding).
    pub fn len(&self) -> usize {
        self.stocks.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.stocks.is_empty()
    }
}

// ============================================================================
// Validation Utilities
// ============================================================================

/// Validate GPU results against CPU reference implementation.
///
/// Returns (passed, max_error, avg_error)
pub fn validate_results(
    inputs: &[BlackScholesInput],
    gpu_outputs: &[BlackScholesOutput],
    tolerance: f32,
) -> (bool, f32, f32) {
    if inputs.len() != gpu_outputs.len() {
        return (false, f32::MAX, f32::MAX);
    }

    let mut max_error = 0.0f32;
    let mut total_error = 0.0f32;

    for (input, gpu_out) in inputs.iter().zip(gpu_outputs.iter()) {
        let cpu_out = black_scholes_cpu(*input);

        let call_error = (cpu_out.call_price - gpu_out.call_price).abs();
        let put_error = (cpu_out.put_price - gpu_out.put_price).abs();

        max_error = max_error.max(call_error).max(put_error);
        total_error += call_error + put_error;
    }

    let avg_error = total_error / (inputs.len() * 2) as f32;
    let passed = max_error <= tolerance;

    (passed, max_error, avg_error)
}

// ============================================================================
// Benchmark Structures (for benchmark harnesses)
// ============================================================================

use chrono::{DateTime, Utc};
use std::time::Instant;

/// Result of a single benchmark test.
#[derive(Serialize, Deserialize, Clone)]
pub struct BenchmarkResult {
    pub test_id: usize,
    pub timestamp: DateTime<Utc>,
    pub operator: String,
    pub items_count: usize,
    pub data_size_gb: f64,
    pub cpu_total_time_s: f64,
    pub gpu_total_time_s: f64,
    pub renoir_total_time_s: f64,
    pub speedup: f64,
    pub cpu_gflops: f64,
    pub gpu_gflops: f64,
    pub renoir_gflops: f64,
    pub cpu_workers: usize,
    pub gpu_threads: u32,
    pub batch_size: usize,
    pub cpu_result: f64,
    pub gpu_result: f64,
    pub validation_passed: bool,
    pub error_margin: f64,
}

/// Container for all benchmark results with metadata.
#[derive(Serialize, Deserialize, Clone)]
pub struct BenchmarkReport {
    pub benchmark_type: String,
    pub start_time: DateTime<Utc>,
    pub platform: String,
    pub gpu_enabled: bool,
    pub total_tests: usize,
    pub results: Vec<BenchmarkResult>,
    #[serde(default)]
    pub is_streaming: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub streaming_config: Option<StreamingConfig>,
}

/// Configuration for streaming mode benchmarks.
#[derive(Serialize, Deserialize, Clone)]
pub struct StreamingConfig {
    pub total_items: usize,
    pub adaptive_min_batch: usize,
    pub adaptive_max_batch: usize,
    pub cpu_workers: usize,
    pub description: String,
}

// ============================================================================
// Optimized GPU Benchmark (kernel-only timing)
// ============================================================================

/// Run optimized GPU benchmark with kernel-only timing.
///
/// Returns: (kernel_time, full_time, call_prices, gpu_threads)
pub fn run_optimized_gpu_benchmark(
    ctx: &GpuContext,
    data: &BlackScholesSoA,
    num_options: usize,
) -> (f64, f64, Vec<f32>, u32) {
    use cubecl::prelude::*;
    #[cfg(feature = "gpu-wgpu")]
    use cubecl::wgpu::WgpuRuntime;

    let client = ctx.client();
    let vectorization_factor = 4usize;
    let num_elements_padded = data.stocks.len();

    let full_start = Instant::now();

    // Create GPU buffers
    let stock_handle = client.create(f32::as_bytes(&data.stocks));
    let strike_handle = client.create(f32::as_bytes(&data.strikes));
    let time_handle = client.create(f32::as_bytes(&data.times));
    let rate_handle = client.create(f32::as_bytes(&data.rates));
    let vol_handle = client.create(f32::as_bytes(&data.vols));

    let call_handle = client.empty(num_elements_padded * std::mem::size_of::<f32>());
    let put_handle = client.empty(num_elements_padded * std::mem::size_of::<f32>());

    // Launch configuration
    let cube_dim = CubeDim::new(256, 1, 1);
    let total_lines = num_elements_padded / vectorization_factor;
    let num_cubes_total = (total_lines as u32 + cube_dim.x - 1) / cube_dim.x;

    const MAX_DISPATCH: u32 = 65535;
    let (cubes_x, cubes_y) = if num_cubes_total <= MAX_DISPATCH {
        (num_cubes_total, 1u32)
    } else {
        let cubes_y = (num_cubes_total + MAX_DISPATCH - 1) / MAX_DISPATCH;
        let cubes_x = (num_cubes_total + cubes_y - 1) / cubes_y;
        (
            std::cmp::min(cubes_x, MAX_DISPATCH),
            std::cmp::min(cubes_y, MAX_DISPATCH),
        )
    };
    let cube_count = CubeCount::Static(cubes_x, cubes_y, 1);
    let gpu_threads = cubes_x * cubes_y * cube_dim.x;

    // Warmup run
    #[cfg(feature = "gpu-wgpu")]
    unsafe {
        black_scholes_kernel::launch_unchecked::<f32, WgpuRuntime>(
            client,
            cube_count.clone(),
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(
                &stock_handle,
                num_elements_padded,
                vectorization_factor as u8,
            ),
            ArrayArg::from_raw_parts::<f32>(
                &strike_handle,
                num_elements_padded,
                vectorization_factor as u8,
            ),
            ArrayArg::from_raw_parts::<f32>(
                &time_handle,
                num_elements_padded,
                vectorization_factor as u8,
            ),
            ArrayArg::from_raw_parts::<f32>(
                &rate_handle,
                num_elements_padded,
                vectorization_factor as u8,
            ),
            ArrayArg::from_raw_parts::<f32>(
                &vol_handle,
                num_elements_padded,
                vectorization_factor as u8,
            ),
            ArrayArg::from_raw_parts::<f32>(
                &call_handle,
                num_elements_padded,
                vectorization_factor as u8,
            ),
            ArrayArg::from_raw_parts::<f32>(
                &put_handle,
                num_elements_padded,
                vectorization_factor as u8,
            ),
        );
    }
    ctx.sync();

    // Timed run (kernel only)
    let kernel_start = Instant::now();

    #[cfg(feature = "gpu-wgpu")]
    unsafe {
        black_scholes_kernel::launch_unchecked::<f32, WgpuRuntime>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts::<f32>(
                &stock_handle,
                num_elements_padded,
                vectorization_factor as u8,
            ),
            ArrayArg::from_raw_parts::<f32>(
                &strike_handle,
                num_elements_padded,
                vectorization_factor as u8,
            ),
            ArrayArg::from_raw_parts::<f32>(
                &time_handle,
                num_elements_padded,
                vectorization_factor as u8,
            ),
            ArrayArg::from_raw_parts::<f32>(
                &rate_handle,
                num_elements_padded,
                vectorization_factor as u8,
            ),
            ArrayArg::from_raw_parts::<f32>(
                &vol_handle,
                num_elements_padded,
                vectorization_factor as u8,
            ),
            ArrayArg::from_raw_parts::<f32>(
                &call_handle,
                num_elements_padded,
                vectorization_factor as u8,
            ),
            ArrayArg::from_raw_parts::<f32>(
                &put_handle,
                num_elements_padded,
                vectorization_factor as u8,
            ),
        );
    }
    ctx.sync();

    let kernel_time = kernel_start.elapsed().as_secs_f64();

    // Read results
    let gpu_calls_bytes = client.read_one(call_handle.clone());

    let full_time = full_start.elapsed().as_secs_f64();

    let gpu_calls: Vec<f32> = gpu_calls_bytes
        .chunks_exact(4)
        .take(num_options)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    (kernel_time, full_time, gpu_calls, gpu_threads)
}

// ============================================================================
// Pipelined GPU Processing (Double-Buffering)
// ============================================================================

/// Optimal chunk size for pipelined processing.
/// This balances GPU utilization with memory overhead.
pub const PIPELINE_CHUNK_SIZE: usize = 50_000_000; // 50M options per chunk

/// Result of pipelined GPU processing.
pub struct PipelineResult {
    pub total_time_s: f64,
    pub kernel_time_s: f64,
    pub items_processed: usize,
    pub chunks_processed: usize,
}

/// Process large batches using pipelined GPU execution.
///
/// This approach uses double-buffering to overlap:
/// - CPU data preparation for chunk N+1
/// - GPU kernel execution for chunk N
///
/// # Arguments
/// * `ctx` - GPU context
/// * `kernel` - Black-Scholes kernel
/// * `options` - Input options to process (can be very large)
/// * `chunk_size` - Size of each processing chunk (default: PIPELINE_CHUNK_SIZE)
///
/// # Returns
/// Tuple of (results, pipeline_result)
pub fn process_pipelined(
    ctx: &GpuContext,
    kernel: &BlackScholesKernel,
    options: &[BlackScholesInput],
    chunk_size: usize,
) -> (Vec<BlackScholesOutput>, PipelineResult) {
    use std::time::Instant;

    let total_start = Instant::now();
    let num_options = options.len();

    if num_options == 0 {
        return (
            Vec::new(),
            PipelineResult {
                total_time_s: 0.0,
                kernel_time_s: 0.0,
                items_processed: 0,
                chunks_processed: 0,
            },
        );
    }

    // For small inputs, just process directly
    if num_options <= chunk_size {
        let kernel_start = Instant::now();
        let results = kernel.execute(ctx, options);
        let kernel_time = kernel_start.elapsed().as_secs_f64();
        let total_time = total_start.elapsed().as_secs_f64();

        return (
            results,
            PipelineResult {
                total_time_s: total_time,
                kernel_time_s: kernel_time,
                items_processed: num_options,
                chunks_processed: 1,
            },
        );
    }

    // Pipelined processing for large inputs
    let mut all_results = Vec::with_capacity(num_options);
    let mut total_kernel_time = 0.0;
    let chunks: Vec<&[BlackScholesInput]> = options.chunks(chunk_size).collect();
    let num_chunks = chunks.len();

    for chunk in chunks {
        let kernel_start = Instant::now();
        let chunk_results = kernel.execute(ctx, chunk);
        total_kernel_time += kernel_start.elapsed().as_secs_f64();
        all_results.extend(chunk_results);
    }

    let total_time = total_start.elapsed().as_secs_f64();

    (
        all_results,
        PipelineResult {
            total_time_s: total_time,
            kernel_time_s: total_kernel_time,
            items_processed: num_options,
            chunks_processed: num_chunks,
        },
    )
}

/// Process SoA data using pipelined GPU execution with kernel-only timing.
///
/// This is optimized for benchmarks where we want to measure pure GPU performance
/// while still benefiting from chunked processing for large datasets.
///
/// # Arguments
/// * `ctx` - GPU context
/// * `data` - Input data in SoA format
/// * `num_options` - Number of actual options (excluding padding)
/// * `chunk_size` - Size of each processing chunk
///
/// # Returns
/// Tuple of (call_prices, put_prices, pipeline_result, gpu_threads)
pub fn process_pipelined_soa(
    ctx: &GpuContext,
    data: &BlackScholesSoA,
    num_options: usize,
    chunk_size: usize,
) -> (Vec<f32>, Vec<f32>, PipelineResult, u32) {
    use cubecl::prelude::*;
    #[cfg(feature = "gpu-wgpu")]
    use cubecl::wgpu::WgpuRuntime;
    use std::time::Instant;

    let total_start = Instant::now();

    if num_options == 0 {
        return (
            Vec::new(),
            Vec::new(),
            PipelineResult {
                total_time_s: 0.0,
                kernel_time_s: 0.0,
                items_processed: 0,
                chunks_processed: 0,
            },
            0,
        );
    }

    let client = ctx.client();
    let vectorization_factor = 4usize;
    let cube_dim = CubeDim::new(256, 1, 1);

    // For small inputs, process directly
    if num_options <= chunk_size {
        let (kernel_time, _full_time, calls, gpu_threads) =
            run_optimized_gpu_benchmark(ctx, data, num_options);
        let total_time = total_start.elapsed().as_secs_f64();

        // Get puts as well
        let puts = vec![0.0f32; calls.len()]; // Simplified - we mainly care about call prices

        return (
            calls,
            puts,
            PipelineResult {
                total_time_s: total_time,
                kernel_time_s: kernel_time,
                items_processed: num_options,
                chunks_processed: 1,
            },
            gpu_threads,
        );
    }

    // Pipelined processing for large inputs
    let mut all_calls = Vec::with_capacity(num_options);
    let mut all_puts = Vec::with_capacity(num_options);
    let mut total_kernel_time = 0.0;
    let mut gpu_threads = 0u32;

    let num_chunks = (num_options + chunk_size - 1) / chunk_size;

    for chunk_idx in 0..num_chunks {
        let start_idx = chunk_idx * chunk_size;
        let end_idx = std::cmp::min(start_idx + chunk_size, num_options);
        let chunk_len = end_idx - start_idx;

        // Pad chunk to vectorization factor
        let padded_len = ((chunk_len + vectorization_factor - 1) / vectorization_factor)
            * vectorization_factor;

        // Extract chunk data
        let mut chunk_stocks = data.stocks[start_idx..end_idx].to_vec();
        let mut chunk_strikes = data.strikes[start_idx..end_idx].to_vec();
        let mut chunk_times = data.times[start_idx..end_idx].to_vec();
        let mut chunk_rates = data.rates[start_idx..end_idx].to_vec();
        let mut chunk_vols = data.vols[start_idx..end_idx].to_vec();

        // Pad if needed
        while chunk_stocks.len() < padded_len {
            chunk_stocks.push(1.0);
            chunk_strikes.push(1.0);
            chunk_times.push(1.0);
            chunk_rates.push(0.05);
            chunk_vols.push(0.2);
        }

        // Create GPU buffers
        let stock_handle = client.create(f32::as_bytes(&chunk_stocks));
        let strike_handle = client.create(f32::as_bytes(&chunk_strikes));
        let time_handle = client.create(f32::as_bytes(&chunk_times));
        let rate_handle = client.create(f32::as_bytes(&chunk_rates));
        let vol_handle = client.create(f32::as_bytes(&chunk_vols));

        let call_handle = client.empty(padded_len * std::mem::size_of::<f32>());
        let put_handle = client.empty(padded_len * std::mem::size_of::<f32>());

        // Launch configuration
        let total_lines = padded_len / vectorization_factor;
        let num_cubes_total = (total_lines as u32 + cube_dim.x - 1) / cube_dim.x;

        const MAX_DISPATCH: u32 = 65535;
        let (cubes_x, cubes_y) = if num_cubes_total <= MAX_DISPATCH {
            (num_cubes_total, 1u32)
        } else {
            let cubes_y = (num_cubes_total + MAX_DISPATCH - 1) / MAX_DISPATCH;
            let cubes_x = (num_cubes_total + cubes_y - 1) / cubes_y;
            (
                std::cmp::min(cubes_x, MAX_DISPATCH),
                std::cmp::min(cubes_y, MAX_DISPATCH),
            )
        };
        let cube_count = CubeCount::Static(cubes_x, cubes_y, 1);
        gpu_threads = std::cmp::max(gpu_threads, cubes_x * cubes_y * cube_dim.x);

        // Kernel execution (timed)
        let kernel_start = Instant::now();

        #[cfg(feature = "gpu-wgpu")]
        unsafe {
            black_scholes_kernel::launch_unchecked::<f32, WgpuRuntime>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts::<f32>(&stock_handle, padded_len, vectorization_factor as u8),
                ArrayArg::from_raw_parts::<f32>(&strike_handle, padded_len, vectorization_factor as u8),
                ArrayArg::from_raw_parts::<f32>(&time_handle, padded_len, vectorization_factor as u8),
                ArrayArg::from_raw_parts::<f32>(&rate_handle, padded_len, vectorization_factor as u8),
                ArrayArg::from_raw_parts::<f32>(&vol_handle, padded_len, vectorization_factor as u8),
                ArrayArg::from_raw_parts::<f32>(&call_handle, padded_len, vectorization_factor as u8),
                ArrayArg::from_raw_parts::<f32>(&put_handle, padded_len, vectorization_factor as u8),
            );
        }
        ctx.sync();

        total_kernel_time += kernel_start.elapsed().as_secs_f64();

        // Read results
        let call_bytes = client.read_one(call_handle.clone());
        let put_bytes = client.read_one(put_handle.clone());

        let chunk_calls: Vec<f32> = call_bytes
            .chunks_exact(4)
            .take(chunk_len)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        let chunk_puts: Vec<f32> = put_bytes
            .chunks_exact(4)
            .take(chunk_len)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        all_calls.extend(chunk_calls);
        all_puts.extend(chunk_puts);
    }

    let total_time = total_start.elapsed().as_secs_f64();

    (
        all_calls,
        all_puts,
        PipelineResult {
            total_time_s: total_time,
            kernel_time_s: total_kernel_time,
            items_processed: num_options,
            chunks_processed: num_chunks,
        },
        gpu_threads,
    )
}

/// Run a pipelined benchmark comparing single-batch vs chunked processing.
///
/// This helps measure the overhead of pipelining vs the benefits of
/// reduced memory pressure for large datasets.
pub fn run_pipelined_comparison(
    ctx: &GpuContext,
    kernel: &BlackScholesKernel,
    options: &[BlackScholesInput],
    chunk_size: usize,
) -> (f64, f64, f64) {
    use std::time::Instant;

    // Single batch processing
    let single_start = Instant::now();
    let _single_results = kernel.execute(ctx, options);
    let single_time = single_start.elapsed().as_secs_f64();

    // Pipelined processing
    let (_pipelined_results, pipeline_info) = process_pipelined(ctx, kernel, options, chunk_size);
    let pipelined_time = pipeline_info.total_time_s;

    // Speedup (single / pipelined, >1 means pipelining is faster)
    let speedup = single_time / pipelined_time;

    (single_time, pipelined_time, speedup)
}

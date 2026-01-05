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

use bytemuck::{Pod, Zeroable};
use cubecl::prelude::*;
use serde::{Deserialize, Serialize};

use renoir::operator::gpu::{GpuContext, GpuKernel};

#[cfg(feature = "gpu-wgpu")]
use cubecl::wgpu::WgpuRuntime;
#[cfg(feature = "gpu-wgpu")]
type GpuRuntime = WgpuRuntime;

#[cfg(all(feature = "gpu-cuda", not(feature = "gpu-wgpu")))]
use cubecl::cuda::CudaRuntime;
#[cfg(all(feature = "gpu-cuda", not(feature = "gpu-wgpu")))]
type GpuRuntime = CudaRuntime;

// ============================================================================
// Constants
// ============================================================================

/// Default risk-free interest rate (5%)
pub const DEFAULT_RISK_FREE_RATE: f32 = 0.05;

/// Default volatility (20%)
pub const DEFAULT_VOLATILITY: f32 = 0.2;

/// Default padding value for stock prices, strikes, and times
pub const PADDING_VALUE_ONE: f32 = 1.0;

/// Default cube dimension (threads per block)
pub const DEFAULT_CUBE_DIM: u32 = 256;

/// Default vectorization factor
pub const DEFAULT_VECTORIZATION_FACTOR: usize = 4;

/// Default preferred batch size for GPU execution
pub const DEFAULT_BATCH_SIZE: usize = 10_000_000;

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
    volatilises: &Array<Line<F>>,
    call_results: &mut Array<Line<F>>,
    put_results: &mut Array<Line<F>>,
) {
    if ABSOLUTE_POS < stock_prices.len() {
        let s = stock_prices[ABSOLUTE_POS];
        let k = strike_prices[ABSOLUTE_POS];
        let t = time_to_expirations[ABSOLUTE_POS];
        let r = risk_free_rates[ABSOLUTE_POS];
        let v = volatilises[ABSOLUTE_POS];

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
}

impl SoABuffers {
    /// Create new empty buffers.
    #[allow(dead_code)]
    fn new() -> Self {
        Self {
            stocks: Vec::new(),
            strikes: Vec::new(),
            times: Vec::new(),
            rates: Vec::new(),
            vols: Vec::new(),
        }
    }

    /// Create buffers with pre-allocated capacity.
    /// Avoids reallocations when pushing items up to capacity.
    fn with_capacity(capacity: usize) -> Self {
        Self {
            stocks: Vec::with_capacity(capacity),
            strikes: Vec::with_capacity(capacity),
            times: Vec::with_capacity(capacity),
            rates: Vec::with_capacity(capacity),
            vols: Vec::with_capacity(capacity),
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

    /// Push a single item directly into SoA format.
    /// This avoids the AoS→SoA conversion phase by accumulating directly.
    #[inline]
    fn push(&mut self, input: &BlackScholesInput) {
        self.stocks.push(input.stock_price);
        self.strikes.push(input.strike_price);
        self.times.push(input.time_to_expiry);
        self.rates.push(input.risk_free_rate);
        self.vols.push(input.volatility);
    }

    /// Finalize the buffer by adding padding for vectorization alignment.
    fn pad_for_vectorization(&mut self, vectorization_factor: usize) {
        let len = self.stocks.len();
        let aligned_len = (len + vectorization_factor - 1) / vectorization_factor * vectorization_factor;
        
        self.stocks.resize(aligned_len, PADDING_VALUE_ONE);
        self.strikes.resize(aligned_len, PADDING_VALUE_ONE);
        self.times.resize(aligned_len, PADDING_VALUE_ONE);
        self.rates.resize(aligned_len, DEFAULT_RISK_FREE_RATE);
        self.vols.resize(aligned_len, DEFAULT_VOLATILITY);
    }

    /// Get number of elements in the buffer.
    fn len(&self) -> usize {
        self.stocks.len()
    }
}


// ============================================================================
// GPU Kernel Implementation
// ============================================================================

/// Black-Scholes GPU kernel implementing the `GpuKernel` trait.
///
/// This kernel processes batches of option contracts on the GPU,
/// converting input data from AoS to SoA format for optimal GPU memory access.
///
/// ## Execution Flow
///
/// ```text
/// Time ──────────────────────────────────────────────────►
///
/// Batch 1:  [push items] [flush → GPU Execute]
/// Batch 2:               [push items] [flush → GPU Execute]
/// ```
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
    /// Vectorization factor (SIMD width): 4, 8, or 16
    pub vectorization_factor: usize,
    /// SoA buffer for accumulating input data
    buffer: SoABuffers,
    /// Track number of items pushed (before padding)
    items_pushed: usize,
    /// Pending GPU results: (call_handle, put_handle, count)
    /// None if no computation is in flight
    pending: Option<(cubecl::server::Handle, cubecl::server::Handle, usize)>,
}

impl Default for BlackScholesKernel {
    fn default() -> Self {
        Self {
            cube_dim: DEFAULT_CUBE_DIM,
            vectorization_factor: DEFAULT_VECTORIZATION_FACTOR,
            buffer: SoABuffers::with_capacity(DEFAULT_BATCH_SIZE),
            items_pushed: 0,
            pending: None,
        }
    }
}

impl BlackScholesKernel {
    #[allow(dead_code)]
    /// Set the vectorization factor for GPU tuning.
    pub fn with_vectorization(mut self, factor: usize) -> Self {
        self.vectorization_factor = factor;
        self
    }
}

impl GpuKernel for BlackScholesKernel {
    type Input = BlackScholesInput;
    type Output = BlackScholesOutput;

    fn push(&mut self, item: Self::Input) {
        self.buffer.push(&item);
        self.items_pushed += 1;
    }

    fn buffer_len(&self) -> usize {
        self.items_pushed
    }

    fn flush(&mut self, ctx: &GpuContext) -> Vec<BlackScholesOutput> {
        let client = ctx.client();
        let n = self.items_pushed;
        let vectorization_factor = self.vectorization_factor;
        
        // === Step 1: Upload NEW batch data FIRST (async) ===
        // This happens WHILE the GPU is still computing the previous batch!
        // The upload is queued and may overlap with ongoing computation.
        let new_batch = if n > 0 {
            self.buffer.pad_for_vectorization(vectorization_factor);
            let num_padded = self.buffer.len();
            
            // Create GPU buffers from SoA data.
            // IMPORTANT: client.create() COPIES the data into GPU memory immediately!
            // The returned Handle references the GPU-side copy, not the original Vec.
            let stock_h = client.create(f32::as_bytes(&self.buffer.stocks));
            let strike_h = client.create(f32::as_bytes(&self.buffer.strikes));
            let time_h = client.create(f32::as_bytes(&self.buffer.times));
            let rate_h = client.create(f32::as_bytes(&self.buffer.rates));
            let vol_h = client.create(f32::as_bytes(&self.buffer.vols));
            let call_h = client.empty(num_padded * size_of::<f32>());
            let put_h = client.empty(num_padded * size_of::<f32>());
            
            // Configure GPU grid (with 2D dispatch for large batches)
            let cube_dim = CubeDim::new(self.cube_dim, 1, 1);
            let total_lines = num_padded / vectorization_factor;
            let num_cubes_total = (total_lines as u32 + cube_dim.x - 1) / cube_dim.x;
            
            const MAX_DISPATCH: u32 = 65535;
            let cube_count = if num_cubes_total <= MAX_DISPATCH {
                CubeCount::Static(num_cubes_total, 1, 1)
            } else {
                let cubes_y = (num_cubes_total + MAX_DISPATCH - 1) / MAX_DISPATCH;
                let cubes_x = (num_cubes_total + cubes_y - 1) / cubes_y;
                CubeCount::Static(
                    std::cmp::min(cubes_x, MAX_DISPATCH),
                    std::cmp::min(cubes_y, MAX_DISPATCH),
                    1,
                )
            };
            
            Some((stock_h, strike_h, time_h, rate_h, vol_h, 
                  call_h, put_h, cube_dim, cube_count, num_padded, n))
        } else {
            None
        };
        
        // === Step 2: NOW read previous results (blocking) ===
        // By now, upload is queued. GPU may be finishing previous batch.
        // The read_one() call blocks until GPU is done.
        let mut results = Vec::new();
        if let Some((call_h, put_h, count)) = self.pending.take() {
            let call_bytes = client.read_one(call_h);
            let put_bytes = client.read_one(put_h);
            let calls: &[f32] = bytemuck::cast_slice(&call_bytes);
            let puts: &[f32] = bytemuck::cast_slice(&put_bytes);
            
            results = calls[..count]
                .iter()
                .zip(&puts[..count])
                .map(|(&call_price, &put_price)| BlackScholesOutput { call_price, put_price })
                .collect();
        }
        
        // === Step 3: Launch new kernel with already-uploaded data ===
        if let Some((stock_h, strike_h, time_h, rate_h, vol_h, 
                     call_h, put_h, cube_dim, cube_count, num_padded, n)) = new_batch {
            #[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]
            unsafe {
                black_scholes_kernel::launch_unchecked::<f32, GpuRuntime>(
                    client,
                    cube_count,
                    cube_dim,
                    ArrayArg::from_raw_parts::<f32>(&stock_h, num_padded, vectorization_factor as u8),
                    ArrayArg::from_raw_parts::<f32>(&strike_h, num_padded, vectorization_factor as u8),
                    ArrayArg::from_raw_parts::<f32>(&time_h, num_padded, vectorization_factor as u8),
                    ArrayArg::from_raw_parts::<f32>(&rate_h, num_padded, vectorization_factor as u8),
                    ArrayArg::from_raw_parts::<f32>(&vol_h, num_padded, vectorization_factor as u8),
                    ArrayArg::from_raw_parts::<f32>(&call_h, num_padded, vectorization_factor as u8),
                    ArrayArg::from_raw_parts::<f32>(&put_h, num_padded, vectorization_factor as u8),
                );
            }
            
            // Store handles for next flush() to collect
            self.pending = Some((call_h, put_h, n));
        }
        
        self.buffer.clear();
        self.items_pushed = 0;
        
        // Return results from PREVIOUS batch (true double-buffering)
        results
    }

    fn preferred_batch_size(&self) -> Option<usize> {
        Some(DEFAULT_BATCH_SIZE)
    }
    
    fn drain(&mut self, ctx: &GpuContext) -> Vec<BlackScholesOutput> {
        let client = ctx.client();
        let mut results = Vec::new();
        
        if let Some((call_h, put_h, count)) = self.pending.take() {
            let call_bytes = client.read_one(call_h);
            let put_bytes = client.read_one(put_h);
            let calls: &[f32] = bytemuck::cast_slice(&call_bytes);
            let puts: &[f32] = bytemuck::cast_slice(&put_bytes);
            
            results = calls[..count]
                .iter()
                .zip(&puts[..count])
                .map(|(&call_price, &put_price)| BlackScholesOutput { call_price, put_price })
                .collect();
        }
        
        results
    }
}

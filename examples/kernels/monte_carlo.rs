//! # Monte Carlo Option Pricing Kernel
//!
//! GPU-accelerated Monte Carlo simulation for option pricing.
//! Uses reduced complexity for practical benchmark execution times.

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
// Configuration - Reduced for practical execution
// ============================================================================

/// Number of simulation paths per option
pub const MC_NUM_PATHS: u32 = 1_000;
/// Number of time steps per path
pub const MC_TIME_STEPS: u32 = 50;

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Clone, Copy, Debug, Pod, Zeroable, Serialize, Deserialize)]
#[repr(C)]
pub struct MonteCarloInput {
    pub stock_price: f32,
    pub strike_price: f32,
    pub time_to_expiry: f32,
    pub risk_free_rate: f32,
    pub volatility: f32,
}

#[derive(Clone, Copy, Debug, Pod, Zeroable, Serialize, Deserialize)]
#[repr(C)]
pub struct MonteCarloOutput {
    pub call_price: f32,
    pub put_price: f32,
}

// ============================================================================
// Deterministic Seed Generation
// ============================================================================

/// Generate a deterministic seed from input parameters.
///
/// This function creates a unique, reproducible seed for each Monte Carlo simulation
/// based on the option's input parameters. Both CPU and GPU implementations use this
/// same formula to ensure identical random number sequences and thus identical results.
///
/// # Algorithm: Hash Combining (Boost-style)
///
/// The algorithm uses a technique similar to `boost::hash_combine` from the C++ Boost library:
///
/// ```text
/// new_seed = old_seed XOR (value + 0x9e3779b9 + (old_seed << 6) + (old_seed >> 2))
/// ```
///
/// ## Why This Works
///
/// 1. **Golden Ratio Constant (0x9e3779b9)**:
///    - This is `2^32 / φ` where φ ≈ 1.618 (golden ratio)
///    - Binary: `10011110001101110111100110111001`
///    - Provides excellent bit mixing with no obvious patterns
///    - Used because it's coprime to 2^32, ensuring good distribution
///
/// 2. **Bit Shifts (`<< 6` and `>> 2`)**:
///    - Left shift by 6: spreads influence of lower bits upward
///    - Right shift by 2: spreads influence of upper bits downward
///    - Combined with XOR: ensures all input bits affect the output
///
/// 3. **XOR Combining**:
///    - Reversible operation (important for good hash properties)
///    - Each bit of output depends on multiple input bits
///    - Avoids the "cancellation" problem of simple addition
///
/// ## Visualization
///
/// ```text
/// Input parameters:
///   stock_price:    100.0  → 0x42C80000 (IEEE 754 bits)
///   strike_price:   105.0  → 0x42D20000
///   time_to_expiry: 1.0    → 0x3F800000
///   risk_free_rate: 0.05   → 0x3D4CCCCD
///   volatility:     0.2    → 0x3E4CCCCD
///
/// Hash combining chain:
///   seed₀ = stock_bits                           = 0x42C80000
///   seed₁ = seed₀ ⊕ hash_combine(strike_bits)    = 0x...
///   seed₂ = seed₁ ⊕ hash_combine(time_bits)      = 0x...
///   seed₃ = seed₂ ⊕ hash_combine(rate_bits)      = 0x...
///   seed₄ = seed₃ ⊕ hash_combine(vol_bits)       = final seed
/// ```
///
/// ## Properties
///
/// - **Deterministic**: Same inputs always produce the same seed
/// - **Unique**: Different inputs produce different seeds (with high probability)
/// - **Fast**: Only bitwise operations, no division or modulo
/// - **Non-zero guarantee**: Returns 1 if result would be 0 (xorshift requirement)
///
/// # Example
///
/// ```rust
/// let input = MonteCarloInput {
///     stock_price: 100.0,
///     strike_price: 105.0,
///     time_to_expiry: 1.0,
///     risk_free_rate: 0.05,
///     volatility: 0.2,
/// };
/// let seed = deterministic_seed(&input);
/// // seed is now a unique 32-bit value for this specific option
/// ```
#[inline]
pub fn deterministic_seed(input: &MonteCarloInput) -> u32 {
    // Convert each f32 parameter to its IEEE 754 bit representation
    let stock_bits = input.stock_price.to_bits();
    let strike_bits = input.strike_price.to_bits();
    let time_bits = input.time_to_expiry.to_bits();
    let rate_bits = input.risk_free_rate.to_bits();
    let vol_bits = input.volatility.to_bits();
    
    // Hash combining using boost::hash_combine algorithm
    // The magic constant 0x9e3779b9 is 2^32 / golden_ratio
    let mut seed = stock_bits;
    seed ^= strike_bits.wrapping_add(0x9e3779b9).wrapping_add(seed << 6).wrapping_add(seed >> 2);
    seed ^= time_bits.wrapping_add(0x9e3779b9).wrapping_add(seed << 6).wrapping_add(seed >> 2);
    seed ^= rate_bits.wrapping_add(0x9e3779b9).wrapping_add(seed << 6).wrapping_add(seed >> 2);
    seed ^= vol_bits.wrapping_add(0x9e3779b9).wrapping_add(seed << 6).wrapping_add(seed >> 2);
    
    // Ensure non-zero (xorshift RNG breaks on zero seed)
    if seed == 0 { 1 } else { seed }
}

// ============================================================================
// CPU Reference Implementation - Uses Same RNG as GPU
// ============================================================================

/// CPU xorshift32 RNG - identical algorithm to GPU kernel
/// CPU xorshift32 Random Number Generator.
///
/// This is a fast, high-quality PRNG that produces 32-bit pseudo-random integers
/// and converts them to uniform f32 values in the range [0, 1).
///
/// # Algorithm: xorshift32
///
/// The xorshift family of PRNGs was introduced by George Marsaglia in 2003.
/// xorshift32 uses three XOR operations with bit shifts to mix the state:
///
/// ```text
/// x ^= x << 13;  // Mix upper bits into lower bits
/// x ^= x >> 17;  // Mix lower bits into upper bits  
/// x ^= x << 5;   // Final mixing pass
/// ```
///
/// ## Properties
///
/// - **Period**: 2³² - 1 (all 32-bit values except 0)
/// - **Speed**: 3 XOR + 3 shift operations (very fast)
/// - **Quality**: Passes most statistical tests; suitable for Monte Carlo
/// - **State**: Single 32-bit word
///
/// ## Why `x >> 9` and `1.1920929e-7`?
///
/// The conversion to f32 uses the upper 23 bits:
/// ```text
/// bits = x >> 9           // Extract upper 23 bits (f32 mantissa size)
/// result = bits * 2^(-23) // Scale to [0, 1) range
///        = bits * 1.1920929e-7
/// ```
///
/// Using upper bits is important because xorshift has better randomness
/// in the high bits than the low bits.
///
/// # Reference
///
/// Marsaglia, G. (2003). "Xorshift RNGs". Journal of Statistical Software.
/// https://www.jstatsoft.org/article/view/v008i14
#[inline]
fn xorshift_cpu(state: &mut u32) -> f32 {
    let mut x = *state;
    // Three-round xorshift mixing
    x ^= x << 13;  // Left shift: spread lower bits upward
    x ^= x >> 17;  // Right shift: spread upper bits downward
    x ^= x << 5;   // Final mixing: additional bit dispersion
    *state = x;
    
    // Convert to f32 in [0, 1) using upper 23 bits
    // 23 bits = f32 mantissa precision
    // 1.1920929e-7 = 2^(-23) = 1 / 8388608
    let bits = x >> 9;
    bits as f32 * 1.1920929e-7
}

/// CPU Box-Muller Transform for Normal Distribution.
///
/// Converts two uniform random numbers U₁, U₂ ∈ [0, 1) into 
/// a standard normal random variable Z ~ N(0, 1).
///
/// # Algorithm: Box-Muller Transform
///
/// The Box-Muller transform is a method for generating pairs of independent
/// standard normal random variables from uniform random variables:
///
/// ```text
/// Z₁ = √(-2 ln U₁) × cos(2π U₂)
/// Z₂ = √(-2 ln U₁) × sin(2π U₂)
/// ```
///
/// We only use Z₁ (the cosine form) for simplicity.
///
/// ## Mathematical Derivation
///
/// The transform works by:
/// 1. `-2 ln U₁` follows an exponential distribution with λ=2
/// 2. `√(-2 ln U₁)` transforms this to a Rayleigh distribution
/// 3. `2π U₂` provides uniform angle in [0, 2π)
/// 4. Combining with cos/sin projects to normal distribution
///
/// ## Implementation Notes
///
/// - `u1_safe = u1.max(1e-10)`: Prevents ln(0) = -∞
/// - `6.283185 = 2π`: Full circle in radians
/// - Returns only Z₁ (cosine); Z₂ (sine) is discarded
///
/// ## Properties
///
/// - **Input**: Two uniform U(0,1) samples
/// - **Output**: Two standard normal N(0,1) samples (Z₁, Z₂)
/// - **Efficiency**: 100% (uses both outputs)
///
/// # Reference
///
/// Box, G.E.P.; Muller, M.E. (1958). "A Note on the Generation of Random Normal Deviates".
/// The Annals of Mathematical Statistics. 29 (2): 610–611.
#[inline]
fn box_muller_pair_cpu(state: &mut u32) -> (f32, f32) {
    // Generate two uniform random numbers
    let u1 = xorshift_cpu(state);
    let u2 = xorshift_cpu(state);
    
    // Clamp u1 to prevent ln(0) = -∞
    let u1_safe = u1.max(1e-10);
    
    // Box-Muller transform produces TWO independent normal deviates:
    // Z₁ = √(-2 ln U₁) × cos(2π U₂)
    // Z₂ = √(-2 ln U₁) × sin(2π U₂)
    let r = (-2.0 * u1_safe.ln()).sqrt();
    let theta = 6.283185 * u2;
    
    (r * theta.cos(), r * theta.sin())
}

pub fn monte_carlo_cpu(input: MonteCarloInput) -> MonteCarloOutput {
    // Use deterministic seed from input - same as GPU
    let mut seed = deterministic_seed(&input);
    
    let s0 = input.stock_price;
    let k = input.strike_price;
    let t = input.time_to_expiry;
    let r = input.risk_free_rate;
    let v = input.volatility;
    
    let dt = t / MC_TIME_STEPS as f32;
    let drift = (r - 0.5 * v * v) * dt;
    let diffusion = v * dt.sqrt();
    
    // Pre-compute number of Box-Muller pairs needed
    let half_steps = MC_TIME_STEPS / 2;
    let odd_step = MC_TIME_STEPS % 2 == 1;
    
    let mut sum_payoff = 0.0f32;
    
    for _ in 0..MC_NUM_PATHS {
        // LOG-PRICE FORMULATION: Track cumulative log-return instead of price
        // This reduces exp() calls from MC_TIME_STEPS per path to just 1
        let mut log_return = 0.0f32;
        
        // Process two steps at a time using both Box-Muller outputs
        for _ in 0..half_steps {
            let (z1, z2) = box_muller_pair_cpu(&mut seed);
            log_return += drift + diffusion * z1;
            log_return += drift + diffusion * z2;
        }
        
        // Handle odd step if MC_TIME_STEPS is odd
        if odd_step {
            let (z1, _) = box_muller_pair_cpu(&mut seed);
            log_return += drift + diffusion * z1;
        }
        
        // Single exp() at the end: S_T = S_0 × exp(log_return)
        let s_final = s0 * log_return.exp();
        sum_payoff += (s_final - k).max(0.0);
    }
    
    let df = (-r * t).exp();
    let call_price = df * sum_payoff / MC_NUM_PATHS as f32;
    let put_price = call_price - s0 + k * df;

    MonteCarloOutput { call_price, put_price }
}


// ============================================================================
// GPU Kernel
// ============================================================================

/// GPU xorshift32 Random Number Generator.
///
/// Generates a pseudo-random f32 in [0, 1) using the xorshift32 algorithm.
/// This is identical to the CPU version to ensure deterministic results.
///
/// # Algorithm
/// The xorshift32 PRNG uses three XOR-with-shift operations:
/// - `x ^= x << 13`: Mix bits upward
/// - `x ^= x >> 17`: Mix bits downward
/// - `x ^= x << 5`:  Final mixing pass
///
/// # Conversion to f32
/// Uses upper 23 bits (x >> 9) because:
/// - f32 mantissa has exactly 23 bits of precision
/// - xorshift has better randomness in high bits
/// - Multiplying by 2^(-23) = 1.1920929e-7 scales to [0, 1)
#[cube]
fn xorshift(state: &mut u32) -> f32 {
    let mut x = *state;
    
    // Three-round xorshift mixing (Marsaglia, 2003)
    x ^= x << 13;  // Spread lower bits upward
    x ^= x >> 17;  // Spread upper bits downward
    x ^= x << 5;   // Final dispersion
    *state = x;
    
    // Convert to f32 in [0, 1) using upper 23 bits
    // bits = x >> 9 gives 23 bits of precision
    // 1.1920929e-7 = 2^(-23) = 1.0 / 8388608.0
    let bits = x >> 9;
    let bits_f = bits as f32;
    bits_f * 1.1920929e-7
}

/// GPU Box-Muller Transform for Normal Distribution (Pair Version).
///
/// Converts two uniform random numbers U₁, U₂ ∈ [0, 1) into TWO standard 
/// normal random variables Z₁, Z₂ ~ N(0, 1).
///
/// # Algorithm (Box-Muller, 1958)
/// Given uniform U₁, U₂:
///   Z₁ = √(-2 ln U₁) × cos(2π U₂)
///   Z₂ = √(-2 ln U₁) × sin(2π U₂)
///
/// Returns BOTH outputs for 100% efficiency.
///
/// # Implementation Notes
/// - `u1_safe = max(u1, 1e-10)`: Prevents ln(0) = -∞
/// - `6.283185 = 2π`: Full circle in radians
#[cube]
fn box_muller_pair<F: Float>(state: &mut u32) -> (F, F) {
    // Generate two uniform random numbers U₁, U₂ ∈ [0, 1)
    let u1 = F::cast_from(xorshift(state));
    let u2 = F::cast_from(xorshift(state));
    
    // Clamp u1 to prevent ln(0) = -∞
    // Note: CubeCL doesn't have max(), so we use conditional
    let u1_safe = if u1 < F::new(1e-10) { F::new(1e-10) } else { u1 };
    
    // Box-Muller transform produces TWO independent normal deviates
    let r = F::sqrt(F::new(-2.0) * F::log(u1_safe));
    let theta = F::new(6.283185) * u2;
    
    (r * F::cos(theta), r * F::sin(theta))
}

/// GPU Monte Carlo Option Pricing Kernel.
///
/// Prices European call and put options using Monte Carlo simulation.
/// Each GPU thread processes ONE option independently (embarrassingly parallel).
///
/// # Parallelization Strategy
/// - ABSOLUTE_POS = unique thread ID (0, 1, 2, ..., N-1)
/// - Thread i processes option i: loads params, runs all paths, writes result
/// - No thread synchronization needed (independent computations)
///
/// # Algorithm (per thread)
/// 1. Load option parameters from SoA arrays (coalesced memory access)
/// 2. Calculate simulation parameters: dt, drift, diffusion
/// 3. For each path (sequential within thread):
///    a. Start with S = S₀
///    b. For each time step: S *= exp(drift + diffusion × Z)
///    c. Calculate payoff: max(S - K, 0)
/// 4. Average payoffs and discount to present value
/// 5. Calculate put price using put-call parity
///
/// # Performance
/// - Each thread: num_paths × num_steps × 2 RNG calls
/// - Typical: 1000 paths × 50 steps = 50,000 iterations per thread
/// - 1M options × 50K iterations = 50 billion operations (parallel!)
#[cube(launch_unchecked)]
pub fn monte_carlo_kernel<F: Float>(
    // Input arrays (Structure of Arrays layout for coalesced access)
    stocks: &Array<F>,      // S₀: Initial stock prices
    strikes: &Array<F>,     // K: Strike prices
    times: &Array<F>,       // T: Time to expiration (years)
    rates: &Array<F>,       // r: Risk-free interest rate
    vols: &Array<F>,        // σ: Volatility
    seeds: &Array<u32>,     // Deterministic seed per option
    
    // Output arrays
    call_out: &mut Array<F>,  // Call option prices
    put_out: &mut Array<F>,   // Put option prices
    
    // Simulation parameters
    num_paths: u32,   // Number of Monte Carlo paths (e.g., 1000)
    num_steps: u32,   // Number of time steps per path (e.g., 50)
) {
    // Bounds check: each thread processes exactly one option
    if ABSOLUTE_POS < stocks.len() {
        // === Step 1: Load option parameters ===
        // Coalesced memory access: adjacent threads read adjacent elements
        let mut seed = seeds[ABSOLUTE_POS];
        let s0 = stocks[ABSOLUTE_POS];   // Initial stock price
        let k = strikes[ABSOLUTE_POS];   // Strike price
        let t = times[ABSOLUTE_POS];     // Time to expiry
        let r = rates[ABSOLUTE_POS];     // Risk-free rate
        let v = vols[ABSOLUTE_POS];      // Volatility
        
        // === Step 2: Calculate simulation parameters ===
        // dt = time per step (e.g., 1 year / 50 steps = 0.02)
        let dt = t / F::cast_from(num_steps as f32);
        
        // GBM drift: (r - σ²/2) × dt
        // This is the expected log-return per step
        let drift = (r - F::new(0.5) * v * v) * dt;
        
        // GBM diffusion: σ × √dt
        // This is the volatility scaling per step
        let diffusion = v * F::sqrt(dt);
        
        // === Step 3: Monte Carlo simulation with LOG-PRICE FORMULATION ===
        // Instead of computing S *= exp(...) each step, we accumulate log-returns
        // and apply a SINGLE exp() at the end. This reduces exp() calls from
        // num_steps per path to just 1 per path.
        let mut sum_payoff = F::new(0.0);
        
        // Pre-compute loop bounds for Box-Muller pair optimization
        let half_steps = num_steps / 2;
        let odd_step = num_steps % 2 == 1;
        
        // Simulate num_paths independent price paths
        for _ in 0..num_paths {
            // Track cumulative log-return instead of price
            let mut log_return = F::new(0.0);
            
            // Process TWO steps at a time using BOTH Box-Muller outputs
            // This eliminates 50% of RNG calls
            for _ in 0..half_steps {
                // Generate TWO normal random Z₁, Z₂ ~ N(0,1) from one Box-Muller call
                let (z1, z2) = box_muller_pair::<F>(&mut seed);
                
                // Accumulate log-returns (just additions!)
                log_return += drift + diffusion * z1;
                log_return += drift + diffusion * z2;
            }
            
            // Handle odd step if num_steps is odd
            if odd_step {
                let (z1, _) = box_muller_pair::<F>(&mut seed);
                log_return += drift + diffusion * z1;
            }
            
            // SINGLE exp() at the end: S_T = S_0 × exp(log_return)
            let s_final = s0 * F::exp(log_return);
            
            // Calculate call payoff: max(S_T - K, 0)
            // Note: CubeCL doesn't have max(), so we use conditional
            let payoff = if s_final > k { s_final - k } else { F::new(0.0) };
            sum_payoff += payoff;
        }
        
        // === Step 4: Calculate option prices ===
        // Discount factor: e^(-rT) (present value of $1 received at time T)
        let df = F::exp(F::new(-1.0) * r * t);
        
        // Call price = discounted average payoff
        let call_price = df * sum_payoff / F::cast_from(num_paths as f32);
        
        // === Step 5: Write results ===
        call_out[ABSOLUTE_POS] = call_price;
        
        // Put price from put-call parity: P = C - S + K × e^(-rT)
        put_out[ABSOLUTE_POS] = call_price - s0 + k * df;
    }
}


// ============================================================================
// Structure of Arrays (SoA) Buffer
// ============================================================================

/// Structure of Arrays (SoA) buffer for GPU-efficient data layout.
///
/// GPU memory access is most efficient when adjacent threads access adjacent
/// memory locations (coalesced access). SoA layout achieves this by storing
/// each field in a separate contiguous array.
///
/// # Memory Layout Comparison
///
/// ```text
/// Array of Structs (AoS) - Poor GPU performance:
/// [Option0{S,K,T,r,σ}, Option1{S,K,T,r,σ}, Option2{S,K,T,r,σ}, ...]
///   └── Thread 0 reads here, but next option is 5 floats away
///
/// Structure of Arrays (SoA) - Good GPU performance:
/// stocks:  [S0, S1, S2, S3, ...]  ← Thread 0,1,2,3 read adjacent elements
/// strikes: [K0, K1, K2, K3, ...]  ← Coalesced memory access!
/// times:   [T0, T1, T2, T3, ...]
/// rates:   [r0, r1, r2, r3, ...]
/// vols:    [σ0, σ1, σ2, σ3, ...]
/// ```
#[derive(Clone)]
struct SoABuffer {
    /// Initial stock prices (S₀) for each option
    stocks: Vec<f32>,
    /// Strike prices (K) for each option
    strikes: Vec<f32>,
    /// Time to expiration in years (T) for each option
    times: Vec<f32>,
    /// Risk-free interest rates (r) for each option
    rates: Vec<f32>,
    /// Volatilities (σ) for each option
    vols: Vec<f32>,
    /// Deterministic seeds for each option's RNG
    seeds: Vec<u32>,
}

impl SoABuffer {
    /// Creates an empty SoA buffer with no options.
    fn new() -> Self {
        Self {
            stocks: Vec::new(),
            strikes: Vec::new(),
            times: Vec::new(),
            rates: Vec::new(),
            vols: Vec::new(),
            seeds: Vec::new(),
        }
    }
    
    /// Creates a SoA buffer with pre-allocated capacity.
    /// 
    /// This avoids repeated allocations during push() operations
    /// when the expected batch size is known in advance.
    #[allow(dead_code)]
    fn with_capacity(capacity: usize) -> Self {
        Self {
            stocks: Vec::with_capacity(capacity),
            strikes: Vec::with_capacity(capacity),
            times: Vec::with_capacity(capacity),
            rates: Vec::with_capacity(capacity),
            vols: Vec::with_capacity(capacity),
            seeds: Vec::with_capacity(capacity),
        }
    }

    /// Adds an option to the buffer.
    ///
    /// # Arguments
    /// * `input` - The option parameters (stock, strike, time, rate, volatility)
    /// * `seed` - Pre-computed deterministic seed for this option's RNG
    fn push(&mut self, input: &MonteCarloInput, seed: u32) {
        self.stocks.push(input.stock_price);
        self.strikes.push(input.strike_price);
        self.times.push(input.time_to_expiry);
        self.rates.push(input.risk_free_rate);
        self.vols.push(input.volatility);
        self.seeds.push(seed);
    }

    /// Clears all options from the buffer.
    /// Called after GPU kernel launch to prepare for next batch.
    fn clear(&mut self) {
        self.stocks.clear();
        self.strikes.clear();
        self.times.clear();
        self.rates.clear();
        self.vols.clear();
        self.seeds.clear();
    }

    /// Returns the number of options currently in the buffer.
    fn len(&self) -> usize {
        self.stocks.len()
    }
}

// ============================================================================
// GpuKernel Implementation
// ============================================================================

/// Monte Carlo GPU kernel wrapper implementing the GpuKernel trait.
///
/// This struct manages the buffering and execution of Monte Carlo option
/// pricing on the GPU. It implements double-buffering for overlapping
/// computation and data transfer.
///
/// # Double-Buffering Strategy
///
/// ```text
/// Time ──────────────────────────────────────────────►
///
/// Batch 1:  [Push items] → [Launch GPU] → [Read results]
/// Batch 2:               [Push items] → [Launch GPU] → [Read results]
///                                      └── overlapped!
/// ```
///
/// While GPU is computing batch N, CPU can:
/// - Push items for batch N+1
/// - Read results from batch N-1
#[derive(Clone)]
pub struct MonteCarloKernel {
    /// SoA buffer holding pending options to be processed
    buffer: SoABuffer,
    
    /// Pending GPU results: (call_handle, put_handle, count)
    /// None if no computation is in flight
    pending: Option<(cubecl::server::Handle, cubecl::server::Handle, usize)>,
    
    /// CUDA/GPU work group size (threads per block)
    /// Default: 256 threads, which works well on most GPUs
    cube_dim: u32,
}

impl Default for MonteCarloKernel {
    fn default() -> Self {
        Self {
            buffer: SoABuffer::new(),
            pending: None,
            cube_dim: 256,  // 256 threads per work group
        }
    }
}

impl GpuKernel for MonteCarloKernel {
    type Input = MonteCarloInput;
    type Output = MonteCarloOutput;

    /// Adds an option to the pending buffer for GPU processing.
    ///
    /// The seed is computed deterministically from the input parameters
    /// to ensure reproducible results across CPU and GPU.
    fn push(&mut self, item: Self::Input) {
        // Compute deterministic seed from input (same as CPU version)
        let seed = deterministic_seed(&item);
        self.buffer.push(&item, seed);
    }

    /// Returns the number of options waiting to be processed.
    fn buffer_len(&self) -> usize {
        self.buffer.len()
    }

    /// Launches GPU kernel and returns results from previous batch.
    ///
    /// This method implements **true double-buffering** with GPU overlap:
    /// 1. First, upload NEW batch data (async, while GPU still computing)
    /// 2. Then, read previous batch results (blocking, GPU should be done by now)
    /// 3. Finally, launch new kernel
    ///
    /// This order ensures that data upload overlaps with GPU computation,
    /// minimizing GPU idle time between batches.
    ///
    /// # Returns
    /// Results from the *previous* flush() call, not the current one.
    /// Call drain() to get results from the final batch.
    fn flush(&mut self, ctx: &GpuContext) -> Vec<Self::Output> {
        let client = ctx.client();
        
        let n = self.buffer.len();
        
        // === Step 1: Upload NEW batch data FIRST (async) ===
        // This happens WHILE the GPU is still computing the previous batch!
        // The upload is queued and may overlap with ongoing computation.
        let new_batch = if n > 0 {
            // Create GPU buffers from SoA data.
            // IMPORTANT: client.create() COPIES the data into GPU memory immediately!
            // The returned Handle references the GPU-side copy, not the original Vec.
            // This means we can safely clear self.buffer after this block.
            let stock_h = client.create(bytemuck::cast_slice(&self.buffer.stocks));
            let strike_h = client.create(bytemuck::cast_slice(&self.buffer.strikes));
            let time_h = client.create(bytemuck::cast_slice(&self.buffer.times));
            let rate_h = client.create(bytemuck::cast_slice(&self.buffer.rates));
            let vol_h = client.create(bytemuck::cast_slice(&self.buffer.vols));
            let seed_h = client.create(bytemuck::cast_slice(&self.buffer.seeds));
            
            // Allocate output buffers (n options × 4 bytes per f32)
            let call_h = client.empty(n * 4);
            let put_h = client.empty(n * 4);
            
            // Configure GPU grid
            let cube_dim = CubeDim::new(self.cube_dim, 1, 1);
            let num_cubes = (n as u32 + self.cube_dim - 1) / self.cube_dim;
            let cube_count = CubeCount::Static(num_cubes, 1, 1);
            
            // Return all handles as a tuple wrapped in Some
            // This does NOT return from the function - it's the value of the if-expression
            // which gets assigned to `new_batch` for use in Step 3
            Some((stock_h, strike_h, time_h, rate_h, vol_h, seed_h, 
                  call_h, put_h, cube_dim, cube_count, n))
        } else {
            // No items to process - new_batch = None, Step 3 will be skipped
            None
        };
        
        // === Step 2: NOW read previous results (blocking) ===
        // By now, upload is queued. GPU may be finishing previous batch.
        // The read_one() call blocks until GPU is done.
        let mut results = Vec::new();
        if let Some((call_h, put_h, count)) = self.pending.take() {
            // Read GPU output buffers back to CPU (BLOCKING)
            let call_bytes = client.read_one(call_h);
            let put_bytes = client.read_one(put_h);
            
            // Reinterpret bytes as f32 slices
            let calls: &[f32] = bytemuck::cast_slice(&call_bytes);
            let puts: &[f32] = bytemuck::cast_slice(&put_bytes);
            
            // Convert to output structs using iterator-based collection
            // More efficient than indexed for-loop: avoids bounds checks and
            // allows the compiler to better optimize the iteration
            results.extend(
                calls[..count].iter()
                    .zip(&puts[..count])
                    .map(|(&call_price, &put_price)| MonteCarloOutput { call_price, put_price })
            );
        }
        
        // === Step 3: Launch new kernel with already-uploaded data ===
        if let Some((stock_h, strike_h, time_h, rate_h, vol_h, seed_h, 
                     call_h, put_h, cube_dim, cube_count, n)) = new_batch {
            #[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]
            unsafe {
                monte_carlo_kernel::launch_unchecked::<f32, GpuRuntime>(
                    client,
                    cube_count,
                    cube_dim,
                    // Input arrays (already uploaded in Step 1)
                    ArrayArg::from_raw_parts::<f32>(&stock_h, n, 1),
                    ArrayArg::from_raw_parts::<f32>(&strike_h, n, 1),
                    ArrayArg::from_raw_parts::<f32>(&time_h, n, 1),
                    ArrayArg::from_raw_parts::<f32>(&rate_h, n, 1),
                    ArrayArg::from_raw_parts::<f32>(&vol_h, n, 1),
                    ArrayArg::from_raw_parts::<u32>(&seed_h, n, 1),
                    // Output arrays
                    ArrayArg::from_raw_parts::<f32>(&call_h, n, 1),
                    ArrayArg::from_raw_parts::<f32>(&put_h, n, 1),
                    // Simulation parameters
                    ScalarArg::new(MC_NUM_PATHS),
                    ScalarArg::new(MC_TIME_STEPS),
                );
            }
            
            // Store handles for next flush() to collect
            self.pending = Some((call_h, put_h, n));
        }
        
        // SAFE to clear: client.create() already COPIED the data to GPU memory.
        // The Handle references the GPU-side copy, not the original Vec.
        // GPU command ordering guarantees the kernel sees the uploaded data.
        self.buffer.clear();
        
        // Return results from PREVIOUS batch (true double-buffering)
        results
    }
    
    /// Collects results from the final pending GPU computation.
    ///
    /// Called at the end of processing to ensure all results are retrieved.
    /// Unlike flush(), this does not launch a new kernel.
    fn drain(&mut self, ctx: &GpuContext) -> Vec<Self::Output> {
        let client = ctx.client();
        let mut results = Vec::new();
        
        // Collect any pending results
        if let Some((call_h, put_h, count)) = self.pending.take() {
            let call_bytes = client.read_one(call_h);
            let put_bytes = client.read_one(put_h);
            let calls: &[f32] = bytemuck::cast_slice(&call_bytes);
            let puts: &[f32] = bytemuck::cast_slice(&put_bytes);
            
            // Iterator-based collection for better performance
            results.extend(
                calls[..count].iter()
                    .zip(&puts[..count])
                    .map(|(&call_price, &put_price)| MonteCarloOutput { call_price, put_price })
            );
        }
        
        results
    }
}

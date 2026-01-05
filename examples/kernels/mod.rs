//! # GPU Kernels
//!
//! This module contains reusable GPU kernel implementations that can be used
//! by both examples and benchmarks.
//!
//! ## Available Kernels
//!
//! - [`black_scholes`] - Black-Scholes option pricing kernel
//! - [`monte_carlo`] - Monte Carlo option pricing kernel

pub mod black_scholes;
pub mod monte_carlo;

#[cfg(test)]
pub mod tests;

// Re-export commonly used items for convenience
#[allow(unused_imports)]
pub use black_scholes::*;

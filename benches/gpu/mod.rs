//! GPU benchmark modules.
//!
//! This module contains GPU benchmarks for different computational kernels:
//! - `black_scholes` - Black-Scholes option pricing benchmark
//! - `common` - Shared utilities for all GPU benchmarks
//! - `monte_carlo` - Monte Carlo option pricing benchmark
//! - `reduce` - GPU reduce (Sum, Product, Min, Max) benchmark

pub mod black_scholes;
pub mod common;
pub mod monte_carlo;
pub mod reduce;

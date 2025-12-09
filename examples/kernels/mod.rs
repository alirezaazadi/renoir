//! # GPU Kernels
//!
//! This module contains reusable GPU kernel implementations that can be used
//! by both examples and benchmarks.
//!
//! ## Available Kernels
//!
//! - [`black_scholes`] - Black-Scholes option pricing kernel

pub mod black_scholes;
pub use black_scholes::*;


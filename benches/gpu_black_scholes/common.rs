//! # Common Benchmark Utilities
//!
//! Shared utilities for GPU Black-Scholes benchmarks including:
//! - Benchmark type definitions
//! - Test size generation
//! - Results persistence with organized folder structure
//! - Filename generation with ISO timestamps

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

// ============================================================================
// Benchmark Test Sizes
// ============================================================================

/// All available benchmark test sizes.
///
/// This is the unified list of test sizes used across all benchmarks.
/// Includes sizes from 10K to 20B for comprehensive GPU testing.
const ALL_TEST_SIZES: &[usize] = &[
    10_000,
    25_000,
    50_000,
    75_000,
    100_000,
    250_000,
    500_000,
    750_000,
    1_000_000,
    2_500_000,
    5_000_000,
    7_500_000,
    10_000_000,
    20_000_000,
    25_000_000,
    50_000_000,
    70_000_000,
    75_000_000,
    100_000_000,
    250_000_000,
    500_000_000,
    750_000_000,
    1_000_000_000,
    2_000_000_000,
    5_000_000_000,
    8_000_000_000,
    10_000_000_000,
    20_000_000_000,
];

/// Default maximum options count when not specified by user.
pub const DEFAULT_MAX_OPTIONS: usize = 1_000_000_000;

/// Generate benchmark test sizes up to the specified maximum.
///
/// If `max_options` is `None`, uses `DEFAULT_MAX_OPTIONS`.
///
/// # Arguments
/// * `max_options` - Maximum number of options to include in test sizes
///
/// # Returns
/// Vector of test sizes up to and including max_options
pub fn generate_benchmark_test_sizes(max_options: Option<usize>) -> Vec<usize> {
    let max = max_options.unwrap_or(DEFAULT_MAX_OPTIONS);
    ALL_TEST_SIZES
        .iter()
        .copied()
        .filter(|&s| s <= max)
        .collect()
}

/// Parse MAX_OPTIONS from environment variable.
///
/// Returns `Some(value)` if the environment variable is set and valid,
/// `None` otherwise (meaning use default sizes).
pub fn parse_max_options_env() -> Option<usize> {
    env::var("MAX_OPTIONS")
        .ok()
        .and_then(|s| s.parse().ok())
}

/// Get test sizes based on environment variable or default.
///
/// This is the main function benchmarks should use to get test sizes.
/// It reads MAX_OPTIONS from environment and filters the unified test sizes.
///
/// # Example
/// ```ignore
/// // Uses default max (1B) if MAX_OPTIONS not set
/// let sizes = get_benchmark_test_sizes();
///
/// // Or with custom max via environment:
/// // MAX_OPTIONS=100000000 cargo bench ...
/// ```
pub fn get_benchmark_test_sizes() -> Vec<usize> {
    generate_benchmark_test_sizes(parse_max_options_env())
}

/// Get the maximum options value (from env or default).
pub fn get_max_options() -> usize {
    parse_max_options_env().unwrap_or(DEFAULT_MAX_OPTIONS)
}

// ============================================================================
// Number Formatting Utilities
// ============================================================================

/// Format a number with thousand separators (commas) for human readability.
///
/// # Examples
/// ```ignore
/// assert_eq!(format_number(1234567), "1,234,567");
/// assert_eq!(format_number(1000), "1,000");
/// assert_eq!(format_number(42), "42");
/// ```
pub fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    let chars: Vec<char> = s.chars().collect();
    
    for (i, c) in chars.iter().enumerate() {
        if i > 0 && (chars.len() - i) % 3 == 0 {
            result.push(',');
        }
        result.push(*c);
    }
    
    result
}

/// Format a number with SI suffix (K, M, B, T) for compact display.
///
/// # Examples
/// ```ignore
/// assert_eq!(format_number_short(1_500_000), "1.5M");
/// assert_eq!(format_number_short(2_500_000_000), "2.5B");
/// ```
pub fn format_number_short(n: usize) -> String {
    if n >= 1_000_000_000_000 {
        format!("{:.1}T", n as f64 / 1_000_000_000_000.0)
    } else if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}

/// Benchmark type enum for organizing results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BenchmarkType {
    Standard,
    Optimized,
    Streaming,
    Comparison,
}

impl BenchmarkType {
    /// Get the folder name for this benchmark type.
    pub fn folder_name(&self) -> &'static str {
        match self {
            BenchmarkType::Standard => "standard",
            BenchmarkType::Optimized => "optimized",
            BenchmarkType::Streaming => "streaming",
            BenchmarkType::Comparison => "comparison",
        }
    }

    /// Get the file prefix for this benchmark type.
    pub fn file_prefix(&self) -> &'static str {
        match self {
            BenchmarkType::Standard => "standard_benchmark",
            BenchmarkType::Optimized => "optimized_benchmark",
            BenchmarkType::Streaming => "streaming_benchmark",
            BenchmarkType::Comparison => "comparison_benchmark",
        }
    }

    /// Get the display name for this benchmark type (for chart titles).
    pub fn display_name(&self) -> &'static str {
        match self {
            BenchmarkType::Standard => "Standard",
            BenchmarkType::Optimized => "Optimized",
            BenchmarkType::Streaming => "Streaming",
            BenchmarkType::Comparison => "Batching Comparison",
        }
    }
}

// ============================================================================
// Results Directory and Filename Generation
// ============================================================================

/// Get the results directory for a specific benchmark type and date.
///
/// Creates folder structure: benches/results/{benchmark_type}/{YYYY-MM-DD}/
/// Example: benches/results/standard/2024-12-04/
pub fn get_results_dir(benchmark_type: BenchmarkType, timestamp: &DateTime<Utc>) -> PathBuf {
    let date_str = timestamp.format("%Y-%m-%d").to_string();
    let results_dir = PathBuf::from("benches/results")
        .join(benchmark_type.folder_name())
        .join(&date_str);
    fs::create_dir_all(&results_dir).expect("Failed to create results directory");
    results_dir
}

/// Generate the filename for a benchmark result file.
///
/// Format: {type}_benchmark_{ISO_timestamp}.json
/// Example: standard_benchmark_2024-12-04T10-30-45.json
pub fn get_benchmark_filename(benchmark_type: BenchmarkType, timestamp: &DateTime<Utc>) -> String {
    let time_str = timestamp.format("%Y-%m-%dT%H-%M-%S").to_string();
    format!("{}_{}.json", benchmark_type.file_prefix(), time_str)
}

/// Get the full path for a benchmark result file.
pub fn get_benchmark_filepath(benchmark_type: BenchmarkType, timestamp: &DateTime<Utc>) -> PathBuf {
    let dir = get_results_dir(benchmark_type, timestamp);
    dir.join(get_benchmark_filename(benchmark_type, timestamp))
}

/// Generate the plot filename for a benchmark.
///
/// Format: plot_{type}_benchmark_{ISO_timestamp}.png
/// Example: plot_standard_benchmark_2024-12-04T10-30-45.png
pub fn get_plot_filename(benchmark_type: BenchmarkType, timestamp: &DateTime<Utc>) -> String {
    let time_str = timestamp.format("%Y-%m-%dT%H-%M-%S").to_string();
    format!("plot_{}_{}.png", benchmark_type.file_prefix(), time_str)
}

/// Get the full path for a plot file.
pub fn get_plot_filepath(benchmark_type: BenchmarkType, timestamp: &DateTime<Utc>) -> PathBuf {
    let dir = get_results_dir(benchmark_type, timestamp);
    dir.join(get_plot_filename(benchmark_type, timestamp))
}

// ============================================================================
// Results Persistence
// ============================================================================

/// Save any serializable report to JSON.
pub fn save_json<T: Serialize>(filename: &PathBuf, report: &T) {
    let json = serde_json::to_string_pretty(report).expect("Failed to serialize results");
    let mut file = File::create(filename).expect("Failed to create output file");
    file.write_all(json.as_bytes())
        .expect("Failed to write results");
}

/// Get platform information string.
pub fn get_platform_info() -> String {
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    #[cfg(feature = "gpu-wgpu")]
    let backend = "WGPU";
    #[cfg(all(feature = "gpu-cuda", not(feature = "gpu-wgpu")))]
    let backend = "CUDA";
    #[cfg(not(any(feature = "gpu-wgpu", feature = "gpu-cuda")))]
    let backend = "CPU";
    format!("{} {} {}", os, arch, backend)
}

// ============================================================================
// Plot Generation
// ============================================================================

/// Run the plot script to generate visualizations from benchmark results.
///
/// This function calls the Python plotting script to generate charts from
/// the benchmark JSON results file.
///
/// # Arguments
/// * `json_path` - Path to the JSON results file
/// * `benchmark_type` - Type of benchmark for determining plot filename
/// * `timestamp` - Timestamp for the plot filename
pub fn run_plotter(
    json_path: &std::path::Path,
    benchmark_type: BenchmarkType,
    timestamp: &DateTime<Utc>,
) {
    let script_path = "benches/tools/plot_benchmark.py";
    let plot_path = get_plot_filepath(benchmark_type, timestamp);

    println!("\nGenerating {} benchmark plots...", benchmark_type.display_name());
    println!("  JSON:  {}", json_path.display());
    println!("  Plot:  {}", plot_path.display());

    let status = std::process::Command::new("python3")
        .args([script_path, json_path.to_str().unwrap_or("")])
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("Plot generation completed successfully.");
            println!("View: {}", plot_path.display());
        }
        Ok(s) => {
            eprintln!("Warning: Plot script exited with status: {}", s);
            eprintln!("You can manually run: python3 {} {}", script_path, json_path.display());
        }
        Err(e) => {
            eprintln!("Warning: Failed to run plot script: {}", e);
            eprintln!("You can manually run: python3 {} {}", script_path, json_path.display());
        }
    }
}

//! # Common GPU Benchmark Utilities
//!
//! Shared utilities for all GPU benchmarks (Black-Scholes, Monte Carlo, etc.):
//! - Benchmark type definitions
//! - Multi-run statistical benchmarking (mean, stddev, min, max, median)
//! - Test size generation
//! - Results persistence with organized folder structure
//! - Filename generation with ISO timestamps
//! - System configuration detection

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;

// ============================================================================
// Multi-Run Benchmark Statistics
// ============================================================================

/// Default number of benchmark runs per test size.
pub const DEFAULT_NUM_RUNS: usize = 5;

/// Number of warmup runs before collecting measurements.
pub const WARMUP_RUNS: usize = 1;

/// Parse BENCH_RUNS from environment variable.
///
/// Returns the configured number of runs, or `DEFAULT_NUM_RUNS` if not set.
pub fn parse_num_runs_env() -> usize {
    env::var("BENCH_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_NUM_RUNS)
}

/// Statistics from multiple benchmark runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunStats {
    /// Mean duration in seconds
    pub mean: f64,
    /// Standard deviation in seconds
    pub stddev: f64,
    /// Minimum duration in seconds
    pub min: f64,
    /// Maximum duration in seconds
    pub max: f64,
    /// Median duration in seconds
    pub median: f64,
    /// Individual run durations in seconds
    pub runs: Vec<f64>,
    /// Number of runs (excluding warmup)
    pub num_runs: usize,
}

impl Default for RunStats {
    fn default() -> Self {
        Self {
            mean: 0.0,
            stddev: 0.0,
            min: 0.0,
            max: 0.0,
            median: 0.0,
            runs: Vec::new(),
            num_runs: 0,
        }
    }
}

/// Compute statistics from a slice of duration measurements (in seconds).
///
/// # Arguments
/// * `durations` - Slice of measured durations in seconds
///
/// # Returns
/// A `RunStats` struct with computed mean, stddev, min, max, median.
pub fn compute_stats(durations: &[f64]) -> RunStats {
    let n = durations.len();
    if n == 0 {
        return RunStats::default();
    }

    let mean = durations.iter().sum::<f64>() / n as f64;
    let variance = if n > 1 {
        durations.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };
    let stddev = variance.sqrt();
    let min = durations.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = durations.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let mut sorted = durations.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };

    RunStats {
        mean,
        stddev,
        min,
        max,
        median,
        runs: durations.to_vec(),
        num_runs: n,
    }
}

/// Format duration showing mean ± stddev for table display.
///
/// # Examples
/// ```ignore
/// // With 3 runs: "  1.234s ± 0.05"
/// assert!(format_duration_with_stddev(1.234, 0.05).contains("±"));
/// ```
pub fn format_duration_with_stddev(mean_secs: f64, stddev_secs: f64) -> String {
    if mean_secs >= 10.0 {
        format!("{:>6.2}s±{:.2}", mean_secs, stddev_secs)
    } else if mean_secs >= 1.0 {
        format!("{:>6.3}s±{:.3}", mean_secs, stddev_secs)
    } else if mean_secs >= 0.001 {
        format!("{:>6.1}ms±{:.1}", mean_secs * 1000.0, stddev_secs * 1000.0)
    } else {
        format!("{:>5.0}μs±{:.0}", mean_secs * 1_000_000.0, stddev_secs * 1_000_000.0)
    }
}

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
    50_231,
    75_254,
    100_232,
    250_256,
    500_257,
    750_251,
    1_000_257,
    2_500_259,
    5_000_002,
    7_500_001,
    10_000_301,
    20_000_401,
    25_000_602,
    50_000_002,
    70_000_002,
    75_000_002,
    100_000_002,
    250_000_002,
    500_000_007,
    750_000_007,
    1_000_000_207,
    2_000_000_607,
    5_000_000_407,
    8_000_000_207,
    10_000_000_708,
    20_000_000_109,
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

// ============================================================================
// Number Formatting Utilities
// ============================================================================

/// Format a number with a thousand separators (commas) for human readability.
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

/// Format duration in seconds with appropriate precision for table display.
///
/// # Examples
/// ```ignore
/// assert_eq!(format_duration(15.5), "   15.50s");
/// assert_eq!(format_duration(1.234), "   1.234s");
/// assert_eq!(format_duration(0.0005), "   0.50ms");
/// ```
#[allow(dead_code)]
pub fn format_duration(secs: f64) -> String {
    if secs >= 10.0 {
        format!("{:>8.2}s", secs)
    } else if secs >= 1.0 {
        format!("{:>8.3}s", secs)
    } else if secs >= 0.001 {
        format!("{:>8.4}s", secs)
    } else {
        format!("{:>7.2}ms", secs * 1000.0)
    }
}

/// Format speedup ratio for table display.
///
/// # Examples
/// ```ignore
/// assert_eq!(format_speedup(5.25), "  5.25x");
/// ```
pub fn format_speedup(speedup: f64) -> String {
    format!("{:>6.2}x", speedup)
}

/// Benchmark type enum for organizing results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BenchmarkType {
    /// Unified Black-Scholes benchmark comparing CPU and GPU strategies
    BlackScholes,
    /// Monte Carlo option pricing benchmark
    MonteCarlo,
    /// GPU Reduce benchmark (Sum, Product, Min, Max)
    Reduce,
}

impl BenchmarkType {
    /// Get the folder name for this benchmark type.
    pub fn folder_name(&self) -> &'static str {
        match self {
            BenchmarkType::BlackScholes => "black_scholes",
            BenchmarkType::MonteCarlo => "monte_carlo",
            BenchmarkType::Reduce => "reduce",
        }
    }

    /// Get the file prefix for this benchmark type.
    pub fn file_prefix(&self) -> &'static str {
        match self {
            BenchmarkType::BlackScholes => "black_scholes_benchmark",
            BenchmarkType::MonteCarlo => "monte_carlo_benchmark",
            BenchmarkType::Reduce => "reduce_benchmark",
        }
    }

    /// Get the display name for this benchmark type (for chart titles).
    pub fn display_name(&self) -> &'static str {
        match self {
            BenchmarkType::BlackScholes => "Black-Scholes CPU vs GPU",
            BenchmarkType::MonteCarlo => "Monte Carlo CPU vs GPU",
            BenchmarkType::Reduce => "Reduce CPU vs GPU",
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
    let os = env::consts::OS;
    let arch = env::consts::ARCH;
    #[cfg(feature = "gpu-wgpu")]
    let backend = "WGPU";
    #[cfg(all(feature = "gpu-cuda", not(feature = "gpu-wgpu")))]
    let backend = "CUDA";
    #[cfg(not(any(feature = "gpu-wgpu", feature = "gpu-cuda")))]
    let backend = "CPU";
    format!("{} {} {}", os, arch, backend)
}

// ============================================================================
// System Configuration
// ============================================================================

/// System configuration information for benchmark reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    /// Operating system name and version
    pub os: String,
    /// CPU architecture (e.g., "aarch64", "x86_64")
    pub arch: String,
    /// Number of logical CPU cores
    pub cpu_cores: usize,
    /// Total system RAM in GB
    pub ram_gb: f64,
    /// GPU backend used (e.g., "WGPU", "CUDA", "None")
    pub gpu_backend: String,
    /// GPU device name (if available)
    pub gpu_device: Option<String>,
    /// GPU VRAM in GB (if available)
    pub gpu_vram_gb: Option<f64>,
    /// Vectorization factor used for GPU kernel
    pub vectorization_factor: usize,
    /// GPU batch size
    pub gpu_batch_size: usize,
    /// Estimated GPU thread count (workgroups × threads-per-workgroup)
    #[serde(default)]
    pub gpu_threads: Option<usize>,
}

impl SystemConfig {
    /// Create a new SystemConfig by detecting system information.
    pub fn detect(vectorization_factor: usize, gpu_batch_size: usize) -> Self {
        let os = env::consts::OS.to_string();
        let arch = env::consts::ARCH.to_string();
        
        // Get CPU core count
        let cpu_cores = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        
        // Estimate RAM (platform-specific)
        let ram_gb = get_system_memory_gb();
        
        // GPU backend
        #[cfg(feature = "gpu-wgpu")]
        let gpu_backend = "WGPU".to_string();
        #[cfg(all(feature = "gpu-cuda", not(feature = "gpu-wgpu")))]
        let gpu_backend = "CUDA".to_string();
        #[cfg(not(any(feature = "gpu-wgpu", feature = "gpu-cuda")))]
        let gpu_backend = "None".to_string();
        
        // GPU device info - query from GpuContext
        #[cfg(any(feature = "gpu-wgpu", feature = "gpu-cuda"))]
        let gpu_device = Some(renoir::operator::gpu::GpuContext::device_name());
        #[cfg(not(any(feature = "gpu-wgpu", feature = "gpu-cuda")))]
        let gpu_device = None;
        
        // VRAM detection not available in CubeCL yet
        let gpu_vram_gb = None;
        
        // GPU thread estimate: workgroups × threads-per-workgroup
        // Common CubeCL default: 256 workgroups × 64 threads = 16384
        let gpu_threads = Some(256 * 64);
        
        Self {
            os,
            arch,
            cpu_cores,
            ram_gb,
            gpu_backend,
            gpu_device,
            gpu_vram_gb,
            vectorization_factor,
            gpu_batch_size,
            gpu_threads,
        }
    }
}

/// Get system memory in GB (platform-specific implementation).
fn get_system_memory_gb() -> f64 {
    #[cfg(target_os = "macos")]
    {
        // Use sysctl on macOS
        use std::process::Command;
        let output = Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output();
        
        if let Ok(output) = output {
            if let Ok(mem_str) = String::from_utf8(output.stdout) {
                if let Ok(bytes) = mem_str.trim().parse::<u64>() {
                    return bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                }
            }
        }
        0.0
    }
    
    #[cfg(target_os = "linux")]
    {
        // Read from /proc/meminfo on Linux
        if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
            for line in contents.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<u64>() {
                            return kb as f64 / (1024.0 * 1024.0);
                        }
                    }
                }
            }
        }
        0.0
    }
    
    #[cfg(target_os = "windows")]
    {
        // Basic fallback for Windows
        0.0
    }
    
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        0.0
    }
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
    // Select the appropriate plotter script based on benchmark type
    let script_path = match benchmark_type {
        BenchmarkType::BlackScholes => "benches/tools/plot_black_scholes.py",
        BenchmarkType::MonteCarlo => "benches/tools/plot_monte_carlo.py",
        BenchmarkType::Reduce => "benches/tools/plot_reduce_benchmark.py",
    };
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

// ============================================================================
// Benchmark Result Structures
// ============================================================================

/// Result for a single test size (all strategies).
///
/// This is the unified result structure used by all GPU benchmarks.
/// Fields that are specific to certain benchmarks (e.g., compute-only timing)
/// are optional.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub test_id: usize,
    pub timestamp: String,
    pub items_count: usize,
    pub data_size_gb: f64,
    
    // Total timing results (in seconds) - mean across runs
    pub renoir_seq_total_time_s: f64,   // Renoir Sequential (1 worker) - mean
    pub renoir_par_total_time_s: f64,   // Renoir Parallel (multi-worker) - mean
    pub gpu_total_time_s: f64,          // GPU - mean
    
    // Multi-run statistics per strategy
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub renoir_seq_stats: Option<RunStats>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub renoir_par_stats: Option<RunStats>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_stats: Option<RunStats>,
    
    // Compute-only timing (excludes data generation overhead) - Optional
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_seq_compute_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_par_compute_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_compute_s: Option<f64>,
    
    // Speedup ratios (total time, based on means)
    pub speedup: f64,               // GPU vs CPU Sequential
    pub speedup_parallel: f64,      // GPU vs CPU Parallel
    
    // Speedup ratios (compute only) - Optional
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speedup_seq_compute: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speedup_par_compute: Option<f64>,
    
    // GFLOPS (based on mean times)
    pub renoir_seq_gflops: f64,     // Renoir Sequential
    pub renoir_par_gflops: f64,     // Renoir Parallel
    pub gpu_gflops: f64,            // GPU
    
    // Configuration
    pub cpu_workers: usize,
    pub batch_size: usize,
    #[serde(default = "default_num_runs")]
    pub num_runs: usize,
    
    // Validation
    pub validation_passed: bool,
    pub validation_max_error: f64,
    pub validation_avg_error: f64,
    pub validation_sample_size: usize,
}

/// Monte Carlo specific configuration for reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloConfig {
    pub num_paths: u32,
    pub time_steps: u32,
    pub flops_per_option: f64,
}

/// Complete benchmark report.
///
/// This is the unified report structure used by all GPU benchmarks.
/// Benchmark-specific config (e.g., Monte Carlo paths/steps) is optional.
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub benchmark_type: String,
    pub start_time: String,
    pub platform: String,
    pub system_config: SystemConfig,
    pub total_tests: usize,
    pub cpu_workers: usize,
    pub batch_size: usize,
    #[serde(default = "default_num_runs")]
    pub num_runs: usize,
    #[serde(default)]
    pub warmup_runs: usize,
    pub results: Vec<TestResult>,
    
    // Benchmark-specific configuration (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub monte_carlo_config: Option<MonteCarloConfig>,
}

fn default_num_runs() -> usize { 1 }

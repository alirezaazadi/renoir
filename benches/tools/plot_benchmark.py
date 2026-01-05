#!/usr/bin/env python3
"""
Unified GPU Benchmark Plotting Tool.

This script visualizes benchmark results from the Black-Scholes CPU vs GPU benchmark.
It compares four strategies: CPU Sequential, CPU Parallel, GPU Sequential, GPU Parallel.

Usage:
    # Plot by benchmark type (uses most recent file)
    python benches/tools/plot_benchmark.py black_scholes

    # Plot a specific results file
    python benches/tools/plot_benchmark.py benches/results/black_scholes/2024-12-04/black_scholes_benchmark_*.json

Output:
    Charts are saved alongside the JSON file as plot_black_scholes_benchmark_{timestamp}.png
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# Benchmark Type Constants
# ============================================================================

BENCHMARK_TYPES = ["black_scholes"]

BENCHMARK_TYPE_DISPLAY = {
    "black_scholes": "Black-Scholes CPU vs GPU",
}

BENCHMARK_FILE_PREFIXES = {
    "black_scholes": "black_scholes_benchmark",
}


# ============================================================================
# Common Utilities
# ============================================================================

def load_results(filename):
    """
    Load benchmark results from JSON file.
    
    Args:
        filename: Path to the JSON results file
        
    Returns:
        Tuple of (results_list, metadata_dict)
    """
    with open(filename, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and "results" in data:
        return data["results"], data
    return data, {"platform": "Unknown", "start_time": "Unknown"}


def get_benchmark_type(metadata: dict, filename: Path) -> str:
    """Determine benchmark type from metadata or filename."""
    # Try to get from metadata first
    if "benchmark_type" in metadata:
        return metadata["benchmark_type"]

    # Fall back to parsing filename
    name = filename.stem.lower()
    if "black_scholes" in name:
        return "black_scholes"
    
    return None


def format_number(n):
    """Format a number with appropriate suffix (K, M, B)."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def find_most_recent_file(benchmark_type: str) -> Path:
    """
    Find the most recent benchmark file for a given type.
    
    Args:
        benchmark_type: One of 'standard', 'optimized', 'streaming', 'comparison'
        
    Returns:
        Path to the most recent file, or None if not found
    """
    script_dir = Path(__file__).parent
    results_base = script_dir.parent / "results"
    
    search_dirs = [
        results_base / benchmark_type,
        Path("benches/results") / benchmark_type,
    ]
    
    prefix = BENCHMARK_FILE_PREFIXES.get(benchmark_type, f"{benchmark_type}_benchmark")
    
    candidates = []
    for search_dir in search_dirs:
        if search_dir.exists():
            candidates.extend(search_dir.rglob(f"*{prefix}*.json"))
    
    if not candidates:
        return None
    
    # Sort by modification time and return most recent
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
    return candidates[-1]


# ============================================================================
# Standard/Optimized/Streaming Benchmark Plots
# ============================================================================

def build_config_description(results, metadata):
    """
    Build a system configuration description string for the chart.
    """
    platform = metadata.get("platform", "Unknown")
    benchmark_type = metadata.get("benchmark_type", "unknown")
    is_streaming = metadata.get("is_streaming", False)
    system_config = metadata.get("system_config", {})
    
    items_counts = [r["items_count"] for r in results]
    min_items = min(items_counts) if items_counts else 0
    max_items = max(items_counts) if items_counts else 0
    
    cpu_workers = results[0].get("cpu_workers", 4) if results else 4
    
    # Extract system config values
    cpu_cores = system_config.get("cpu_cores", cpu_workers)
    ram_gb = system_config.get("ram_gb", 0)
    gpu_device = system_config.get("gpu_device", None)
    vectorization = system_config.get("vectorization_factor", 16)
    gpu_batch = system_config.get("gpu_batch_size", 10_000_000)
    
    lines = [
        f"Type: {BENCHMARK_TYPE_DISPLAY.get(benchmark_type, benchmark_type.title())}",
    ]
    
    # Add system details if available
    if ram_gb > 0:
        lines.append(f"CPU: {cpu_cores} cores, RAM: {ram_gb:.0f}GB")
    else:
        lines.append(f"CPU: {cpu_cores} cores")
    
    # Add GPU device info
    if gpu_device:
        lines.append(f"GPU: {gpu_device}")
    else:
        lines.append(f"GPU: {platform}")
    
    if is_streaming:
        streaming_config = metadata.get("streaming_config", {})
        lines.append("Mode: STREAMING SIMULATION (Lazy Iterator)")
        lines.append(f"Total Items: {format_number(streaming_config.get('total_items', max_items))}")
        lines.append(f"Adaptive Batch: {format_number(streaming_config.get('adaptive_min_batch', 100000))} - {format_number(streaming_config.get('adaptive_max_batch', 500000000))}")
    else:
        lines.append(f"Problem Sizes: {format_number(min_items)} - {format_number(max_items)}")
    
    lines.append(f"Vec: {vectorization}")
    lines.append(f"Batch: {format_number(gpu_batch)}")
    lines.append(f"Tests: {len(results)}")
    
    return "  |  ".join(lines)


def plot_standard_results(results, metadata, source_name: str, output_dir: Path, benchmark_type: str):
    """
    Generate comprehensive benchmark visualization for standard/optimized/streaming benchmarks.
    
    Creates a 2x3 grid of charts showing:
    1. Execution time comparison (log-log)
    2. Speedup vs problem size
    3. GFLOPS comparison
    4. Throughput (options/sec)
    5. Average speedup by size range
    6. Speedup distribution histogram
    """
    type_display = BENCHMARK_TYPE_DISPLAY.get(benchmark_type, benchmark_type.title())

    items_counts = [r["items_count"] for r in results]
    data_sizes = [r["data_size_gb"] for r in results]
    
    # Support multiple field naming conventions (newest to oldest)
    # Newest: renoir_seq_total_time_s, renoir_par_total_time_s, gpu_total_time_s
    # Middle: cpu_total_time_s, renoir_total_time_s, gpu_total_time_s
    # Old: cpu_seq_time_s, cpu_par_time_s, gpu_double_buf_time_s
    def get_field(r, *names, default=0):
        for name in names:
            if name in r:
                return r[name]
        return default
    
    cpu_times = [get_field(r, "renoir_seq_total_time_s", "cpu_total_time_s", "cpu_seq_time_s") for r in results]
    gpu_times = [get_field(r, "gpu_total_time_s", "gpu_double_buf_time_s") for r in results]
    renoir_times = [get_field(r, "renoir_par_total_time_s", "renoir_total_time_s", "cpu_par_time_s") for r in results]
    
    # Calculate speedups if not provided
    speedups = []
    for r, cpu_t, gpu_t in zip(results, cpu_times, gpu_times):
        if "speedup" in r:
            speedups.append(r["speedup"])
        elif "gpu_double_buf_vs_cpu_seq" in r:
            speedups.append(r["gpu_double_buf_vs_cpu_seq"])
        else:
            speedups.append(cpu_t / gpu_t if gpu_t > 0 else 0)
    
    speedups_parallel = []
    for r, renoir_t, gpu_t in zip(results, renoir_times, gpu_times):
        if "speedup_parallel" in r:
            speedups_parallel.append(r["speedup_parallel"])
        elif "gpu_double_buf_vs_cpu_par" in r:
            speedups_parallel.append(r["gpu_double_buf_vs_cpu_par"])
        else:
            speedups_parallel.append(renoir_t / gpu_t if gpu_t > 0 else 0)
    
    cpu_gflops = [get_field(r, "renoir_seq_gflops", "cpu_gflops", "cpu_seq_gflops") for r in results]
    gpu_gflops = [get_field(r, "gpu_gflops", "gpu_double_buf_gflops") for r in results]
    renoir_gflops = [get_field(r, "renoir_par_gflops", "renoir_gflops", "cpu_par_gflops") for r in results]
    validation_passed = [r["validation_passed"] for r in results]
    
    # Optional: GPU Simple (non-double-buffered) for comparison
    has_gpu_simple = "gpu_simple_time_s" in results[0] if results else False
    if has_gpu_simple:
        gpu_simple_times = [r["gpu_simple_time_s"] for r in results]
        gpu_simple_gflops = [r.get("gpu_simple_gflops", 0) for r in results]
        db_gains = [r.get("double_buf_vs_simple", 1.0) for r in results]

    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"Black-Scholes {type_display} Benchmark: GPU vs CPU\n{metadata.get('platform', 'Unknown Platform')}",
        fontsize=14,
        fontweight="bold",
    )

    # Chart 1: Execution Time Comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.loglog(items_counts, cpu_times, "o-", label="CPU Sequential", color="tab:blue", markersize=5, alpha=0.7)
    ax1.loglog(items_counts, renoir_times, "s-", label="CPU Parallel", color="tab:green", markersize=5, alpha=0.7)
    ax1.loglog(items_counts, gpu_times, "^-", label="GPU", color="tab:red", markersize=5, alpha=0.7)
    if has_gpu_simple:
        ax1.loglog(items_counts, gpu_simple_times, "d-", label="GPU Simple", color="tab:orange", markersize=5, alpha=0.7)
    ax1.set_xlabel("Number of Options")
    ax1.set_ylabel("Execution Time (s)")
    ax1.legend(loc="upper left", fontsize=7)
    ax1.set_title("Execution Time vs Problem Size", fontweight="bold")
    ax1.grid(True, alpha=0.3, which="both")

    # Chart 2: Speedup vs Problem Size
    ax2 = plt.subplot(2, 3, 2)
    ax2.semilogx(items_counts, speedups, "o-", label="Seq/GPU", color="tab:blue", markersize=6, alpha=0.8)
    ax2.semilogx(items_counts, speedups_parallel, "s-", label="Par/GPU", color="tab:green", markersize=6, alpha=0.8)
    if has_gpu_simple:
        ax2.semilogx(items_counts, db_gains, "d-", label="DB vs Simple (Gain)", color="tab:purple", markersize=6, alpha=0.8)
    ax2.axhline(y=1.0, color="black", linestyle="--", linewidth=2, label="Break-even")
    ax2.set_xlabel("Number of Options")
    ax2.set_ylabel("Speedup Ratio")
    ax2.set_title("GPU Speedup vs Problem Size", fontweight="bold")
    ax2.legend(loc="best", fontsize=7)
    ax2.grid(True, alpha=0.3)
    ax2.axhspan(0, 1.0, alpha=0.1, color="red", label="_nolegend_")
    ax2.axhspan(1.0, ax2.get_ylim()[1] if ax2.get_ylim()[1] > 1 else 2, alpha=0.1, color="green", label="_nolegend_")

    # Chart 3: GFLOPS Comparison
    ax3 = plt.subplot(2, 3, 3)
    ax3.semilogx(items_counts, cpu_gflops, "o-", label="CPU Sequential", color="tab:blue", markersize=5, alpha=0.7)
    ax3.semilogx(items_counts, renoir_gflops, "s-", label="CPU Parallel", color="tab:green", markersize=5, alpha=0.7)
    if has_gpu_simple:
        ax3.semilogx(items_counts, gpu_simple_gflops, "d-", label="GPU Simple", color="tab:orange", markersize=5, alpha=0.7)
    ax3.semilogx(items_counts, gpu_gflops, "^-", label="GPU", color="tab:red", markersize=5, alpha=0.7)
    ax3.set_xlabel("Number of Options")
    ax3.set_ylabel("GFLOPS")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.set_title("Computational Throughput", fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # Chart 4: Throughput (Options/sec)
    ax4 = plt.subplot(2, 3, 4)
    cpu_throughput = [n / t if t > 0 else 0 for n, t in zip(items_counts, cpu_times)]
    gpu_throughput = [n / t if t > 0 else 0 for n, t in zip(items_counts, gpu_times)]
    par_throughput = [n / t if t > 0 else 0 for n, t in zip(items_counts, renoir_times)]
    if has_gpu_simple:
        gpu_simple_throughput = [n / t if t > 0 else 0 for n, t in zip(items_counts, gpu_simple_times)]

    ax4.loglog(items_counts, cpu_throughput, "o-", label="CPU Sequential", color="tab:blue", markersize=5, alpha=0.7)
    ax4.loglog(items_counts, par_throughput, "s-", label="CPU Parallel", color="tab:green", markersize=5, alpha=0.7)
    if has_gpu_simple:
        ax4.loglog(items_counts, gpu_simple_throughput, "d-", label="GPU Simple", color="tab:orange", markersize=5, alpha=0.7)
    ax4.loglog(items_counts, gpu_throughput, "^-", label="GPU", color="tab:red", markersize=5, alpha=0.7)
    ax4.set_xlabel("Number of Options")
    ax4.set_ylabel("Options/second")
    ax4.legend(loc="upper left", fontsize=8)
    ax4.set_title("Pricing Throughput", fontweight="bold")
    ax4.grid(True, alpha=0.3, which="both")

    # Chart 5: Speedup by Size Range
    ax5 = plt.subplot(2, 3, 5)
    ranges = [
        (0, 10_000, "Tiny\n(<10K)"),
        (10_000, 100_000, "Small\n(10K-100K)"),
        (100_000, 1_000_000, "Medium\n(100K-1M)"),
        (1_000_000, 10_000_000, "Large\n(1M-10M)"),
        (10_000_000, 1_000_000_000, "Huge\n(>10M)"),
    ]
    labels, avg_speedups_seq, avg_speedups_par = [], [], []
    for low, high, label in ranges:
        mask = [i for i, count in enumerate(items_counts) if low <= count < high]
        if mask:
            labels.append(label)
            avg_speedups_seq.append(float(np.mean([speedups[i] for i in mask])))
            avg_speedups_par.append(float(np.mean([speedups_parallel[i] for i in mask])))

    x = np.arange(len(labels))
    width = 0.35
    
    colors_seq = ["#ff6b6b" if s < 1 else "#51cf66" for s in avg_speedups_seq]
    colors_par = ["#ffb3b3" if s < 1 else "#a3e4a3" for s in avg_speedups_par]
    
    bars1 = ax5.bar(x - width/2, avg_speedups_seq, width, label="vs CPU Seq", color=colors_seq, alpha=0.9, edgecolor="black")
    bars2 = ax5.bar(x + width/2, avg_speedups_par, width, label="vs CPU Par", color=colors_par, alpha=0.9, edgecolor="black")
    
    ax5.axhline(y=1.0, color="black", linestyle="--", linewidth=2)
    ax5.set_ylabel("Average Speedup (CPU / GPU)")
    ax5.set_title("Average Speedup by Problem Size", fontweight="bold")
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels)
    ax5.legend(loc="upper left", fontsize=8)
    ax5.grid(True, alpha=0.3, axis="y")

    for bar, value in zip(bars1, avg_speedups_seq):
        ax5.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.1f}x",
                 ha="center", va="bottom", fontweight="bold", fontsize=7)
    for bar, value in zip(bars2, avg_speedups_par):
        ax5.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.1f}x",
                 ha="center", va="bottom", fontweight="bold", fontsize=7)

    # Chart 6: Speedup Distribution
    ax6 = plt.subplot(2, 3, 6)
    all_speedups = speedups + speedups_parallel
    bins = np.linspace(min(all_speedups) * 0.9, max(all_speedups) * 1.1, 20)
    
    ax6.hist(speedups, bins=bins, alpha=0.6, edgecolor="black", color="tab:blue", label=f"vs CPU Seq (mean={np.mean(speedups):.2f}x)")
    ax6.hist(speedups_parallel, bins=bins, alpha=0.6, edgecolor="black", color="tab:green", label=f"vs CPU Par (mean={np.mean(speedups_parallel):.2f}x)")

    ax6.axvline(x=1.0, color="black", linestyle="--", linewidth=2, label="Break-even")
    ax6.set_xlabel("Speedup (CPU / GPU)")
    ax6.set_ylabel("Frequency")
    ax6.legend(fontsize=8)
    ax6.set_title("Speedup Distribution", fontweight="bold")
    ax6.grid(True, alpha=0.3, axis="y")

    # Add System Configuration Description Box
    config_text = build_config_description(results, metadata)
    fig.text(
        0.5, 0.01, config_text,
        ha="center", va="bottom",
        fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8, edgecolor="gray")
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])

    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    if "_benchmark_" in source_name:
        timestamp_part = source_name.split("_benchmark_")[-1]
        output_file = output_dir / f"plot_{benchmark_type}_benchmark_{timestamp_part}.png"
    else:
        safe_name = source_name.replace(" ", "_")
        output_file = output_dir / f"plot_{benchmark_type}_benchmark_{safe_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nChart saved to: {output_file}")

    # Print Summary Statistics
    print_standard_summary(results, metadata, speedups, speedups_parallel, 
                          items_counts, data_sizes, cpu_gflops, gpu_gflops, 
                          renoir_gflops, validation_passed, benchmark_type)


def print_standard_summary(results, metadata, speedups, speedups_parallel,
                          items_counts, data_sizes, cpu_gflops, gpu_gflops,
                          renoir_gflops, validation_passed, benchmark_type):
    """Print summary statistics for standard/optimized/streaming benchmarks."""
    type_display = BENCHMARK_TYPE_DISPLAY.get(benchmark_type, benchmark_type.title())
    
    print("\n" + "=" * 70)
    print(f"          {type_display.upper()} BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Benchmark Type: {type_display}")
    print(f"Platform: {metadata.get('platform', 'Unknown')}")
    print(f"Start time: {metadata.get('start_time', 'Unknown')}")
    print(f"Total benchmarks: {len(results)}")

    if items_counts:
        print(f"\nProblem size range: {min(items_counts):,} - {max(items_counts):,} options")
        print(f"Data size range: {min(data_sizes):.6f} - {max(data_sizes):.4f} GB")

    print(f"\nSpeedup Statistics (CPU Sequential / GPU):")
    print(f"  Min:    {min(speedups):.3f}x")
    print(f"  Max:    {max(speedups):.3f}x")
    print(f"  Mean:   {np.mean(speedups):.3f}x")
    print(f"  Median: {np.median(speedups):.3f}x")

    print(f"\nSpeedup Statistics (CPU Parallel / GPU):")
    print(f"  Min:    {min(speedups_parallel):.3f}x")
    print(f"  Max:    {max(speedups_parallel):.3f}x")
    print(f"  Mean:   {np.mean(speedups_parallel):.3f}x")
    print(f"  Median: {np.median(speedups_parallel):.3f}x")

    gpu_wins_seq = sum(1 for s in speedups if s > 1.0)
    gpu_wins_par = sum(1 for s in speedups_parallel if s > 1.0)
    print(f"\nWin/Loss (vs CPU Sequential):")
    print(f"  GPU wins: {gpu_wins_seq} ({100 * gpu_wins_seq / len(speedups):.1f}%)")
    print(f"  CPU wins: {len(speedups) - gpu_wins_seq} ({100 * (len(speedups) - gpu_wins_seq) / len(speedups):.1f}%)")
    print(f"\nWin/Loss (vs CPU Parallel):")
    print(f"  GPU wins: {gpu_wins_par} ({100 * gpu_wins_par / len(speedups_parallel):.1f}%)")
    print(f"  CPU wins: {len(speedups_parallel) - gpu_wins_par} ({100 * (len(speedups_parallel) - gpu_wins_par) / len(speedups_parallel):.1f}%)")

    validation_failures = len([v for v in validation_passed if not v])
    print(f"  Validation failures: {validation_failures}")

    crossover_idx = None
    for i, s in enumerate(speedups):
        if s > 1.0:
            crossover_idx = i
            break

    if crossover_idx is not None:
        print(f"\nGPU becomes faster at: ~{items_counts[crossover_idx]:,} options")

    best_idx = int(np.argmax(speedups))
    
    # Support multiple field naming conventions
    def get_field(r, *names):
        for name in names:
            if name in r:
                return r[name]
        return 0
    
    cpu_times = [get_field(r, "renoir_seq_total_time_s", "cpu_total_time_s", "cpu_seq_time_s") for r in results]
    gpu_times = [get_field(r, "gpu_total_time_s", "gpu_double_buf_time_s") for r in results]
    print(f"\nBest GPU performance:")
    print(f"  Speedup: {speedups[best_idx]:.3f}x")
    print(f"  Options: {items_counts[best_idx]:,}")
    print(f"  CPU time: {cpu_times[best_idx]:.4f}s")
    print(f"  GPU time: {gpu_times[best_idx]:.4f}s")
    
    # Double-buffer gain statistics if available
    if results and "double_buf_vs_simple" in results[0]:
        db_gains = [r.get("double_buf_vs_simple", 1.0) for r in results]
        print(f"\nDouble-Buffer Gain (vs Simple GPU):")
        print(f"  Min:  {min(db_gains):.2f}x")
        print(f"  Max:  {max(db_gains):.2f}x")
        print(f"  Mean: {np.mean(db_gains):.2f}x")

    print(f"\nPeak GFLOPS:")
    print(f"  Renoir CPU Sequential: {max(cpu_gflops):.2f}")
    print(f"  Renoir CPU Parallel:   {max(renoir_gflops):.2f}")
    print(f"  Renoir GPU:            {max(gpu_gflops):.2f}")
    print("=" * 70)


# ============================================================================
# Comparison (Batching) Benchmark Plots
# ============================================================================

def organize_by_strategy(results):
    """Organize results by strategy name."""
    by_strategy = defaultdict(list)
    for r in results:
        by_strategy[r["strategy_name"]].append(r)
    return by_strategy


def organize_by_size(results):
    """Organize results by input size."""
    by_size = defaultdict(list)
    for r in results:
        by_size[r["items_count"]].append(r)
    return by_size


def plot_comparison_results(results, metadata, source_name: str, output_dir: Path):
    """
    Generate comprehensive batching strategy comparison visualization.
    
    Creates a 2x3 grid of charts showing:
    1. Throughput comparison by strategy (grouped bar)
    2. Throughput vs input size (line chart)
    3. Time per item by strategy
    4. Best strategy heatmap
    5. Strategy ranking by size
    6. Throughput distribution
    """
    by_strategy = organize_by_strategy(results)
    by_size = organize_by_size(results)
    
    strategies = sorted(by_strategy.keys())
    sizes = sorted(by_size.keys())
    
    throughput_matrix = {}
    for strategy in strategies:
        throughput_matrix[strategy] = {}
        for r in by_strategy[strategy]:
            throughput_matrix[strategy][r["items_count"]] = r["throughput"]
    
    strategy_colors = {
        "Fixed(100K)": "#3498db",
        "Fixed(1M)": "#2980b9",
        "Fixed(10M)": "#1a5276",
        "Fixed(Full)": "#27ae60",
        "Timed(1M,100ms)": "#e74c3c",
        "Adaptive(100K-10M)": "#9b59b6",
    }
    
    default_colors = plt.cm.tab10.colors
    for i, s in enumerate(strategies):
        if s not in strategy_colors:
            strategy_colors[s] = default_colors[i % len(default_colors)]
    
    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Black-Scholes Batching Comparison Benchmark\n{metadata.get('platform', 'Unknown Platform')}",
        fontsize=14,
        fontweight="bold",
    )

    # Chart 1: Throughput by Strategy (largest size)
    ax1 = plt.subplot(2, 3, 1)
    
    largest_size = max(sizes)
    largest_results = by_size[largest_size]
    strat_names = [r["strategy_name"] for r in largest_results]
    throughputs = [r["throughput"] / 1_000_000 for r in largest_results]
    
    colors = [strategy_colors.get(s, "gray") for s in strat_names]
    bars = ax1.bar(range(len(strat_names)), throughputs, color=colors, edgecolor="black", alpha=0.8)
    
    ax1.set_ylabel("Throughput (M items/sec)")
    ax1.set_title(f"Throughput by Strategy\n({largest_size:,} items)", fontweight="bold")
    ax1.set_xticks(range(len(strat_names)))
    ax1.set_xticklabels([s.replace("(", "\n(") for s in strat_names], rotation=0, fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")
    
    for bar, val in zip(bars, throughputs):
        ax1.text(bar.get_x() + bar.get_width()/2, val, f"{val:.1f}",
                ha="center", va="bottom", fontweight="bold", fontsize=8)
    
    best_idx = np.argmax(throughputs)
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(3)

    # Chart 2: Throughput vs Input Size
    ax2 = plt.subplot(2, 3, 2)
    
    for strategy in strategies:
        strat_sizes = sorted(throughput_matrix[strategy].keys())
        strat_throughputs = [throughput_matrix[strategy][s] / 1_000_000 for s in strat_sizes]
        ax2.semilogx(strat_sizes, strat_throughputs, "o-", 
                    label=strategy, color=strategy_colors.get(strategy, "gray"),
                    markersize=6, alpha=0.8)
    
    ax2.set_xlabel("Number of Items")
    ax2.set_ylabel("Throughput (M items/sec)")
    ax2.set_title("Throughput vs Input Size", fontweight="bold")
    ax2.legend(loc="best", fontsize=7)
    ax2.grid(True, alpha=0.3)

    # Chart 3: Execution Time by Strategy
    ax3 = plt.subplot(2, 3, 3)
    
    display_sizes = [s for s in sizes if s in [100_000, 1_000_000, 10_000_000]]
    if not display_sizes:
        display_sizes = sizes[:3] if len(sizes) >= 3 else sizes
    
    x = np.arange(len(display_sizes))
    width = 0.12
    
    for i, strategy in enumerate(strategies):
        times = []
        for size in display_sizes:
            if size in throughput_matrix[strategy]:
                for r in by_strategy[strategy]:
                    if r["items_count"] == size:
                        times.append(r["total_time_s"])
                        break
            else:
                times.append(0)
        
        ax3.bar(x + i * width, times, width, label=strategy,
               color=strategy_colors.get(strategy, "gray"), alpha=0.8, edgecolor="black")
    
    ax3.set_ylabel("Execution Time (s)")
    ax3.set_title("Execution Time by Strategy", fontweight="bold")
    ax3.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax3.set_xticklabels([f"{s/1_000_000:.1f}M" if s >= 1_000_000 else f"{s/1_000:.0f}K" for s in display_sizes])
    ax3.set_xlabel("Input Size")
    ax3.legend(loc="upper left", fontsize=6)
    ax3.grid(True, alpha=0.3, axis="y")

    # Chart 4: Best Strategy Heatmap
    ax4 = plt.subplot(2, 3, 4)
    
    rank_matrix = np.zeros((len(strategies), len(sizes)))
    for j, size in enumerate(sizes):
        size_results = by_size[size]
        sorted_results = sorted(size_results, key=lambda r: r["throughput"], reverse=True)
        for rank, r in enumerate(sorted_results):
            i = strategies.index(r["strategy_name"])
            rank_matrix[i, j] = rank + 1
    
    im = ax4.imshow(rank_matrix, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=len(strategies))
    
    ax4.set_xticks(range(len(sizes)))
    ax4.set_xticklabels([f"{s/1_000_000:.1f}M" if s >= 1_000_000 else f"{s/1_000:.0f}K" for s in sizes],
                       rotation=45, ha="right", fontsize=8)
    ax4.set_yticks(range(len(strategies)))
    ax4.set_yticklabels([s.replace("(", "\n(") for s in strategies], fontsize=8)
    ax4.set_xlabel("Input Size")
    ax4.set_title("Strategy Ranking by Input Size\n(1=Best, Green; Higher=Worse, Red)", fontweight="bold")
    
    for i in range(len(strategies)):
        for j in range(len(sizes)):
            ax4.text(j, i, f"{int(rank_matrix[i, j])}", ha="center", va="center", 
                    fontsize=9, fontweight="bold",
                    color="white" if rank_matrix[i, j] <= 2 else "black")
    
    plt.colorbar(im, ax=ax4, label="Rank", shrink=0.8)

    # Chart 5: Average Throughput by Strategy
    ax5 = plt.subplot(2, 3, 5)
    
    avg_throughputs = []
    for strategy in strategies:
        all_throughputs = [r["throughput"] for r in by_strategy[strategy]]
        avg_throughputs.append(np.mean(all_throughputs) / 1_000_000)
    
    colors = [strategy_colors.get(s, "gray") for s in strategies]
    bars = ax5.barh(range(len(strategies)), avg_throughputs, color=colors, edgecolor="black", alpha=0.8)
    
    ax5.set_xlabel("Average Throughput (M items/sec)")
    ax5.set_title("Average Throughput by Strategy\n(Across All Input Sizes)", fontweight="bold")
    ax5.set_yticks(range(len(strategies)))
    ax5.set_yticklabels([s.replace("(", "\n(") for s in strategies], fontsize=8)
    ax5.grid(True, alpha=0.3, axis="x")
    
    for bar, val in zip(bars, avg_throughputs):
        ax5.text(val, bar.get_y() + bar.get_height()/2, f" {val:.2f}",
                va="center", fontweight="bold", fontsize=9)
    
    best_idx = np.argmax(avg_throughputs)
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(3)

    # Chart 6: Throughput Distribution
    ax6 = plt.subplot(2, 3, 6)
    
    throughput_data = []
    for strategy in strategies:
        throughput_data.append([r["throughput"] / 1_000_000 for r in by_strategy[strategy]])
    
    bp = ax6.boxplot(throughput_data, patch_artist=True)
    
    for i, (patch, strategy) in enumerate(zip(bp["boxes"], strategies)):
        patch.set_facecolor(strategy_colors.get(strategy, "gray"))
        patch.set_alpha(0.8)
    
    ax6.set_ylabel("Throughput (M items/sec)")
    ax6.set_title("Throughput Distribution by Strategy", fontweight="bold")
    ax6.set_xticklabels([s.replace("(", "\n(") for s in strategies], rotation=0, fontsize=7)
    ax6.grid(True, alpha=0.3, axis="y")

    # Add config description
    config_text = f"Type: Batching Comparison  |  System: {metadata.get('platform', 'Unknown')}  |  Strategies: {len(strategies)}  |  Input Sizes: {len(sizes)}  |  Tests: {len(results)}"
    fig.text(
        0.5, 0.01, config_text,
        ha="center", va="bottom",
        fontsize=8,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8, edgecolor="gray")
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])

    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    if "_benchmark_" in source_name:
        timestamp_part = source_name.split("_benchmark_")[-1]
        output_file = output_dir / f"plot_comparison_benchmark_{timestamp_part}.png"
    else:
        safe_name = source_name.replace(" ", "_")
        output_file = output_dir / f"plot_comparison_benchmark_{safe_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nChart saved to: {output_file}")

    # Print Summary
    print_comparison_summary(results, metadata, by_strategy, by_size, strategies, sizes)


def print_comparison_summary(results, metadata, by_strategy, by_size, strategies, sizes):
    """Print summary statistics for comparison benchmarks."""
    print("\n" + "=" * 70)
    print("            BATCHING COMPARISON BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Benchmark Type: Batching Comparison")
    print(f"Platform: {metadata.get('platform', 'Unknown')}")
    print(f"Start time: {metadata.get('start_time', 'Unknown')}")
    print(f"Total benchmarks: {len(results)}")
    print(f"Strategies tested: {len(strategies)}")
    print(f"Input sizes tested: {len(sizes)}")
    
    print(f"\nInput size range: {min(sizes):,} - {max(sizes):,} items")
    
    print("\n" + "-" * 70)
    print("STRATEGY PERFORMANCE SUMMARY")
    print("-" * 70)
    print(f"{'Strategy':<25} {'Avg Throughput':>15} {'Best Size':>12} {'Best Tput':>12}")
    print("-" * 70)
    
    for strategy in strategies:
        strat_results = by_strategy[strategy]
        avg_tput = np.mean([r["throughput"] for r in strat_results]) / 1_000_000
        best_result = max(strat_results, key=lambda r: r["throughput"])
        best_size = best_result["items_count"]
        best_tput = best_result["throughput"] / 1_000_000
        
        print(f"{strategy:<25} {avg_tput:>12.2f} M/s {best_size:>10,} {best_tput:>10.2f} M/s")
    
    print("\n" + "-" * 70)
    print("BEST STRATEGY BY INPUT SIZE")
    print("-" * 70)
    
    for size in sizes:
        size_results = by_size[size]
        best = max(size_results, key=lambda r: r["throughput"])
        size_str = f"{size/1_000_000:.1f}M" if size >= 1_000_000 else f"{size/1_000:.0f}K"
        print(f"{size_str:>10} items: {best['strategy_name']:<25} ({best['throughput']/1_000_000:.2f} M/s)")
    
    all_avg = [(s, np.mean([r["throughput"] for r in by_strategy[s]])) for s in strategies]
    overall_best = max(all_avg, key=lambda x: x[1])
    print(f"\n{'='*70}")
    print(f"OVERALL BEST STRATEGY: {overall_best[0]}")
    print(f"Average throughput: {overall_best[1]/1_000_000:.2f} M items/sec")
    print("=" * 70)
    
    print("\nRECOMMENDATIONS:")
    print("-" * 70)
    
    small_sizes = [s for s in sizes if s <= 1_000_000]
    large_sizes = [s for s in sizes if s > 1_000_000]
    
    if small_sizes:
        small_best = {}
        for strategy in strategies:
            small_tputs = [r["throughput"] for r in by_strategy[strategy] if r["items_count"] in small_sizes]
            if small_tputs:
                small_best[strategy] = np.mean(small_tputs)
        if small_best:
            best_small = max(small_best.items(), key=lambda x: x[1])
            print(f"  For small inputs (<= 1M): {best_small[0]}")
    
    if large_sizes:
        large_best = {}
        for strategy in strategies:
            large_tputs = [r["throughput"] for r in by_strategy[strategy] if r["items_count"] in large_sizes]
            if large_tputs:
                large_best[strategy] = np.mean(large_tputs)
        if large_best:
            best_large = max(large_best.items(), key=lambda x: x[1])
            print(f"  For large inputs (> 1M):  {best_large[0]}")
    
    print(f"  For unknown input sizes:  Adaptive strategy")
    print("=" * 70)


# ============================================================================
# Streaming Benchmark Plots (4 strategies)
# ============================================================================

def plot_streaming_results(results, metadata, source_name: str, output_dir: Path):
    """
    Generate streaming benchmark visualization with all 4 strategies.
    
    Shows: CPU Sequential, GPU Sequential (Adaptive), CPU Parallel, GPU Parallel (Adaptive)
    
    Charts:
    1. Execution Time vs Problem Size (log-log)
    2. GPU Speedup vs Problem Size
    3. GFLOPS Comparison
    4. Throughput (Options/sec)
    5. Average Speedup by Size Range (bar chart)
    6. Speedup Distribution (histogram)
    """
    # Check if this is the new streaming format with 4 strategies
    if not results or "cpu_seq_time_s" not in results[0]:
        print("Warning: Streaming results not in expected format, falling back to standard plot")
        return False
    
    items_counts = [r["items_count"] for r in results]
    cpu_seq_times = [r["cpu_seq_time_s"] for r in results]
    gpu_seq_times = [r["gpu_seq_time_s"] for r in results]
    cpu_par_times = [r["cpu_par_time_s"] for r in results]
    gpu_par_times = [r["gpu_par_time_s"] for r in results]
    
    gpu_seq_speedups = [r["gpu_seq_vs_cpu_seq"] for r in results]
    gpu_par_speedups = [r["gpu_par_vs_cpu_par"] for r in results]
    
    cpu_seq_gflops = [r["cpu_seq_gflops"] for r in results]
    gpu_seq_gflops = [r["gpu_seq_gflops"] for r in results]
    cpu_par_gflops = [r["cpu_par_gflops"] for r in results]
    gpu_par_gflops = [r["gpu_par_gflops"] for r in results]
    
    platform = metadata.get("platform", "Unknown")
    cpu_workers = metadata.get("cpu_workers", 8)
    adaptive_min = metadata.get("adaptive_min_batch", 1_000_000)
    adaptive_max = metadata.get("adaptive_max_batch", 50_000_000)
    
    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Black-Scholes Streaming Benchmark: 4-Strategy Comparison (Adaptive Batching)\n"
        f"{platform} | {cpu_workers} CPU Workers | Adaptive: {format_number(adaptive_min)} - {format_number(adaptive_max)}",
        fontsize=14,
        fontweight="bold",
    )
    
    colors = {
        'cpu_seq': '#3498db',  # Blue
        'gpu_seq': '#e74c3c',  # Red
        'cpu_par': '#27ae60',  # Green
        'gpu_par': '#9b59b6',  # Purple
    }
    
    # Chart 1: Execution Time Comparison (log-log)
    ax1 = plt.subplot(2, 3, 1)
    ax1.loglog(items_counts, cpu_seq_times, "o-", label="CPU Sequential", color=colors['cpu_seq'], markersize=5, alpha=0.8)
    ax1.loglog(items_counts, gpu_seq_times, "s-", label="GPU Seq+Adaptive", color=colors['gpu_seq'], markersize=5, alpha=0.8)
    ax1.loglog(items_counts, cpu_par_times, "^-", label="CPU Parallel", color=colors['cpu_par'], markersize=5, alpha=0.8)
    ax1.loglog(items_counts, gpu_par_times, "D-", label="GPU Par+Adaptive", color=colors['gpu_par'], markersize=5, alpha=0.8)
    ax1.set_xlabel("Number of Options")
    ax1.set_ylabel("Execution Time (s)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title("Execution Time vs Problem Size", fontweight="bold")
    ax1.grid(True, alpha=0.3, which="both")
    
    # Chart 2: GPU Speedup vs Problem Size
    ax2 = plt.subplot(2, 3, 2)
    ax2.semilogx(items_counts, gpu_seq_speedups, "s-", label="GPU Seq vs CPU Seq", color=colors['gpu_seq'], markersize=6, alpha=0.8)
    ax2.semilogx(items_counts, gpu_par_speedups, "D-", label="GPU Par vs CPU Par", color=colors['gpu_par'], markersize=6, alpha=0.8)
    ax2.axhline(y=1.0, color="black", linestyle="--", linewidth=2, label="Break-even")
    ax2.set_xlabel("Number of Options")
    ax2.set_ylabel("GPU Speedup (CPU Time / GPU Time)")
    ax2.set_title("GPU Speedup vs Problem Size", fontweight="bold")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)
    # Add shading for GPU win/lose regions
    ylim = ax2.get_ylim()
    ax2.axhspan(0, 1.0, alpha=0.1, color="red")
    ax2.axhspan(1.0, max(ylim[1], 2), alpha=0.1, color="green")
    
    # Chart 3: GFLOPS Comparison
    ax3 = plt.subplot(2, 3, 3)
    ax3.semilogx(items_counts, cpu_seq_gflops, "o-", label="CPU Sequential", color=colors['cpu_seq'], markersize=5, alpha=0.8)
    ax3.semilogx(items_counts, gpu_seq_gflops, "s-", label="GPU Seq+Adaptive", color=colors['gpu_seq'], markersize=5, alpha=0.8)
    ax3.semilogx(items_counts, cpu_par_gflops, "^-", label="CPU Parallel", color=colors['cpu_par'], markersize=5, alpha=0.8)
    ax3.semilogx(items_counts, gpu_par_gflops, "D-", label="GPU Par+Adaptive", color=colors['gpu_par'], markersize=5, alpha=0.8)
    ax3.set_xlabel("Number of Options")
    ax3.set_ylabel("GFLOPS")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.set_title("Computational Throughput (GFLOPS)", fontweight="bold")
    ax3.grid(True, alpha=0.3)
    
    # Chart 4: Throughput (Options/sec)
    ax4 = plt.subplot(2, 3, 4)
    cpu_seq_throughput = [n / t if t > 0 else 0 for n, t in zip(items_counts, cpu_seq_times)]
    gpu_seq_throughput = [n / t if t > 0 else 0 for n, t in zip(items_counts, gpu_seq_times)]
    cpu_par_throughput = [n / t if t > 0 else 0 for n, t in zip(items_counts, cpu_par_times)]
    gpu_par_throughput = [n / t if t > 0 else 0 for n, t in zip(items_counts, gpu_par_times)]
    
    ax4.loglog(items_counts, cpu_seq_throughput, "o-", label="CPU Sequential", color=colors['cpu_seq'], markersize=5, alpha=0.8)
    ax4.loglog(items_counts, gpu_seq_throughput, "s-", label="GPU Seq+Adaptive", color=colors['gpu_seq'], markersize=5, alpha=0.8)
    ax4.loglog(items_counts, cpu_par_throughput, "^-", label="CPU Parallel", color=colors['cpu_par'], markersize=5, alpha=0.8)
    ax4.loglog(items_counts, gpu_par_throughput, "D-", label="GPU Par+Adaptive", color=colors['gpu_par'], markersize=5, alpha=0.8)
    ax4.set_xlabel("Number of Options")
    ax4.set_ylabel("Throughput (Options/s)")
    ax4.legend(loc="upper left", fontsize=8)
    ax4.set_title("Processing Throughput", fontweight="bold")
    ax4.grid(True, alpha=0.3, which="both")
    
    # Chart 5: Average Speedup by Size Range (bar chart)
    ax5 = plt.subplot(2, 3, 5)
    ranges = [
        (0, 10_000, "Tiny\n(<10K)"),
        (10_000, 100_000, "Small\n(10K-100K)"),
        (100_000, 1_000_000, "Medium\n(100K-1M)"),
        (1_000_000, 10_000_000, "Large\n(1M-10M)"),
        (10_000_000, 1_000_000_000_000, "Huge\n(>10M)"),
    ]
    labels, avg_speedups_seq, avg_speedups_par = [], [], []
    for low, high, label in ranges:
        seq_speedups_in_range = [s for s, n in zip(gpu_seq_speedups, items_counts) if low <= n < high]
        par_speedups_in_range = [s for s, n in zip(gpu_par_speedups, items_counts) if low <= n < high]
        if seq_speedups_in_range or par_speedups_in_range:
            labels.append(label)
            avg_speedups_seq.append(np.mean(seq_speedups_in_range) if seq_speedups_in_range else 0)
            avg_speedups_par.append(np.mean(par_speedups_in_range) if par_speedups_in_range else 0)
    
    if labels:
        x = np.arange(len(labels))
        width = 0.35
        
        colors_seq = ["#ff6b6b" if s < 1 else "#51cf66" for s in avg_speedups_seq]
        colors_par = ["#ffb3b3" if s < 1 else "#a3e4a3" for s in avg_speedups_par]
        
        bars1 = ax5.bar(x - width/2, avg_speedups_seq, width, label="GPU Seq vs CPU Seq", color=colors_seq, alpha=0.9, edgecolor="black")
        bars2 = ax5.bar(x + width/2, avg_speedups_par, width, label="GPU Par vs CPU Par", color=colors_par, alpha=0.9, edgecolor="black")
        
        ax5.axhline(y=1.0, color="black", linestyle="--", linewidth=2)
        ax5.set_ylabel("Average Speedup (CPU / GPU)")
        ax5.set_title("Average Speedup by Problem Size", fontweight="bold")
        ax5.set_xticks(x)
        ax5.set_xticklabels(labels, fontsize=9)
        ax5.legend(loc="upper left", fontsize=8)
        ax5.grid(True, alpha=0.3, axis="y")
        
        for bar, value in zip(bars1, avg_speedups_seq):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value:.2f}x',
                    ha='center', va='bottom', fontsize=8)
        for bar, value in zip(bars2, avg_speedups_par):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value:.2f}x',
                    ha='center', va='bottom', fontsize=8)
    
    # Chart 6: Speedup Distribution (histogram)
    ax6 = plt.subplot(2, 3, 6)
    all_speedups = gpu_seq_speedups + gpu_par_speedups
    if all_speedups:
        bins = np.linspace(min(all_speedups) * 0.9, max(all_speedups) * 1.1, 20)
        
        ax6.hist(gpu_seq_speedups, bins=bins, alpha=0.6, edgecolor="black", color=colors['gpu_seq'], 
                label=f"GPU Seq (mean={np.mean(gpu_seq_speedups):.2f}x)")
        ax6.hist(gpu_par_speedups, bins=bins, alpha=0.6, edgecolor="black", color=colors['gpu_par'], 
                label=f"GPU Par (mean={np.mean(gpu_par_speedups):.2f}x)")
        
        ax6.axvline(x=1.0, color="black", linestyle="--", linewidth=2, label="Break-even")
        ax6.set_xlabel("Speedup (CPU / GPU)")
        ax6.set_ylabel("Frequency")
        ax6.legend(fontsize=8)
        ax6.set_title("Speedup Distribution", fontweight="bold")
        ax6.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    # Save
    plot_file = output_dir / f"plot_{source_name}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {plot_file}")
    plt.close()
    
    # Print summary
    print("\n" + "=" * 70)
    print("STREAMING BENCHMARK SUMMARY (Adaptive Batching)")
    print("=" * 70)
    print(f"Total tests: {len(results)}")
    print(f"Adaptive batch range: {format_number(adaptive_min)} - {format_number(adaptive_max)}")
    print(f"\nAverage times:")
    print(f"  CPU Sequential:     {np.mean(cpu_seq_times):.4f}s")
    print(f"  GPU Seq+Adaptive:   {np.mean(gpu_seq_times):.4f}s")
    print(f"  CPU Parallel:       {np.mean(cpu_par_times):.4f}s")
    print(f"  GPU Par+Adaptive:   {np.mean(gpu_par_times):.4f}s")
    print(f"\nAverage speedups:")
    print(f"  GPU Seq vs CPU Seq: {np.mean(gpu_seq_speedups):.2f}x")
    print(f"  GPU Par vs CPU Par: {np.mean(gpu_par_speedups):.2f}x")
    
    # Find best strategy for each test
    best_strategies = []
    for i in range(len(results)):
        times = {
            "CPU Seq": cpu_seq_times[i],
            "GPU Seq": gpu_seq_times[i],
            "CPU Par": cpu_par_times[i],
            "GPU Par": gpu_par_times[i],
        }
        best = min(times.items(), key=lambda x: x[1])[0]
        best_strategies.append(best)
    
    from collections import Counter
    win_counts = Counter(best_strategies)
    print(f"\nBest strategy wins: {dict(win_counts)}")
    print("=" * 70)
    
    return True


# ============================================================================
# Unified Black-Scholes Benchmark Plots
# ============================================================================

def plot_unified_results(results, metadata, source_name, output_dir):
    """
    Generate comprehensive benchmark visualization for the unified benchmark.
    
    The results format has:
    - strategy: "cpu_sequential", "cpu_parallel", "gpu_sequential", "gpu_parallel"
    - num_options: number of options processed
    - duration_ms: execution time in milliseconds
    - throughput_mops: million options per second
    - num_workers: number of workers used
    """
    
    # Strategy display names and colors
    strategy_display = {
        "cpu_sequential": "CPU Sequential",
        "cpu_parallel": "CPU Parallel",
        "gpu_sequential": "GPU Sequential",
        "gpu_parallel": "GPU Parallel",
    }
    
    strategy_colors = {
        "cpu_sequential": "#3498db",  # Blue
        "cpu_parallel": "#2ecc71",    # Green
        "gpu_sequential": "#e74c3c",  # Red
        "gpu_parallel": "#9b59b6",    # Purple
    }
    
    # Group results by strategy and size
    by_strategy = defaultdict(list)
    by_size = defaultdict(list)
    
    for r in results:
        strategy = r.get("strategy", "unknown")
        by_strategy[strategy].append(r)
        by_size[r["num_options"]].append(r)
    
    # Get sorted unique sizes
    sizes = sorted(set(r["num_options"] for r in results))
    strategies = ["cpu_sequential", "cpu_parallel", "gpu_sequential", "gpu_parallel"]
    
    # Extract data series for each strategy
    data_by_strategy = {}
    for strategy in strategies:
        strat_results = sorted(by_strategy.get(strategy, []), key=lambda r: r["num_options"])
        data_by_strategy[strategy] = {
            "sizes": [r["num_options"] for r in strat_results],
            "times": [r["duration_ms"] / 1000.0 for r in strat_results],  # Convert to seconds
            "throughputs": [r["throughput_mops"] for r in strat_results],
        }
    
    # Create figure with 2x3 grid
    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"Black-Scholes Benchmark: CPU vs GPU Comparison\n{metadata.get('platform', 'Unknown Platform')}",
        fontsize=14,
        fontweight="bold",
    )
    
    # Chart 1: Execution Time Comparison (log-log)
    ax1 = plt.subplot(2, 3, 1)
    for strategy in strategies:
        data = data_by_strategy[strategy]
        if data["sizes"]:
            ax1.loglog(data["sizes"], data["times"], "o-", 
                      label=strategy_display.get(strategy, strategy),
                      color=strategy_colors.get(strategy, "gray"),
                      markersize=5, alpha=0.8)
    
    ax1.set_xlabel("Number of Options")
    ax1.set_ylabel("Execution Time (s)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_title("Execution Time vs Problem Size", fontweight="bold")
    ax1.grid(True, alpha=0.3, which="both")
    
    # Chart 2: Throughput Comparison (Million ops/sec)
    ax2 = plt.subplot(2, 3, 2)
    for strategy in strategies:
        data = data_by_strategy[strategy]
        if data["sizes"]:
            ax2.semilogx(data["sizes"], data["throughputs"], "o-",
                        label=strategy_display.get(strategy, strategy),
                        color=strategy_colors.get(strategy, "gray"),
                        markersize=5, alpha=0.8)
    
    ax2.set_xlabel("Number of Options")
    ax2.set_ylabel("Throughput (M opts/s)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.set_title("Throughput vs Problem Size", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    
    # Chart 3: GPU Speedup vs CPU
    ax3 = plt.subplot(2, 3, 3)
    
    # Calculate speedups: GPU Seq vs CPU Seq, GPU Par vs CPU Par
    speedups_seq = []
    speedups_par = []
    speedup_sizes = []
    
    for size in sizes:
        size_results = {r["strategy"]: r for r in by_size[size]}
        
        cpu_seq = size_results.get("cpu_sequential")
        gpu_seq = size_results.get("gpu_sequential")
        cpu_par = size_results.get("cpu_parallel")
        gpu_par = size_results.get("gpu_parallel")
        
        if cpu_seq and gpu_seq:
            speedup = cpu_seq["duration_ms"] / gpu_seq["duration_ms"] if gpu_seq["duration_ms"] > 0 else 0
            speedups_seq.append(speedup)
        else:
            speedups_seq.append(0)
            
        if cpu_par and gpu_par:
            speedup = cpu_par["duration_ms"] / gpu_par["duration_ms"] if gpu_par["duration_ms"] > 0 else 0
            speedups_par.append(speedup)
        else:
            speedups_par.append(0)
            
        speedup_sizes.append(size)
    
    ax3.semilogx(speedup_sizes, speedups_seq, "o-", label="GPU Seq vs CPU Seq",
                color=strategy_colors["gpu_sequential"], markersize=6, alpha=0.8)
    ax3.semilogx(speedup_sizes, speedups_par, "s-", label="GPU Par vs CPU Par",
                color=strategy_colors["gpu_parallel"], markersize=6, alpha=0.8)
    ax3.axhline(y=1.0, color="black", linestyle="--", linewidth=2, label="Break-even")
    ax3.set_xlabel("Number of Options")
    ax3.set_ylabel("Speedup (CPU / GPU)")
    ax3.set_title("GPU Speedup vs Problem Size", fontweight="bold")
    ax3.legend(loc="best", fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Shade regions
    ylim = ax3.get_ylim()
    ax3.axhspan(0, 1.0, alpha=0.1, color="red")
    ax3.axhspan(1.0, max(ylim[1], 2), alpha=0.1, color="green")
    ax3.set_ylim(ylim)
    
    # Chart 4: Best Strategy Bar Chart
    ax4 = plt.subplot(2, 3, 4)
    
    # Find best strategy for each size
    best_by_size = []
    for size in sizes:
        size_results = by_size[size]
        best = min(size_results, key=lambda r: r["duration_ms"])
        best_by_size.append(best["strategy"])
    
    # Count wins per strategy
    from collections import Counter
    win_counts = Counter(best_by_size)
    
    strategies_ordered = ["cpu_sequential", "cpu_parallel", "gpu_sequential", "gpu_parallel"]
    wins = [win_counts.get(s, 0) for s in strategies_ordered]
    colors = [strategy_colors.get(s, "gray") for s in strategies_ordered]
    labels = [strategy_display.get(s, s) for s in strategies_ordered]
    
    bars = ax4.bar(labels, wins, color=colors, alpha=0.8, edgecolor="black")
    ax4.set_ylabel("Number of Wins")
    ax4.set_title("Best Strategy by Problem Size", fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")
    
    for bar, win in zip(bars, wins):
        if win > 0:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, str(win),
                    ha='center', va='bottom', fontweight='bold')
    
    # Chart 5: Average Speedup by Size Range
    ax5 = plt.subplot(2, 3, 5)
    
    ranges = [
        (0, 10_000, "Tiny\n(<10K)"),
        (10_000, 100_000, "Small\n(10K-100K)"),
        (100_000, 1_000_000, "Medium\n(100K-1M)"),
        (1_000_000, 10_000_000, "Large\n(1M-10M)"),
        (10_000_000, 100_000_000, "Very Large\n(10M-100M)"),
        (100_000_000, float('inf'), "Huge\n(>100M)"),
    ]
    
    range_labels, avg_seq, avg_par = [], [], []
    
    for low, high, label in ranges:
        mask = [i for i, size in enumerate(speedup_sizes) if low <= size < high]
        if mask:
            range_labels.append(label)
            avg_seq.append(float(np.mean([speedups_seq[i] for i in mask])))
            avg_par.append(float(np.mean([speedups_par[i] for i in mask])))
    
    if range_labels:
        x = np.arange(len(range_labels))
        width = 0.35
        
        colors_seq = ["#ff6b6b" if s < 1 else "#51cf66" for s in avg_seq]
        colors_par = ["#ffb3b3" if s < 1 else "#a3e4a3" for s in avg_par]
        
        bars1 = ax5.bar(x - width/2, avg_seq, width, label="GPU Seq vs CPU Seq", color=colors_seq, alpha=0.9, edgecolor="black")
        bars2 = ax5.bar(x + width/2, avg_par, width, label="GPU Par vs CPU Par", color=colors_par, alpha=0.9, edgecolor="black")
        
        ax5.axhline(y=1.0, color="black", linestyle="--", linewidth=2)
        ax5.set_ylabel("Average Speedup (CPU / GPU)")
        ax5.set_title("Average Speedup by Problem Size", fontweight="bold")
        ax5.set_xticks(x)
        ax5.set_xticklabels(range_labels, fontsize=8)
        ax5.legend(loc="upper left", fontsize=8)
        ax5.grid(True, alpha=0.3, axis="y")
        
        for bar, value in zip(bars1, avg_seq):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value:.1f}x',
                    ha='center', va='bottom', fontsize=7)
        for bar, value in zip(bars2, avg_par):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{value:.1f}x',
                    ha='center', va='bottom', fontsize=7)
    
    # Chart 6: Throughput Distribution (box plot)
    ax6 = plt.subplot(2, 3, 6)
    
    throughput_data = []
    box_labels = []
    box_colors = []
    
    for strategy in strategies:
        if by_strategy.get(strategy):
            throughputs = [r["throughput_mops"] for r in by_strategy[strategy]]
            throughput_data.append(throughputs)
            box_labels.append(strategy_display.get(strategy, strategy).replace(" ", "\n"))
            box_colors.append(strategy_colors.get(strategy, "gray"))
    
    if throughput_data:
        bp = ax6.boxplot(throughput_data, patch_artist=True)
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax6.set_ylabel("Throughput (M opts/s)")
        ax6.set_title("Throughput Distribution by Strategy", fontweight="bold")
        ax6.set_xticklabels(box_labels, fontsize=8)
        ax6.grid(True, alpha=0.3, axis="y")
    
    # Add config description
    gpu_batch_size = metadata.get("gpu_batch_size", 10_000_000)
    config_text = f"Platform: {metadata.get('platform', 'Unknown')}  |  GPU Batch Size: {format_number(gpu_batch_size)}  |  Test Sizes: {len(sizes)}  |  Total Tests: {len(results)}"
    fig.text(
        0.5, 0.01, config_text,
        ha="center", va="bottom",
        fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8, edgecolor="gray")
    )
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    
    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_file = output_dir / f"plot_{source_name}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nChart saved to: {plot_file}")
    plt.close()
    
    # Print summary
    print_unified_summary(results, metadata, speedups_seq, speedups_par, speedup_sizes, by_strategy)


def print_unified_summary(results, metadata, speedups_seq, speedups_par, sizes, by_strategy):
    """Print summary statistics for unified benchmark."""
    print("\n" + "=" * 70)
    print("          BLACK-SCHOLES BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Platform: {metadata.get('platform', 'Unknown')}")
    print(f"Timestamp: {metadata.get('timestamp', 'Unknown')}")
    print(f"GPU Batch Size: {format_number(metadata.get('gpu_batch_size', 10_000_000))}")
    print(f"Total benchmarks: {len(results)}")
    print(f"Problem sizes tested: {len(sizes)}")
    
    if sizes:
        print(f"Problem size range: {format_number(min(sizes))} - {format_number(max(sizes))}")
    
    # Speedup statistics
    if speedups_seq:
        valid_seq = [s for s in speedups_seq if s > 0]
        if valid_seq:
            print(f"\nSpeedup Statistics (GPU Seq vs CPU Seq):")
            print(f"  Min:    {min(valid_seq):.3f}x")
            print(f"  Max:    {max(valid_seq):.3f}x")
            print(f"  Mean:   {np.mean(valid_seq):.3f}x")
            print(f"  Median: {np.median(valid_seq):.3f}x")
    
    if speedups_par:
        valid_par = [s for s in speedups_par if s > 0]
        if valid_par:
            print(f"\nSpeedup Statistics (GPU Par vs CPU Par):")
            print(f"  Min:    {min(valid_par):.3f}x")
            print(f"  Max:    {max(valid_par):.3f}x")
            print(f"  Mean:   {np.mean(valid_par):.3f}x")
            print(f"  Median: {np.median(valid_par):.3f}x")
    
    # Win/Loss
    if speedups_seq:
        valid_seq = [s for s in speedups_seq if s > 0]
        gpu_wins = sum(1 for s in valid_seq if s > 1.0)
        print(f"\nWin/Loss (GPU Seq vs CPU Seq):")
        print(f"  GPU wins: {gpu_wins} ({100 * gpu_wins / len(valid_seq):.1f}%)")
        print(f"  CPU wins: {len(valid_seq) - gpu_wins} ({100 * (len(valid_seq) - gpu_wins) / len(valid_seq):.1f}%)")
    
    if speedups_par:
        valid_par = [s for s in speedups_par if s > 0]
        gpu_wins = sum(1 for s in valid_par if s > 1.0)
        print(f"\nWin/Loss (GPU Par vs CPU Par):")
        print(f"  GPU wins: {gpu_wins} ({100 * gpu_wins / len(valid_par):.1f}%)")
        print(f"  CPU wins: {len(valid_par) - gpu_wins} ({100 * (len(valid_par) - gpu_wins) / len(valid_par):.1f}%)")
    
    # Best throughput per strategy
    print(f"\nBest Throughput per Strategy:")
    for strategy, strat_results in by_strategy.items():
        if strat_results:
            best = max(strat_results, key=lambda r: r["throughput_mops"])
            print(f"  {strategy.replace('_', ' ').title()}: {best['throughput_mops']:.2f} M opts/s @ {format_number(best['num_options'])} options")
    
    # Find crossover point
    crossover_idx = None
    for i, s in enumerate(speedups_seq):
        if s > 1.0:
            crossover_idx = i
            break
    
    if crossover_idx is not None:
        print(f"\nGPU becomes faster at: ~{format_number(sizes[crossover_idx])} options")
    
    print("=" * 70)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the unified plotting script."""
    if len(sys.argv) < 2:
        print("Usage: python benches/tools/plot_benchmark.py <benchmark_type|json_file>")
        print("\nBenchmark types: black_scholes")
        print("\nExamples:")
        print("  python benches/tools/plot_benchmark.py black_scholes")
        print("  python benches/tools/plot_benchmark.py path/to/benchmark.json")
        sys.exit(1)

    arg = sys.argv[1]
    
    # Check if argument is a benchmark type or a file path
    if arg.lower() in BENCHMARK_TYPES:
        # Find most recent file for this benchmark type
        benchmark_type = arg.lower()
        filename = find_most_recent_file(benchmark_type)
        
        if not filename:
            print(f"Error: No benchmark files found for type '{benchmark_type}'")
            print(f"\nRun the benchmark first with:")
            print(f"  cargo bench --bench gpu_black_scholes --features gpu-wgpu")
            sys.exit(1)
        
        print(f"Using most recent {benchmark_type} file: {filename}")
    else:
        # Treat as file path
        filename = Path(arg)
        
        if not filename.exists():
            print(f"Error: file '{filename}' not found")
            sys.exit(1)

    # Load results
    results, metadata = load_results(filename)
    if not results:
        print("No benchmark results found in file!")
        sys.exit(1)

    print(f"Loaded {len(results)} benchmark results from {filename}")

    # Determine benchmark type
    benchmark_type = get_benchmark_type(metadata, filename)
    
    if not benchmark_type:
        # Default to black_scholes for new format
        benchmark_type = "black_scholes"

    print(f"Detected benchmark type: {benchmark_type}")

    # Generate appropriate plots
    output_dir = filename.parent
    source_name = filename.stem
    
    # Use standard plotting for black_scholes (handles items_count format)
    plot_standard_results(results, metadata, source_name, output_dir, benchmark_type)


if __name__ == "__main__":
    main()


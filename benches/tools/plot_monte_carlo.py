#!/usr/bin/env python3
"""
Monte Carlo Benchmark Plotting Tool.

Visualizes benchmark results from the Monte Carlo CPU vs GPU benchmark.

Usage:
    python benches/tools/plot_monte_carlo.py benches/results/monte_carlo/2024-12-31/monte_carlo_benchmark_*.json
    python benches/tools/plot_monte_carlo.py monte_carlo  # Uses most recent file

Output:
    Charts are saved alongside the JSON file as plot_monte_carlo_benchmark_{timestamp}.png
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_results(filename):
    """Load benchmark results from JSON file."""
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "results" in data:
        return data["results"], data
    return data, {"platform": "Unknown", "start_time": "Unknown"}


def format_number(n):
    """Format a number with appropriate suffix (K, M, B)."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def format_time(seconds):
    """Format time in appropriate units."""
    if seconds >= 60:
        return f"{seconds / 60:.1f}min"
    elif seconds >= 1:
        return f"{seconds:.2f}s"
    elif seconds >= 0.001:
        return f"{seconds * 1000:.1f}ms"
    else:
        return f"{seconds * 1_000_000:.1f}µs"


def find_most_recent_file(benchmark_type: str) -> Path:
    """Find the most recent benchmark file for a given type."""
    script_dir = Path(__file__).parent
    results_base = script_dir.parent / "results"
    
    search_dirs = [
        results_base / benchmark_type,
        Path("benches/results") / benchmark_type,
    ]
    
    prefix = f"{benchmark_type}_benchmark"
    
    candidates = []
    for search_dir in search_dirs:
        if search_dir.exists():
            candidates.extend(search_dir.rglob(f"*{prefix}*.json"))
    
    if not candidates:
        return None
    
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def build_config_description(results, metadata):
    """
    Build a system configuration description string.

    Shows system specs, parallelisation factors, Monte Carlo parameters,
    and benchmark load parameters as a single-line metadata footer.

    Unified format across all GPU benchmark plotters:
      Platform | CPU | GPU | Workers | GPU Threads | Batch | [specific] | Sizes | Tests
    """
    platform = metadata.get("platform", "Unknown")
    mc_config = metadata.get("monte_carlo_config", {})
    system_config = metadata.get("system_config", {})

    items_counts = [r["items_count"] for r in results]
    min_items = min(items_counts) if items_counts else 0
    max_items = max(items_counts) if items_counts else 0

    cpu_workers = metadata.get("cpu_workers", results[0].get("cpu_workers", 4) if results else 4)
    cpu_cores = system_config.get("cpu_cores", cpu_workers)
    ram_gb = system_config.get("ram_gb", 0)
    gpu_device = system_config.get("gpu_device", None)
    gpu_backend = system_config.get("gpu_backend", "")
    gpu_batch = metadata.get("batch_size", system_config.get("gpu_batch_size", "?"))

    # GPU threads: try system_config first, then per-result field, then default estimate
    gpu_threads = system_config.get("gpu_threads", None)
    if gpu_threads is None and results:
        gpu_threads = results[0].get("gpu_threads", None)
    if gpu_threads is None:
        gpu_threads = 256 * 64  # default CubeCL estimate

    num_paths = mc_config.get("num_paths", "?")
    time_steps = mc_config.get("time_steps", "?")

    parts = [f"Platform: {platform}"]

    # CPU + RAM
    if ram_gb > 0:
        parts.append(f"CPU: {cpu_cores} cores, RAM: {ram_gb:.0f} GB")
    else:
        parts.append(f"CPU: {cpu_cores} cores")

    # GPU device
    if gpu_device:
        parts.append(f"GPU: {gpu_device}")
    elif gpu_backend:
        parts.append(f"GPU Backend: {gpu_backend}")
    else:
        parts.append(f"GPU: {platform}")

    # Parallelisation
    parts.append(f"Workers: {cpu_workers}")
    parts.append(f"GPU Threads: {format_number(gpu_threads)}")
    parts.append(f"Batch: {format_number(gpu_batch)}")

    # Monte Carlo specific
    parts.append(f"MC: {num_paths} paths \u00d7 {time_steps} steps")

    # Benchmark load
    parts.append(f"Sizes: {format_number(min_items)}\u2013{format_number(max_items)}")
    parts.append(f"Tests: {len(results)}")

    return "  |  ".join(parts)


def plot_monte_carlo_results(results, metadata, source_name: str, output_dir: Path):
    """Generate benchmark visualization for Monte Carlo results."""
    
    items_counts = [r["items_count"] for r in results]
    
    # Helper to get field with multiple naming conventions
    def get_field(r, *names, default=0):
        for name in names:
            if name in r:
                return r[name]
        return default
    
    # Total timing (newest to oldest field names)
    cpu_times = [get_field(r, "renoir_seq_total_time_s", "cpu_total_time_s") for r in results]
    gpu_times = [get_field(r, "gpu_total_time_s") for r in results]
    par_times = [get_field(r, "renoir_par_total_time_s", "renoir_total_time_s") for r in results]
    
    # Speedups
    speedups = [r.get("speedup", 0) for r in results]
    speedups_parallel = [r.get("speedup_parallel", 0) for r in results]
    
    cpu_gflops = [get_field(r, "renoir_seq_gflops", "cpu_gflops") for r in results]
    gpu_gflops = [get_field(r, "gpu_gflops") for r in results]
    par_gflops = [get_field(r, "renoir_par_gflops", "renoir_gflops") for r in results]
    
    validation_passed = [r.get("validation_passed", False) for r in results]
    
    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=(16, 10))
    
    fig.suptitle(
        f"Monte Carlo CPU vs GPU Benchmark\n{metadata.get('platform', 'Unknown Platform')}",
        fontsize=14,
        fontweight="bold",
    )

    # Chart 1: Execution Time Comparison (Log-Log)
    ax1 = plt.subplot(2, 3, 1)
    ax1.loglog(items_counts, cpu_times, "o-", label="CPU Sequential", color="tab:blue", markersize=6, linewidth=2)
    ax1.loglog(items_counts, par_times, "s-", label="CPU Parallel", color="tab:green", markersize=6, linewidth=2)
    ax1.loglog(items_counts, gpu_times, "^-", label="GPU", color="tab:red", markersize=6, linewidth=2)
    ax1.set_xlabel("Number of Options")
    ax1.set_ylabel("Time (s)")
    ax1.legend(loc="upper left")
    ax1.set_title("Execution Time (Log Scale)", fontweight="bold")
    ax1.grid(True, alpha=0.3, which="both")

    # Chart 2: GPU Speedup vs Problem Size
    ax2 = plt.subplot(2, 3, 2)
    ax2.semilogx(items_counts, speedups, "o-", label="vs CPU Seq", color="tab:blue", markersize=6, linewidth=2)
    ax2.semilogx(items_counts, speedups_parallel, "s-", label="vs CPU Par", color="tab:green", markersize=6, linewidth=2)
    ax2.axhline(y=1.0, color="black", linestyle="--", linewidth=2, label="Break-even")
    ax2.set_xlabel("Number of Options")
    ax2.set_ylabel("Speedup (X times faster)")
    ax2.set_title("GPU Speedup", fontweight="bold")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.axhspan(0, 1.0, alpha=0.1, color="red")
    if max(speedups) > 1:
        ax2.axhspan(1.0, max(speedups) * 1.1, alpha=0.1, color="green")
    
    # Add speedup annotations
    for i, (x, y) in enumerate(zip(items_counts, speedups)):
        if i == len(items_counts) - 1 or i == 0:
            ax2.annotate(f"{y:.0f}x", (x, y), textcoords="offset points", 
                        xytext=(0, 10), ha='center', fontsize=8, fontweight='bold')

    # Chart 3: GFLOPS Comparison (Log scale Y)
    ax3 = plt.subplot(2, 3, 3)
    ax3.semilogy(items_counts, cpu_gflops, "o-", label="CPU Seq", color="tab:blue", markersize=6, linewidth=2)
    ax3.semilogy(items_counts, par_gflops, "s-", label="CPU Par", color="tab:green", markersize=6, linewidth=2)
    ax3.semilogy(items_counts, gpu_gflops, "^-", label="GPU", color="tab:red", markersize=6, linewidth=2)
    ax3.set_xscale('log')
    ax3.set_xlabel("Number of Options")
    ax3.set_ylabel("GFLOPS (Log Scale)")
    ax3.legend(loc="best")
    ax3.set_title("Computational Throughput", fontweight="bold")
    ax3.grid(True, alpha=0.3, which="both")

    # Chart 4: Bar Chart (Largest Test)
    ax4 = plt.subplot(2, 3, 4)
    if results:
        last = results[-1]
        strategies = ["CPU Seq", "CPU Par", "GPU"]
        times = [cpu_times[-1], par_times[-1], gpu_times[-1]]
        colors = ["tab:blue", "tab:green", "tab:red"]
        bars = ax4.bar(strategies, times, color=colors, edgecolor="black", alpha=0.8)
        ax4.set_ylabel("Time (s) - Log Scale")
        ax4.set_yscale('log')
        ax4.set_title(f"Time @ {format_number(last['items_count'])} Options", fontweight="bold")
        ax4.grid(True, alpha=0.3, axis="y", which="both")
        
        # Add time labels
        for bar, t in zip(bars, times):
            label_y = t * 1.2 if t > 0 else 1
            ax4.text(bar.get_x() + bar.get_width()/2, label_y, format_time(t),
                    ha="center", va="bottom", fontweight="bold", fontsize=9)
        
        # Add speedup annotation
        if times[2] > 0:
            speedup_seq = times[0] / times[2]
            speedup_par = times[1] / times[2]
            ax4.text(0.5, 0.95, f"GPU is {speedup_seq:.0f}x faster than CPU Seq\n"
                                f"GPU is {speedup_par:.0f}x faster than CPU Par",
                    transform=ax4.transAxes, ha='center', va='top',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Chart 5: Relative Time Comparison (Normalized to GPU)
    ax5 = plt.subplot(2, 3, 5)
    if results:
        x_labels = [format_number(r["items_count"]) for r in results]
        x_pos = np.arange(len(x_labels))
        
        # Normalize to GPU time
        cpu_relative = [c / g if g > 0 else 0 for c, g in zip(cpu_times, gpu_times)]
        par_relative = [p / g if g > 0 else 0 for p, g in zip(par_times, gpu_times)]
        gpu_relative = [1.0] * len(results)
        
        width = 0.25
        bars1 = ax5.bar(x_pos - width, cpu_relative, width, label="CPU Seq", color="tab:blue", alpha=0.8)
        bars2 = ax5.bar(x_pos, par_relative, width, label="CPU Par", color="tab:green", alpha=0.8)
        bars3 = ax5.bar(x_pos + width, gpu_relative, width, label="GPU", color="tab:red", alpha=0.8)
        
        ax5.set_xlabel("Problem Size")
        ax5.set_ylabel("Time Relative to GPU (X times slower)")
        ax5.set_title("CPU Slowdown vs GPU", fontweight="bold")
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(x_labels, rotation=45, ha="right")
        ax5.legend(loc="upper left")
        ax5.grid(True, alpha=0.3, axis="y")
        ax5.set_yscale('log')
        
        # Add value labels on bars
        for bars, values in [(bars1, cpu_relative), (bars2, par_relative)]:
            for bar, val in zip(bars, values):
                if val > 10:
                    ax5.text(bar.get_x() + bar.get_width()/2, val, f"{val:.0f}x",
                            ha="center", va="bottom", fontsize=7, rotation=90)

    # Chart 6: Validation Summary
    ax6 = plt.subplot(2, 3, 6)
    valid_count = sum(validation_passed)
    total_count = len(validation_passed)
    
    if total_count > 0:
        sizes = [valid_count, total_count - valid_count]
        labels = [f"Passed ({valid_count})", f"Failed ({total_count - valid_count})"]
        colors = ["tab:green", "tab:red"]
        explode = (0.05, 0) if valid_count > 0 else (0, 0.05)
        
        # Filter out zero values
        non_zero = [(s, l, c, e) for s, l, c, e in zip(sizes, labels, colors, explode) if s > 0]
        if non_zero:
            sizes, labels, colors, explode = zip(*non_zero)
            ax6.pie(sizes, labels=labels, colors=colors, explode=explode,
                   autopct='%1.0f%%', startangle=90)
        ax6.set_title("Validation Results", fontweight="bold")
        
        # Add max speedup annotation
        if max(speedups) > 0:
            ax6.text(0.5, -0.15, f"Peak Speedup: {max(speedups):.0f}x\n"
                                f"Peak GFLOPS: {max(gpu_gflops):.0f}",
                    transform=ax6.transAxes, ha='center', fontsize=9, fontweight='bold')

    # Add config description
    config_text = build_config_description(results, metadata)
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
        output_file = output_dir / f"plot_monte_carlo_benchmark_{timestamp_part}.png"
    else:
        output_file = output_dir / f"plot_monte_carlo_{source_name}.png"
    
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nChart saved to: {output_file}")

    # Print Summary
    print_summary(results, metadata, speedups, speedups_parallel, 
                  gpu_gflops, validation_passed)


def print_summary(results, metadata, speedups, speedups_parallel, gpu_gflops, 
                  validation_passed):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("          MONTE CARLO BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Platform: {metadata.get('platform', 'Unknown')}")
    print(f"Start time: {metadata.get('start_time', 'Unknown')}")
    print(f"Total tests: {len(results)}")
    
    mc_config = metadata.get("monte_carlo_config", {})
    print(f"\nMonte Carlo Config:")
    print(f"  Paths/Option: {mc_config.get('num_paths', '?')}")
    print(f"  Steps/Path:   {mc_config.get('time_steps', '?')}")
    flops = mc_config.get('flops_per_option', 0)
    if flops > 0:
        print(f"  FLOPs/Option: {flops / 1000:.1f}K")

    print(f"\nSpeedup vs CPU Sequential:")
    print(f"  Min:    {min(speedups):.1f}x")
    print(f"  Max:    {max(speedups):.1f}x")
    print(f"  Mean:   {np.mean(speedups):.1f}x")

    print(f"\nSpeedup vs CPU Parallel:")
    print(f"  Min:    {min(speedups_parallel):.1f}x")
    print(f"  Max:    {max(speedups_parallel):.1f}x")
    print(f"  Mean:   {np.mean(speedups_parallel):.1f}x")

    print(f"\nPeak GPU GFLOPS: {max(gpu_gflops):.1f}")
    
    valid_count = sum(validation_passed)
    print(f"\nValidation: {valid_count}/{len(validation_passed)} passed")
    print("=" * 60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_monte_carlo.py <json_file_or_type>")
        print("Types: monte_carlo")
        sys.exit(1)

    arg = sys.argv[1]
    
    # Check if it's a file path or a type name
    if Path(arg).exists() and arg.endswith(".json"):
        file_path = Path(arg)
    elif arg == "monte_carlo":
        file_path = find_most_recent_file("monte_carlo")
        if not file_path:
            print(f"Error: No benchmark files found for type '{arg}'")
            sys.exit(1)
        print(f"Using most recent file: {file_path}")
    else:
        print(f"Error: Unknown argument '{arg}'")
        sys.exit(1)

    results, metadata = load_results(file_path)
    
    if not results:
        print("Error: No results found in file")
        sys.exit(1)

    output_dir = file_path.parent
    source_name = file_path.stem
    
    plot_monte_carlo_results(results, metadata, source_name, output_dir)


if __name__ == "__main__":
    main()

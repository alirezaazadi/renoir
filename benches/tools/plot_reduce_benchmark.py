#!/usr/bin/env python3
"""
GPU Reduce Benchmark Plotting Tool.

Visualizes benchmark results from the GPU reduce benchmark, comparing
three strategies for each reduce operator (Sum, Product, Min, Max):
  - Renoir Sequential (single worker CPU fold)
  - Renoir Parallel   (multi-worker reduce_assoc)
  - GPU               (CubeCL reduce_gpu)

Usage:
    # Plot by benchmark type (uses most recent file)
    python benches/tools/plot_reduce_benchmark.py reduce

    # Plot a specific results file
    python benches/tools/plot_reduce_benchmark.py path/to/reduce_benchmark_*.json

Output:
    Charts are saved alongside the JSON file as
    plot_reduce_benchmark_{timestamp}.png
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# Constants
# ============================================================================

OPERATOR_COLORS = {
    "sum": "#3498db",
    "product": "#e74c3c",
    "min": "#2ecc71",
    "max": "#9b59b6",
}

STRATEGY_COLORS = {
    "seq": "#3498db",   # blue   – Renoir Sequential
    "par": "#2ecc71",   # green  – Renoir Parallel
    "gpu": "#e74c3c",   # red    – GPU
}

STRATEGY_LABELS = {
    "seq": "Renoir Sequential",
    "par": "Renoir Parallel",
    "gpu": "GPU (CubeCL)",
}


# ============================================================================
# Data Loading
# ============================================================================

def load_results(filename):
    """Load benchmark results from JSON file."""
    with open(filename, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "results" in data:
        return data["results"], data
    return data, {"platform": "Unknown", "start_time": "Unknown"}


def format_number(n):
    """Format a number with appropriate suffix."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.1f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def find_most_recent_file():
    """Find the most recent reduce benchmark JSON file."""
    script_dir = Path(__file__).parent
    results_base = script_dir.parent / "results" / "reduce"
    candidates = list(results_base.rglob("reduce_benchmark_*.json"))

    # Also check from CWD
    for extra in [
        Path("benches/results/reduce"),
        Path("benchmark_results"),
    ]:
        if extra.exists():
            candidates.extend(extra.rglob("*reduce*.json"))

    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


# ============================================================================
# Metadata / Config Description
# ============================================================================

def build_metadata_text(results, metadata):
    """
    Build a system + benchmark configuration string to display
    at the bottom of the chart.

    Unified format across all GPU benchmark plotters:
      Platform | CPU | GPU | Workers | GPU Threads | Batch | [specific] | Sizes | Tests
    """
    platform = metadata.get("platform", "Unknown")
    system_config = metadata.get("system_config", {})

    cpu_cores = system_config.get("cpu_cores", metadata.get("cpu_workers", "?"))
    ram_gb = system_config.get("ram_gb", 0)
    gpu_device = system_config.get("gpu_device", None)
    gpu_backend = system_config.get("gpu_backend", "")
    batch_size = metadata.get("batch_size", "?")
    tile_size = metadata.get("tile_size", "?")
    cpu_workers = metadata.get("cpu_workers", "?")
    total_tests = metadata.get("total_tests", len(results))

    # GPU threads: try system_config first, then per-result field, then default estimate
    gpu_threads = system_config.get("gpu_threads", None)
    if gpu_threads is None and results:
        gpu_threads = results[0].get("gpu_threads", None)
    if gpu_threads is None:
        gpu_threads = 256 * 64  # default CubeCL estimate

    items_counts = [r["items_count"] for r in results]
    min_items = min(items_counts) if items_counts else 0
    max_items = max(items_counts) if items_counts else 0

    operators = sorted(set(r.get("operator", "?") for r in results))

    parts = []
    parts.append(f"Platform: {platform}")

    if ram_gb > 0:
        parts.append(f"CPU: {cpu_cores} cores, RAM: {ram_gb:.0f} GB")
    else:
        parts.append(f"CPU: {cpu_cores} cores")

    if gpu_device:
        parts.append(f"GPU: {gpu_device}")
    elif gpu_backend:
        parts.append(f"GPU Backend: {gpu_backend}")

    parts.append(f"Workers: {cpu_workers}")
    parts.append(f"GPU Threads: {format_number(gpu_threads)}")
    parts.append(f"Batch: {format_number(batch_size)}")
    parts.append(f"Tile: {format_number(tile_size)}")
    parts.append(f"Sizes: {format_number(min_items)}\u2013{format_number(max_items)}")
    parts.append(f"Ops: {', '.join(operators)}")
    parts.append(f"Tests: {total_tests}")

    return "  |  ".join(parts)


# ============================================================================
# Plotting
# ============================================================================

def plot_reduce_results(results, metadata, source_name, output_dir):
    """
    Generate a 2x3 grid of charts for the reduce benchmark.

    Charts:
      1. Execution Time vs Problem Size (log-log) – all operators, 3 strategies
      2. GPU Speedup vs Problem Size (Seq/GPU and Par/GPU)
      3. GFLOPS Comparison
      4. Throughput (elements/sec)
      5. Average Speedup by Size Range (bar chart)
      6. Per-Operator Speedup Distribution (box plot)
    """

    # ── Organise data by operator ────────────────────────────────────────
    by_op = defaultdict(list)
    for r in results:
        by_op[r.get("operator", "sum")].append(r)

    all_ops = sorted(by_op.keys())

    # Aggregate across all operators for combined charts
    items_all = [r["items_count"] for r in results]
    seq_times = [r["renoir_seq_total_time_s"] for r in results]
    par_times = [r["renoir_par_total_time_s"] for r in results]
    gpu_times = [r["gpu_total_time_s"] for r in results]

    speedups_seq = [r.get("speedup_seq", r.get("speedup", 1.0)) for r in results]
    speedups_par = [r.get("speedup_par", r.get("speedup_parallel", 1.0)) for r in results]

    seq_gflops = [r.get("renoir_seq_gflops", 0) for r in results]
    par_gflops = [r.get("renoir_par_gflops", 0) for r in results]
    gpu_gflops = [r.get("gpu_gflops", 0) for r in results]

    # ── Figure setup ─────────────────────────────────────────────────────
    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "GPU Reduce Benchmark: Renoir Sequential vs Parallel vs GPU\n"
        f"{metadata.get('platform', 'Unknown Platform')}",
        fontsize=14,
        fontweight="bold",
    )

    # ── Chart 1: Execution Time (log-log) ────────────────────────────────
    ax1 = plt.subplot(2, 3, 1)
    for op_name in all_ops:
        op_results = sorted(by_op[op_name], key=lambda r: r["items_count"])
        sizes = [r["items_count"] for r in op_results]
        color = OPERATOR_COLORS.get(op_name, "gray")

        ax1.loglog(sizes,
                   [r["renoir_seq_total_time_s"] for r in op_results],
                   "o--", color=color, alpha=0.35, markersize=3, linewidth=1)
        ax1.loglog(sizes,
                   [r["renoir_par_total_time_s"] for r in op_results],
                   "s--", color=color, alpha=0.35, markersize=3, linewidth=1)
        ax1.loglog(sizes,
                   [r["gpu_total_time_s"] for r in op_results],
                   "^-", color=color, alpha=0.8, markersize=4, linewidth=1.5,
                   label=f"GPU {op_name}")

    # Add legend proxy for strategy
    from matplotlib.lines import Line2D
    proxy = [
        Line2D([0], [0], marker="o", linestyle="--", color="gray", label="Renoir Seq"),
        Line2D([0], [0], marker="s", linestyle="--", color="gray", label="Renoir Par"),
        Line2D([0], [0], marker="^", linestyle="-", color="gray", label="GPU"),
    ]
    handles = proxy + ax1.get_legend_handles_labels()[0][:len(all_ops)]
    ax1.legend(handles=proxy + [
        Line2D([0], [0], marker="^", linestyle="-",
               color=OPERATOR_COLORS.get(op, "gray"), label=f"GPU {op}")
        for op in all_ops
    ], loc="upper left", fontsize=6, ncol=2)
    ax1.set_xlabel("Number of Elements")
    ax1.set_ylabel("Execution Time (s)")
    ax1.set_title("Execution Time vs Problem Size", fontweight="bold")
    ax1.grid(True, alpha=0.3, which="both")

    # ── Chart 2: GPU Speedup ─────────────────────────────────────────────
    ax2 = plt.subplot(2, 3, 2)
    for op_name in all_ops:
        op_results = sorted(by_op[op_name], key=lambda r: r["items_count"])
        sizes = [r["items_count"] for r in op_results]
        color = OPERATOR_COLORS.get(op_name, "gray")

        ax2.semilogx(sizes,
                     [r.get("speedup_seq", 1) for r in op_results],
                     "o-", color=color, markersize=4, alpha=0.7,
                     label=f"Seq/GPU {op_name}")
        ax2.semilogx(sizes,
                     [r.get("speedup_par", 1) for r in op_results],
                     "s--", color=color, markersize=4, alpha=0.45)

    ax2.axhline(y=1.0, color="black", linestyle="--", linewidth=2, label="Break-even")
    ylim = ax2.get_ylim()
    ax2.axhspan(0, 1.0, alpha=0.08, color="red")
    ax2.axhspan(1.0, max(ylim[1], 2), alpha=0.08, color="green")
    ax2.set_xlabel("Number of Elements")
    ax2.set_ylabel("Speedup (CPU / GPU)")
    ax2.set_title("GPU Speedup vs Problem Size\n(solid = Seq/GPU, dashed = Par/GPU)",
                  fontweight="bold", fontsize=10)
    ax2.legend(loc="best", fontsize=6, ncol=2)
    ax2.grid(True, alpha=0.3)

    # ── Chart 3: GFLOPS ──────────────────────────────────────────────────
    ax3 = plt.subplot(2, 3, 3)
    for op_name in all_ops:
        op_results = sorted(by_op[op_name], key=lambda r: r["items_count"])
        sizes = [r["items_count"] for r in op_results]
        color = OPERATOR_COLORS.get(op_name, "gray")

        ax3.semilogx(sizes,
                     [r.get("renoir_seq_gflops", 0) for r in op_results],
                     "o--", color=color, alpha=0.35, markersize=3)
        ax3.semilogx(sizes,
                     [r.get("renoir_par_gflops", 0) for r in op_results],
                     "s--", color=color, alpha=0.35, markersize=3)
        ax3.semilogx(sizes,
                     [r.get("gpu_gflops", 0) for r in op_results],
                     "^-", color=color, alpha=0.8, markersize=4,
                     label=f"GPU {op_name}")

    ax3.set_xlabel("Number of Elements")
    ax3.set_ylabel("GFLOPS")
    ax3.legend(loc="upper left", fontsize=7)
    ax3.set_title("Computational Throughput", fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # ── Chart 4: Throughput (elements/sec) ───────────────────────────────
    ax4 = plt.subplot(2, 3, 4)
    for op_name in all_ops:
        op_results = sorted(by_op[op_name], key=lambda r: r["items_count"])
        sizes = [r["items_count"] for r in op_results]
        color = OPERATOR_COLORS.get(op_name, "gray")

        seq_tp = [n / t if t > 0 else 0
                  for n, t in zip(sizes, [r["renoir_seq_total_time_s"] for r in op_results])]
        par_tp = [n / t if t > 0 else 0
                  for n, t in zip(sizes, [r["renoir_par_total_time_s"] for r in op_results])]
        gpu_tp = [n / t if t > 0 else 0
                  for n, t in zip(sizes, [r["gpu_total_time_s"] for r in op_results])]

        ax4.loglog(sizes, seq_tp, "o--", color=color, alpha=0.35, markersize=3)
        ax4.loglog(sizes, par_tp, "s--", color=color, alpha=0.35, markersize=3)
        ax4.loglog(sizes, gpu_tp, "^-", color=color, alpha=0.8, markersize=4,
                   label=f"GPU {op_name}")

    ax4.set_xlabel("Number of Elements")
    ax4.set_ylabel("Throughput (elements/s)")
    ax4.legend(loc="upper left", fontsize=7)
    ax4.set_title("Processing Throughput\n(hollow = Seq/Par, solid = GPU)",
                  fontweight="bold", fontsize=10)
    ax4.grid(True, alpha=0.3, which="both")

    # ── Chart 5: Average Speedup by Size Range ───────────────────────────
    ax5 = plt.subplot(2, 3, 5)
    ranges = [
        (0,           100_000,       "Tiny\n(<100K)"),
        (100_000,     1_000_000,     "Small\n(100K\u20131M)"),
        (1_000_000,   10_000_000,    "Med\n(1M\u201310M)"),
        (10_000_000,  100_000_000,   "Large\n(10M\u2013100M)"),
        (100_000_000, 10**15,        "Huge\n(>100M)"),
    ]
    labels, avg_seq, avg_par = [], [], []
    for lo, hi, label in ranges:
        idxs = [i for i, n in enumerate(items_all) if lo <= n < hi]
        if idxs:
            labels.append(label)
            avg_seq.append(float(np.mean([speedups_seq[i] for i in idxs])))
            avg_par.append(float(np.mean([speedups_par[i] for i in idxs])))

    if labels:
        x = np.arange(len(labels))
        width = 0.35
        colors_seq = ["#ff6b6b" if s < 1 else "#51cf66" for s in avg_seq]
        colors_par = ["#ffb3b3" if s < 1 else "#a3e4a3" for s in avg_par]

        bars1 = ax5.bar(x - width / 2, avg_seq, width, label="Seq / GPU",
                        color=colors_seq, alpha=0.9, edgecolor="black")
        bars2 = ax5.bar(x + width / 2, avg_par, width, label="Par / GPU",
                        color=colors_par, alpha=0.9, edgecolor="black")
        ax5.axhline(y=1.0, color="black", linestyle="--", linewidth=2)
        ax5.set_ylabel("Average Speedup (CPU / GPU)")
        ax5.set_title("Average Speedup by Problem Size", fontweight="bold")
        ax5.set_xticks(x)
        ax5.set_xticklabels(labels, fontsize=9)
        ax5.legend(loc="upper left", fontsize=8)
        ax5.grid(True, alpha=0.3, axis="y")

        for bar, val in zip(bars1, avg_seq):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.2f}x", ha="center", va="bottom", fontsize=7,
                     fontweight="bold")
        for bar, val in zip(bars2, avg_par):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.2f}x", ha="center", va="bottom", fontsize=7,
                     fontweight="bold")

    # ── Chart 6: Per-Operator Speedup Distribution (box plot) ────────────
    ax6 = plt.subplot(2, 3, 6)
    box_data_seq, box_data_par, box_labels = [], [], []
    for op_name in all_ops:
        op_results = by_op[op_name]
        box_data_seq.append([r.get("speedup_seq", 1) for r in op_results])
        box_data_par.append([r.get("speedup_par", 1) for r in op_results])
        box_labels.append(op_name.capitalize())

    if box_data_seq:
        positions_seq = np.arange(len(box_labels)) * 2
        positions_par = positions_seq + 0.6

        bp1 = ax6.boxplot(box_data_seq, positions=positions_seq, widths=0.5,
                          patch_artist=True, showfliers=True)
        bp2 = ax6.boxplot(box_data_par, positions=positions_par, widths=0.5,
                          patch_artist=True, showfliers=True)

        for patch in bp1["boxes"]:
            patch.set_facecolor(STRATEGY_COLORS["seq"])
            patch.set_alpha(0.7)
        for patch in bp2["boxes"]:
            patch.set_facecolor(STRATEGY_COLORS["par"])
            patch.set_alpha(0.7)

        ax6.axhline(y=1.0, color="black", linestyle="--", linewidth=2,
                    label="Break-even")
        ax6.set_xticks((positions_seq + positions_par) / 2)
        ax6.set_xticklabels(box_labels, fontsize=10)
        ax6.set_ylabel("Speedup (CPU / GPU)")
        ax6.set_title("Speedup Distribution per Operator\n"
                      "(blue = Seq/GPU, green = Par/GPU)",
                      fontweight="bold", fontsize=10)
        ax6.legend(
            handles=[
                plt.Rectangle((0, 0), 1, 1, fc=STRATEGY_COLORS["seq"], alpha=0.7,
                               label="Seq / GPU"),
                plt.Rectangle((0, 0), 1, 1, fc=STRATEGY_COLORS["par"], alpha=0.7,
                               label="Par / GPU"),
            ],
            loc="upper left", fontsize=8,
        )
        ax6.grid(True, alpha=0.3, axis="y")

    # ── Metadata footer ──────────────────────────────────────────────────
    meta_text = build_metadata_text(results, metadata)
    fig.text(
        0.5, 0.01, meta_text,
        ha="center", va="bottom",
        fontsize=7.5,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                  alpha=0.8, edgecolor="gray"),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # ── Save ─────────────────────────────────────────────────────────────
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if "_benchmark_" in source_name:
        ts_part = source_name.split("_benchmark_")[-1]
        output_file = output_dir / f"plot_reduce_benchmark_{ts_part}.png"
    else:
        output_file = output_dir / f"plot_reduce_benchmark_{source_name}.png"

    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nChart saved to: {output_file}")
    plt.close()

    # ── Terminal summary ─────────────────────────────────────────────────
    print_summary(results, metadata)


# ============================================================================
# Terminal Summary
# ============================================================================

def print_summary(results, metadata):
    """Print a concise summary of the benchmark to the terminal."""
    print("\n" + "=" * 70)
    print("            GPU REDUCE BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Platform:    {metadata.get('platform', 'Unknown')}")
    print(f"Start time:  {metadata.get('start_time', 'Unknown')}")
    print(f"Total tests: {len(results)}")

    items_counts = [r["items_count"] for r in results]
    if items_counts:
        print(f"Size range:  {format_number(min(items_counts))} \u2013 "
              f"{format_number(max(items_counts))}")

    by_op = defaultdict(list)
    for r in results:
        by_op[r.get("operator", "?")].append(r)

    for op_name in sorted(by_op.keys()):
        op_results = by_op[op_name]
        speedups_seq = [r.get("speedup_seq", 1) for r in op_results]
        speedups_par = [r.get("speedup_par", 1) for r in op_results]

        print(f"\n  ── {op_name.upper()} ({len(op_results)} tests) ──")
        print(f"    Speedup Seq/GPU:  min={min(speedups_seq):.3f}x  "
              f"max={max(speedups_seq):.3f}x  "
              f"mean={np.mean(speedups_seq):.3f}x  "
              f"median={np.median(speedups_seq):.3f}x")
        print(f"    Speedup Par/GPU:  min={min(speedups_par):.3f}x  "
              f"max={max(speedups_par):.3f}x  "
              f"mean={np.mean(speedups_par):.3f}x  "
              f"median={np.median(speedups_par):.3f}x")

        gpu_wins_seq = sum(1 for s in speedups_seq if s > 1)
        gpu_wins_par = sum(1 for s in speedups_par if s > 1)
        print(f"    GPU wins (seq):   {gpu_wins_seq}/{len(speedups_seq)} "
              f"({100 * gpu_wins_seq / max(len(speedups_seq), 1):.0f}%)")
        print(f"    GPU wins (par):   {gpu_wins_par}/{len(speedups_par)} "
              f"({100 * gpu_wins_par / max(len(speedups_par), 1):.0f}%)")

        val_ok = sum(1 for r in op_results if r.get("validation_passed", True))
        print(f"    Validation OK:    {val_ok}/{len(op_results)}")

    # Overall crossover estimate (first size where avg GPU speedup > 1)
    unique_sizes = sorted(set(r["items_count"] for r in results))
    for size in unique_sizes:
        size_results = [r for r in results if r["items_count"] == size]
        avg_sp = np.mean([r.get("speedup_seq", 1) for r in size_results])
        if avg_sp > 1.0:
            print(f"\n  GPU becomes faster (avg) at: ~{format_number(size)} elements")
            break

    print("=" * 70)


# ============================================================================
# Entry Point
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python benches/tools/plot_reduce_benchmark.py "
              "<reduce|json_file>")
        print("\nExamples:")
        print("  python benches/tools/plot_reduce_benchmark.py reduce")
        print("  python benches/tools/plot_reduce_benchmark.py "
              "benches/results/reduce/2025-01-15/reduce_benchmark_*.json")
        sys.exit(1)

    arg = sys.argv[1]

    if arg.lower() == "reduce":
        filename = find_most_recent_file()
        if not filename:
            print("Error: no reduce benchmark files found.")
            print("Run the benchmark first with:")
            print("  cargo bench --bench gpu_reduce --features gpu-wgpu")
            sys.exit(1)
        print(f"Using most recent file: {filename}")
    else:
        filename = Path(arg)
        if not filename.exists():
            print(f"Error: file '{filename}' not found")
            sys.exit(1)

    results, metadata = load_results(filename)
    if not results:
        print("No benchmark results found in file!")
        sys.exit(1)

    print(f"Loaded {len(results)} benchmark results from {filename}")

    output_dir = filename.parent
    source_name = filename.stem

    plot_reduce_results(results, metadata, source_name, output_dir)


if __name__ == "__main__":
    main()

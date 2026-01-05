#!/usr/bin/env python3
"""
Monte Carlo Stock Price Path Visualization.

Generates a chart showing multiple stock price paths for Monte Carlo option pricing,
with both ITM (in-the-money) and OTM (out-of-the-money) paths highlighted.

This uses the same algorithm as the Rust implementation for reproducibility.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def xorshift32(state: int) -> tuple[int, float]:
    """xorshift32 RNG - identical to Rust implementation."""
    x = state & 0xFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17) & 0xFFFFFFFF
    x ^= (x << 5) & 0xFFFFFFFF
    # Upper 23 bits for f32
    bits = x >> 9
    u = bits * 1.1920929e-7
    return x, u


def box_muller(state: int) -> tuple[int, float]:
    """Box-Muller transform - identical to Rust implementation."""
    state, u1 = xorshift32(state)
    state, u2 = xorshift32(state)
    u1_safe = max(u1, 1e-10)
    z = np.sqrt(-2.0 * np.log(u1_safe)) * np.cos(6.283185 * u2)
    return state, z


def deterministic_seed(stock: float, strike: float, time: float, rate: float, vol: float) -> int:
    """Generate deterministic seed from input parameters - identical to Rust."""
    def to_bits(f):
        """Convert f32 to its IEEE 754 bit representation."""
        import struct
        return struct.unpack('<I', struct.pack('<f', f))[0]
    
    stock_bits = to_bits(stock)
    strike_bits = to_bits(strike)
    time_bits = to_bits(time)
    rate_bits = to_bits(rate)
    vol_bits = to_bits(vol)
    
    # Hash combining (boost::hash_combine style)
    def combine(seed, value):
        return (seed ^ ((value + 0x9e3779b9 + (seed << 6) + (seed >> 2)) & 0xFFFFFFFF)) & 0xFFFFFFFF
    
    seed = stock_bits
    seed = combine(seed, strike_bits)
    seed = combine(seed, time_bits)
    seed = combine(seed, rate_bits)
    seed = combine(seed, vol_bits)
    
    return seed if seed != 0 else 1


def simulate_path(seed: int, s0: float, r: float, v: float, t: float, num_steps: int) -> tuple[int, list[float]]:
    """Simulate one stock price path."""
    dt = t / num_steps
    drift = (r - 0.5 * v * v) * dt
    diffusion = v * np.sqrt(dt)
    
    prices = [s0]
    s = s0
    
    for _ in range(num_steps):
        seed, z = box_muller(seed)
        s *= np.exp(drift + diffusion * z)
        prices.append(s)
    
    return seed, prices


def generate_path_chart():
    """Generate Monte Carlo path visualization chart."""
    # Option parameters (same as documentation example)
    S0 = 100.0
    K = 105.0
    T = 1.0
    r = 0.05
    v = 0.2
    num_steps = 50
    num_paths = 20
    
    # Get deterministic seed
    seed = deterministic_seed(S0, K, T, r, v)
    print(f"Seed: 0x{seed:08X}")
    
    # Simulate paths
    paths = []
    for _ in range(num_paths):
        seed, prices = simulate_path(seed, S0, r, v, T, num_steps)
        paths.append(prices)
    
    # Classify ITM vs OTM
    itm_paths = [(i, p) for i, p in enumerate(paths) if p[-1] > K]
    otm_paths = [(i, p) for i, p in enumerate(paths) if p[-1] <= K]
    
    print(f"ITM paths: {len(itm_paths)} / {num_paths}")
    print(f"OTM paths: {len(otm_paths)} / {num_paths}")
    
    # Create figure
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    time_axis = np.linspace(0, T, num_steps + 1)
    
    # Plot OTM paths (gray, thin)
    for i, prices in otm_paths:
        ax.plot(time_axis, prices, color='#888888', alpha=0.4, linewidth=1, zorder=1)
    
    # Plot ITM paths (green, thicker)
    for i, prices in itm_paths:
        ax.plot(time_axis, prices, color='#22AA22', alpha=0.7, linewidth=1.5, zorder=2)
    
    # Highlight specific paths
    # Best ITM path
    if itm_paths:
        best_itm_idx = max(range(len(itm_paths)), key=lambda x: itm_paths[x][1][-1])
        i, prices = itm_paths[best_itm_idx]
        ax.plot(time_axis, prices, color='#006600', linewidth=2.5, zorder=3,
                label=f'Path {i+1}: ends ${prices[-1]:.2f} → payoff ${prices[-1]-K:.2f}')
        ax.scatter([T], [prices[-1]], color='#006600', s=80, zorder=4)
    
    # Worst OTM path (first one - the documented one)
    if otm_paths:
        i, prices = otm_paths[0]
        ax.plot(time_axis, prices, color='#CC0000', linewidth=2.5, zorder=3,
                label=f'Path {i+1}: ends ${prices[-1]:.2f} → payoff $0.00')
        ax.scatter([T], [prices[-1]], color='#CC0000', s=80, zorder=4)
    
    # Strike price line
    ax.axhline(y=K, color='black', linestyle='--', linewidth=2, label=f'Strike K = ${K:.0f}')
    
    # Shading for ITM/OTM regions
    ax.axhspan(K, ax.get_ylim()[1] if ax.get_ylim()[1] > K else 180, 
               alpha=0.1, color='green', label='ITM Region')
    ax.axhspan(ax.get_ylim()[0] if ax.get_ylim()[0] < K else 60, K, 
               alpha=0.1, color='red', label='OTM Region')
    
    # Labels and styling
    ax.set_xlabel('Time (years)', fontsize=12)
    ax.set_ylabel('Stock Price ($)', fontsize=12)
    ax.set_title('Monte Carlo Simulation: Stock Price Paths\n'
                 f'S₀=${S0:.0f}, K=${K:.0f}, σ={v*100:.0f}%, r={r*100:.0f}%, T={T:.0f}yr, '
                 f'{num_steps} steps',
                 fontsize=14, fontweight='bold')
    
    # Legend
    ax.legend(loc='upper left', framealpha=0.9)
    
    # Add annotation box
    itm_count = len(itm_paths)
    otm_count = len(otm_paths)
    avg_payoff = sum(max(p[-1] - K, 0) for _, p in itm_paths + otm_paths) / num_paths
    call_price = avg_payoff * np.exp(-r * T)
    
    textstr = '\n'.join([
        f'Paths: {num_paths}',
        f'ITM: {itm_count} ({100*itm_count/num_paths:.0f}%)',
        f'OTM: {otm_count} ({100*otm_count/num_paths:.0f}%)',
        f'Avg payoff: ${avg_payoff:.2f}',
        f'Call price: ${call_price:.2f}',
    ])
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    # Set reasonable y limits
    all_prices = [p for path in paths for p in path]
    y_min = min(all_prices) * 0.9
    y_max = max(all_prices) * 1.1
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, T)
    
    plt.tight_layout()
    
    # Save to docs directory
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent.parent  # benches/tools -> benches -> repo_root
    output_dir = repo_root / 'docs' / 'images'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'monte_carlo_paths.png'
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nChart saved to: {output_file}")
    
    return output_file


if __name__ == '__main__':
    generate_path_chart()

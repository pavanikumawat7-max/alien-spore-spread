"""Person C Deliverable: Containment Chance vs. Laser Delay Experiment.

Simulates 1,000 Monte Carlo futures across varying laser activation delays
(t_delay = 30 to 70 steps) using GPU batch simulation on NVIDIA GTX 1650 Ti.
Captures the phase transition from 100% containment to complete breach.

Writes:
  - results/containment_experiment.csv
  - results/containment_vs_delay.png

Run from project root:
  python scripts/experiment_containment.py
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import spore
from spore import make_grid, SporeParams, INFECTED, TORCH_AVAILABLE


def run_experiment(futures=1000, n_steps=85, delays=None, device="auto", h=60, w=120, wall_col=60):
    if delays is None:
        # High resolution around the phase transition (50 to 64)
        delays = [30, 36, 42, 46, 50, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 68, 72]

    params = SporeParams(spread_prob=0.3, wind_dir=90, wind_strength=0.4, seed=42)
    laser_barrier = [(r, wall_col) for r in range(h)]

    use_gpu = False
    if TORCH_AVAILABLE:
        import torch
        dev = spore.resolve_device(device)
        use_gpu = (dev.type == "cuda")
    else:
        dev = None

    dev_label = torch.cuda.get_device_name(dev) if (use_gpu and torch.cuda.is_available()) else "CPU (NumPy)"
    print(f"===========================================================")
    print(f" Person C: Containment Chance vs. Laser Delay Experiment")
    print(f"===========================================================")
    print(f"  Futures per delay point : {futures:,}")
    print(f"  Total simulations       : {len(delays) * futures:,}")
    print(f"  Grid dimensions         : {h} x {w}")
    print(f"  Laser barrier position  : column {wall_col} (steps 0 to {n_steps})")
    print(f"  Wind dynamics           : 90° (East), strength {params.wind_strength}")
    print(f"  Compute Device          : {dev_label}")
    print(f"-----------------------------------------------------------")

    results = []

    if use_gpu:
        from spore import GpuSimulator
        sim = GpuSimulator(params, device=dev, exact=False)

        def check_contained(g):
            # Check if any cell to the right of the laser barrier is infected
            escaped = (g[:, :, (wall_col + 1):] == INFECTED).any(dim=1).any(dim=1)
            return (~escaped).to(torch.float32)

        t_total_start = time.perf_counter()
        for delay in delays:
            t0 = time.perf_counter()
            sched = {delay: laser_barrier}

            contained_flags = sim.run_batch(
                height=h,
                width=w,
                batch=futures,
                n_steps=n_steps,
                seed_cells=[(h // 2, 5)],
                laser_schedule=sched,
                reduce=check_contained,
                seed=42 + delay
            )
            elapsed = time.perf_counter() - t0
            p_contain = float(np.mean(contained_flags)) * 100.0
            
            # 95% Wilson Score Confidence Interval
            n = futures
            p = p_contain / 100.0
            z = 1.96
            ci = z * np.sqrt(p * (1 - p) / n) * 100.0 if n > 0 else 0.0

            results.append({
                "delay": delay,
                "containment_pct": p_contain,
                "ci_95": ci,
                "futures": futures,
                "elapsed_sec": elapsed
            })
            print(f"  Delay: {delay:2d} steps | Containment: {p_contain:5.1f}% (±{ci:4.1f}%) | Time: {elapsed:.3f}s")

        total_time = time.perf_counter() - t_total_start
        print(f"-----------------------------------------------------------")
        print(f"  Total GPU Execution Time: {total_time:.2f}s ({len(delays)*futures/total_time:,.0f} sims/sec)")

    else:
        # Fallback NumPy
        t_total_start = time.perf_counter()
        for delay in delays:
            t0 = time.perf_counter()
            sched = {delay: laser_barrier}
            g = make_grid(h, w, batch=futures, seed_cells=[(h // 2, 5)], seed=42 + delay)
            out = spore.simulation.run(g, params, n_steps=n_steps, laser_schedule=sched)
            escaped = (out[:, :, (wall_col + 1):] == INFECTED).any(axis=(1, 2))
            contained_flags = ~escaped
            elapsed = time.perf_counter() - t0
            p_contain = float(np.mean(contained_flags)) * 100.0
            p = p_contain / 100.0
            ci = 1.96 * np.sqrt(p * (1 - p) / futures) * 100.0

            results.append({
                "delay": delay,
                "containment_pct": p_contain,
                "ci_95": ci,
                "futures": futures,
                "elapsed_sec": elapsed
            })
            print(f"  Delay: {delay:2d} steps | Containment: {p_contain:5.1f}% (±{ci:4.1f}%) | Time: {elapsed:.3f}s")
        total_time = time.perf_counter() - t_total_start

    print(f"-----------------------------------------------------------")

    # Save CSV
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", "containment_experiment.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["delay", "containment_pct", "ci_95", "futures", "elapsed_sec"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved numerical data to: {csv_path}")

    # Plot
    plot_path = os.path.join("results", "containment_vs_delay.png")
    plot_containment_curve(results, plot_path, dev_label)
    print(f"Saved visualization to:  {plot_path}")
    print(f"===========================================================\n")
    return results


def plot_containment_curve(results, out_path, dev_label):
    delays = [r["delay"] for r in results]
    probs = [r["containment_pct"] for r in results]
    cis = [r["ci_95"] for r in results]

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=200)
    
    # Modern dark theme matching live demo
    fig.patch.set_facecolor("#0f0e17")
    ax.set_facecolor("#181724")
    ax.grid(True, linestyle="--", alpha=0.25, color="#6b6882")

    probs_arr = np.array(probs)
    cis_arr = np.array(cis)

    # 95% Confidence Band
    ax.fill_between(delays, np.clip(probs_arr - cis_arr, 0, 100), np.clip(probs_arr + cis_arr, 0, 100),
                    color="#ff4500", alpha=0.20, label="95% Wilson Confidence Band")

    # Containment Curve
    ax.plot(delays, probs, color="#ff4500", lw=3.0, marker="o", markersize=6.5,
            markerfacecolor="#39ff14", markeredgecolor="#ffffff", markeredgewidth=1.2,
            label=f"Containment Rate (N = {results[0]['futures']:,} futures/point)")

    # 50% Threshold line
    ax.axhline(50, color="#7fd4ff", linestyle=":", lw=1.6, alpha=0.85, label="50% Critical Failure Threshold")
    
    # Interpolate exact 50% crossover
    crossover_delay = None
    for i in range(len(probs) - 1):
        if probs[i] >= 50 and probs[i+1] <= 50:
            d1, d2 = delays[i], delays[i+1]
            p1, p2 = probs[i], probs[i+1]
            crossover_delay = d1 + (50 - p1) * (d2 - d1) / (p2 - p1)
            break
            
    if crossover_delay is not None:
        ax.axvline(crossover_delay, color="#7fd4ff", linestyle="--", lw=1.8, alpha=0.8)
        ax.annotate(f"Critical Delay: $t_{{50\\%}} \\approx {crossover_delay:.1f}$ steps\n(Phase Transition)",
                    xy=(crossover_delay, 50),
                    xytext=(crossover_delay - 14, 25),
                    arrowprops=dict(facecolor="#7fd4ff", shrink=0.08, width=1.8, headwidth=8),
                    color="#ffffff", fontsize=10.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.6", facecolor="#242038", edgecolor="#7fd4ff", lw=1.4))

    # Highlight operational regimes
    ax.axvspan(min(delays), 54, color="#39ff14", alpha=0.06, label="Safe Window (100% Contained)")
    ax.axvspan(54, 60, color="#f4a261", alpha=0.12, label="Phase Transition (Stochastic Risk)")
    ax.axvspan(60, max(delays), color="#e63946", alpha=0.07, label="Breach Window (< 5% Contained)")

    ax.set_title("Alien Spore Containment Probability vs. Laser Barrier Response Delay",
                 color="#ffffff", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Laser Barrier Activation Delay ($t_{\\mathrm{delay}}$ in simulation steps)",
                  color="#e0e0e0", fontsize=11, labelpad=10)
    ax.set_ylabel("Containment Success Rate (%)", color="#e0e0e0", fontsize=11, labelpad=10)

    ax.set_xlim(min(delays), max(delays))
    ax.set_ylim(-3, 105)
    ax.tick_params(colors="#e0e0e0", labelsize=10)
    
    for spine in ax.spines.values():
        spine.set_color("#4a4760")

    ax.legend(facecolor="#242038", edgecolor="#4a4760", labelcolor="#ffffff",
              loc="lower left", fontsize=9.5, framealpha=0.92)

    fig.text(0.99, 0.015, f"Engine: PyTorch CUDA | Device: {dev_label} | Grid: 60x120 | Wind: 90° East (0.4)",
             color="#8c88a8", fontsize=8.5, ha="right", va="bottom")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Containment vs Laser Delay Experiment")
    parser.add_argument("--futures", type=int, default=1000, help="Futures per delay point (default: 1000)")
    parser.add_argument("--steps", type=int, default=85, help="Simulation steps (default: 85)")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, cpu)")
    args = parser.parse_args()

    run_experiment(futures=args.futures, n_steps=args.steps, device=args.device)

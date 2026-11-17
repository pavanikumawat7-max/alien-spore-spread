"""Benchmark: Python loops vs vectorised NumPy vs PyTorch (GPU).

Run from project root:  python scripts/benchmark.py
Options:  --sizes 64 128 256 512 1024 2000   --batches 1 10 100   --steps 10 --repeats 3
          --max-cells 3e7  (skip combos with more cells than this)   --quick   --device auto|cuda|cpu

Writes results/benchmark.csv, results/benchmark_time.png, results/benchmark_speedup.png,
results/benchmark_batch.png. Without PyTorch only the CPU columns are produced.
If no CUDA GPU is found, the 'gpu' columns are torch running on the CPU and are labelled so.

What is timed (ms per simulation step, median of repeats, after warm-up):
  loop      pure Python loops (only for small grids, it is very slow)
  numpy     vectorised NumPy step()
  gpu_fast  float32 + device RNG, data already on the device (kernel time)
  gpu_exact float64 + host RNG (what the validation mode costs)
  gpu_e2e   a whole `steps`-step run from NumPy input to NumPy output, i.e. including
            host->device and device->host transfers, divided by steps
  h2d / d2h one-off transfer times for the grid
"""
import argparse, csv, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import spore
from spore import make_grid, SporeParams, step, step_loop

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2000])
ap.add_argument("--batches", type=int, nargs="+", default=[1, 10, 100])
ap.add_argument("--steps", type=int, default=10)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--max-cells", type=float, default=3e7)
ap.add_argument("--loop-max-cells", type=float, default=2e5, help="largest grid (cells x batch) the loop baseline runs on")
ap.add_argument("--device", default="auto")
ap.add_argument("--quick", action="store_true", help="tiny run to check the script works")
args = ap.parse_args()
if args.quick:
    args.sizes, args.batches, args.steps, args.repeats = [32, 64], [1, 4], 3, 2

KW = dict(spread_prob=0.3, wind_dir=90, wind_strength=0.4)
HAVE_TORCH = spore.TORCH_AVAILABLE
if HAVE_TORCH:
    import torch
    dev = spore.resolve_device(args.device)
    dev_label = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "torch-on-CPU (NO GPU FOUND)"
    sync = lambda: spore.synchronize(dev)
else:
    dev, dev_label, sync = None, "none (PyTorch not installed)", (lambda: None)
print("GPU device:", dev_label)
if HAVE_TORCH and dev.type != "cuda":
    print("WARNING: no CUDA GPU, so the gpu_* columns are NOT GPU results. Run on a GPU machine for the real numbers.")


def median_ms(fn, repeats, per=1):
    ts = []
    for _ in range(repeats):
        sync(); t0 = time.perf_counter(); fn(); sync()
        ts.append((time.perf_counter() - t0) / per * 1e3)
    return float(np.median(ts))


rows = []
for n in args.sizes:
    for b in args.batches:
        cells = n * n * b
        if cells > args.max_cells:
            print(f"skip {n}x{n} batch {b}: {cells:.1e} cells > --max-cells"); continue
        g0 = make_grid(n, n, batch=b, n_seeds=max(1, n * n // 1000), seed=0)
        row = dict(size=n, batch=b, cells=cells, device=dev_label)

        # numpy
        p = SporeParams(seed=1, **KW); g = g0
        for _ in range(2): g = step(g, p)                                   # warm-up
        def np_run(): 
            gg = g0
            for _ in range(args.steps): gg = step(gg, p)
        row["numpy_ms"] = median_ms(np_run, args.repeats, args.steps)

        # python loops (small only)
        if cells <= args.loop_max_cells:
            pl = SporeParams(seed=1, **KW)
            def loop_run(): step_loop(g0, pl)
            row["loop_ms"] = median_ms(loop_run, max(1, args.repeats - 1))

        # gpu
        if HAVE_TORCH:
            for name, exact in (("gpu_fast_ms", False), ("gpu_exact_ms", True)):
                sim = spore.GpuSimulator(SporeParams(seed=1, **KW), device=dev, exact=exact)
                t = sim.to_device(g0)
                for _ in range(3): t = sim.step(t)                           # warm-up (CUDA init, allocator)
                def gpu_run():
                    tt = t
                    for _ in range(args.steps): tt = sim.step(tt)
                row[name] = median_ms(gpu_run, args.repeats, args.steps)
            sim = spore.GpuSimulator(SporeParams(seed=1, **KW), device=dev, exact=False)
            row["h2d_ms"] = median_ms(lambda: sim.to_device(g0), args.repeats)
            t = sim.to_device(g0)
            row["d2h_ms"] = median_ms(lambda: sim.to_host(t), args.repeats)
            row["gpu_e2e_ms"] = median_ms(lambda: sim.run(g0, args.steps), args.repeats, args.steps)
            row["speedup_fast_vs_numpy"] = row["numpy_ms"] / row["gpu_fast_ms"]
            row["speedup_e2e_vs_numpy"] = row["numpy_ms"] / row["gpu_e2e_ms"]
            del t, sim
            if dev.type == "cuda": torch.cuda.empty_cache()
        if "loop_ms" in row: row["speedup_numpy_vs_loop"] = row["loop_ms"] / row["numpy_ms"]
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "device"})

os.makedirs("results", exist_ok=True)
cols = ["size", "batch", "cells", "device", "loop_ms", "numpy_ms", "gpu_fast_ms", "gpu_exact_ms", "gpu_e2e_ms",
        "h2d_ms", "d2h_ms", "speedup_numpy_vs_loop", "speedup_fast_vs_numpy", "speedup_e2e_vs_numpy"]
with open("results/benchmark.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=cols); wr.writeheader(); wr.writerows(rows)
print("saved results/benchmark.csv")

# ----------------------------------------------------------------- graphs
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def series(key, batch):
    xs = [r["size"] for r in rows if r["batch"] == batch and key in r]
    ys = [r[key] for r in rows if r["batch"] == batch and key in r]
    return xs, ys

batches = sorted({r["batch"] for r in rows})
styles = [("loop_ms", "Python loops", "tab:red"), ("numpy_ms", "NumPy (vectorised)", "tab:blue"),
          ("gpu_fast_ms", "GPU kernel (float32)", "tab:green"), ("gpu_e2e_ms", "GPU incl. transfers", "tab:olive")]

fig, axes = plt.subplots(1, len(batches), figsize=(5 * len(batches), 4.2), squeeze=False)
for ax, b in zip(axes[0], batches):
    for key, label, col in styles:
        xs, ys = series(key, b)
        if xs: ax.plot(xs, ys, "o-", label=label, color=col)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("grid side (cells)"); ax.set_ylabel("ms per step"); ax.set_title(f"batch = {b}"); ax.grid(alpha=.3)
axes[0, 0].legend(fontsize=8)
fig.suptitle(f"Time per simulation step ({dev_label})"); fig.tight_layout(); fig.savefig("results/benchmark_time.png", dpi=110)

if HAVE_TORCH:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for b in batches:
        xs, ys = series("speedup_fast_vs_numpy", b)
        if xs: ax.plot(xs, ys, "o-", label=f"batch {b}")
    ax.axhline(1, color="k", lw=.8, ls="--")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("grid side (cells)"); ax.set_ylabel("speedup over NumPy (x)"); ax.grid(alpha=.3); ax.legend()
    ax.set_title(f"GPU kernel speedup vs NumPy ({dev_label})")
    fig.tight_layout(); fig.savefig("results/benchmark_speedup.png", dpi=110)

    sizes = sorted({r["size"] for r in rows})
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    n_ref = sizes[len(sizes) // 2]
    for key, label, col in styles[1:]:
        xs = [r["batch"] for r in rows if r["size"] == n_ref and key in r]
        ys = [r["cells"] / (r[key] / 1e3) / 1e6 for r in rows if r["size"] == n_ref and key in r]
        if xs: ax.plot(xs, ys, "o-", label=label, color=col)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("batch size (simulations at once)"); ax.set_ylabel("million cell-updates / s")
    ax.set_title(f"Throughput vs batch, {n_ref}x{n_ref} grid"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("results/benchmark_batch.png", dpi=110)
print("saved graphs in results/")

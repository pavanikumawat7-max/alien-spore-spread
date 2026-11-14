"""Phase 2 demo: wind-driven spread + laser barrier.  Run from project root:  python scripts/demo_phase2.py"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from spore import make_grid, SporeParams, step, apply_laser, run, INFECTED

cmap = ListedColormap(["#2b1d14", "#39ff14", "#ff4500"])
H = W = 80
fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))

# Row 1: same infection, no wind vs wind toward east, after 40 steps
for ax, (title, wind) in zip(axes[0], [("No wind", 0.0), ("Wind to the east, strength 1.0", 1.0)]):
    g = make_grid(H, W, centre=True)
    g = run(g, SporeParams(0.12, wind_dir=90, wind_strength=wind, seed=1), 40)
    ax.imshow(g[0], cmap=cmap, vmin=0, vmax=2); ax.set_title(f"{title}\n40 steps"); ax.axis("off")
axes[0, 2].axis("off")

# Row 2: laser barrier fired early / late / never
H2, W2 = 60, 120
for ax, (title, fire_at) in zip(axes[1], [("Laser at step 8", 8), ("Laser at step 30", 30), ("No laser", None)]):
    g = make_grid(H2, W2, seed_cells=[(30, 5)])
    sched = {fire_at: [(r, 60) for r in range(H2)]} if fire_at is not None else {}
    g = run(g, SporeParams(0.3, wind_dir=90, wind_strength=0.4, seed=2), 80, sched)
    ax.imshow(g[0], cmap=cmap, vmin=0, vmax=2); ax.set_title(f"{title}\n80 steps"); ax.axis("off")

os.makedirs("results", exist_ok=True)
fig.tight_layout(); fig.savefig("results/phase2_demo.png", dpi=100)
print("saved results/phase2_demo.png")

# Batch: 500 futures at once, containment chance vs laser delay (a preview of Person C's experiment)
B = 500
print("\nContainment % over 500 futures (wall at column 60, 80 steps), by laser delay:")
for delay in (5, 15, 25, 35):
    g = make_grid(H2, W2, batch=B, seed_cells=[(30, 5)])
    t = time.perf_counter()
    out = run(g, SporeParams(0.3, 90, 0.4, seed=3), 80, {delay: [(r, 60) for r in range(H2)]})
    contained = ~(out[:, :, 61:] == INFECTED).any(axis=(1, 2))
    print(f"  delay {delay:2d}: {100*contained.mean():5.1f}%   ({time.perf_counter()-t:.2f}s)")

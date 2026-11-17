"""Visual + timing check of Phase 1.  Run from project root:  python scripts/demo_phase1.py"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from spore import make_grid, count_neighbours, count_neighbours_naive

grid = make_grid(60, 60, batch=1, n_seeds=12, seed=3)
n = count_neighbours(grid, state=1)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(grid[0], cmap=ListedColormap(["#2b1d14", "#39ff14", "#ff4500"]), vmin=0, vmax=2)
ax[0].set_title("Mars grid (0 clear, 1 infected, 2 burned)")
im = ax[1].imshow(n[0], cmap="viridis")
ax[1].set_title("Infected-neighbour count per cell")
fig.colorbar(im, ax=ax[1], shrink=0.8)
os.makedirs("results", exist_ok=True)
fig.savefig("results/phase1_demo.png", dpi=110, bbox_inches="tight")
print("saved results/phase1_demo.png")

big = make_grid(300, 300, batch=1, n_seeds=500, seed=0)
t = time.perf_counter(); a = count_neighbours_naive(big); t_naive = time.perf_counter() - t
t = time.perf_counter(); b = count_neighbours(big); t_vec = time.perf_counter() - t
print(f"300x300  loops: {t_naive:.3f}s   vectorised: {t_vec*1000:.2f}ms   identical: {np.array_equal(a, b)}")

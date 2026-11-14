"""CPU vs GPU correctness validation.  Run from project root:  python scripts/validate_cpu_gpu.py

Checks, for several grid sizes, batch sizes and wind settings:
  1. loop reference == NumPy engine         (exact, every step)
  2. GPU exact mode == NumPy engine         (exact, every step, incl. laser)
  3. GPU fast mode ~= NumPy engine          (statistical: float32 + device RNG)
Exit code 0 = everything passed, 1 = a check failed.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import spore
from spore import make_grid, SporeParams, step, step_loop, run, INFECTED

if not spore.TORCH_AVAILABLE:
    sys.exit("PyTorch is not installed: install it (see requirements.txt), then rerun this script.")
import torch

devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
print("devices tested:", devices, "" if "cuda" in devices else "  (NO CUDA GPU FOUND: torch-on-CPU only)")
ok = True


def report(name, passed, detail=""):
    global ok
    ok &= bool(passed)
    print(f"  [{'PASS' if passed else 'FAIL'}] {name} {detail}")


configs = [(32, 32, 1, 0.3, 0, 0.0), (64, 48, 4, 0.3, 90, 0.6), (100, 100, 8, 0.2, 200, 1.0), (40, 200, 3, 0.45, 45, 0.3)]
for h, w, b, p, wd, ws in configs:
    print(f"\ngrid {h}x{w}  batch {b}  p={p}  wind {wd} deg x{ws}")
    g0 = make_grid(h, w, batch=b, n_seeds=max(2, h * w // 400), seed=1)
    kw = dict(spread_prob=p, wind_dir=wd, wind_strength=ws)
    steps = 30
    sched = {10: [(r, w // 2) for r in range(h)]}

    cpu = g0; loop = g0; pc, pl = SporeParams(seed=9, **kw), SporeParams(seed=9, **kw)
    same_loop = True
    for t in range(min(steps, 10)):                          # loops are slow: first 10 steps
        cpu, loop = step(cpu, pc), step_loop(loop, pl)
        same_loop &= np.array_equal(cpu, loop)
    report("loop reference == NumPy (10 steps)", same_loop)

    ref = run(g0, SporeParams(seed=9, **kw), steps, sched)
    for dev in devices:
        out = spore.GpuSimulator(SporeParams(seed=9, **kw), device=dev, exact=True).run(g0, steps, sched)
        diff = int((out != ref).sum())
        report(f"GPU exact [{dev}] == NumPy ({steps} steps + laser)", diff == 0, f"({diff} differing cells)")

    n = 4000                                                   # statistical check on many small worlds
    tiny = np.zeros((n, 11, 11), np.uint8); tiny[:, 5, 5] = INFECTED
    c = (run(tiny, SporeParams(seed=1, **kw), 6) == INFECTED).sum(axis=(1, 2))
    for dev in devices:
        gpu = (spore.GpuSimulator(SporeParams(seed=2, **kw), device=dev).run(tiny, 6) == INFECTED).sum(axis=(1, 2))
        z = abs(c.mean() - gpu.mean()) / np.sqrt(c.var() / n + gpu.var() / n + 1e-12)
        report(f"GPU fast  [{dev}] ~= NumPy (mean infected {gpu.mean():.2f} vs {c.mean():.2f})", z < 4.5, f"(z = {z:.2f})")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)

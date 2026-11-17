# Alien Spore-Spread: GPU-Accelerated Reaction-Diffusion Matrix

Parallel Programming course project. Phase 1 = foundation (setup, grid, neighbours). Phase 2 = simulation logic (spread, wind, step, laser). Phase 3 = PyTorch GPU engine, batch simulations, CPU/GPU validation, benchmarks.

## Setup
```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # for GPU use the CUDA build of torch from pytorch.org
python -m unittest discover -s tests -v
python scripts/demo_phase1.py    # writes results/phase1_demo.png
python scripts/demo_phase2.py    # writes results/phase2_demo.png, prints containment table
python scripts/validate_cpu_gpu.py   # Phase 3: CPU vs GPU correctness (needs torch)
python scripts/benchmark.py          # Phase 3: loops vs NumPy vs GPU (use --quick to smoke-test)
```

## Structure
```
spore/grid.py        make_grid(), validate_grid(), CLEAR/INFECTED/BURNED
spore/neighbours.py  count_neighbours() (vectorised), count_neighbours_naive() (loops), neighbour_coords(), overlap_slices()
spore/params.py      SporeParams (spread_prob, wind_dir, wind_strength, seed)
spore/simulation.py  wind_weights(), neighbour_probs(), spread_probability(), step(), apply_laser(), run()
spore/reference.py   step_loop(): pure-Python loop step (slowest baseline + independent check)
spore/gpu.py         GpuSimulator: PyTorch engine (step, apply_laser, run, run_batch)
tests/               51 unit tests (stdlib unittest, pytest-compatible); GPU tests skip if torch is missing
scripts/             demo_phase1.py, demo_phase2.py, validate_cpu_gpu.py, benchmark.py
results/             graphs and images (Person C fills this later)
```

## Agreed interfaces
- Cell values: 0 clear, 1 infected, 2 burned
- Grid shape: (batch, height, width), uint8, batch = 1 for the live demo
- `step(grid, params) -> new grid` and `apply_laser(grid, cells) -> new grid` (both return copies, never modify the input)

## Phase 2 model
- Each infected neighbour independently infects a clear cell with p = spread_prob x wind_weight, so
  P(infected) = 1 - prod(1 - p_k). With no wind and n infected neighbours: 1 - (1 - p)^n.
- Wind: `wind_dir` is the compass direction the wind blows toward (0 N, 90 E, 180 S, 270 W);
  weight = 1 + wind_strength x cos(angle between spore travel and wind). Straight downwind x(1+s), upwind x(1-s).
- Infected cells stay infected until the laser burns them (no recovery, no self burn-out). Burned cells stay burned
  and cannot be re-infected, so a burned line is a firebreak.
- Randomness: `SporeParams(seed=...)` owns a NumPy Generator that advances each step; `params.reset_rng()` replays a run.
- `apply_laser(grid, [(r, c), ...])` hits every batch item; `[(b, r, c), ...]` hits one.

## Design decisions
- **uint8 grids**: values are 0..2, so 1 byte per cell. 2000x2000 x 1000 batch = 4 GB, not 16 GB+.
- **Non-wrapping edges**: Mars is flat; edge cells have 5 neighbours, corners 3.
- **Two neighbour implementations**: the loop version is the ground truth and the "CPU loop" benchmark baseline; the vectorised shifted-slice version is checked against it and is the one `step()` will use. It uses only slicing and `+=`, so the PyTorch port is a near copy.
- **Seeded RNG** (`numpy.random.default_rng`): same seed, same grid, needed for the CPU vs GPU identity check.

## Phase 3: GPU engine
```python
from spore import make_grid, SporeParams, GpuSimulator
sim = GpuSimulator(SporeParams(0.3, wind_dir=90, wind_strength=0.4, seed=1), device="auto", exact=False)
final = sim.run(make_grid(512, 512, batch=100, n_seeds=20, seed=0), n_steps=100)    # numpy in, numpy out
t = sim.to_device(grid); t = sim.step(t)                                            # stay on the GPU (fastest)
counts = sim.run_batch(1000, 1000, batch=1000, n_steps=80, seed_cells=[(500, 5)],   # 1,000 futures, chunked to fit memory
                       laser_schedule={20: [(r, 600) for r in range(1000)]})
```
- Same spread rule as the CPU, written as tensor ops, so all cells of all simulations update in parallel. Phase 1-2 code works without torch.
- **`exact=True`**: float64, random numbers drawn on the host with NumPy and copied over. Bit-for-bit identical to the NumPy engine for the same seed. Use it to validate (slower).
- **`exact=False`** (default): float32, random numbers generated on the device. Same distribution, different random stream, so it matches the CPU statistically, not cell by cell. Use it for benchmarks and the live demo.
- `run_batch` splits large batches into chunks sized from free device memory and returns only a small per-simulation result (default: infected count; pass `reduce=` for e.g. "did it cross column X").
- `step_loop` (pure Python) is checked against NumPy, and NumPy against the GPU, giving a three-way validation chain.
- Device is `auto` (CUDA if available, else CPU); the engine also runs on CPU so the tests work without a GPU. Apple MPS has no float64, so `exact=True` is not supported there.
- GPU timings: CUDA is asynchronous, so the benchmark calls `torch.cuda.synchronize()` around every timer and does warm-up steps first.

"""Phase 3: PyTorch engine (runs on a CUDA GPU, or on the CPU for testing).

It is the same spread rule as spore.simulation.step(), written with tensor
operations so every cell of every simulation is updated in parallel on the device.

Two modes (choose with `exact`):

  exact=True   float64 + random numbers drawn on the HOST with NumPy (params.rng)
               and copied to the device. Bit-for-bit identical to the NumPy CPU
               engine for the same seed. Slower (float64 + a transfer every step);
               used for CORRECTNESS VALIDATION.
  exact=False  float32 + random numbers generated directly on the device
               (torch.Generator). Same distribution as the CPU engine, but a
               different random stream, so grids match only statistically.
               This is the fast mode for benchmarks and the live demo.
"""
import numpy as np
import torch

from .grid import CLEAR, INFECTED, BURNED, validate_grid, make_grid
from .neighbours import OFFSETS_8, overlap_slices
from .simulation import neighbour_probs

# step() relies on 0 -> 1 being "add one" for newly infected cells.
assert CLEAR == 0 and INFECTED == 1 and BURNED == 2


def resolve_device(device="auto"):
    """'auto' -> CUDA GPU if available, otherwise CPU. Also accepts 'cuda', 'cuda:1', 'cpu'."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def synchronize(device):
    """Wait for queued GPU work (CUDA launches are asynchronous). No-op on CPU.
    Always call before reading a timer."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class GpuSimulator:
    """Runs the spore simulation on a torch device.

    sim = GpuSimulator(SporeParams(0.3, 90, 0.4, seed=1), device="auto", exact=False)
    final = sim.run(grid_numpy, n_steps=50)           # numpy in, numpy out
    t = sim.to_device(grid_numpy); t = sim.step(t)    # keep data on the device (fastest)
    """

    def __init__(self, params, device="auto", exact=False):
        self.params = params
        self.device = resolve_device(device)
        self.exact = exact
        self.dtype = torch.float64 if exact else torch.float32
        self._gen = None
        self.reset()

    # ---------------------------------------------------------------- RNG
    def reset(self):
        """Restart the random stream so a run can be replayed from params.seed."""
        self.params.reset_rng()
        self._gen = torch.Generator(device=self.device)
        if self.params.seed is None:
            self._gen.seed()
        else:
            self._gen.manual_seed(int(self.params.seed))

    def _roll(self, shape):
        shape = tuple(shape)
        if self.exact:                                   # same numbers the CPU engine draws
            return torch.from_numpy(self.params.rng.random(shape)).to(self.device)
        return torch.rand(shape, generator=self._gen, device=self.device, dtype=self.dtype)

    # ---------------------------------------------------------- transfers
    def to_device(self, grid):
        """NumPy grid (batch, h, w) -> uint8 tensor on the device (always a copy)."""
        validate_grid(grid)
        return torch.from_numpy(np.ascontiguousarray(grid, dtype=np.uint8)).to(self.device, copy=True)

    @staticmethod
    def to_host(tensor):
        """Device tensor -> NumPy array (blocks until the GPU is done)."""
        return np.array(tensor.detach().cpu().numpy())

    # ----------------------------------------------------------- the model
    def spread_probability(self, grid):
        """Tensor (batch, h, w): chance each cell is infected this step (same maths as the CPU)."""
        h, w = grid.shape[-2:]
        probs = [float(p) for p in neighbour_probs(self.params)]   # re-read each step: wind can change live
        infected = (grid == INFECTED).to(self.dtype)
        survive = torch.ones(grid.shape, dtype=self.dtype, device=grid.device)
        for (dr, dc), p in zip(OFFSETS_8, probs):
            r_dst, c_dst, r_src, c_src = overlap_slices(dr, dc, h, w)
            survive[..., r_dst, c_dst] *= 1.0 - p * infected[..., r_src, c_src]
        return 1.0 - survive

    def step(self, grid):
        """One time step for the whole batch. Returns a NEW tensor (input untouched)."""
        p_infect = self.spread_probability(grid)
        roll = self._roll(grid.shape)
        newly = (grid == CLEAR) & (roll < p_infect)
        return grid + newly.to(torch.uint8)               # 0 -> 1 only where newly infected

    def apply_laser(self, grid, cells):
        """Burn target cells. cells: [(row, col), ...] for every batch item, or [(batch, row, col), ...]."""
        _, h, w = grid.shape
        two, three = [], []
        for cell in cells:
            r, c = (cell[0], cell[1]) if len(cell) == 2 else (cell[1], cell[2])
            if not (0 <= r < h and 0 <= c < w):
                raise ValueError(f"laser cell {(r, c)} is outside the grid")
            (two if len(cell) == 2 else three).append(tuple(cell))
        new = grid.clone()
        if two:
            idx = torch.tensor(two, dtype=torch.long, device=grid.device)
            new[:, idx[:, 0], idx[:, 1]] = BURNED
        if three:
            idx = torch.tensor(three, dtype=torch.long, device=grid.device)
            new[idx[:, 0], idx[:, 1], idx[:, 2]] = BURNED
        return new

    def run(self, grid, n_steps, laser_schedule=None, return_numpy=True):
        """Run n_steps (same laser_schedule semantics as spore.run). Accepts a NumPy grid or a tensor."""
        laser_schedule = laser_schedule or {}
        g = self.to_device(grid) if isinstance(grid, np.ndarray) else grid
        for t in range(n_steps):
            if t in laser_schedule:
                g = self.apply_laser(g, laser_schedule[t])
            g = self.step(g)
        return self.to_host(g) if return_numpy else g

    # ------------------------------------------------ large batch simulations
    def auto_chunk(self, height, width, batch):
        """How many simulations fit on the device at once (rough memory model, 50% of free memory)."""
        isize = 8 if self.exact else 4
        bytes_per_cell = 5 * isize + 4                    # float temporaries + grids + bool masks
        if self.device.type == "cuda":
            free = torch.cuda.mem_get_info(self.device)[0]
            budget = 0.5 * free
        else:
            budget = 2e9
        return int(max(1, min(batch, budget // (height * width * bytes_per_cell))))

    def run_batch(self, height, width, batch, n_steps, n_seeds=1, seed=None, seed_cells=None,
                  laser_schedule=None, reduce=None, chunk=None):
        """Run `batch` independent simulations (e.g. the '1,000 futures' button).

        Simulations are processed in chunks that fit in device memory; only the
        small per-simulation result goes back to the host, not the grids.

        reduce : function(grid_tensor) -> 1-D tensor with one number per simulation.
                 Default: number of infected cells at the end.
        Returns a NumPy array of length `batch`.
        Note: with chunking the random numbers are drawn chunk by chunk, so results
        are statistically (not bit-for-bit) equal to one big batch.
        """
        if reduce is None:
            reduce = lambda g: (g == INFECTED).sum(dim=(1, 2))
        chunk = chunk or self.auto_chunk(height, width, batch)
        out, done, k = [], 0, 0
        while done < batch:
            n = min(chunk, batch - done)
            grid = make_grid(height, width, batch=n, n_seeds=n_seeds,
                             seed=None if seed is None else seed + k, seed_cells=seed_cells)
            final = self.run(grid, n_steps, laser_schedule, return_numpy=False)
            out.append(reduce(final).detach().cpu())
            done += n
            k += 1
        return torch.cat(out).numpy()

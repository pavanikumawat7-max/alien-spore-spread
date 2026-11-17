"""Grid generation.

Agreed project interface (do not change without telling the team):
    cell values : 0 = clear, 1 = infected (weed), 2 = burned
    grid shape  : (batch, height, width), batch = 1 for the live demo
    dtype       : uint8 (tiny, fast on GPU, and plenty for values 0..2)
"""
import numpy as np

CLEAR, INFECTED, BURNED = 0, 1, 2
GRID_DTYPE = np.uint8


def make_grid(height=200, width=200, batch=1, n_seeds=1, seed=None, centre=False,
              seed_cells=None):
    """Create a Mars grid of shape (batch, height, width) with initial infection.

    Parameters
    ----------
    n_seeds : number of initially infected cells per simulation.
    seed    : integer RNG seed -> identical grids every run (needed for CPU/GPU checks).
    centre  : if True, put a single infection in the middle (useful for demos/tests).
    seed_cells : explicit list of (row, col) to infect in every batch item;
                 overrides n_seeds/centre.
    """
    if min(height, width, batch) < 1:
        raise ValueError("height, width and batch must all be >= 1")

    grid = np.zeros((batch, height, width), dtype=GRID_DTYPE)

    if seed_cells is not None:
        for r, c in seed_cells:
            if not (0 <= r < height and 0 <= c < width):
                raise ValueError(f"seed cell {(r, c)} is outside the grid")
            grid[:, r, c] = INFECTED
    elif centre:
        grid[:, height // 2, width // 2] = INFECTED
    else:
        if n_seeds > height * width:
            raise ValueError("n_seeds exceeds number of cells")
        rng = np.random.default_rng(seed)
        for b in range(batch):  # each simulation gets its own random landing site
            flat = rng.choice(height * width, size=n_seeds, replace=False)
            grid[b].flat[flat] = INFECTED
    return grid


def validate_grid(grid):
    """Raise ValueError unless the grid obeys the agreed interface."""
    if grid.ndim != 3:
        raise ValueError(f"grid must be 3-D (batch, height, width), got {grid.ndim}-D")
    if grid.shape[0] < 1:
        raise ValueError("batch dimension must be >= 1")
    if int(grid.min()) < CLEAR or int(grid.max()) > BURNED:
        raise ValueError("grid contains values outside {0, 1, 2}")
    return grid

"""Pure-Python loop version of step(): the slowest baseline ("CPU loops") and a
third, independent implementation for validation.

It draws its random numbers exactly like step() does (one params.rng.random(shape)
per step, then compares cell by cell), so with the same seed it must produce
the same grid as the NumPy and GPU engines.
"""
from .grid import CLEAR, INFECTED, validate_grid
from .neighbours import OFFSETS_8
from .simulation import neighbour_probs


def step_loop(grid, params):
    """One step using plain Python loops over batch, rows, columns and neighbours."""
    validate_grid(grid)
    b_, h, w = grid.shape
    probs = [float(p) for p in neighbour_probs(params)]
    cells = grid.tolist()                                  # plain lists: fairer than numpy scalar indexing
    roll = params.rng.random(grid.shape).tolist()
    new = grid.copy()
    for b in range(b_):
        for r in range(h):
            for c in range(w):
                if cells[b][r][c] != CLEAR:
                    continue
                survive = 1.0
                for (dr, dc), p in zip(OFFSETS_8, probs):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and cells[b][rr][cc] == INFECTED:
                        survive *= 1.0 - p
                if roll[b][r][c] < 1.0 - survive:
                    new[b, r, c] = INFECTED
    return new

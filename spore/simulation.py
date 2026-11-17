"""Phase 2: spread rule, wind, step(), apply_laser() and run().

Spread rule (per time step, per CLEAR cell, all cells in parallel):
    every infected neighbour k independently tries to infect the cell with
    probability  p_k = clip(spread_prob * wind_weight_k, 0, 1)
    P(cell infected) = 1 - prod_k (1 - p_k)         (k over its infected neighbours)
With no wind and n infected neighbours this is 1 - (1 - p)^n, so more infected
neighbours means a higher chance, which is the neighbour-count rule.

Cell lifecycle:  0 clear -> 1 infected  (by spreading)
                 1 infected stays 1 until the laser hits it -> 2 burned
                 2 burned stays 2 forever and cannot be re-infected (firebreak).
Infected cells never recover or burn out on their own.
"""
import numpy as np
from .grid import CLEAR, INFECTED, BURNED, GRID_DTYPE, validate_grid
from .neighbours import OFFSETS_8, overlap_slices
from .params import SporeParams


def wind_weights(wind_dir, wind_strength):
    """Multiplier for each of the 8 neighbours, in the order of OFFSETS_8.

    For the neighbour at offset (dr, dc), spores travel TO the target cell in
    direction (-dr, -dc). weight = 1 + strength * cos(angle between that travel
    direction and the wind direction), so downwind neighbours push harder.
    Rows grow downward, so compass north is (-1, 0) and east is (0, +1).
    """
    th = np.deg2rad(wind_dir)
    wind = np.array([-np.cos(th), np.sin(th)])            # (d_row, d_col)
    weights = []
    for dr, dc in OFFSETS_8:
        travel = np.array([-dr, -dc], dtype=float)
        travel /= np.linalg.norm(travel)                  # diagonals count as unit length
        weights.append(1.0 + wind_strength * float(travel @ wind))
    return np.array(weights)


def neighbour_probs(params):
    """Per-neighbour infection probability, in the order of OFFSETS_8 (8 floats).

    Single source of truth shared by the NumPy, loop-reference and PyTorch
    engines, so all of them use bit-identical numbers.
    """
    return np.clip(params.spread_prob * wind_weights(params.wind_dir, params.wind_strength), 0.0, 1.0)


def spread_probability(grid, params):
    """Float array (batch, h, w): chance each cell becomes infected this step.

    Computed for every cell regardless of its state; step() only applies it to
    clear cells.
    """
    validate_grid(grid)
    h, w = grid.shape[-2:]
    probs = neighbour_probs(params)
    infected = (grid == INFECTED)
    survive = np.ones(grid.shape, dtype=np.float64)       # chance of NOT being infected
    for (dr, dc), p in zip(OFFSETS_8, probs):
        r_dst, c_dst, r_src, c_src = overlap_slices(dr, dc, h, w)
        survive[..., r_dst, c_dst] *= 1.0 - p * infected[..., r_src, c_src]
    return 1.0 - survive


def step(grid, params):
    """Advance every simulation in the batch by one time step. Returns a NEW grid.

    Agreed interface: step(grid, params), grid shape (batch, height, width).
    Randomness comes from params.rng, so runs are reproducible via params.seed.
    """
    p_infect = spread_probability(grid, params)
    roll = params.rng.random(grid.shape)
    new = grid.copy()
    new[(grid == CLEAR) & (roll < p_infect)] = INFECTED
    return new


def apply_laser(grid, cells):
    """Fire the laser at `cells`: every targeted cell becomes BURNED. Returns a NEW grid.

    cells : iterable of (row, col), applied to EVERY simulation in the batch
            (so one click on the demo can be applied to all 'futures'), or an
            iterable of (batch, row, col) to hit single simulations.
    Burning clear cells too is deliberate: it builds a firebreak that spores cannot cross.
    """
    new = grid.copy()
    _, h, w = grid.shape
    for cell in cells:
        if len(cell) == 2:
            r, c = cell
            b = slice(None)
        else:
            b, r, c = cell
        if not (0 <= r < h and 0 <= c < w):
            raise ValueError(f"laser cell {(r, c)} is outside the grid")
        new[b, r, c] = BURNED
    return new


def run(grid, params, n_steps, laser_schedule=None):
    """Run n_steps and return the final grid.

    laser_schedule : optional {step_number: cells} fired BEFORE that step's spread
                     (step numbers start at 0), e.g. {10: [(r, 5) for r in range(H)]}.
    """
    laser_schedule = laser_schedule or {}
    for t in range(n_steps):
        if t in laser_schedule:
            grid = apply_laser(grid, laser_schedule[t])
        grid = step(grid, params)
    return grid

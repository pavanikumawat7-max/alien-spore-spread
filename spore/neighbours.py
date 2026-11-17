"""8-neighbour (Moore neighbourhood) detection.

Two implementations that must always agree:
  * count_neighbours_naive : obvious loops. The ground truth, and the
                             "CPU loop" baseline for Person C's benchmarks.
  * count_neighbours       : vectorised shifted-slice version. Uses ONLY slicing
                             and +=, so the same logic ports directly to PyTorch
                             tensors in Phase 2.

Boundary rule: cells beyond the edge count as empty (no wrap-around). Mars
doesn't wrap, so edge cells simply have fewer neighbours (edges 5, corners 3).
"""
import numpy as np

# (d_row, d_col) for all 8 neighbours
OFFSETS_8 = [(-1, -1), (-1, 0), (-1, 1),
             (0, -1),           (0, 1),
             (1, -1),  (1, 0),  (1, 1)]


def overlap_slices(dr, dc, h, w):
    """Slices pairing every cell with its (dr, dc) neighbour, edges excluded.

    Returns (r_dst, c_dst, r_src, c_src): for all cells in [r_dst, c_dst], the
    neighbour at offset (dr, dc) lives at the same position in [r_src, c_src].
    Shared by count_neighbours and the Phase 2 spread logic.
    """
    r_dst = slice(max(0, -dr), h - max(0, dr))
    r_src = slice(max(0, dr), h - max(0, -dr))
    c_dst = slice(max(0, -dc), w - max(0, dc))
    c_src = slice(max(0, dc), w - max(0, -dc))
    return r_dst, c_dst, r_src, c_src


def neighbour_coords(r, c, height, width):
    """List the in-bounds (row, col) neighbours of one cell."""
    return [(r + dr, c + dc) for dr, dc in OFFSETS_8
            if 0 <= r + dr < height and 0 <= c + dc < width]


def count_neighbours_naive(grid, state=1):
    """Reference implementation: for every cell, count neighbours equal to `state`."""
    batch, h, w = grid.shape
    out = np.zeros((batch, h, w), dtype=np.uint8)
    for b in range(batch):
        for r in range(h):
            for c in range(w):
                n = 0
                for rr, cc in neighbour_coords(r, c, h, w):
                    if grid[b, rr, cc] == state:
                        n += 1
                out[b, r, c] = n
    return out


def count_neighbours(grid, state=1):
    """Vectorised: out[b, r, c] = number of the 8 neighbours of (r, c) equal to `state`.

    Handles all batch items and all cells at once. For an offset (dr, dc) the
    neighbour of cell (r, c) is (r+dr, c+dc), so we add the shifted overlap of
    the mask into the matching region of the output. Cells whose neighbour would
    fall off the grid are left untouched (= edge treated as empty).
    """
    mask = (grid == state).astype(np.uint8)
    h, w = grid.shape[-2], grid.shape[-1]
    out = np.zeros_like(mask)

    for dr, dc in OFFSETS_8:
        r_dst, c_dst, r_src, c_src = overlap_slices(dr, dc, h, w)
        out[..., r_dst, c_dst] += mask[..., r_src, c_src]
    return out

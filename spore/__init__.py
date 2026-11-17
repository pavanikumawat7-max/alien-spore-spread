"""Alien Spore-Spread: GPU-accelerated cellular-automaton spread simulation."""
from .grid import CLEAR, INFECTED, BURNED, make_grid, validate_grid
from .neighbours import (
    OFFSETS_8, count_neighbours, count_neighbours_naive, neighbour_coords,
)

from .params import SporeParams
from .simulation import wind_weights, neighbour_probs, spread_probability, step, apply_laser, run
from .reference import step_loop

# The GPU engine needs PyTorch. Phase 1-2 code keeps working without it.
try:
    from .gpu import GpuSimulator, resolve_device, synchronize
    TORCH_AVAILABLE = True
except ModuleNotFoundError as e:   # only swallow 'torch is not installed'
    if e.name != "torch":
        raise
    TORCH_AVAILABLE = False

__all__ = [
    "CLEAR", "INFECTED", "BURNED", "make_grid", "validate_grid",
    "OFFSETS_8", "count_neighbours", "count_neighbours_naive", "neighbour_coords",
    "SporeParams", "wind_weights", "neighbour_probs", "step_loop", "TORCH_AVAILABLE",
    "GpuSimulator", "resolve_device", "synchronize", "spread_probability", "step", "apply_laser", "run",
]

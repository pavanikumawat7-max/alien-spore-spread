"""Simulation parameters (the `params` argument of step())."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SporeParams:
    """Parameters for one simulation run.

    spread_prob   : base chance that ONE infected neighbour infects a clear cell
                    in one time step (0..1), before wind is applied.
    wind_dir      : direction the wind blows TOWARD, in compass degrees
                    (0 = north / up, 90 = east / right, 180 = south, 270 = west).
    wind_strength : 0 = no wind, 1 = full wind. Spread straight downwind is
                    multiplied by (1 + strength), straight upwind by (1 - strength).
    seed          : RNG seed. The generator lives in `rng` and advances with every
                    step, so a run is reproducible from the seed alone while each
                    step still gets fresh random numbers.
    """
    spread_prob: float = 0.2
    wind_dir: float = 0.0
    wind_strength: float = 0.0
    seed: int | None = None
    rng: np.random.Generator = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not 0.0 <= self.spread_prob <= 1.0:
            raise ValueError("spread_prob must be in [0, 1]")
        if not 0.0 <= self.wind_strength <= 1.0:
            raise ValueError("wind_strength must be in [0, 1]")
        self.rng = np.random.default_rng(self.seed)

    def reset_rng(self):
        """Restart the random stream from `seed` (to replay a run)."""
        self.rng = np.random.default_rng(self.seed)

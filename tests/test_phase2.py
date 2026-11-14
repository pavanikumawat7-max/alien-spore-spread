"""Phase 2 tests. Run:  python -m unittest discover -s tests -v   (or: pytest)"""
import os, sys, unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from spore import (CLEAR, INFECTED, BURNED, OFFSETS_8, make_grid, SporeParams, wind_weights,
                   spread_probability, step, apply_laser, run, neighbour_coords)


def spread_probability_naive(grid, params):
    """Loop reference for the spread rule (slow, obviously correct)."""
    wts = dict(zip(OFFSETS_8, wind_weights(params.wind_dir, params.wind_strength)))
    b_, h, w = grid.shape
    out = np.zeros(grid.shape)
    for b in range(b_):
        for r in range(h):
            for c in range(w):
                surv = 1.0
                for rr, cc in neighbour_coords(r, c, h, w):
                    if grid[b, rr, cc] == INFECTED:
                        p = min(1.0, max(0.0, params.spread_prob * wts[(rr - r, cc - c)]))
                        surv *= 1 - p
                out[b, r, c] = 1 - surv
    return out


class TestWind(unittest.TestCase):
    def idx(self, d):
        return OFFSETS_8.index(d)

    def test_no_wind_is_uniform(self):
        self.assertTrue(np.allclose(wind_weights(123, 0.0), 1.0))

    def test_wind_blowing_east(self):
        w = wind_weights(90, 0.5)                      # blows toward +col
        self.assertAlmostEqual(w[self.idx((0, -1))], 1.5)   # western neighbour pushes east
        self.assertAlmostEqual(w[self.idx((0, 1))], 0.5)    # eastern neighbour pushes west
        self.assertAlmostEqual(w[self.idx((-1, 0))], 1.0)   # sideways: unchanged

    def test_wind_blowing_north(self):
        w = wind_weights(0, 1.0)                       # blows toward -row (up)
        self.assertAlmostEqual(w[self.idx((1, 0))], 2.0)    # southern neighbour pushes north
        self.assertAlmostEqual(w[self.idx((-1, 0))], 0.0)

    def test_average_weight_is_one(self):
        for d in (0, 37, 90, 211):
            self.assertAlmostEqual(float(wind_weights(d, 0.8).mean()), 1.0)


class TestSpreadProbability(unittest.TestCase):
    def test_matches_naive_with_wind(self):
        g = make_grid(12, 9, batch=3, n_seeds=10, seed=4)
        g[0, 3, 3] = BURNED
        for params in (SporeParams(0.3), SporeParams(0.6, 45, 0.7), SporeParams(0.9, 270, 1.0)):
            self.assertTrue(np.allclose(spread_probability(g, params), spread_probability_naive(g, params)))

    def test_no_wind_formula(self):
        g = np.zeros((1, 3, 3), np.uint8)
        g[0, 0, 0] = g[0, 0, 1] = INFECTED            # centre (1,1) has 2 infected neighbours
        p = spread_probability(g, SporeParams(0.3))
        self.assertAlmostEqual(p[0, 1, 1], 1 - 0.7 ** 2)

    def test_probability_in_range(self):
        g = make_grid(20, 20, batch=2, n_seeds=100, seed=1)
        p = spread_probability(g, SporeParams(1.0, 10, 1.0))
        self.assertTrue(p.min() >= 0.0 and p.max() <= 1.0)


class TestStep(unittest.TestCase):
    def test_interface_shape_dtype_values(self):
        g = make_grid(20, 30, batch=4, n_seeds=3, seed=2)
        out = step(g, SporeParams(0.5, seed=0))
        self.assertEqual(out.shape, g.shape)
        self.assertEqual(out.dtype, np.uint8)
        self.assertTrue(set(np.unique(out)) <= {0, 1, 2})

    def test_input_not_modified(self):
        g = make_grid(20, 20, n_seeds=5, seed=1)
        before = g.copy()
        step(g, SporeParams(0.9, seed=1))
        self.assertTrue(np.array_equal(g, before))

    def test_zero_prob_no_spread(self):
        g = make_grid(15, 15, n_seeds=4, seed=3)
        self.assertTrue(np.array_equal(step(g, SporeParams(0.0, seed=1)), g))

    def test_certain_spread_fills_neighbourhood(self):
        g = make_grid(7, 7, centre=True)
        out = step(g, SporeParams(1.0, seed=1))
        self.assertTrue(np.all(out[0, 2:5, 2:5] == INFECTED))
        self.assertEqual(int((out == INFECTED).sum()), 9)       # but not beyond 1 step

    def test_infected_stay_infected_until_laser(self):
        g = make_grid(25, 25, batch=3, n_seeds=3, seed=6)
        params = SporeParams(0.4, 90, 0.5, seed=9)
        for _ in range(40):
            new = step(g, params)
            self.assertTrue(np.all(new[g == INFECTED] == INFECTED))   # nobody recovers or burns
            self.assertTrue(np.all((new == INFECTED).sum(axis=(1, 2)) >= (g == INFECTED).sum(axis=(1, 2))))
            g = new
        self.assertFalse(np.any(g == BURNED))                          # only the laser burns

    def test_burned_cells_are_never_reinfected(self):
        g = make_grid(9, 9, centre=True)
        g[0, 4, 5] = BURNED
        for _ in range(10):
            g = step(g, SporeParams(1.0, seed=0))
        self.assertEqual(g[0, 4, 5], BURNED)

    def test_reproducible_with_seed(self):
        g = make_grid(40, 40, batch=2, n_seeds=2, seed=8)
        a = run(g, SporeParams(0.3, 90, 0.5, seed=77), 25)
        b = run(g, SporeParams(0.3, 90, 0.5, seed=77), 25)
        c = run(g, SporeParams(0.3, 90, 0.5, seed=78), 25)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))

    def test_reset_rng_replays_run(self):
        g = make_grid(30, 30, n_seeds=2, seed=8)
        params = SporeParams(0.3, seed=5)
        a = run(g, params, 10)
        params.reset_rng()
        self.assertTrue(np.array_equal(a, run(g, params, 10)))

    def test_statistics_match_formula(self):
        # Many independent 3x3 worlds: centre cell has 1 or 2 infected neighbours.
        n = 40000
        g = np.zeros((n, 3, 3), np.uint8)
        g[:, 0, 1] = INFECTED                                 # 1 infected neighbour of (1,1)
        g[:2000, 1, 0] = INFECTED                             # 2 for the first 2000
        out = step(g, SporeParams(0.3, seed=123))
        frac1 = (out[2000:, 1, 1] == INFECTED).mean()
        frac2 = (out[:2000, 1, 1] == INFECTED).mean()
        self.assertAlmostEqual(frac1, 0.3, delta=0.01)
        self.assertAlmostEqual(frac2, 1 - 0.7 ** 2, delta=0.03)

    def test_wind_makes_downwind_spread_faster(self):
        n = 40000
        g = np.zeros((n, 5, 5), np.uint8)
        g[:, 2, 2] = INFECTED
        out = step(g, SporeParams(0.3, wind_dir=90, wind_strength=0.8, seed=5))   # wind to the east
        east = (out[:, 2, 3] == INFECTED).mean()
        west = (out[:, 2, 1] == INFECTED).mean()
        north = (out[:, 1, 2] == INFECTED).mean()
        self.assertAlmostEqual(east, 0.3 * 1.8, delta=0.015)
        self.assertAlmostEqual(west, 0.3 * 0.2, delta=0.015)
        self.assertAlmostEqual(north, 0.3, delta=0.015)

    def test_bad_params(self):
        with self.assertRaises(ValueError): SporeParams(1.5)
        with self.assertRaises(ValueError): SporeParams(0.2, wind_strength=-0.1)


class TestLaser(unittest.TestCase):
    def test_burns_targets_for_all_batch_items(self):
        g = make_grid(10, 10, batch=3, n_seeds=5, seed=1)
        out = apply_laser(g, [(2, 2), (5, 7)])
        self.assertTrue(np.all(out[:, 2, 2] == BURNED) and np.all(out[:, 5, 7] == BURNED))
        self.assertEqual(g.max(), INFECTED)                    # input untouched

    def test_single_batch_target_and_bounds(self):
        g = make_grid(10, 10, batch=3, n_seeds=1, seed=1)
        out = apply_laser(g, [(1, 4, 4)])
        self.assertEqual(out[1, 4, 4], BURNED)
        self.assertNotEqual(out[0, 4, 4], BURNED)
        with self.assertRaises(ValueError): apply_laser(g, [(10, 0)])

    def test_barrier_contains_spread(self):
        H, W = 15, 30
        g = make_grid(H, W, batch=5, seed_cells=[(7, 2)])
        wall = [(r, 10) for r in range(H)]                     # full-height firebreak
        out = run(g, SporeParams(1.0, seed=0), 60, laser_schedule={3: wall})
        self.assertFalse(np.any(out[:, :, 11:] == INFECTED))   # nothing crosses
        self.assertTrue(np.all(out[:, :, :10] == INFECTED))    # near side fully overrun
        self.assertTrue(np.all(out[:, 7, 5] == INFECTED))      # but the near side is overrun

    def test_laser_burns_existing_infection(self):
        g = make_grid(9, 9, centre=True)
        out = apply_laser(g, [(4, 4)])
        self.assertEqual(int((out == INFECTED).sum()), 0)

    def test_without_laser_spore_crosses(self):
        g = make_grid(15, 30, batch=2, seed_cells=[(7, 2)])
        out = run(g, SporeParams(1.0, seed=0), 60)
        self.assertTrue(np.all(out[:, 7, 29] == INFECTED))


if __name__ == "__main__":
    unittest.main()

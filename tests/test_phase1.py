"""Phase 1 tests. Run:  python -m unittest discover -s tests -v   (or: pytest)"""
import os, sys, unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from spore import (CLEAR, INFECTED, BURNED, make_grid, validate_grid,
                   OFFSETS_8, count_neighbours, count_neighbours_naive, neighbour_coords)


class TestGrid(unittest.TestCase):
    def test_shape_dtype_values(self):
        g = make_grid(30, 40, batch=5, n_seeds=3, seed=1)
        self.assertEqual(g.shape, (5, 30, 40))
        self.assertEqual(g.dtype, np.uint8)
        self.assertTrue(set(np.unique(g)) <= {CLEAR, INFECTED, BURNED})

    def test_seed_counts_per_batch_item(self):
        g = make_grid(30, 40, batch=7, n_seeds=4, seed=2)
        self.assertTrue(np.all((g == INFECTED).sum(axis=(1, 2)) == 4))

    def test_reproducible_with_seed(self):
        a = make_grid(50, 50, batch=3, n_seeds=5, seed=42)
        b = make_grid(50, 50, batch=3, n_seeds=5, seed=42)
        c = make_grid(50, 50, batch=3, n_seeds=5, seed=43)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))

    def test_batch_items_differ(self):
        g = make_grid(50, 50, batch=2, n_seeds=5, seed=0)
        self.assertFalse(np.array_equal(g[0], g[1]))

    def test_centre_and_explicit_seeds(self):
        g = make_grid(11, 11, centre=True)
        self.assertEqual(g[0, 5, 5], INFECTED)
        self.assertEqual(int((g == INFECTED).sum()), 1)
        g = make_grid(10, 10, batch=2, seed_cells=[(0, 0), (9, 9)])
        self.assertTrue(np.all(g[:, 0, 0] == INFECTED) and np.all(g[:, 9, 9] == INFECTED))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError): make_grid(0, 5)
        with self.assertRaises(ValueError): make_grid(3, 3, n_seeds=10)
        with self.assertRaises(ValueError): make_grid(3, 3, seed_cells=[(3, 0)])
        with self.assertRaises(ValueError): validate_grid(np.zeros((4, 4), np.uint8))
        with self.assertRaises(ValueError): validate_grid(np.full((1, 4, 4), 3, np.uint8))


class TestNeighbours(unittest.TestCase):
    def test_eight_distinct_offsets(self):
        self.assertEqual(len(set(OFFSETS_8)), 8)
        self.assertNotIn((0, 0), OFFSETS_8)

    def test_coords_interior_edge_corner(self):
        self.assertEqual(len(neighbour_coords(5, 5, 10, 10)), 8)
        self.assertEqual(len(neighbour_coords(0, 5, 10, 10)), 5)
        self.assertEqual(len(neighbour_coords(0, 0, 10, 10)), 3)
        self.assertEqual(len(neighbour_coords(9, 9, 10, 10)), 3)
        self.assertEqual(sorted(neighbour_coords(1, 1, 3, 3)),
                         sorted((r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)))

    def test_single_infection_hits_exactly_its_8_neighbours(self):
        g = make_grid(7, 7, centre=True)          # infection at (3, 3)
        n = count_neighbours(g)[0]
        expected = np.zeros((7, 7), np.uint8)
        expected[2:5, 2:5] = 1
        expected[3, 3] = 0                        # a cell is not its own neighbour
        self.assertTrue(np.array_equal(n, expected))

    def test_corner_infection(self):
        g = make_grid(5, 5, seed_cells=[(0, 0)])
        n = count_neighbours(g)[0]
        self.assertEqual(int(n.sum()), 3)
        self.assertEqual((n[0, 1], n[1, 0], n[1, 1]), (1, 1, 1))

    def test_full_grid_counts(self):
        g = np.ones((1, 6, 6), np.uint8)
        n = count_neighbours(g)[0]
        self.assertEqual((n[3, 3], n[0, 3], n[0, 0]), (8, 5, 3))

    def test_matches_naive_random_grids(self):
        rng = np.random.default_rng(7)
        for shape in [(1, 1, 1), (1, 1, 9), (2, 9, 1), (3, 8, 13), (4, 20, 20)]:
            g = rng.integers(0, 3, size=shape).astype(np.uint8)
            for state in (CLEAR, INFECTED, BURNED):
                self.assertTrue(np.array_equal(count_neighbours(g, state),
                                               count_neighbours_naive(g, state)),
                                f"mismatch shape={shape} state={state}")

    def test_batch_items_independent(self):
        g = make_grid(15, 15, batch=3, n_seeds=10, seed=5)
        full = count_neighbours(g)
        for b in range(3):
            self.assertTrue(np.array_equal(full[b], count_neighbours(g[b:b + 1])[0]))

    def test_input_not_modified(self):
        g = make_grid(10, 10, n_seeds=5, seed=1)
        before = g.copy()
        count_neighbours(g)
        self.assertTrue(np.array_equal(g, before))


if __name__ == "__main__":
    unittest.main()

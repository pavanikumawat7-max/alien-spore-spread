"""Phase 3 tests. Run:  python -m unittest discover -s tests -v   (or: pytest)

Loop-reference tests always run. GPU-engine tests need PyTorch and are skipped
without it; they run on the CPU device always and on CUDA when a GPU is present.
"""
import os, sys, unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import spore
from spore import (INFECTED, BURNED, make_grid, SporeParams, step, apply_laser, run,
                   step_loop, neighbour_probs, OFFSETS_8)

CONFIGS = [  # (h, w, batch, n_seeds, params kwargs, steps)
    (9, 9, 1, 1, dict(spread_prob=0.4), 12),
    (12, 7, 3, 4, dict(spread_prob=0.3, wind_dir=90, wind_strength=0.6), 15),
    (1, 10, 2, 1, dict(spread_prob=0.5, wind_dir=270, wind_strength=1.0), 10),
    (16, 16, 4, 6, dict(spread_prob=0.9, wind_dir=33, wind_strength=0.3), 8),
]


class TestLoopReference(unittest.TestCase):
    def test_probs_shared_by_numpy_engine(self):
        p = SporeParams(0.5, 120, 0.7)
        probs = neighbour_probs(p)
        self.assertEqual(len(probs), len(OFFSETS_8))
        self.assertTrue(probs.min() >= 0 and probs.max() <= 1)

    def test_loop_step_equals_numpy_step_every_step(self):
        for h, w, b, ns, kw, steps in CONFIGS:
            g0 = make_grid(h, w, batch=b, n_seeds=ns, seed=11)
            pa, pb = SporeParams(seed=5, **kw), SporeParams(seed=5, **kw)
            a = b_ = g0
            for t in range(steps):
                a, b_ = step(a, pa), step_loop(b_, pb)
                self.assertTrue(np.array_equal(a, b_), f"loop != numpy at step {t}, config {(h, w, b)}")

    def test_loop_step_does_not_modify_input(self):
        g = make_grid(8, 8, n_seeds=3, seed=1)
        before = g.copy()
        step_loop(g, SporeParams(0.9, seed=1))
        self.assertTrue(np.array_equal(g, before))


@unittest.skipUnless(spore.TORCH_AVAILABLE, "PyTorch not installed: GPU engine tests skipped")
class TestGpuEngine(unittest.TestCase):
    @classmethod
    def devices(cls):
        import torch
        return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    def test_exact_mode_identical_to_numpy_every_step(self):
        for dev in self.devices():
            for h, w, b, ns, kw, steps in CONFIGS:
                g0 = make_grid(h, w, batch=b, n_seeds=ns, seed=21)
                cpu_p = SporeParams(seed=7, **kw)
                sim = spore.GpuSimulator(SporeParams(seed=7, **kw), device=dev, exact=True)
                cpu = g0
                gpu = sim.to_device(g0)
                for t in range(steps):
                    cpu = step(cpu, cpu_p)
                    gpu = sim.step(gpu)
                    self.assertTrue(np.array_equal(cpu, sim.to_host(gpu)),
                                    f"{dev}: GPU != CPU at step {t}, config {(h, w, b)}")

    def test_exact_mode_with_laser_schedule_identical(self):
        for dev in self.devices():
            g0 = make_grid(15, 30, batch=3, seed_cells=[(7, 2)])
            wall = [(r, 10) for r in range(15)] + [(1, 4, 4), (2, 5, 5)]
            sched = {3: wall[:15], 6: wall[15:]}
            kw = dict(spread_prob=0.5, wind_dir=90, wind_strength=0.4)
            cpu = run(g0, SporeParams(seed=3, **kw), 25, sched)
            gpu = spore.GpuSimulator(SporeParams(seed=3, **kw), device=dev, exact=True).run(g0, 25, sched)
            self.assertTrue(np.array_equal(cpu, gpu))

    def test_spread_probability_matches_cpu(self):
        for dev in self.devices():
            g = make_grid(14, 11, batch=2, n_seeds=9, seed=4)
            p = SporeParams(0.6, 45, 0.7)
            sim = spore.GpuSimulator(p, device=dev, exact=True)
            self.assertTrue(np.array_equal(sim.to_host(sim.spread_probability(sim.to_device(g))),
                                           spore.spread_probability(g, p)))

    def test_input_not_modified_and_dtype(self):
        for dev in self.devices():
            sim = spore.GpuSimulator(SporeParams(0.9, seed=1), device=dev)
            g = make_grid(10, 10, batch=2, n_seeds=3, seed=1)
            before = g.copy()
            t = sim.to_device(g)
            t_before = t.clone()
            out = sim.step(t)
            self.assertTrue((t == t_before).all())
            self.assertTrue(np.array_equal(g, before))
            self.assertEqual(sim.to_host(out).dtype, np.uint8)

    def test_fast_mode_reproducible_and_seed_sensitive(self):
        for dev in self.devices():
            g = make_grid(30, 30, batch=2, n_seeds=2, seed=8)
            kw = dict(spread_prob=0.3, wind_dir=90, wind_strength=0.5)
            a = spore.GpuSimulator(SporeParams(seed=77, **kw), device=dev).run(g, 20)
            b = spore.GpuSimulator(SporeParams(seed=77, **kw), device=dev).run(g, 20)
            c = spore.GpuSimulator(SporeParams(seed=78, **kw), device=dev).run(g, 20)
            self.assertTrue(np.array_equal(a, b))
            self.assertFalse(np.array_equal(a, c))

    def test_reset_replays_run(self):
        for dev in self.devices():
            for exact in (True, False):
                g = make_grid(20, 20, n_seeds=2, seed=8)
                sim = spore.GpuSimulator(SporeParams(0.3, seed=5), device=dev, exact=exact)
                a = sim.run(g, 8)
                sim.reset()
                self.assertTrue(np.array_equal(a, sim.run(g, 8)))

    def test_fast_mode_statistics_match_cpu(self):
        # Thousands of tiny independent worlds: compare mean infected count after a few steps.
        n = 6000
        g = np.zeros((n, 9, 9), np.uint8)
        g[:, 4, 4] = INFECTED
        kw = dict(spread_prob=0.25, wind_dir=90, wind_strength=0.6)
        cpu = (run(g, SporeParams(seed=1, **kw), 6) == INFECTED).sum(axis=(1, 2))
        for dev in self.devices():
            gpu = (spore.GpuSimulator(SporeParams(seed=2, **kw), device=dev).run(g, 6) == INFECTED).sum(axis=(1, 2))
            se = np.sqrt(cpu.var() / n + gpu.var() / n)
            self.assertLess(abs(cpu.mean() - gpu.mean()), 4.5 * se,
                            f"{dev}: means {cpu.mean():.2f} vs {gpu.mean():.2f}")

    def test_laser_barrier_contains_on_device(self):
        for dev in self.devices():
            g = make_grid(15, 30, batch=4, seed_cells=[(7, 2)])
            sim = spore.GpuSimulator(SporeParams(1.0, seed=0), device=dev)
            out = sim.run(g, 60, {3: [(r, 10) for r in range(15)]})
            self.assertFalse(np.any(out[:, :, 11:] == INFECTED))
            self.assertTrue(np.all(out[:, 7, 5] == INFECTED))

    def test_laser_bounds_checked(self):
        sim = spore.GpuSimulator(SporeParams(0.5, seed=0), device="cpu")
        with self.assertRaises(ValueError):
            sim.apply_laser(sim.to_device(make_grid(5, 5, centre=True)), [(5, 0)])

    def test_run_batch_chunking(self):
        # p = 1 makes the run deterministic, so any chunk size must give the CPU answer.
        kw = dict(spread_prob=1.0)
        g = make_grid(11, 11, batch=1, seed_cells=[(5, 5)])
        cpu = int((run(g, SporeParams(seed=0, **kw), 4) == INFECTED).sum())
        for dev in self.devices():
            sim = spore.GpuSimulator(SporeParams(seed=0, **kw), device=dev)
            res = sim.run_batch(11, 11, batch=7, n_steps=4, seed_cells=[(5, 5)], chunk=3)
            self.assertEqual(res.shape, (7,))
            self.assertTrue(np.all(res == cpu))

    def test_run_batch_custom_reducer_and_auto_chunk(self):
        sim = spore.GpuSimulator(SporeParams(0.3, seed=1), device="cpu")
        self.assertGreaterEqual(sim.auto_chunk(100, 100, 50), 1)
        res = sim.run_batch(20, 20, batch=10, n_steps=3, n_seeds=2, seed=1,
                            reduce=lambda g: (g == BURNED).sum(dim=(1, 2)))
        self.assertTrue(np.all(res == 0))                       # no laser fired


if __name__ == "__main__":
    unittest.main()

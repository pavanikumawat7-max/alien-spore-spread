"""Person B: live demo with glowing display, click-to-fire laser, controls and the
"Run 1,000 futures" button.

Run from the project root:
    python scripts/demo_live.py
    python scripts/demo_live.py --size 120 --futures 500     # slower laptop

Uses only numpy + matplotlib for the live view (batch = 1, so the GPU would not
help here). The futures button uses the GPU engine if a CUDA GPU is present,
otherwise the NumPy engine in chunks.

Mouse : click or drag on the map to fire the laser (burns a small round spot).
Keys  : space = pause, r = reset, f = run futures.
"""
import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

from spore import (make_grid, SporeParams, step, apply_laser,
                   CLEAR, INFECTED, BURNED, TORCH_AVAILABLE)

# ---- definition of "contained" (agree this with Person A and C) -------------
# A future counts as contained if, after FUTURE_STEPS more steps, fewer than
# CONTAIN_FRACTION of all cells are infected.
FUTURE_STEPS = 80
CONTAIN_FRACTION = 0.25
LASER_RADIUS = 2

BG = np.array([0.05, 0.035, 0.03])        # dark Martian soil
GREEN = np.array([0.22, 1.0, 0.08])       # glowing spores
EMBER = np.array([1.0, 0.27, 0.0])        # burned / laser cells


# ------------------------------------------------------------------ pure helpers
def blur(a, passes=2):
    """Cheap separable blur (no scipy): average with the 4 neighbours, repeated."""
    a = a.astype(np.float32)
    for _ in range(passes):
        p = np.pad(a, 1, mode="edge")
        a = (4 * p[1:-1, 1:-1] + p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 8.0
    return a


def render(grid2d, pulse=1.0):
    """(h, w) uint8 grid -> (h, w, 3) float image with a glow around spores."""
    inf = (grid2d == INFECTED).astype(np.float32)
    burn = (grid2d == BURNED).astype(np.float32)
    glow = blur(inf, passes=4) * 1.3
    ember_glow = blur(burn, passes=1) * 0.6
    img = np.empty(grid2d.shape + (3,), dtype=np.float32)
    img[:] = BG
    img += glow[..., None] * GREEN * 0.45 * pulse
    img += inf[..., None] * GREEN * 0.40 * pulse
    img += ember_glow[..., None] * EMBER * 0.5
    img += burn[..., None] * EMBER * 0.55
    return np.clip(img, 0, 1)


def disc_cells(r, c, radius, h, w):
    """Grid cells within `radius` of (r, c), clipped to the grid."""
    cells = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr * dr + dc * dc <= radius * radius and 0 <= r + dr < h and 0 <= c + dc < w:
                cells.append((r + dr, c + dc))
    return cells


def line_points(r0, c0, r1, c1):
    """Cells on the straight line between two cells (so fast drags leave no gaps)."""
    n = max(abs(r1 - r0), abs(c1 - c0), 1)
    return [(int(round(r0 + (r1 - r0) * t / n)), int(round(c0 + (c1 - c0) * t / n))) for t in range(n + 1)]


def contained(final_grids, fraction=CONTAIN_FRACTION):
    """final_grids: (batch, h, w) -> bool array, one entry per future."""
    infected = (final_grids == INFECTED).sum(axis=(1, 2))
    return infected < fraction * final_grids.shape[1] * final_grids.shape[2]


def run_futures(grid, spread_prob, wind_dir, wind_strength, n_futures, n_steps,
                progress=None, chunk=250):
    """Copy the CURRENT grid n_futures times and let each evolve independently.

    Returns (containment_percent, seconds, engine_name). The laser barriers
    already on the grid are part of the starting state of every future.
    """
    start = grid[:1]
    t0 = time.perf_counter()
    params = SporeParams(spread_prob, wind_dir, wind_strength, seed=None)  # fresh randomness
    use_gpu = False
    if TORCH_AVAILABLE:
        import torch
        use_gpu = torch.cuda.is_available()

    if use_gpu:
        from spore import GpuSimulator
        sim = GpuSimulator(params, device="cuda", exact=False)
        final = sim.run(np.repeat(start, n_futures, axis=0), n_steps)
        flags = contained(final)
        engine = "GPU"
        if progress:
            progress(1.0)
    else:
        flags, done = [], 0
        while done < n_futures:
            n = min(chunk, n_futures - done)
            g = np.repeat(start, n, axis=0)
            for _ in range(n_steps):
                g = step(g, params)
            flags.append(contained(g))
            done += n
            if progress:
                progress(done / n_futures)
        flags = np.concatenate(flags)
        engine = "CPU (NumPy)"
    return 100.0 * float(np.mean(flags)), time.perf_counter() - t0, engine


# ------------------------------------------------------------------------ the app
class Demo:
    def __init__(self, size, n_futures, fps):
        self.h = self.w = size
        self.n_futures = n_futures
        self.paused = False
        self.frame = 0
        self.steps = 0
        self.last_cell = None
        self.firing = False
        self.futures_state = "idle"        # idle | running | done
        self.futures_progress = 0.0
        self.futures_result = None

        self.params = SporeParams(0.12, wind_dir=90, wind_strength=0.3, seed=None)
        self.grid = self.fresh_grid()

        self.fig = plt.figure(figsize=(8, 9.4), facecolor="#0b0806")
        self.fig.canvas.manager.set_window_title("Alien Spore-Spread")
        self.ax = self.fig.add_axes([0.04, 0.30, 0.92, 0.66])
        self.ax.set_axis_off()
        self.im = self.ax.imshow(render(self.grid[0]), interpolation="nearest", origin="upper")
        self.title = self.ax.set_title("", color="#9dff7a", fontsize=11)
        self.wind_arrow = None
        self.result_text = self.fig.text(0.5, 0.275, "", ha="center", color="#ffd27a",
                                         fontsize=13, fontweight="bold")
        self.help_text = self.fig.text(
            0.5, 0.005, "drag on the map = laser    space = pause    r = reset    f = futures",
            ha="center", color="#7a6a5a", fontsize=8)

        self.build_controls()
        self.draw_wind_arrow()

        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.anim = FuncAnimation(self.fig, self.update, interval=int(1000 / fps),
                                  cache_frame_data=False)

    # -- setup helpers
    def fresh_grid(self):
        return make_grid(self.h, self.w, centre=True)

    def build_controls(self):
        style = dict(color="#2a1f18", hovercolor="#4a3626")
        def slider(y, label, lo, hi, init, step_=None):
            ax = self.fig.add_axes([0.22, y, 0.60, 0.025], facecolor="#2a1f18")
            s = Slider(ax, label, lo, hi, valinit=init, valstep=step_, color="#39ff14")
            s.label.set_color("#c9b8a8"); s.valtext.set_color("#c9b8a8")
            return s
        self.s_spread = slider(0.21, "Spread rate", 0.02, 0.40, self.params.spread_prob)
        self.s_wdir = slider(0.17, "Wind dir (deg)", 0, 359, self.params.wind_dir, 1)
        self.s_wstr = slider(0.13, "Wind strength", 0.0, 1.0, self.params.wind_strength)
        self.s_speed = slider(0.09, "Steps / second", 1, 30, 12, 1)
        self.s_wdir.on_changed(lambda _: self.draw_wind_arrow())

        def button(x, label):
            ax = self.fig.add_axes([x, 0.025, 0.2, 0.045])
            b = Button(ax, label, **style)
            b.label.set_color("#e8d8c8")
            return b
        self.b_pause = button(0.08, "Pause")
        self.b_reset = button(0.30, "Reset")
        self.b_fut = button(0.52, f"Run {self.n_futures:,} futures")
        self.b_fut.ax.set_position([0.52, 0.025, 0.40, 0.045])
        self.b_pause.on_clicked(lambda _: self.toggle_pause())
        self.b_reset.on_clicked(lambda _: self.reset())
        self.b_fut.on_clicked(lambda _: self.start_futures())

    def draw_wind_arrow(self):
        """Arrow in the corner showing where the wind blows TOWARD (0 = up, 90 = right)."""
        if self.wind_arrow is not None:
            self.wind_arrow.remove()
        th = np.deg2rad(self.s_wdir.val)
        dx, dy = np.sin(th), -np.cos(th)               # image y axis points down
        cx, cy, L = self.w * 0.9, self.h * 0.1, self.w * 0.07
        self.wind_arrow = self.ax.annotate(
            "", xy=(cx + dx * L, cy + dy * L), xytext=(cx - dx * L, cy - dy * L),
            arrowprops=dict(arrowstyle="-|>", color="#7fd4ff", lw=2.5))
        self.fig.canvas.draw_idle()

    # -- controls
    def toggle_pause(self):
        self.paused = not self.paused
        self.b_pause.label.set_text("Resume" if self.paused else "Pause")

    def reset(self):
        self.grid = self.fresh_grid()
        self.steps = 0
        self.result_text.set_text("")
        self.futures_state = "idle"
        self.im.set_data(render(self.grid[0]))
        self.fig.canvas.draw_idle()

    def start_futures(self):
        if self.futures_state == "running":
            return
        self.futures_state, self.futures_progress = "running", 0.0
        snapshot = self.grid.copy()
        args = (snapshot, self.s_spread.val, self.s_wdir.val, self.s_wstr.val,
                self.n_futures, FUTURE_STEPS)

        def work():
            pct, secs, engine = run_futures(*args, progress=self._set_progress)
            self.futures_result = (pct, secs, engine)
            self.futures_state = "done"

        threading.Thread(target=work, daemon=True).start()

    def _set_progress(self, p):
        self.futures_progress = p

    # -- mouse / keyboard
    def cell_at(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return None
        r, c = int(round(event.ydata)), int(round(event.xdata))
        return (r, c) if 0 <= r < self.h and 0 <= c < self.w else None

    def fire(self, cell):
        """Burn a line from the previous cell to this one (leaves no gaps when dragging)."""
        pts = line_points(*(self.last_cell or cell), *cell)
        cells = {rc for p in pts for rc in disc_cells(p[0], p[1], LASER_RADIUS, self.h, self.w)}
        self.grid = apply_laser(self.grid, sorted(cells))
        self.last_cell = cell

    def on_press(self, event):
        cell = self.cell_at(event)
        if cell is not None and event.button == 1:
            self.firing, self.last_cell = True, None
            self.fire(cell)

    def on_move(self, event):
        if self.firing:
            cell = self.cell_at(event)
            if cell is not None:
                self.fire(cell)

    def on_release(self, _):
        self.firing, self.last_cell = False, None

    def on_key(self, event):
        if event.key == " ":
            self.toggle_pause()
        elif event.key == "r":
            self.reset()
        elif event.key == "f":
            self.start_futures()

    # -- animation loop
    def update(self, _):
        self.frame += 1
        self.params.spread_prob = float(self.s_spread.val)
        self.params.wind_dir = float(self.s_wdir.val)
        self.params.wind_strength = float(self.s_wstr.val)

        if not self.paused:
            # run several steps per frame at higher speeds so the slider has real range
            n = max(1, int(round(self.s_speed.val / 12)))
            for _ in range(n):
                self.grid = step(self.grid, self.params)
                self.steps += 1

        pulse = 0.92 + 0.08 * np.sin(self.frame / 5.0)
        self.im.set_data(render(self.grid[0], pulse))

        n_inf = int((self.grid == INFECTED).sum())
        self.title.set_text(f"step {self.steps}    infected {n_inf}    "
                            f"({100 * n_inf / (self.h * self.w):.1f}% of the map)")

        if self.futures_state == "running":
            self.result_text.set_text(f"simulating futures... {100 * self.futures_progress:.0f}%")
        elif self.futures_state == "done":
            pct, secs, engine = self.futures_result
            self.result_text.set_text(
                f"Containment: {pct:.1f}% of {self.n_futures:,} futures\n"
                f"(<{int(CONTAIN_FRACTION * 100)}% infected after {FUTURE_STEPS} more steps; "
                f"{engine}, {secs:.1f}s)")
        return (self.im,)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=100, help="grid size (square), default 100")
    ap.add_argument("--futures", type=int, default=1000, help="number of futures, default 1000")
    ap.add_argument("--fps", type=int, default=24, help="display frames per second")
    args = ap.parse_args()
    demo = Demo(args.size, args.futures, args.fps)
    plt.show()


if __name__ == "__main__":
    main()
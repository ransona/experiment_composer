import os
import pickle
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np

from core.base_source import DataSource


class SleepScoreSource(DataSource):
    """
    SleepScoreSource — plots sleep state across time with colored segments.
    Expects sleep_score/sleep_state.pickle with fields:
      - state_epoch_t : times (seconds)
      - state_epoch   : states (0..3)
    """

    def __init__(
        self,
        exp_dir_processed: str,
        *,
        time_window: Tuple[float, float] = (-5.0, 5.0),
        colors: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        line_width: float = 3.0,
        figure_size: Tuple[int, int] = (4, 2),
        dpi: int = 100,
        bg_color: str = "black",
        font_color: str = "white",
        show_y_axis: bool = True,
    ):
        super().__init__()
        path = os.path.join(exp_dir_processed, "sleep_score", "sleep_state.pickle")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Sleep state file not found: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)
        if "state_epoch_t" not in data or "state_epoch" not in data:
            raise ValueError("sleep_state.pickle missing required keys: state_epoch_t, state_epoch")

        self.t = np.asarray(data["state_epoch_t"], dtype=float)
        self.state = np.asarray(data["state_epoch"], dtype=int)
        if self.t.ndim != 1 or self.state.ndim != 1 or self.t.shape[0] != self.state.shape[0]:
            raise ValueError("state_epoch_t and state_epoch must be 1D arrays of same length.")

        self.time_window = time_window
        self.colors = colors or ["#E76F51", "#F4A261", "#E9C46A", "#2A9D8F"]
        self.labels = labels or ["AW", "QW", "NREM", "REM"]
        self.line_width = float(line_width)
        self.figure_size = figure_size
        self.dpi = int(dpi)
        self.bg_color = bg_color
        self.font_color = font_color
        self.show_y_axis = bool(show_y_axis)

        self.tmin = float(np.nanmin(self.t))
        self.tmax = float(np.nanmax(self.t))

        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
        })
        self._fig = None
        self._ax = None
        self._vline = None
        self._lines = []
        self._buf = BytesIO()

    def initialize(self):
        return

    def draw_frame(self, t: float):
        H = int(self.figure_size[1] * self.dpi)
        W = int(self.figure_size[0] * self.dpi)
        if t < self.tmin or t > self.tmax:
            return np.zeros((H, W, 3), dtype=np.uint8)

        t_start = t + self.time_window[0]
        t_end = t + self.time_window[1]
        mask = (self.t >= t_start) & (self.t <= t_end)
        if not np.any(mask):
            return np.zeros((H, W, 3), dtype=np.uint8)

        t_seg = self.t[mask]
        s_seg = self.state[mask]

        if self._fig is None or self._ax is None:
            self._fig, self._ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
            self._fig.patch.set_facecolor(self.bg_color)
            self._ax.set_facecolor(self.bg_color)
        else:
            self._ax.clear()
            self._ax.set_facecolor(self.bg_color)

        # draw colored segments with step transitions (no gaps)
        # determine starting state at t_start
        idx0 = np.searchsorted(self.t, t_start, side="right") - 1
        if idx0 < 0:
            start_state = int(self.state[0])
        else:
            start_state = int(self.state[idx0])

        times = [t_start]
        states = [start_state]
        if len(t_seg) > 0:
            times.extend(t_seg.tolist())
            states.extend([int(s) for s in s_seg.tolist()])
        times.append(t_end)
        states.append(states[-1])

        for i in range(len(times) - 1):
            s = int(states[i])
            c = self.colors[s % len(self.colors)]
            t0 = times[i]
            t1 = times[i + 1]
            self._ax.plot([t0, t1], [s, s], color=c, lw=self.line_width)
            if states[i + 1] != states[i]:
                self._ax.plot([t1, t1], [states[i], states[i + 1]], color=c, lw=self.line_width)

        self._ax.axvline(t, color="white" if self.bg_color == "black" else "gray",
                         lw=0.8, ls="--", alpha=0.6)

        self._ax.set_xlim(t_start, t_end)
        self._ax.set_ylim(-0.5, 3.5)
        self._ax.set_xticks([t_start, t, t_end])
        self._ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
        self._ax.set_yticks([0, 1, 2, 3])
        if self.show_y_axis:
            self._ax.set_yticklabels(self.labels[:4], color=self.font_color)
            for tick, idx in zip(self._ax.yaxis.get_major_ticks(), [0, 1, 2, 3]):
                tick.label1.set_color(self.colors[idx % len(self.colors)])
        else:
            self._ax.yaxis.set_visible(False)

        self._ax.tick_params(axis="x", colors=self.font_color)
        self._ax.tick_params(axis="y", colors=self.font_color)

        self._buf.seek(0)
        self._buf.truncate(0)
        self._fig.savefig(
            self._buf,
            format="png",
            facecolor=self._fig.get_facecolor(),
            bbox_inches=None,
            pad_inches=0,
        )
        self._buf.seek(0)
        img = plt.imread(self._buf)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        return (img * 255).astype(np.uint8)

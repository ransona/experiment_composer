import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from io import BytesIO
from PIL import Image
from typing import List, Optional, Tuple, Literal
from core.base_source import DataSource


class LinePlotSource(DataSource):
    """
    LinePlotSource — generates a time-aligned line plot image for one or more traces.
    Produces consistent, non-jittering frames suitable for video rendering.
    """

    def __init__(
        self,
        config: dict,
        time_vector: np.ndarray,
        y_values: List[np.ndarray],
        colors: Optional[List[str]] = None,
        time_window: Tuple[float, float] = (-2.0, 2.0),
        y_range_mode: Literal["global", "local", "fixed"] = "global",
        fixed_y_range: Optional[Tuple[float, float]] = None,
        y_label: Optional[str] = None,
        title: Optional[str] = None,
        show_y_axis: bool = True,
        line_width: float = 1.5,
        figure_size: Tuple[int, int] = (4, 2),
        dpi: int = 100,
        bg_color: str = "black",
        grid: bool = False,
        font_color: str = "white",
        interpolate: bool = False,
        backend: Literal["matplotlib", "fast"] = "matplotlib",
    ):
        self.config = config
        self.t = time_vector
        self.y = y_values
        self.colors = colors or plt.cm.tab10(np.linspace(0, 1, len(y_values)))
        self.time_window = time_window
        self.y_range_mode = y_range_mode
        self.fixed_y_range = fixed_y_range
        self.y_label = y_label
        self.title = title
        self.show_y_axis = show_y_axis
        self.line_width = line_width
        self.figure_size = figure_size
        self.dpi = dpi
        self.bg_color = bg_color
        self.grid = grid
        self.font_color = font_color
        self.interpolate = interpolate
        self.backend = backend

        # Precompute global Y range if needed
        if y_range_mode == "global":
            all_y = np.concatenate(y_values)
            self.global_ymin, self.global_ymax = np.nanmin(all_y), np.nanmax(all_y)
        else:
            self.global_ymin, self.global_ymax = None, None

        # Cache time range for boundary checks
        self.tmin, self.tmax = np.nanmin(time_vector), np.nanmax(time_vector)
        self._time_sorted = bool(np.all(np.diff(self.t) >= 0))

        # Consistent font and layout
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
        })

    def initialize(self):
        pass

    def draw_frame(self, t: float) -> np.ndarray:
        if self.backend == "fast":
            return self._draw_frame_fast(t)
        return self._draw_frame_matplotlib(t)

    def _get_window(self, t_start: float, t_end: float):
        if self._time_sorted:
            i0 = int(np.searchsorted(self.t, t_start, side="left"))
            i1 = int(np.searchsorted(self.t, t_end, side="right"))
            if i1 <= i0:
                return None, None
            return self.t[i0:i1], [yi[i0:i1] for yi in self.y]
        mask = (self.t >= t_start) & (self.t <= t_end)
        if not np.any(mask):
            return None, None
        return self.t[mask], [yi[mask] for yi in self.y]

    def _draw_frame_matplotlib(self, t: float) -> np.ndarray:
        profile = os.environ.get("COMPOSER_PROFILE", "").strip() in ("1", "true", "yes")
        t0 = time.perf_counter() if profile else None
        if t < self.tmin or t > self.tmax:
            print(f"[LinePlotSource WARNING] Requested time {t:.3f}s outside data range "
                  f"({self.tmin:.3f}–{self.tmax:.3f}s)")
            return np.zeros((self.figure_size[1]*self.dpi,
                             self.figure_size[0]*self.dpi, 3), dtype=np.uint8)

        t_start = t + self.time_window[0]
        t_end = t + self.time_window[1]
        t_seg, y_seg = self._get_window(t_start, t_end)
        if t_seg is None:
            print(f"[LinePlotSource WARNING] No samples in window for t={t:.3f}s")
            return np.zeros((self.figure_size[1]*self.dpi,
                             self.figure_size[0]*self.dpi, 3), dtype=np.uint8)

        # Determine Y-range
        if self.y_range_mode == "fixed" and self.fixed_y_range:
            ymin, ymax = self.fixed_y_range
        elif self.y_range_mode == "global" and self.global_ymin is not None:
            ymin, ymax = self.global_ymin, self.global_ymax
        else:
            seg_y = np.concatenate(y_seg)
            ymin, ymax = np.nanmin(seg_y), np.nanmax(seg_y)

        # Optional interpolation for smoothness
        if self.interpolate:
            t_interp = np.linspace(t_start, t_end, 400)
            y_interp = [np.interp(t_interp, t_seg, yi_seg) for yi_seg in y_seg]
            t_plot = t_interp
        else:
            t_plot = t_seg
            y_interp = y_seg

        # Create plot
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        fig.patch.set_facecolor(self.bg_color)
        ax.set_facecolor(self.bg_color)

        for yi, color in zip(y_interp, self.colors):
            ax.plot(t_plot, yi, color=color, lw=self.line_width)

        # Vertical line for current time
        ax.axvline(t, color="white" if self.bg_color == "black" else "gray",
                   lw=0.8, ls="--", alpha=0.6)

        # Fixed axis ranges and ticks (no jitter)
        ax.set_xlim(t_start, t_end)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(np.linspace(t_start, t_end, 5))
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))

        ax.tick_params(axis="x", colors=self.font_color)
        ax.tick_params(axis="y", colors=self.font_color)
        if not self.show_y_axis:
            ax.yaxis.set_visible(False)
        if self.y_label:
            ax.set_ylabel(self.y_label, color=self.font_color)
        if self.title:
            ax.set_title(self.title, color=self.font_color, pad=8)
        if self.grid:
            ax.grid(True, color="gray", alpha=0.3, lw=0.5)

        # Save without cropping — keeps dimensions constant
        buf = BytesIO()
        plt.savefig(
            buf,
            format="png",
            facecolor=fig.get_facecolor(),
            bbox_inches=None,  # <- important: no jitter
            pad_inches=0,
        )
        plt.close(fig)
        buf.seek(0)

        img = Image.open(buf).convert("RGB")
        out = np.array(img, dtype=np.uint8)
        if profile and t0 is not None:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[Profile] LinePlotSource(matplotlib): {dt_ms:.1f}ms")
        return out

    def _draw_frame_fast(self, t: float) -> np.ndarray:
        profile = os.environ.get("COMPOSER_PROFILE", "").strip() in ("1", "true", "yes")
        t0 = time.perf_counter() if profile else None
        H = int(self.figure_size[1] * self.dpi)
        W = int(self.figure_size[0] * self.dpi)
        if t < self.tmin or t > self.tmax:
            return np.zeros((H, W, 3), dtype=np.uint8)

        t_start = t + self.time_window[0]
        t_end = t + self.time_window[1]
        t_seg, y_seg = self._get_window(t_start, t_end)
        if t_seg is None:
            return np.zeros((H, W, 3), dtype=np.uint8)

        if self.y_range_mode == "fixed" and self.fixed_y_range:
            ymin, ymax = self.fixed_y_range
        elif self.y_range_mode == "global" and self.global_ymin is not None:
            ymin, ymax = self.global_ymin, self.global_ymax
        else:
            seg_y = np.concatenate(y_seg)
            ymin, ymax = np.nanmin(seg_y), np.nanmax(seg_y)
        if ymax <= ymin:
            ymax = ymin + 1.0

        if self.interpolate:
            t_plot = np.linspace(t_start, t_end, 400)
            y_plot = [np.interp(t_plot, t_seg, yi_seg) for yi_seg in y_seg]
        else:
            t_plot = t_seg
            y_plot = y_seg

        bg = self._color_to_bgr(self.bg_color)
        img = np.full((H, W, 3), bg, dtype=np.uint8)

        left = 40 if self.show_y_axis else 10
        right = 10
        top = 20 if self.title else 10
        bottom = 20
        plot_w = max(1, W - left - right)
        plot_h = max(1, H - top - bottom)

        def x_to_px(x):
            return left + (x - t_start) / (t_end - t_start) * plot_w

        def y_to_px(y):
            return top + (1.0 - (y - ymin) / (ymax - ymin)) * plot_h

        if self.grid:
            for i in range(5):
                x = left + int(i * plot_w / 4)
                cv2.line(img, (x, top), (x, top + plot_h), (80, 80, 80), 1)
            for i in range(5):
                y = top + int(i * plot_h / 4)
                cv2.line(img, (left, y), (left + plot_w, y), (80, 80, 80), 1)

        for yi, color in zip(y_plot, self.colors):
            if yi.size == 0:
                continue
            pts = np.column_stack([x_to_px(t_plot), y_to_px(yi)])
            pts = np.round(pts).astype(np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
            if len(pts) >= 2:
                cv2.polylines(img, [pts], False, self._color_to_bgr(color), int(max(1, round(self.line_width))))

        x_now = int(np.clip(x_to_px(t), left, left + plot_w))
        cv2.line(img, (x_now, top), (x_now, top + plot_h), (200, 200, 200), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = self._color_to_bgr(self.font_color)
        if self.title:
            cv2.putText(img, self.title, (left, max(12, top - 5)), font, 0.4, font_color, 1, cv2.LINE_AA)
        if self.show_y_axis and self.y_label:
            cv2.putText(img, self.y_label, (5, top + 12), font, 0.35, font_color, 1, cv2.LINE_AA)

        if profile and t0 is not None:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            print(f"[Profile] LinePlotSource(fast): {dt_ms:.1f}ms")
        return img

    def _color_to_bgr(self, color):
        if isinstance(color, (list, tuple, np.ndarray)):
            vals = list(color)
            if len(vals) >= 3:
                if max(vals) <= 1.0:
                    vals = [int(v * 255) for v in vals[:3]]
                else:
                    vals = [int(v) for v in vals[:3]]
                return (vals[2], vals[1], vals[0])
        if isinstance(color, str):
            name = color.strip().lower()
            if name.startswith("#") and len(name) in (7, 9):
                r = int(name[1:3], 16)
                g = int(name[3:5], 16)
                b = int(name[5:7], 16)
                return (b, g, r)
            lut = {
                "black": (0, 0, 0),
                "white": (255, 255, 255),
                "gray": (128, 128, 128),
                "grey": (128, 128, 128),
                "red": (0, 0, 255),
                "green": (0, 255, 0),
                "blue": (255, 0, 0),
                "cyan": (255, 255, 0),
                "magenta": (255, 0, 255),
                "yellow": (0, 255, 255),
            }
            if name in lut:
                return lut[name]
        return (255, 255, 255)

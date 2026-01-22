import numpy as np
import matplotlib.pyplot as plt
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

        # Precompute global Y range if needed
        if y_range_mode == "global":
            all_y = np.concatenate(y_values)
            self.global_ymin, self.global_ymax = np.nanmin(all_y), np.nanmax(all_y)
        else:
            self.global_ymin, self.global_ymax = None, None

        # Cache time range for boundary checks
        self.tmin, self.tmax = np.nanmin(time_vector), np.nanmax(time_vector)

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
        if t < self.tmin or t > self.tmax:
            print(f"[LinePlotSource WARNING] Requested time {t:.3f}s outside data range "
                  f"({self.tmin:.3f}–{self.tmax:.3f}s)")
            return np.zeros((self.figure_size[1]*self.dpi,
                             self.figure_size[0]*self.dpi, 3), dtype=np.uint8)

        t_start = t + self.time_window[0]
        t_end = t + self.time_window[1]
        mask = (self.t >= t_start) & (self.t <= t_end)
        if not np.any(mask):
            print(f"[LinePlotSource WARNING] No samples in window for t={t:.3f}s")
            return np.zeros((self.figure_size[1]*self.dpi,
                             self.figure_size[0]*self.dpi, 3), dtype=np.uint8)

        # Determine Y-range
        if self.y_range_mode == "fixed" and self.fixed_y_range:
            ymin, ymax = self.fixed_y_range
        elif self.y_range_mode == "global" and self.global_ymin is not None:
            ymin, ymax = self.global_ymin, self.global_ymax
        else:
            seg_y = np.concatenate([yi[mask] for yi in self.y])
            ymin, ymax = np.nanmin(seg_y), np.nanmax(seg_y)

        # Optional interpolation for smoothness
        if self.interpolate:
            t_interp = np.linspace(t_start, t_end, 400)
            y_interp = [np.interp(t_interp, self.t, yi) for yi in self.y]
            t_plot = t_interp
        else:
            t_plot = self.t[mask]
            y_interp = [yi[mask] for yi in self.y]

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
        return np.array(img, dtype=np.uint8)

import os
import pickle
from typing import List, Optional, Tuple, Literal

import numpy as np

from core.base_source import DataSource
from sources.line_plot_source import LinePlotSource


class WheelSpeedSource(DataSource):
    """
    WheelSpeedSource — loads wheel speed from recordings/wheel.pickle and renders
    a time-aligned line plot using LinePlotSource.
    """

    def __init__(
        self,
        exp_dir_processed: str,
        *,
        time_window: Tuple[float, float] = (-5.0, 5.0),
        y_range_mode: Literal["global", "local", "fixed"] = "global",
        fixed_y_range: Optional[Tuple[float, float]] = None,
        y_label: str = "",
        title: str = "Run speed",
        show_y_axis: bool = True,
        line_width: float = 1.5,
        figure_size: Tuple[int, int] = (4, 2),
        dpi: int = 100,
        bg_color: str = "black",
        grid: bool = False,
        font_color: str = "white",
        interpolate: bool = True,
        colors: Optional[List[str]] = None,
    ):
        super().__init__()
        wheel_path = os.path.join(exp_dir_processed, "recordings", "wheel.pickle")
        if not os.path.exists(wheel_path):
            raise FileNotFoundError(f"Wheel data not found: {wheel_path}")

        with open(wheel_path, "rb") as f:
            wheel_data = pickle.load(f)
        if "t" not in wheel_data or "speed" not in wheel_data:
            raise ValueError("wheel.pickle missing required keys: 't' and 'speed'.")

        wheel_time = np.asarray(wheel_data["t"])
        wheel_trace = np.asarray(wheel_data["speed"])

        self._plotter = LinePlotSource(
            config={},
            time_vector=wheel_time,
            y_values=[wheel_trace],
            colors=colors or ["cyan"],
            title=title,
            y_label=y_label,
            time_window=time_window,
            y_range_mode=y_range_mode,
            fixed_y_range=fixed_y_range,
            show_y_axis=show_y_axis,
            line_width=line_width,
            figure_size=figure_size,
            dpi=dpi,
            bg_color=bg_color,
            grid=grid,
            font_color=font_color,
            interpolate=interpolate,
        )

    def initialize(self):
        return

    def draw_frame(self, t):
        return self._plotter.draw_frame(t)

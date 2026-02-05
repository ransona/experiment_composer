import os
from typing import List, Optional, Tuple, Literal

import numpy as np

from core.base_source import DataSource
from sources.line_plot_source import LinePlotSource


class NumpyTraceSource(DataSource):
    """
    NumpyTraceSource — loads a 2D array where col0 = time and cols1..N are signals.
    Supports .npy or .npz (optional key).

    If columns is provided, it refers to original file column indices (so 0 is time).
    """

    def __init__(
        self,
        path: str,
        *,
        key: str = "",
        columns: Optional[List[int]] = None,
        time_window: Tuple[float, float] = (-5.0, 0.0),
        y_range_mode: Literal["global", "local", "fixed"] = "global",
        fixed_y_range: Optional[Tuple[float, float]] = None,
        y_label: str = "",
        title: str = "Numpy trace",
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
        if not path:
            raise ValueError("path is required for NumpyTraceSource.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Numpy trace file not found: {path}")

        data = np.load(path, allow_pickle=False)
        if isinstance(data, np.lib.npyio.NpzFile):
            if key:
                if key not in data:
                    raise ValueError(f"Key '{key}' not found in {path}.")
                arr = data[key]
            else:
                keys = list(data.keys())
                if not keys:
                    raise ValueError(f"No arrays found in {path}.")
                arr = data[keys[0]]
        else:
            arr = data

        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError("Array must be 2D with at least 2 columns (time + signals).")

        t = np.asarray(arr[:, 0])
        y = np.asarray(arr[:, 1:])

        if columns:
            cols = [int(c) for c in columns]
            y_sel = np.asarray(arr[:, cols])
        else:
            y_sel = y

        if y_sel.ndim == 1:
            y_traces = [y_sel]
        else:
            y_traces = [y_sel[:, i] for i in range(y_sel.shape[1])]

        self._plotter = LinePlotSource(
            config={},
            time_vector=t,
            y_values=y_traces,
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

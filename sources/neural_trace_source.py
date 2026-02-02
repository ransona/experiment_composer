import os
import pickle
from typing import List, Optional, Tuple, Literal

import numpy as np

from core.base_source import DataSource
from sources.line_plot_source import LinePlotSource


class NeuralTraceSource(DataSource):
    """
    NeuralTraceSource — loads s2p_chX.pickle and renders the mean activity
    across selected neurons for a chosen signal key.
    """

    def __init__(
        self,
        exp_dir_processed: str,
        *,
        channel: int = 0,
        signal_key: str = "Spikes",
        neuron_indices: Optional[List[int]] = None,
        time_window: Tuple[float, float] = (-5.0, 0.0),
        y_range_mode: Literal["global", "local", "fixed"] = "global",
        fixed_y_range: Optional[Tuple[float, float]] = None,
        y_label: str = "",
        title: str = "Mean population activity",
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
        data_path = os.path.join(exp_dir_processed, "recordings", f"s2p_ch{channel}.pickle")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Neural trace file not found: {data_path}")

        with open(data_path, "rb") as f:
            ca_data = pickle.load(f)
        if "t" not in ca_data:
            raise ValueError("s2p_chX.pickle missing required key: 't'.")
        if signal_key not in ca_data:
            raise ValueError(f"s2p_chX.pickle missing required key: '{signal_key}'.")

        time_vector = np.asarray(ca_data["t"])
        traces = np.asarray(ca_data[signal_key])
        if traces.ndim != 2:
            raise ValueError("Signal array must be 2D (neurons x time or time x neurons).")

        if neuron_indices:
            neuron_indices = [int(i) for i in neuron_indices]
        else:
            neuron_indices = []

        if traces.shape[0] == time_vector.shape[0] and traces.shape[1] != time_vector.shape[0]:
            traces = traces.T

        if neuron_indices:
            traces_sel = traces[neuron_indices, :]
        else:
            traces_sel = traces

        mean_trace = np.nanmean(traces_sel, axis=0)

        self._plotter = LinePlotSource(
            config={},
            time_vector=time_vector,
            y_values=[mean_trace],
            colors=colors or ["cyan", "magenta"],
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

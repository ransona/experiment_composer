"""
VideoBinSource — Suite2p binary video reader for Experiment Composer.

Loads a .bin file (typically Suite2p output) and exposes frames as
time-indexed images compatible with the core composer system.
"""

import os
import numpy as np
import cv2
from typing import Any, Dict, Optional
from core.base_source import DataSource


class VideoBinSource(DataSource):
    """
    Data source for Suite2p binary videos, using known frame timestamps.

    Configuration dictionary (cfg) fields:
        path:              str, path to data.bin
        height:            int, image height
        width:             int, image width
        frame_times:       np.ndarray, per-frame timeline times (seconds)
        stride:            int, spatial downsampling factor (default 1)
        spatial_sigma:     float, Gaussian blur sigma (optional)
        temporal_window:   int, temporal median window size (optional)
        label:             str, optional label for overlay
        vmin, vmax:        float, manual intensity scaling (optional)
        auto_scale_sample: int, number of frames to sample for auto-scaling
    """

    # ---------------------------------------------------------------
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.mm = None
        self.frame_times = np.asarray(cfg["frame_times"])
        self.frame_size = None
        self.temporal_buffer = []
        self._parse_cfg()

    # ---------------------------------------------------------------
    def _parse_cfg(self):
        c = self.cfg
        self.path = c["path"]
        self.height = int(c.get("height", 512))
        self.width = int(c.get("width", 512))
        self.stride = int(c.get("stride", 1))
        self.spatial_sigma = float(c.get("spatial_sigma", 0.0))
        self.temporal_window = int(c.get("temporal_window", 0))
        self.label = c.get("label", None)
        self.vmin = c.get("vmin", None)
        self.vmax = c.get("vmax", None)
        self.auto_scale_sample = int(c.get("auto_scale_sample", 200))

    # ---------------------------------------------------------------
    def initialize(self):
        """Open memmap and compute scaling parameters."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Missing binary file: {self.path}")

        self.mm = np.memmap(self.path, dtype=np.int16, mode="r")
        self.frame_size = self.height * self.width
        n_frames = self.mm.size // self.frame_size

        # --- Handle mismatch between frame count and timestamps ---
        if len(self.frame_times) != n_frames:
            diff = n_frames - len(self.frame_times)
            print(f"[Warning] Frame count mismatch: bin={n_frames}, "
                  f"times={len(self.frame_times)} (Δ={diff})")

            if abs(diff) == 1:
                # One frame mismatch → trim to match
                if diff > 0:
                    # Bin longer → ignore last frame
                    n_frames = len(self.frame_times)
                else:
                    # frame_times longer → trim timestamps
                    self.frame_times = self.frame_times[:n_frames]
                print(f"[Info] Adjusted to {n_frames} frames for alignment.")
            else:
                raise ValueError(
                    f"Frame count mismatch too large (Δ={diff}). "
                    "Check acquisition logs."
                )

        # --- Compute auto contrast scaling if needed ---
        if self.vmin is None or self.vmax is None:
            idxs = np.linspace(0, n_frames - 1,
                               min(self.auto_scale_sample, n_frames),
                               dtype=int)
            samples = [self._get_frame(i) for i in idxs]
            stack = np.stack(samples, axis=0).astype(np.float32)
            self.vmin = float(np.percentile(stack, 1))
            self.vmax = float(np.percentile(stack, 99))
            if self.vmax <= self.vmin:
                self.vmax = self.vmin + 1.0
            print(f"[Auto scale] vmin={self.vmin:.1f}, vmax={self.vmax:.1f}")

        self.temporal_buffer = []

    # ---------------------------------------------------------------
    def _get_frame(self, idx: int) -> np.ndarray:
        """Return raw 2D frame from memmap."""
        s = idx * self.frame_size
        e = s + self.frame_size
        f = self.mm[s:e].reshape(self.height, self.width)
        if self.stride > 1:
            f = f[::self.stride, ::self.stride]
        return f

    # ---------------------------------------------------------------
    def _apply_filters(self, f: np.ndarray) -> np.ndarray:
        """Apply optional spatial and temporal filtering."""
        if self.spatial_sigma > 0:
            from scipy.ndimage import gaussian_filter
            f = gaussian_filter(f, self.spatial_sigma)

        if self.temporal_window > 1:
            self.temporal_buffer.append(f)
            if len(self.temporal_buffer) > self.temporal_window:
                self.temporal_buffer.pop(0)
            if len(self.temporal_buffer) > 1:
                f = np.median(np.stack(self.temporal_buffer, axis=0), axis=0)

        return f

    # ---------------------------------------------------------------
    def draw_frame(self, t: float) -> np.ndarray:
        """
        Return RGB frame corresponding to nearest timeline time t.
        """
        if self.mm is None:
            raise RuntimeError("VideoBinSource not initialized.")

        idx = np.searchsorted(self.frame_times, t, side="right") - 1
        idx = int(np.clip(idx, 0, len(self.frame_times) - 1))

        f = self._get_frame(idx)
        f = self._apply_filters(f)

        # Normalize to 8-bit
        img = np.clip((f - self.vmin) / (self.vmax - self.vmin) * 255, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if self.label:
            cv2.putText(img, self.label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)

        return img

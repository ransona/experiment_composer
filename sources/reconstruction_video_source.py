# sources/reconstruction_video_source.py
import os
import cv2
import numpy as np
from collections import OrderedDict
from typing import Optional, Tuple
from core.base_source import DataSource


def _odd_kernel_from_sigma(sigma: float, min_ksize: int = 3) -> int:
    if sigma <= 0:
        return 0
    # 6*sigma rule → odd kernel
    k = int(np.ceil(sigma * 6))
    if k % 2 == 0:
        k += 1
    return max(min_ksize, k)


class ReconstructionVideoSource(DataSource):
    """
    Returns the frame from a video whose frame timestamps (timeline time)
    are provided as a 1D NumPy array. Supports:
      - on-demand decoding (cv2.VideoCapture)
      - optional temporal box smoothing over frames
      - optional linear time interpolation between neighboring frames
      - optional spatial Gaussian blur on the output frame
      - optional polygon outline overlay from mask_edges.npy
      - optional high-res (supersampled) overlay rendering independent of video res
    """

    def __init__(
        self,
        video_path: str,
        timestamps_path: str,
        *,
        enable_temporal_filter: bool = False,
        temporal_window: int = 0,          # frames on each side (total = 2*window+1)
        enable_spatial_filter: bool = False,
        spatial_sigma: float = 0.0,        # Gaussian sigma in pixels
        interpolate: bool = False,         # linear blend using fractional timeline
        cache_size: int = 128,             # decoded-frame LRU cache size
        # --- overlay options ---
        overlay_edges: bool = False,
        edges_path: Optional[str] = None,
        edge_color: Tuple[int, int, int] = (0, 255, 0),   # BGR red
        edge_thickness: int = 3,
        close_edges: bool = True,          # draw closed polygon outline
        overlay_render_scale: float = 1.0, # >1.0 draws overlay at higher res, then downsamples
        overlay_alpha: float = 1.0,        # 0..1 when compositing overlay onto frame
    ):
        super().__init__()

        self.video_path = video_path
        self.timestamps_path = timestamps_path

        self.enable_temporal = bool(enable_temporal_filter)
        self.twin = int(max(0, temporal_window))
        self.enable_spatial = bool(enable_spatial_filter)
        self.sigma = float(max(0.0, spatial_sigma))
        self.interpolate = bool(interpolate)

        # LRU decode cache (frame_idx -> np.ndarray(H,W,3), uint8)
        self.cache_size = max(cache_size, 2 * self.twin + 4)
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

        # overlay config
        self.overlay_edges = bool(overlay_edges)
        self.edges_path = edges_path  # may be None → resolved below
        self.edge_color = tuple(int(c) for c in edge_color)
        self.edge_thickness = int(max(1, edge_thickness))
        self.close_edges = bool(close_edges)
        self.overlay_render_scale = float(max(1.0, overlay_render_scale))
        self.overlay_alpha = float(min(max(overlay_alpha, 0.0), 1.0))
        self._edges_pts: Optional[np.ndarray] = None  # Nx2 int32
        self._edges_pts_hr: Optional[np.ndarray] = None  # Nx2 int32 for high-res draw

        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        if not os.path.exists(self.timestamps_path):
            raise FileNotFoundError(f"Timestamps file not found: {self.timestamps_path}")

        # Load timeline timestamps (seconds in timeline time; 1D, non-decreasing)
        stamps = np.load(self.timestamps_path)
        if stamps.ndim != 1:
            raise ValueError("Timestamps array must be 1D (one timestamp per frame).")
        if np.any(np.diff(stamps) < 0):
            raise ValueError("Timestamps must be non-decreasing.")
        self.timeline_stamps = stamps.astype(np.float64, copy=False)

        # Video capture
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"Failed to open video: {self.video_path}")

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.n_frames = int(min(total_frames, self.timeline_stamps.shape[0]))
        if total_frames != self.timeline_stamps.shape[0]:
            print(
                f"[ReconstructionVideoSource] Warning: video has {total_frames} frames, "
                f"timestamps array has {self.timeline_stamps.shape[0]}. Using n_frames={self.n_frames}."
            )

        # Frame size (for clipping overlay coords)
        self.frame_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        self.frame_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None

        # Precompute bounds and defaults
        self.t_min = float(self.timeline_stamps[0])
        self.t_max = float(self.timeline_stamps[self.n_frames - 1])
        self._default_grey = np.full((256, 256, 3), 127, dtype=np.uint8)
        self._ksize = _odd_kernel_from_sigma(self.sigma)

        print(
            f"[ReconstructionVideoSource] Opened '{self.video_path}' "
            f"frames={total_frames}, usable={self.n_frames}; timestamps={self.timeline_stamps.shape[0]}"
        )

        # Load overlay points if requested
        if self.overlay_edges:
            resolved = self.edges_path or os.path.join(os.path.dirname(self.video_path), "mask_edges.npy")
            if not os.path.exists(resolved):
                print(f"[ReconstructionVideoSource] Overlay enabled, but edges file not found: {resolved}")
            else:
                try:
                    edges = np.load(resolved)
                    if edges.ndim != 2 or edges.shape[1] != 2:
                        raise ValueError("mask_edges.npy must be an Nx2 array of (x,y).")
                    pts = np.round(edges).astype(np.int32)

                    # clip to frame
                    if self.frame_w and self.frame_h:
                        pts[:, 0] = np.clip(pts[:, 0], 0, self.frame_w - 1)
                        pts[:, 1] = np.clip(pts[:, 1], 0, self.frame_h - 1)
                    self._edges_pts = pts

                    # precompute high-res points if needed
                    if self.overlay_render_scale > 1.0:
                        s = self.overlay_render_scale
                        pts_hr = np.round(pts.astype(np.float64) * s).astype(np.int32)
                        self._edges_pts_hr = pts_hr

                    print(f"[ReconstructionVideoSource] Loaded {len(pts)} edge points from {resolved}")
                except Exception as e:
                    print(f"[ReconstructionVideoSource] Failed to load edges from {resolved}: {e}")

    # --------------------------------------------------------------
    def initialize(self):
        """Optional hook for composer; nothing to do for this source."""
        return

    # --------------------------------------------------------------
    def _cache_get(self, idx: int) -> Optional[np.ndarray]:
        f = self._cache.get(idx)
        if f is not None:
            self._cache.move_to_end(idx)
        return f

    def _cache_put(self, idx: int, frame: np.ndarray) -> None:
        self._cache[idx] = frame
        self._cache.move_to_end(idx)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    # --------------------------------------------------------------
    def _read_frame(self, frame_idx: int) -> np.ndarray:
        """Decode a single frame (uint8 BGR). Uses LRU cache."""
        if frame_idx < 0 or frame_idx >= self.n_frames:
            return self._default_grey

        cached = self._cache_get(frame_idx)
        if cached is not None:
            return cached

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = self.cap.read()
        if not ok or frame is None:
            print(f"[ReconstructionVideoSource] ⚠️ Failed to read frame {frame_idx}")
            return self._default_grey

        frame_u8 = frame.copy()
        self._cache_put(frame_idx, frame_u8)
        return frame_u8

    # --------------------------------------------------------------
    def _temporal_aggregate(self, center_idx: int) -> np.ndarray:
        """
        Return float32 frame after optional temporal smoothing.
        - If temporal filter disabled: returns center frame as float32.
        - If enabled: boxcar mean over [center - twin, center + twin], clamped to valid range.
        """
        if not self.enable_temporal or self.twin <= 0:
            return self._read_frame(center_idx).astype(np.float32)

        lo = max(0, center_idx - self.twin)
        hi = min(self.n_frames - 1, center_idx + self.twin)
        count = hi - lo + 1

        acc = None
        for i in range(lo, hi + 1):
            f = self._read_frame(i).astype(np.float32)
            acc = f if acc is None else acc + f
        return acc / float(count)

    # --------------------------------------------------------------
    def _apply_spatial(self, frame_float: np.ndarray) -> np.ndarray:
        """Apply optional Gaussian blur; returns uint8."""
        out = frame_float
        if self.enable_spatial and self.sigma > 0.0:
            if self._ksize <= 1:
                self._ksize = _odd_kernel_from_sigma(self.sigma)
            out = cv2.GaussianBlur(out, (self._ksize, self._ksize), self.sigma)
        return np.clip(out, 0, 255).astype(np.uint8)

    # --------------------------------------------------------------
    def _render_edges_overlay(self, h: int, w: int) -> Optional[np.ndarray]:
        """
        Render the polygon outline to a separate overlay image.
        If overlay_render_scale > 1, draw at higher res then downsample with INTER_AREA.
        Returns BGR uint8 overlay of shape (h, w, 3) with black background,
        or None if no edges.
        """
        if not (self.overlay_edges and self._edges_pts is not None and len(self._edges_pts) >= 2):
            return None

        s = self.overlay_render_scale
        if s <= 1.0:
            # draw directly at native size
            overlay = np.zeros((h, w, 3), dtype=np.uint8)
            pts = self._edges_pts.reshape(-1, 1, 2)
            cv2.polylines(
                overlay, [pts],
                isClosed=self.close_edges,
                color=self.edge_color,
                thickness=self.edge_thickness,
                lineType=cv2.LINE_AA,
            )
            return overlay

        # high-res draw
        hr_h, hr_w = int(round(h * s)), int(round(w * s))
        overlay_hr = np.zeros((hr_h, hr_w, 3), dtype=np.uint8)
        pts_hr = (self._edges_pts_hr if self._edges_pts_hr is not None
                  else np.round(self._edges_pts.astype(np.float64) * s).astype(np.int32))
        t_hr = max(1, int(round(self.edge_thickness * s)))

        cv2.polylines(
            overlay_hr, [pts_hr.reshape(-1, 1, 2)],
            isClosed=self.close_edges,
            color=self.edge_color,
            thickness=t_hr,
            lineType=cv2.LINE_AA,
        )

        # downsample overlay back to native size (area averaging → crisp AA edges)
        overlay = cv2.resize(overlay_hr, (w, h), interpolation=cv2.INTER_AREA)
        return overlay

    # --------------------------------------------------------------
    def _apply_overlay(self, frame_u8: np.ndarray) -> np.ndarray:
        """
        Composite the (possibly supersampled) overlay onto the original frame.
        Uses full replacement where overlay nonzero if overlay_alpha == 1.0,
        else alpha blend only at overlay pixels.
        """
        if not self.overlay_edges:
            return frame_u8

        h, w = frame_u8.shape[:2]
        overlay = self._render_edges_overlay(h, w)
        if overlay is None:
            return frame_u8

        # mask where overlay draws
        mask_gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)

        if self.overlay_alpha >= 1.0:
            out = frame_u8.copy()
            cv2.copyTo(overlay, mask_gray, out)  # copy only where mask > 0
            return out

        # partial alpha
        alpha = self.overlay_alpha
        out = frame_u8.astype(np.float32)
        ol = overlay.astype(np.float32)

        m = (mask_gray > 0).astype(np.float32)[..., None]  # (H,W,1)
        blended = out * (1.0 - alpha) + ol * alpha
        out = np.where(m > 0, blended, out)
        return np.clip(out, 0, 255).astype(np.uint8)

    # --------------------------------------------------------------
    def _map_time_to_indices(self, timeline_time: float):
        """
        Map timeline_time to (left_idx, right_idx, alpha) for linear interpolation.
        If out of range, returns (None, None, 0).
        """
        if timeline_time < self.t_min or timeline_time > self.t_max:
            return None, None, 0.0

        stamps = self.timeline_stamps
        n = self.n_frames

        r = int(np.searchsorted(stamps[:n], timeline_time, side="right"))
        r = min(r, n - 1)
        l = max(0, r - 1)

        t0 = float(stamps[l])
        t1 = float(stamps[r])

        if self.interpolate and r != l and t1 > t0:
            alpha = float((timeline_time - t0) / (t1 - t0))
            alpha = min(max(alpha, 0.0), 1.0)
        else:
            alpha = 0.0  # use left only
            if abs(timeline_time - t1) < abs(timeline_time - t0):
                l = r

        return l, r, alpha

    # --------------------------------------------------------------
    def get_frame_at_time(self, timeline_time: float) -> np.ndarray:
        """
        Return frame (uint8 BGR) for the requested timeline time.
        Order:
          1) map time → indices (l,r,alpha)
          2) temporal smoothing on l and r (if enabled)
          3) optional linear interpolation (if enabled and l!=r)
          4) optional spatial blur
          5) overlay outline (possibly supersampled), composited over original
        """
        l, r, alpha = self._map_time_to_indices(timeline_time)
        if l is None:
            return self._default_grey

        f0 = self._temporal_aggregate(l).astype(np.float32)

        if self.interpolate and r is not None and r != l and 0.0 < alpha < 1.0:
            f1 = self._temporal_aggregate(r).astype(np.float32)
            mixed = (1.0 - alpha) * f0 + alpha * f1
            out = self._apply_spatial(mixed)
            out = np.clip(out, 0, 255).astype(np.uint8)
            return self._apply_overlay(out)

        out = self._apply_spatial(f0)
        out = np.clip(out, 0, 255).astype(np.uint8)
        return self._apply_overlay(out)

    # --------------------------------------------------------------
    def draw_frame(self, timeline_time: float):
        return self.get_frame_at_time(timeline_time)

    # --------------------------------------------------------------
    def set_filters(
        self,
        *,
        enable_temporal_filter: Optional[bool] = None,
        temporal_window: Optional[int] = None,
        enable_spatial_filter: Optional[bool] = None,
        spatial_sigma: Optional[float] = None,
        interpolate: Optional[bool] = None,
        cache_size: Optional[int] = None,
    ):
        """Runtime update of filter options."""
        if enable_temporal_filter is not None:
            self.enable_temporal = bool(enable_temporal_filter)
        if temporal_window is not None:
            self.twin = int(max(0, temporal_window))
        if enable_spatial_filter is not None:
            self.enable_spatial = bool(enable_spatial_filter)
        if spatial_sigma is not None:
            self.sigma = float(max(0.0, spatial_sigma))
            self._ksize = _odd_kernel_from_sigma(self.sigma)
        if interpolate is not None:
            self.interpolate = bool(interpolate)
        if cache_size is not None:
            self.cache_size = max(int(cache_size), 2 * self.twin + 4)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    # --------------------------------------------------------------
    def __del__(self):
        try:
            if hasattr(self, "cap") and self.cap is not None and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass

# sources/eye_source.py

import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from typing import Optional, Tuple, Sequence, Union
from core.base_source import DataSource


class EyeSource(DataSource):
    """
    Eye video with REQUIRED per-frame timestamps, optional ROI crop (matplotlib),
    contrast clipping, and DLC overlays.

    Video path resolver (correct mapping):
        eye='right' → {exp_dir_processed}/{expID}_eye1_right.avi
        eye='left'  → {exp_dir_processed}/{expID}_eye1_left.avi

    DLC file (as provided earlier):
        {exp_dir_processed}/recordings/dlcEyeRight.pickle
    """

    def __init__(
        self,
        exp_dir_processed: str,
        expID: str,
        eye: str,                         # 'right' or 'left'
        timestamps_path: str,             # REQUIRED: 1D npy (timeline time per frame)
        *,
        crop: Union[Sequence[float], bool] = False,  # (x,y,w,h) px OR normalized if all in [0,1]
        plot_detected_pupil: bool = False,
        plot_detected_eye: bool = False,
        overlay_thickness: int = 2,
        contrast_clip_percentiles: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()

        eye_norm = eye.lower().strip()
        if eye_norm not in ("right", "left"):
            raise ValueError("eye must be 'right' or 'left'")

        # ---- correct video mapping (no reversal) ----
        suffix = "right" if eye_norm == "right" else "left"
        self.video_path = os.path.join(exp_dir_processed, f"{expID}_eye1_{suffix}.avi")
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Eye video not found: {self.video_path}")

        # ---- required timestamps ----
        if not os.path.exists(timestamps_path):
            raise FileNotFoundError(f"Timestamps file not found: {timestamps_path}")
        ts = np.load(timestamps_path)
        if ts.ndim != 1:
            raise ValueError("timestamps .npy must be 1D (one per frame).")
        if np.any(np.diff(ts) < 0):
            raise ValueError("timestamps must be non-decreasing.")
        self.timestamps = ts.astype(np.float64, copy=False)

        # ---- DLC overlays ----
        dlc_suffix = "Right" if eye_norm == "right" else "Left"
        self.dlc_path = os.path.join(exp_dir_processed, "recordings", f"dlcEye{dlc_suffix}.pickle")
        if os.path.exists(self.dlc_path):
            with open(self.dlc_path, "rb") as f:
                self.dlc = pickle.load(f)
        else:
            print(f"[EyeSource] Warning: DLC file not found: {self.dlc_path}")
            self.dlc = None

        # ---- options ----
        self._requested_crop = crop
        self.crop_rect: Optional[Tuple[int, int, int, int]] = None
        self.plot_pupil = bool(plot_detected_pupil)
        self.plot_eye = bool(plot_detected_eye)
        self.overlay_thickness = int(max(1, overlay_thickness))
        self.clip_percentiles = contrast_clip_percentiles
        self._clip_range: Optional[Tuple[float, float]] = None

        # ---- open video ----
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"Failed to open video: {self.video_path}")

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total_frames <= 0:
            raise IOError("Video reports zero frames.")
        if total_frames != self.timestamps.size:
            print(f"[EyeSource] Warning: video frames={total_frames} != timestamps={self.timestamps.size}. Using min length.")
        self.n_frames = min(total_frames, self.timestamps.size)

        # first frame → record size, then crop/contrast setup
        first = self._read_frame(0)
        if first is None:
            raise IOError("Could not read first frame from eye video.")
        self.full_H, self.full_W = first.shape[:2]

        self._init_crop(first)
        if self.clip_percentiles is not None:
            self._init_clip_range(first)

        # align with DLC length if present (avoid index errors)
        if self.dlc is not None:
            try:
                dlc_len = len(self.dlc.get("eyeX", []))
                if dlc_len and dlc_len != self.n_frames:
                    m = min(self.n_frames, dlc_len)
                    print(f"[EyeSource] Adjusting usable frames to {m} to match DLC length.")
                    self.n_frames = m
            except Exception:
                pass

        self.t_min = float(self.timestamps[0])
        self.t_max = float(self.timestamps[self.n_frames - 1])

        print(
            f"[EyeSource] open={os.path.basename(self.video_path)}, "
            f"frames={self.n_frames}, crop={self.crop_rect}, clip={self._clip_range}"
        )

    # -------------------------------------------------------------------------
    def initialize(self):
        return

    # -------------------------------------------------------------------------
    def _select_roi_matplotlib(self, frame_bgr: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        img = frame_bgr
        if img.ndim == 2:
            img_rgb = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img[..., :3]

        H, W = img_rgb.shape[:2]
        roi = {"rect": None}

        fig, ax = plt.subplots()
        ax.imshow(img_rgb)
        ax.set_title("Drag to select ROI. Enter=confirm, Esc=cancel.")
        plt.tight_layout()

        def onselect(eclick, erelease):
            if None in (eclick.xdata, eclick.ydata, erelease.xdata, erelease.ydata):
                return
            x0, y0 = float(eclick.xdata), float(eclick.ydata)
            x1, y1 = float(erelease.xdata), float(erelease.ydata)
            x_min, x_max = sorted([x0, x1])
            y_min, y_max = sorted([y0, y1])
            x = int(np.clip(np.floor(x_min), 0, W - 1))
            y = int(np.clip(np.floor(y_min), 0, H - 1))
            x2 = int(np.clip(np.ceil(x_max), 0, W))
            y2 = int(np.clip(np.ceil(y_max), 0, H))
            roi["rect"] = (x, y, max(0, x2 - x), max(0, y2 - y))

        def onkey(event):
            if event.key in ("enter", "return"):
                plt.close(fig)
            elif event.key == "escape":
                roi["rect"] = None
                plt.close(fig)

        rect_sel = RectangleSelector(ax, onselect, useblit=False, button=[1], interactive=True, drag_from_anywhere=True)
        cid = fig.canvas.mpl_connect("key_press_event", onkey)
        try:
            plt.show(block=True)
        finally:
            try:
                fig.canvas.mpl_disconnect(cid)
                rect_sel.disconnect_events()
                plt.close(fig)
            except Exception:
                pass

        r = roi["rect"]
        if not r:
            return None
        x, y, w, h = r
        if w <= 0 or h <= 0:
            return None
        return (x, y, w, h)

    # -------------------------------------------------------------------------
    def _init_crop(self, first_frame: np.ndarray):
        H, W = first_frame.shape[:2]

        if isinstance(self._requested_crop, (list, tuple)) and len(self._requested_crop) == 4:
            vals = [float(v) for v in self._requested_crop]
            is_normalized = all(0.0 <= v <= 1.0 for v in vals)
            if is_normalized:
                x = int(round(vals[0] * W))
                y = int(round(vals[1] * H))
                w = int(round(vals[2] * W))
                h = int(round(vals[3] * H))
            else:
                x, y, w, h = map(int, vals)
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))
            w = max(0, min(w, W - x))
            h = max(0, min(h, H - y))
            self.crop_rect = (x, y, w, h)
            return

        if self._requested_crop is True:
            try:
                rect = self._select_roi_matplotlib(first_frame)
                self.crop_rect = rect if rect is not None else None
                if rect is None:
                    print("[EyeSource] ROI cancelled or empty; using full frame.")
                return
            except Exception as e:
                print(f"[EyeSource] Matplotlib ROI failed ({e}); using full frame.")
                self.crop_rect = None
                return

        self.crop_rect = None

    # -------------------------------------------------------------------------
    def _init_clip_range(self, first_frame: np.ndarray):
        f0 = self._apply_crop(first_frame)
        lo, hi = self.clip_percentiles
        lo = float(np.clip(lo, 0.0, 100.0))
        hi = float(np.clip(hi, 0.0, 100.0))
        if hi <= lo:
            print("[EyeSource] Invalid contrast percentiles; skipping clipping setup.")
            self._clip_range = None
            return
        vals = f0.astype(np.float32).reshape(-1)
        vmin = float(np.percentile(vals, lo))
        vmax = float(np.percentile(vals, hi))
        self._clip_range = (vmin, vmax) if vmax > vmin else None
        if self._clip_range is None:
            print("[EyeSource] Degenerate clip range; skipping clipping.")

    # -------------------------------------------------------------------------
    def _apply_crop(self, frame: np.ndarray) -> np.ndarray:
        if self.crop_rect is None:
            return frame
        x, y, w, h = self.crop_rect
        return frame[y:y+h, x:x+w]

    def _apply_contrast(self, frame: np.ndarray) -> np.ndarray:
        if self._clip_range is None:
            return frame
        vmin, vmax = self._clip_range
        f = frame.astype(np.float32)
        f = np.clip(f, vmin, vmax)
        f = (f - vmin) / (vmax - vmin + 1e-12) * 255.0
        return np.clip(f, 0, 255).astype(np.uint8)

    def _ensure_color(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        return frame

    # -------------------------------------------------------------------------
    def _overlay(self, frame: np.ndarray, idx: int) -> np.ndarray:
        if self.dlc is None:
            return frame

        frame = self._ensure_color(frame)
        Hc, Wc = frame.shape[:2]

        ox = oy = 0
        if self.crop_rect is not None:
            ox, oy, _, _ = self.crop_rect

        # pupil circle (blue)
        if self.plot_pupil:
            try:
                cx = float(self.dlc["x"][idx]) - ox
                cy = float(self.dlc["y"][idx]) - oy
                r  = float(self.dlc["radius"][idx])
                icx, icy, ir = int(round(cx)), int(round(cy)), int(round(r))
                if 0 <= icx < Wc and 0 <= icy < Hc and ir > 0:
                    cv2.circle(frame, (icx, icy), ir, (255, 0, 0), self.overlay_thickness)
            except Exception:
                pass

        # eyelid polyline (red)
        if self.plot_eye:
            try:
                xs = np.asarray(self.dlc["eye_lid_x"][idx], dtype=np.float32) - ox
                ys = np.asarray(self.dlc["eye_lid_y"][idx], dtype=np.float32) - oy
                pts = []
                for x, y in zip(xs, ys):
                    xi, yi = int(round(x)), int(round(y))
                    if 0 <= xi < Wc and 0 <= yi < Hc:
                        pts.append([xi, yi])
                if len(pts) >= 2:
                    arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                    cv2.polylines(frame, [arr], isClosed=False, color=(0, 0, 255), thickness=self.overlay_thickness)
            except Exception:
                pass

        return frame

    # -------------------------------------------------------------------------
    def _read_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        if frame_idx < 0 or frame_idx >= self.n_frames:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return frame

    def _map_time_to_index(self, timeline_time: float) -> int:
        idx = int(np.searchsorted(self.timestamps[:self.n_frames], timeline_time, side="right") - 1)
        if idx < 0:
            idx = 0
        if idx >= self.n_frames:
            idx = self.n_frames - 1
        return idx

    def get_frame_at_time(self, timeline_time: float) -> np.ndarray:
        idx = self._map_time_to_index(timeline_time)
        frame = self._read_frame(idx)
        if frame is None:
            return np.full((256, 256, 3), 127, np.uint8)

        frame = self._apply_crop(frame)
        frame = self._apply_contrast(frame)
        frame = self._overlay(frame, idx)
        return frame

    def draw_frame(self, timeline_time: float):
        return self.get_frame_at_time(timeline_time)

    def __del__(self):
        try:
            if hasattr(self, "cap") and self.cap is not None and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass

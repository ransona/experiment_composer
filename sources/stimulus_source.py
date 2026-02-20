import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from core.base_source import DataSource


class StimulusSource(DataSource):
    """
    General stimulus compositor driven by trial CSV feature columns (F#_*).

    Supports at least:
    - movie features (F#_type == 'movie')
    - grating features (F#_type == 'grating')

    Spatial units are in visual degrees. Features are rendered to a base visual field
    canvas and then cropped to requested output FOV.
    """
    NAS_BV_NATURAL_ROOT = "//ar-lab-nas1/dataserver/remote_repository/bv_resources/natural_video_set/"

    def __init__(
        self,
        config: dict,
        bonsai_root: str,
        stimulus_base_dir: str,
        fps: int = 30,
        field_azimuth_range: Tuple[float, float] = (-180.0, 180.0),
        field_elevation_range: Tuple[float, float] = (-180.0, 180.0),
        output_azimuth_center: float = 0.0,
        output_azimuth_span: float = 360.0,
        output_elevation_center: float = 0.0,
        output_elevation_span: float = 360.0,
        output_azimuth_range: Optional[Tuple[float, float]] = None,
        output_elevation_range: Optional[Tuple[float, float]] = None,
        pixels_per_degree: float = 2.0,
        background_gray: int = 127,
    ):
        super().__init__()
        self.user = config.get("user")
        self.exp_id = config.get("expID")
        self.bonsai_root = bonsai_root.lower().replace("\\", "/")
        self.stimulus_base_dir = stimulus_base_dir
        self.default_movie_fps = float(fps)

        self.field_az = (float(field_azimuth_range[0]), float(field_azimuth_range[1]))
        self.field_el = (float(field_elevation_range[0]), float(field_elevation_range[1]))
        if output_azimuth_range is not None:
            self.out_az = (float(output_azimuth_range[0]), float(output_azimuth_range[1]))
        else:
            az_span = max(0.1, float(output_azimuth_span))
            az_center = float(output_azimuth_center)
            self.out_az = (az_center - az_span / 2.0, az_center + az_span / 2.0)

        if output_elevation_range is not None:
            self.out_el = (float(output_elevation_range[0]), float(output_elevation_range[1]))
        else:
            el_span = max(0.1, float(output_elevation_span))
            el_center = float(output_elevation_center)
            self.out_el = (el_center - el_span / 2.0, el_center + el_span / 2.0)
        self.ppd = float(pixels_per_degree)
        if self.ppd <= 0:
            raise ValueError("pixels_per_degree must be > 0")

        self.bg = int(np.clip(background_gray, 0, 255))

        animal_id = self.exp_id.split("_")[-1]
        self.csv_path = os.path.join(
            f"/home/{self.user}/data/Repository",
            animal_id,
            self.exp_id,
            f"{self.exp_id}_all_trials.csv",
        )
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Trial CSV not found: {self.csv_path}")

        self.trials_df = pd.read_csv(self.csv_path, header=0)
        self.trials = self._parse_trials(self.trials_df)

        self._frame_cache: Dict[str, List[str]] = {}

        self._field_h, self._field_w = self._shape_from_ranges(self.field_az, self.field_el)
        self._crop = self._compute_crop(self.field_az, self.field_el, self.out_az, self.out_el)

        print(f"[StimulusSource] Loaded {len(self.trials)} trials from {self.csv_path}")

    def initialize(self):
        return

    def draw_frame(self, timeline_time: float):
        return self.get_frame_at_time(timeline_time)

    def get_frame_at_time(self, timeline_time: float) -> np.ndarray:
        trial = self._get_trial_for_time(timeline_time)
        if trial is None:
            return self._blank_output()

        t_rel = float(timeline_time) - float(trial["onset"])
        active = self._active_features(trial, t_rel)
        if not active:
            return self._blank_output()

        canvas = np.full((self._field_h, self._field_w, 3), self.bg, dtype=np.uint8)

        for feat in active:
            ftype = feat["type"]
            if ftype == "movie":
                frame = self._render_movie(feat, t_rel)
                if frame is None:
                    continue
                self._blit_feature(canvas, frame, feat)
            elif ftype == "grating":
                frame, alpha = self._render_grating(feat, t_rel)
                if frame is None:
                    continue
                self._blit_feature(canvas, frame, feat, alpha=alpha)

        y0, y1, x0, x1 = self._crop
        return canvas[y0:y1, x0:x1]

    def _parse_trials(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        feature_ids = set()
        for col in df.columns:
            m = re.match(r"^F(\d+)_", str(col))
            if m:
                feature_ids.add(int(m.group(1)))
        feature_ids = sorted(feature_ids)

        trials: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            onset = self._to_float(row.iloc[0], 0.0)
            row_trial_duration = self._to_float(row.iloc[2], np.nan)
            features: List[Dict[str, Any]] = []

            for fid in feature_ids:
                p = f"F{fid}_"
                ftype = self._to_str(row.get(p + "type", "")).lower()
                if not ftype:
                    continue

                feat = {
                    "id": fid,
                    "type": ftype,
                    "onset": self._to_float(row.get(p + "onset", 0.0), 0.0),
                    "duration": self._to_float(row.get(p + "duration", 0.0), 0.0),
                    "x": self._to_float(row.get(p + "x", 0.0), 0.0),
                    "y": self._to_float(row.get(p + "y", 0.0), 0.0),
                    "width": max(0.1, self._to_float(row.get(p + "width", 10.0), 10.0)),
                    "height": max(0.1, self._to_float(row.get(p + "height", 10.0), 10.0)),
                    "angle": self._to_float(row.get(p + "angle", 0.0), 0.0),
                    "speed": self._to_float(row.get(p + "speed", 0.0), 0.0),
                    "opacity": self._normalize_unit(row.get(p + "opacity", 1.0), default=1.0),
                    "loop": self._to_bool(row.get(p + "loop", False)),
                    "name": self._to_str(row.get(p + "name", "")),
                    "phase": self._to_float(row.get(p + "phase", 0.0), 0.0),
                    "contrast": self._normalize_unit(row.get(p + "contrast", 1.0), default=1.0),
                    "dcycle": self._normalize_unit(row.get(p + "dcycle", 0.5), default=0.5),
                    "freq": self._to_float(row.get(p + "freq", 0.1), 0.1),
                }
                features.append(feat)

            if not features:
                continue

            if np.isfinite(row_trial_duration):
                trial_duration = float(row_trial_duration)
            else:
                trial_duration = max((f["onset"] + f["duration"]) for f in features)

            trials.append(
                {
                    "onset": onset,
                    "duration": max(0.0, trial_duration),
                    "features": sorted(features, key=lambda x: x["id"]),
                }
            )

        return sorted(trials, key=lambda t: t["onset"])

    def _get_trial_for_time(self, timeline_time: float) -> Optional[Dict[str, Any]]:
        t = float(timeline_time)
        for trial in self.trials:
            if trial["onset"] <= t < trial["onset"] + trial["duration"]:
                return trial
        return None

    def _active_features(self, trial: Dict[str, Any], t_rel: float) -> List[Dict[str, Any]]:
        active: List[Dict[str, Any]] = []
        for feat in trial["features"]:
            on = feat["onset"]
            off = on + feat["duration"]
            if on <= t_rel < off:
                active.append(feat)
        return active

    def _render_movie(self, feat: Dict[str, Any], t_rel: float) -> Optional[np.ndarray]:
        stim_dir = self._resolve_movie_path(feat["name"])
        frames = self._get_frame_files(stim_dir)
        if not frames:
            return None

        elapsed = max(0.0, t_rel - feat["onset"])
        playback_fps = feat["speed"] if feat["speed"] > 0 else self.default_movie_fps
        idx = int(elapsed * playback_fps)

        if feat["loop"]:
            idx = idx % len(frames)
        elif idx >= len(frames):
            return None

        img = cv2.imread(frames[idx], cv2.IMREAD_COLOR)
        if img is None:
            return None

        return img

    def _render_grating(self, feat: Dict[str, Any], t_rel: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        contrast = float(np.clip(feat["contrast"], 0.0, 1.0))
        opacity = float(np.clip(feat["opacity"], 0.0, 1.0))
        if contrast <= 0.0 or opacity <= 0.0:
            return None, None

        w = max(2, int(round(feat["width"] * self.ppd)))
        h = max(2, int(round(feat["height"] * self.ppd)))

        x = (np.arange(w, dtype=np.float32) - (w - 1) / 2.0) / self.ppd
        y = (np.arange(h, dtype=np.float32) - (h - 1) / 2.0) / self.ppd
        xx, yy = np.meshgrid(x, y)

        # Orientation is applied later in _blit_feature via affine rotation.
        # Keep grating generation axis-aligned here to avoid double-rotating.
        xr = xx

        elapsed = max(0.0, t_rel - feat["onset"])
        phase_cycles = feat["phase"] / 360.0
        phase = (feat["freq"] * xr + feat["speed"] * elapsed + phase_cycles) % 1.0

        duty = float(np.clip(feat["dcycle"], 1e-6, 1.0 - 1e-6))
        wave = np.where(phase < duty, 1.0, -1.0)

        lum = 0.5 + 0.5 * contrast * wave
        gray = np.clip(np.round(lum * 255.0), 0, 255).astype(np.uint8)
        rgb = np.repeat(gray[..., None], 3, axis=2)

        rr = min(w, h) * 0.5
        cx = (w - 1) / 2.0
        cy = (h - 1) / 2.0
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        circle = (((gx - cx) ** 2 + (gy - cy) ** 2) <= (rr ** 2)).astype(np.float32)

        alpha = circle * opacity
        return rgb, alpha

    def _blit_feature(
        self,
        canvas: np.ndarray,
        img: np.ndarray,
        feat: Dict[str, Any],
        alpha: Optional[np.ndarray] = None,
    ):
        target_w = max(1, int(round(feat["width"] * self.ppd)))
        target_h = max(1, int(round(feat["height"] * self.ppd)))

        patch = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        if alpha is None:
            alpha_patch = np.ones((target_h, target_w), dtype=np.float32)
            alpha_patch *= float(np.clip(feat.get("opacity", 1.0), 0.0, 1.0))
        else:
            alpha_patch = cv2.resize(alpha.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        patch, alpha_patch = self._rotate_patch(patch, alpha_patch, feat["angle"])

        cx, cy = self._deg_to_px(feat["x"], feat["y"])
        ph, pw = patch.shape[:2]
        x0 = int(round(cx - pw / 2.0))
        y0 = int(round(cy - ph / 2.0))
        x1 = x0 + pw
        y1 = y0 + ph

        cx0 = max(0, x0)
        cy0 = max(0, y0)
        cx1 = min(self._field_w, x1)
        cy1 = min(self._field_h, y1)
        if cx0 >= cx1 or cy0 >= cy1:
            return

        px0 = cx0 - x0
        py0 = cy0 - y0
        px1 = px0 + (cx1 - cx0)
        py1 = py0 + (cy1 - cy0)

        src = patch[py0:py1, px0:px1].astype(np.float32)
        a = alpha_patch[py0:py1, px0:px1].astype(np.float32)[..., None]
        dst = canvas[cy0:cy1, cx0:cx1].astype(np.float32)

        out = src * a + dst * (1.0 - a)
        canvas[cy0:cy1, cx0:cx1] = np.clip(out, 0, 255).astype(np.uint8)

    def _rotate_patch(self, patch: np.ndarray, alpha: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
        if abs(float(angle_deg)) < 1e-8:
            return patch, alpha

        h, w = patch.shape[:2]
        center = (w * 0.5, h * 0.5)
        M = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)

        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int(round((h * sin) + (w * cos)))
        new_h = int(round((h * cos) + (w * sin)))

        M[0, 2] += (new_w / 2.0) - center[0]
        M[1, 2] += (new_h / 2.0) - center[1]

        rot_patch = cv2.warpAffine(
            patch,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        rot_alpha = cv2.warpAffine(
            alpha,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return rot_patch, np.clip(rot_alpha, 0.0, 1.0)

    def _resolve_movie_path(self, name: str) -> str:
        p = str(name).strip().lower().replace("\\", "/")
        if not p:
            return ""
        if p.startswith(self.NAS_BV_NATURAL_ROOT):
            suffix = p[len(self.NAS_BV_NATURAL_ROOT):].lstrip("/")
            p = os.path.join(self.stimulus_base_dir, "natural_video_set", suffix)
        if self.bonsai_root and p.startswith(self.bonsai_root):
            p = p.replace(self.bonsai_root, self.stimulus_base_dir, 1)
        elif not os.path.isabs(p):
            p = os.path.join(self.stimulus_base_dir, p)
        return p

    def _get_frame_files(self, stim_dir: str) -> List[str]:
        if stim_dir in self._frame_cache:
            return self._frame_cache[stim_dir]

        if not stim_dir or not os.path.isdir(stim_dir):
            self._frame_cache[stim_dir] = []
            return []

        files = sorted(
            [
                os.path.join(stim_dir, f)
                for f in os.listdir(stim_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
            ]
        )
        self._frame_cache[stim_dir] = files
        return files

    def _blank_output(self) -> np.ndarray:
        y0, y1, x0, x1 = self._crop
        h = y1 - y0
        w = x1 - x0
        return np.full((h, w, 3), self.bg, dtype=np.uint8)

    def _deg_to_px(self, az_deg: float, el_deg: float) -> Tuple[int, int]:
        x = int(round((float(az_deg) - self.field_az[0]) * self.ppd))
        y = int(round((self.field_el[1] - float(el_deg)) * self.ppd))
        return x, y

    def _shape_from_ranges(self, az: Tuple[float, float], el: Tuple[float, float]) -> Tuple[int, int]:
        w = max(1, int(round((az[1] - az[0]) * self.ppd)))
        h = max(1, int(round((el[1] - el[0]) * self.ppd)))
        return h, w

    def _compute_crop(
        self,
        field_az: Tuple[float, float],
        field_el: Tuple[float, float],
        out_az: Tuple[float, float],
        out_el: Tuple[float, float],
    ) -> Tuple[int, int, int, int]:
        x0 = int(round((out_az[0] - field_az[0]) * self.ppd))
        x1 = int(round((out_az[1] - field_az[0]) * self.ppd))
        y0 = int(round((field_el[1] - out_el[1]) * self.ppd))
        y1 = int(round((field_el[1] - out_el[0]) * self.ppd))

        x0 = max(0, min(x0, self._field_w - 1))
        x1 = max(x0 + 1, min(x1, self._field_w))
        y0 = max(0, min(y0, self._field_h - 1))
        y1 = max(y0 + 1, min(y1, self._field_h))
        return y0, y1, x0, x1

    @staticmethod
    def _to_float(v: Any, default: float) -> float:
        try:
            if v is None:
                return float(default)
            if isinstance(v, str) and v.strip() == "":
                return float(default)
            return float(v)
        except Exception:
            return float(default)

    @staticmethod
    def _to_str(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float) and np.isnan(v):
            return ""
        return str(v).strip()

    @staticmethod
    def _to_bool(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        return s in ("1", "true", "yes", "y")

    @staticmethod
    def _normalize_unit(v: Any, default: float) -> float:
        x = StimulusSource._to_float(v, default)
        if x > 1.0:
            x = x / 100.0
        return float(np.clip(x, 0.0, 1.0))

"""
VideoBinSource — Suite2p binary video loader and synchronizer with:
- Per-plane contrast autoscaling (cached on disk per plane)
- Subsampled pixel percentile estimation for faster scaling
- Filtering, interpolation, and flexible tiling as before
"""

import os
import numpy as np
import cv2
from scipy.io import loadmat
from typing import Any, Dict, Optional, List
from core.base_source import DataSource


class VideoBinSource(DataSource):
    """Suite2p binary video reader with Timeline synchronization, filtering, tiling, and interpolation."""

    # ---------------------------------------------------------------
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.mm_list: List[np.memmap] = []
        self.frame_times: Optional[np.ndarray] = None
        self.frame_size = None
        self.temporal_buffers: List[List[np.ndarray]] = []
        self.paths = []
        self._parse_cfg()

    # ---------------------------------------------------------------
    def _parse_cfg(self):
        c = self.cfg
        self.user = c.get("user", None)
        self.expID = c.get("expID", None)
        self.planes = c.get("planes", [0])
        if isinstance(self.planes, int):
            self.planes = [self.planes]

        self.paths = c.get("paths", [])
        self.height = int(c.get("height", 512))
        self.width = int(c.get("width", 512))
        self.stride = int(c.get("stride", 1))
        self.fps = float(c.get("fps", 3.333))
        self.spatial_sigma = float(c.get("spatial_sigma", 0.0))
        self.temporal_window = int(c.get("temporal_window", 0))
        self.label = c.get("label", None)
        self.auto_scale_sample = int(c.get("auto_scale_sample", 200))
        self.concat_axis = c.get("concat_axis", "h")
        self.tile_layout = c.get("tile_layout", None)
        self.interpolate = bool(c.get("interpolate", True))

        # Filtering control
        self.filter_opts = {
            "enable_spatial": c.get("enable_spatial_filter", True),
            "enable_temporal": c.get("enable_temporal_filter", True),
        }

        # Subsampling for faster autoscale (downsample factor)
        self.autoscale_subsample = int(c.get("autoscale_subsample", 4))

        ft = c.get("frame_times", None)
        self.frame_times = np.asarray(ft) if ft is not None else None

        self.vmin: List[float] = []
        self.vmax: List[float] = []

    # ---------------------------------------------------------------
    def _infer_animal_id(self, expID: str) -> str:
        if len(expID) < 15:
            raise ValueError("expID too short to extract animal ID (need ≥15 chars)")
        return expID[14:]

    # ---------------------------------------------------------------
    def _find_bin_paths(self) -> List[str]:
        if self.paths:
            return self.paths
        if not (self.user and self.expID):
            raise ValueError("Either explicit 'paths' or ('user' + 'expID') required.")
        animalID = self._infer_animal_id(self.expID)
        base = os.path.join("/home", self.user, "data", "Repository",
                            animalID, self.expID, "suite2p")

        paths = []
        for p in self.planes:
            pth = os.path.join(base, f"plane{p}", "data.bin")
            if not os.path.exists(pth):
                raise FileNotFoundError(f"Missing file: {pth}")
            paths.append(pth)
        return paths

    # ---------------------------------------------------------------
    def _get_depth_count(self) -> int:
        animalID = self._infer_animal_id(self.expID)
        suite2p_dir = os.path.join("/home", self.user, "data", "Repository",
                                   animalID, self.expID, "suite2p")
        if not os.path.exists(suite2p_dir):
            raise FileNotFoundError(f"Suite2p folder not found: {suite2p_dir}")

        plane_dirs = [d for d in os.listdir(suite2p_dir)
                      if d.startswith("plane") and os.path.isdir(os.path.join(suite2p_dir, d))]
        depth_count = len(plane_dirs)
        if depth_count == 0:
            raise RuntimeError(f"No plane folders found in {suite2p_dir}")
        print(f"[Info] Found {depth_count} total imaging depths in suite2p folder.")
        return depth_count

    # ---------------------------------------------------------------
    def _load_frame_times_from_timeline(self) -> np.ndarray:
        animalID = self._infer_animal_id(self.expID)
        tl_path = os.path.join("/data", "Remote_Repository",
                               animalID, self.expID, f"{self.expID}_Timeline.mat")

        if not os.path.exists(tl_path):
            raise FileNotFoundError(f"Timeline file not found: {tl_path}")

        print(f"[Info] Loading frame times from {tl_path}")
        TL = loadmat(tl_path)
        TL = TL["timelineSession"]

        ch_names = [n[0] for n in TL["chNames"][0, 0][0]]
        daq_data = TL["daqData"][0, 0]
        tl_time = np.squeeze(TL["time"][0, 0])

        if "MicroscopeFrames" not in ch_names:
            raise ValueError("Channel 'MicroscopeFrames' not found in Timeline.")
        ch_idx = ch_names.index("MicroscopeFrames")

        pulses = np.squeeze((daq_data[:, ch_idx] > 1).astype(int))
        rising_edges = np.where(np.diff(pulses) == 1)[0]
        frame_times_all = tl_time[rising_edges]

        depth_count = self._get_depth_count()
        frame_times = frame_times_all[0::depth_count]

        print(f"[Info] Timeline: detected {len(frame_times_all)} pulses total "
              f"→ {len(frame_times)} per depth (depthCount={depth_count})")
        return frame_times

    # ---------------------------------------------------------------
    def initialize(self):
        self.paths = self._find_bin_paths()
        self.mm_list = []
        frame_counts = []

        for path in self.paths:
            mm = np.memmap(path, dtype=np.int16, mode="r")
            self.mm_list.append(mm)
            n_frames = mm.size // (self.height * self.width)
            frame_counts.append(n_frames)

        n_frames = int(min(frame_counts))
        if len(set(frame_counts)) > 1:
            print(f"[Info] Truncating planes to {n_frames} frames (shortest plane).")

        if self.frame_times is None:
            self.frame_times = self._load_frame_times_from_timeline()

        diff = n_frames - len(self.frame_times)
        if diff == 0:
            print(f"[OK] Frames in .bin match Timeline pulses ({n_frames} each).")
        elif abs(diff) < 5:
            print(f"[Warn] Minor mismatch: bin={n_frames}, Timeline={len(self.frame_times)} (Δ={diff})")
            n_frames = min(n_frames, len(self.frame_times))
            self.frame_times = self.frame_times[:n_frames]
        else:
            raise ValueError(f"Frame count mismatch too large: bin={n_frames}, Timeline={len(self.frame_times)} (Δ={diff})")

        # --- Per-plane autoscaling with caching ---
        self.vmin = []
        self.vmax = []
        for pi, path in enumerate(self.paths):
            cache_path = os.path.join(os.path.dirname(path), "autoscale_cache.npz")
            if os.path.exists(cache_path):
                cache = np.load(cache_path)
                vmin, vmax = float(cache["vmin"]), float(cache["vmax"])
                print(f"[Cache] Using cached autoscale for plane {pi}: vmin={vmin:.1f}, vmax={vmax:.1f}")
            else:
                print(f"[Compute] Calculating autoscale for plane {pi} (this may take a moment)...")
                idxs = np.linspace(0, n_frames - 1,
                                   min(self.auto_scale_sample, n_frames),
                                   dtype=int)
                samples = [self._get_frame(pi, i)[::self.autoscale_subsample,
                                                  ::self.autoscale_subsample] for i in idxs]
                stack = np.stack(samples, axis=0).astype(np.float32)
                vmin = float(np.percentile(stack, 1))
                vmax = float(np.percentile(stack, 99))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                np.savez(cache_path, vmin=vmin, vmax=vmax)
                print(f"[Cache] Saved autoscale for plane {pi}: {cache_path}")
            self.vmin.append(vmin)
            self.vmax.append(vmax)

        self.temporal_buffers = [[] for _ in self.mm_list]

    # ---------------------------------------------------------------
    def _get_frame(self, plane_idx: int, frame_idx: int) -> np.ndarray:
        mm = self.mm_list[plane_idx]
        s = frame_idx * self.height * self.width
        e = s + self.height * self.width
        f = mm[s:e].reshape(self.height, self.width)
        if self.stride > 1:
            f = f[::self.stride, ::self.stride]
        return f

    # ---------------------------------------------------------------
    def _apply_filters(self, plane_idx: int, f: np.ndarray) -> np.ndarray:
        if self.filter_opts["enable_spatial"] and self.spatial_sigma > 0:
            from scipy.ndimage import gaussian_filter
            f = gaussian_filter(f, self.spatial_sigma)

        if self.filter_opts["enable_temporal"] and self.temporal_window > 1:
            buf = self.temporal_buffers[plane_idx]
            buf.append(f)
            if len(buf) > self.temporal_window:
                buf.pop(0)
            if len(buf) > 1:
                f = np.median(np.stack(buf, axis=0), axis=0)
        return f

    # ---------------------------------------------------------------
    def _tile_planes(self, plane_imgs: List[np.ndarray]) -> np.ndarray:
        if not self.tile_layout:
            if len(plane_imgs) == 1:
                return plane_imgs[0]
            return np.hstack(plane_imgs) if self.concat_axis == "h" else np.vstack(plane_imgs)

        layout = self.tile_layout
        rows = layout.get("rows", 1)
        cols = layout.get("cols", len(plane_imgs))
        order = layout.get("order", list(range(len(plane_imgs))))
        gap = layout.get("gap", 0)

        h, w = plane_imgs[0].shape[:2]
        canvas_h = rows * h + (rows - 1) * gap
        canvas_w = cols * w + (cols - 1) * gap
        canvas = np.zeros((canvas_h, canvas_w), dtype=plane_imgs[0].dtype)

        for i, pidx in enumerate(order):
            if pidx >= len(plane_imgs):
                continue
            r = i // cols
            c = i % cols
            y = r * (h + gap)
            x = c * (w + gap)
            canvas[y:y + h, x:x + w] = plane_imgs[pidx]

        return canvas

    # ---------------------------------------------------------------
    def _interpolate_frames(self, plane_idx: int, t: float) -> np.ndarray:
        ft = self.frame_times
        if t <= ft[0]:
            return self._get_frame(plane_idx, 0)
        if t >= ft[-1]:
            return self._get_frame(plane_idx, len(ft) - 1)

        i1 = np.searchsorted(ft, t) - 1
        i2 = i1 + 1
        alpha = (t - ft[i1]) / (ft[i2] - ft[i1])
        f1 = self._get_frame(plane_idx, i1)
        f2 = self._get_frame(plane_idx, i2)
        return (1 - alpha) * f1 + alpha * f2

    # ---------------------------------------------------------------
    def draw_frame(self, t: float) -> np.ndarray:
        if not self.mm_list:
            raise RuntimeError("VideoBinSource not initialized.")

        frames_8u = []
        for pi in range(len(self.mm_list)):
            if self.interpolate:
                f = self._interpolate_frames(pi, t)
            else:
                idx = np.searchsorted(self.frame_times, t, side="right") - 1
                idx = int(np.clip(idx, 0, len(self.frame_times) - 1))
                f = self._get_frame(pi, idx)

            f = self._apply_filters(pi, f)
            img = np.clip((f - self.vmin[pi]) / (self.vmax[pi] - self.vmin[pi]) * 255, 0, 255).astype(np.uint8)
            frames_8u.append(img)

        img_all = self._tile_planes(frames_8u)
        img_all = cv2.cvtColor(img_all, cv2.COLOR_GRAY2BGR)

        if self.label:
            cv2.putText(img_all, self.label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return img_all

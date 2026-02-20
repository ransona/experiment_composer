import os
import cv2
import numpy as np
import pandas as pd
from typing import Optional
from core.base_source import DataSource


class StimulusVideoSource(DataSource):
    """
    StimulusVideoSource — returns the visual stimulus frame corresponding
    to a given timeline time, based on the experiment trial CSV.

    Each row in the CSV specifies:
        [0] trial onset time (Timeline)
        [2] duration (s)
        [7] path to stimulus frames directory (Bonsai path)

    The Bonsai path root (e.g. 'D:\\bonsai_resources\\') is replaced by
    a configurable local base directory (e.g. '/home/adamranson/data/vid_for_decoder/').
    """
    NAS_BV_NATURAL_ROOT = "//ar-lab-nas1/dataserver/remote_repository/bv_resources/natural_video_set/"

    def __init__(self, config: dict, bonsai_root: str, stimulus_base_dir: str, fps: int = 30):
        """
        Args:
            config: dict containing "user" and "expID"
            bonsai_root: root prefix in Bonsai paths (e.g. 'D:\\bonsai_resources\\')
            stimulus_base_dir: local directory base to replace Bonsai root
            fps: stimulus playback rate (frames per second)
        """
        super().__init__()
        self.user = config.get("user")
        self.exp_id = config.get("expID")
        self.bonsai_root = bonsai_root.lower().replace("\\", "/")
        self.stimulus_base_dir = stimulus_base_dir
        self.fps = fps

        # ----------------------------------------------------------
        # Build CSV path automatically from expID
        # expID = '2025-07-07_05_ESPM154' → animal_id = 'ESPM154'
        # CSV = /home/<user>/data/Repository/<animal_id>/<expID>/<expID>_all_trials.csv
        # ----------------------------------------------------------
        animal_id = self.exp_id.split("_")[-1]
        self.csv_path = os.path.join(
            f"/home/{self.user}/data/Repository",
            animal_id,
            self.exp_id,
            f"{self.exp_id}_all_trials.csv"
        )

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Trial CSV not found: {self.csv_path}")

        # ----------------------------------------------------------
        # Load and parse CSV
        # ----------------------------------------------------------
        self.trials = pd.read_csv(self.csv_path, header=0)
        self.trials.columns = [f"col{i}" for i in range(len(self.trials.columns))]

        # Extract relevant info from each row
        self.trials_info = []
        for _, row in self.trials.iterrows():
            try:
                onset = float(row[0])
                duration = float(row[2])
                local_path = self._resolve_movie_path(str(row[7]))
                self.trials_info.append((onset, duration, local_path))
            except Exception as e:
                print(f"[StimulusVideoSource] ⚠️ Failed parsing row: {e}")

        # Cache of directory listings
        self._frame_cache = {}
        self._default_grey = np.full((256, 256, 3), 127, dtype=np.uint8)

        print(f"[StimulusVideoSource] Loaded {len(self.trials_info)} trials from {self.csv_path}")

    # ----------------------------------------------------------
    def _get_frame_files(self, stim_dir: str):
        """Return sorted list of frame file paths in directory."""
        if stim_dir in self._frame_cache:
            return self._frame_cache[stim_dir]

        if not os.path.isdir(stim_dir):
            print(f"[StimulusVideoSource] ⚠️ Missing stimulus dir: {stim_dir}")
            self._frame_cache[stim_dir] = []
            return []

        files = sorted([
            os.path.join(stim_dir, f)
            for f in os.listdir(stim_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ])
        self._frame_cache[stim_dir] = files
        return files

    def _resolve_movie_path(self, name: str) -> str:
        p = str(name).strip().lower().replace("\\", "/")
        if p.startswith(self.NAS_BV_NATURAL_ROOT):
            suffix = p[len(self.NAS_BV_NATURAL_ROOT):].lstrip("/")
            return os.path.join(self.stimulus_base_dir, "natural_video_set", suffix)
        if self.bonsai_root and p.startswith(self.bonsai_root):
            return p.replace(self.bonsai_root, self.stimulus_base_dir, 1)
        return p

    # ----------------------------------------------------------
    def get_frame_at_time(self, timeline_time: float) -> np.ndarray:
        """Return stimulus frame (numpy array) shown at given timeline_time."""
        # Find trial corresponding to this time
        trial = None
        for onset, duration, stim_dir in self.trials_info:
            if onset <= timeline_time < onset + duration:
                trial = (onset, duration, stim_dir)
                break

        if trial is None:
            # Out of trial → grey frame
            return self._default_grey

        onset, duration, stim_dir = trial
        elapsed = timeline_time - onset
        frame_idx = int(elapsed * self.fps)

        frame_files = self._get_frame_files(stim_dir)
        if not frame_files or frame_idx >= len(frame_files):
            return self._default_grey

        frame_path = frame_files[frame_idx]
        img = cv2.imread(frame_path)
        if img is None:
            print(f"[StimulusVideoSource] ⚠️ Failed to read frame: {frame_path}")
            return self._default_grey

        return img
    
    def draw_frame(self, timeline_time: float):
        """Interface method expected by CanvasComposer."""
        return self.get_frame_at_time(timeline_time)    

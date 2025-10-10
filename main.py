"""
Example main script to test VideoBinSource with the core framework.
Creates a short MP4 from a Suite2p .bin file using synthetic timestamps.
"""

import numpy as np
import cv2
import os

from core.timeline import Timeline
from core.composer import Composer
from core.writer import VideoWriter
from sources.video_bin_source import VideoBinSource


def main():
    # --- experiment parameters ---
    user_home = "/home/pmateosaparicio"
    exp_path = (
        f"{user_home}/data/Repository/ESPM154/"
        f"2025-07-07_05_ESPM154/suite2p/plane0/data.bin"
    )

    # --- create artificial frame times ---
    # 100 frames spaced by 0.3 s -> covers 30 seconds
    frame_times = np.arange(0, 100) * 0.3

    # --- configure the source ---
    cfg = {
        "path": exp_path,
        "height": 512,
        "width": 512,
        "frame_times": frame_times,
        "label": "Plane 0",
        "spatial_sigma": 0.0,     # no filtering
        "temporal_window": 0,
    }

    # --- initialize video source ---
    video_source = VideoBinSource(cfg)
    video_source.initialize()

    # --- define global timeline (first 20 s, output 10 fps) ---
    timeline = Timeline(start_time=0.0, stop_time=20.0, fps=10.0)

    # --- create composer ---
    composer = Composer([video_source], layout="v", pad=2, bg=0)
    composer.initialize()

    # --- prepare video writer ---
    test_frame = composer.draw_composite(timeline.times[0])
    H, W = test_frame.shape[:2]
    out_path = os.path.join(os.getcwd(), "video_test_output.mp4")

    writer = VideoWriter(out_path, fps=timeline.fps, frame_size=(W, H))

    # --- iterate over timeline and write frames ---
    for i, t in enumerate(timeline.times):
        frame = composer.draw_composite(t)
        writer.write(frame)
        if i % 10 == 0 or i == len(timeline.times) - 1:
            print(f"[{i+1}/{len(timeline.times)}] t={t:.2f}s")

    writer.close()
    print(f"✅ Video saved to: {out_path}")


if __name__ == "__main__":
    main()

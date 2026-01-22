"""
Simple wrapper around imageio for MP4 output with libx264 + yuv420p,
ensuring broad compatibility (VS Code, browsers, etc.).
"""

import numpy as np
import imageio.v3 as iio

class VideoWriter:
    """MP4 writer using libx264 and yuv420p pixel format."""

    def __init__(self, out_path, fps, frame_size):
        """
        Args:
            out_path: Path to output .mp4 file
            fps: Frames per second
            frame_size: (width, height)
        """
        self.out_path = out_path
        self.fps = fps
        self.frames = []  # collect frames before writing
        self.frame_size = frame_size

    def write(self, frame):
        """Append a frame (uint8 HxWx3)."""
        if frame.shape[1::-1] != self.frame_size:
            raise ValueError(f"Frame size {frame.shape[1::-1]} does not match expected {self.frame_size}")
        self.frames.append(np.ascontiguousarray(frame))

    def close(self):
        """Encode all frames to MP4 with H.264 (libx264, yuv420p)."""
        if not self.frames:
            raise RuntimeError("No frames written before close().")
        arr = np.stack(self.frames, axis=0).astype(np.uint8)
        iio.imwrite(
            self.out_path,
            arr,
            fps=self.fps,
            codec="libx264",
            ffmpeg_params=["-pix_fmt", "yuv420p"],
        )
        self.frames.clear()

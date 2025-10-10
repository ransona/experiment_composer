"""
Simple wrapper around OpenCV VideoWriter for MP4 output.
"""

import cv2

class VideoWriter:
    """Manages MP4 video output."""

    def __init__(self, out_path, fps, frame_size):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(out_path, fourcc, fps, frame_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {out_path}")

    def write(self, frame):
        self.writer.write(frame)

    def close(self):
        self.writer.release()

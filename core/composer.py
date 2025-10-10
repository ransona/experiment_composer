"""
Combines multiple DataSource frames into one composite frame.
"""

import cv2
import numpy as np

class Composer:
    """
    Handles synchronization and composition of multiple data sources.
    """

    def __init__(self, sources, layout="v", pad=2, bg=0):
        """
        Parameters
        ----------
        sources : list of DataSource
            The list of sources to combine.
        layout : str
            'v' = vertical stack, 'h' = horizontal stack, 'grid' = square grid
        pad : int
            Pixel padding between panels.
        bg : int
            Background gray level (0–255).
        """
        self.sources = sources
        self.layout = layout
        self.pad = pad
        self.bg = bg

    # -------------------------------------------------------------
    def initialize(self):
        for s in self.sources:
            s.initialize()

    # -------------------------------------------------------------
    def _make_layout(self, n):
        if n == 1:
            return 1, 1
        if self.layout == "h":
            return 1, n
        if self.layout == "v":
            return n, 1
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(float(n) / cols))
        return rows, cols

    # -------------------------------------------------------------
    def draw_composite(self, t):
        """
        Query each source at time t and compose them into one frame.
        """
        imgs = [s.draw_frame(t) for s in self.sources]
        h_min = min(im.shape[0] for im in imgs)
        w_min = min(im.shape[1] for im in imgs)
        imgs = [cv2.resize(im, (w_min, h_min)) for im in imgs]

        rows, cols = self._make_layout(len(imgs))
        H = rows * h_min + (rows - 1) * self.pad
        W = cols * w_min + (cols - 1) * self.pad
        canvas = np.full((H, W, 3), self.bg, np.uint8)

        k = 0
        for r in range(rows):
            for c in range(cols):
                if k >= len(imgs):
                    break
                y = r * (h_min + self.pad)
                x = c * (w_min + self.pad)
                canvas[y:y + h_min, x:x + w_min] = imgs[k]
                k += 1
        return canvas

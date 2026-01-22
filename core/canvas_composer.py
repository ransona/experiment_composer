"""
CanvasComposer — explicit spatial composition of visual sources on a fixed canvas.
"""

import cv2
import numpy as np

class CanvasComposer:
    def __init__(self, sources_dict, layout_cfg, bg=0):
        """
        Parameters
        ----------
        sources_dict : dict[str, DataSource]
            Mapping of source names to initialized DataSource objects.
        layout_cfg : dict
            {
                "canvas_size": (H, W),
                "elements": {
                    "name1": {"source": "video0", "x": 0, "y": 0, "w": 300, "h": 300},
                    ...
                }
            }
        bg : int
            Background gray value (0–255)
        """
        self.sources = sources_dict
        self.canvas_h, self.canvas_w = layout_cfg["canvas_size"]
        self.elements = layout_cfg["elements"]
        self.bg = bg

    # ---------------------------------------------------------
    def initialize(self):
        """Initialize all sources."""
        for src in self.sources.values():
            src.initialize()

    # ---------------------------------------------------------
    def draw_composite(self, t):
        """Render a full canvas frame for time t."""
        canvas = np.full((self.canvas_h, self.canvas_w, 3), self.bg, np.uint8)

        for name, elem in self.elements.items():
            src_name = elem["source"]
            if src_name not in self.sources:
                raise KeyError(f"Source '{src_name}' not found for element '{name}'")

            x, y, w, h = elem["x"], elem["y"], elem["w"], elem["h"]
            y2, x2 = y + h, x + w

            # Verbose bounds check
            if x < 0 or y < 0 or x2 > self.canvas_w or y2 > self.canvas_h:
                reasons = []
                if x < 0:
                    reasons.append(f"x ({x}) < 0")
                if y < 0:
                    reasons.append(f"y ({y}) < 0")
                if x2 > self.canvas_w:
                    reasons.append(f"x+w ({x2}) > canvas_w ({self.canvas_w}) by {x2 - self.canvas_w}px")
                if y2 > self.canvas_h:
                    reasons.append(f"y+h ({y2}) > canvas_h ({self.canvas_h}) by {y2 - self.canvas_h}px")

                valid_x_max = max(0, self.canvas_w - w)
                valid_y_max = max(0, self.canvas_h - h)

                raise ValueError(
                    "Element exceeds canvas bounds: "
                    f"element='{name}', source='{src_name}'; "
                    f"canvas(H,W)=({self.canvas_h},{self.canvas_w}); "
                    f"requested [x,y,w,h]=({x},{y},{w},{h}) -> "
                    f"occupies x:[{x},{x2}), y:[{y},{y2}). "
                    f"Issue: {', '.join(reasons)}. "
                    f"Valid top-left ranges: x∈[0,{valid_x_max}], y∈[0,{valid_y_max}]."
                )

            frame = self.sources[src_name].draw_frame(t)
            frame_resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            canvas[y:y2, x:x2] = frame_resized

        return canvas

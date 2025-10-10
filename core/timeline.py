"""
Defines the master Timeline used to coordinate all sources.
"""

import numpy as np

class Timeline:
    """Global time base for the composed output video."""

    def __init__(self, start_time: float, stop_time: float, fps: float):
        """
        Parameters
        ----------
        start_time : float
            Beginning of timeline (seconds)
        stop_time : float
            End of timeline (seconds)
        fps : float
            Temporal sampling rate for output video (frames per second)
        """
        self.start = float(start_time)
        self.stop = float(stop_time)
        self.fps = float(fps)
        self.dt = 1.0 / self.fps
        self.times = np.arange(self.start, self.stop, self.dt)

    def __len__(self):
        return len(self.times)

    def __getitem__(self, i):
        return self.times[i]

    def __iter__(self):
        for t in self.times:
            yield t

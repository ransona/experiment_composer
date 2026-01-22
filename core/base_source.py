"""
Base class for all time-dependent data sources.
Every source (e.g. VideoBinSource, PlotSource, EphysSource)
inherits from this.
"""

class DataSource:
    """
    Abstract base class defining the interface for data sources.
    """

    def initialize(self):
        """
        Prepare the source (load data, allocate memory, open files, etc.)
        Called once before the main rendering loop.
        """
        pass

    def draw_frame(self, t):
        """
        Return a numpy.ndarray (H×W×3 uint8) image corresponding
        to the given timeline time `t` (in seconds).
        Subclasses must implement this.
        """
        raise NotImplementedError("draw_frame(t) not implemented in this source.")

import numpy as np
from motion import Stationary

class Point:
    """Represents an object using a set of points"""
    def __init__(self, pos, motion=None):
        """
        Parameters:
            pos: location of the points.
            motion: motion of the points.
        """
        self.pos = np.array(pos)
        if len(self.pos.shape) == 1:
            self.pos = np.expand_dims(pos, 0)
        self.size = self.pos.shape[0]   # number of points
        self.path = None
        if not motion:  # no motion by default
            self.motion = Stationary()
        else:
            self.motion = motion

    def get_init_pos(self):
        """Get the initial position of the points."""
        return self.pos

    def get_size(self):
        """Get the number of points."""
        return self.size

    def get_path(self):
        """Get the path of all points."""
        if self.path is None:
            self.path = self.motion.move(self.pos)
        return self.path

    def register(self, *info):
        """Register the object with frame information."""
        self.motion.register(*info)

    def get_average_pos(self):
        """Get the path of the centroid of the object."""
        if self.path is None:
            self.path = self.motion.move(self.pos)
        return np.mean(self.path, axis=0)
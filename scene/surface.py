from .point import Point
import numpy as np
import matplotlib.pyplot as plt

class Circle(Point):
    """A circular collection of points to simulate a surface"""
    def __init__(self, pos, radius, step=1e-3, motion=None):
        """
        Parameters:
            pos: centriod of the circle.
            radius: radius of the circle. 
            step: distance between every two points.
            motion: motion of the circle.
        """
        super().__init__(pos, motion)
        x0, y, z0 = pos
        x = np.arange(x0-radius, x0+radius, step)
        z = np.arange(z0-radius, z0+radius, step)
        # define a rectangular grid of points
        pts = np.stack(np.meshgrid(x, y, z), -1).reshape((-1, 3))
        # define a circular grid of points
        dist = np.linalg.norm(pts-pos, axis=-1)
        self.pos = pts[dist<=radius]
        self.size = self.pos.shape[0]
        self.cen = pos

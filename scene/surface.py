from .point import Point
import numpy as np
import matplotlib.pyplot as plt

class Circle(Point):
    def __init__(self, pos, radius, step=1e-3, motion=None):
        super().__init__(pos, motion)
        x0, y, z0 = pos
        x = np.arange(x0-radius, x0+radius, step)
        z = np.arange(z0-radius, z0+radius, step)
        pts = np.stack(np.meshgrid(x, y, z), -1).reshape((-1, 3))
        dist = np.linalg.norm(pts-pos, axis=-1)
        self.pos = pts[dist<=radius]

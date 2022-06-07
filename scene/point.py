import numpy as np
from motion import Stationary

class Point:
    def __init__(self, pos, motion=None):
        self.pos = np.array(pos)
        if len(self.pos.shape) == 1:
            self.pos = np.expand_dims(pos, 0)
        self.size = self.pos.shape[0]
        self.path = None
        if not motion:
            self.motion = Stationary()
        else:
            self.motion = motion

    def get_init_pos(self):
        return self.pos

    def get_size(self):
        return self.size

    def get_path(self):
        if self.path is None:
            self.path = self.motion.move(self.pos)
        return self.path

    def register(self, *info):
        self.motion.register(*info)

    def get_average_pos(self):
        if self.path is None:
            self.path = self.motion.move(self.pos)
        return np.mean(self.path, axis=0)
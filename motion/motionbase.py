import numpy as np
import warnings

class Stationary:
    """The default motion that keeps an object stationary."""
    cnt = 0
    def __init__(self, name=None):
        self.registered = False
        self.path = None
        if not name:
            self.name = f'{self.__class__.__name__}_{type(self).cnt}'
        else:
            self.name = name
        type(self).cnt += 1

    def move(self, pos):
        """Take (n, 3) pos, return (steps, n, 3) pos"""
        if not self.registered:
            raise RuntimeError('Err: Motion initialised but not registered with the simulator')
        pos = np.expand_dims(pos, 0)
        res = np.repeat(pos, self.steps, 0)
        return res

    def register(self, t, fps):
        """Register frame information."""
        self.t = t
        self.fps = fps
        self.steps = int(t*fps)
        self.registered = True

    def get_path(self):
        if not self.registered:
            raise RuntimeError('Err: Motion not registered with the simulator, path not available')
        return self.path


class MotionBase(Stationary):
    """Base class to represent a motion"""
    def __init__(self, name=None, delay=None):
        """
        Parameters:
            name: name of the motion.
            delay: start the motion after `delay` seconds.
        """
        super().__init__(name=name)
        self.delay = delay

    def move(self, pos):
        """Apply the motion path to the object"""
        if not self.registered:
            raise RuntimeError('Err: Motion initialised but not registered with the simulator')
        n = pos.shape[0]
        res = np.zeros((self.steps, n, 3))
        for i in range(self.steps):
            res[i] = pos + self.path[i]
        return res

    def make_path(self):
        """Calculate the path of the motion"""
        raise NotImplementedError

    def register(self, t, fps):
        super().register(t, fps)
        self.make_path()
        max_d = 1e-3
        d = np.linalg.norm(self.path[1:]-self.path[:-1], axis=1)
        if np.max(d) > max_d:
            warnings.warn(f'[{self.name}] Warning: Peak velocity {np.max(d)*fps:.2f} m/s may cause ambiguous phase wrapping.')
        if self.delay:
            delay_steps = int(self.fps*self.delay)
            self.path[delay_steps:] = self.path[:-delay_steps]
            self.path[:delay_steps] = 0

class MotionList(MotionBase):
    """A list of motion to be combined"""
    def __init__(self, motions: list, **kwargs):
        """
        Parameters:
            name: name of the motion.
            delay: start the motion after `delay` seconds.
            motions: list of motions to be combined.
        """
        super().__init__(**kwargs)
        self.motions = motions

    def make_path(self):
        self.path = np.zeros((self.steps, 3))
        for motion in self.motions:
            self.path += motion.get_path()

    def register(self, t, fps):
        for motion in self.motions:
            motion.register(t, fps)
        super().register(t, fps)

class Line(MotionBase):
    """Motion along a straight line"""
    def __init__(self, velocity: tuple[float], **kwargs):
        """
        Parameters:
            name: name of the motion.
            delay: start the motion after `delay` seconds.
            velocity: a vector of size 3, defines the velocity in m/s.
        """
        super().__init__(**kwargs)
        self.velocity = np.array(velocity)

    def make_path(self):
        self.path = np.zeros((self.steps, 3))
        for i in range(self.steps):
            self.path[i] = self.velocity*i*(1/self.fps)

class LineBF(Line):
    """
    A linear motion that turns back every `turn` seconds
    """
    def __init__(self, velocity: tuple[float], turn: float, **kwargs):
        """
        Parameters:
            name: name of the motion.
            delay: start the motion after `delay` seconds.
            velocity: a vector of size 3, defines the velocity in m/s.
            turn: period of the turn.
        """
        super().__init__(velocity, **kwargs)
        self.turn = turn

    def make_path(self):
        t1 = self.turn*self.fps
        t2 = t1*2
        self.path = np.zeros((self.steps, 3))
        for i in range(self.steps):
            j = i % t2
            self.path[i] = self.velocity*j*(1/self.fps) if j < t1 else self.velocity*(t2-j)*(1/self.fps)
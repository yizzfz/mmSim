import numpy as np

class Simulator:
    def __init__(self, radars, scene, period, fps):
        self.T = period
        self.fps = fps
        self.steps = self.T*self.fps
        self.radars = radars
        for radar in radars:
            radar.register(self.T, self.fps)
        self.scene = scene
        for obj in self.scene:
            obj.register(self.T, self.fps)
        self.name = f'{self.__class__.__name__}'
        self.max_v = 1e-3*fps
        print(f'[{self.name}] Allowed maximum velocity is {self.max_v} m/s')

    def run(self):
        signals, freqzs = [], []
        for radar in self.radars:
            s, f = radar.reflect_motion_multi(self.scene)
            signals.append(s)
            freqzs.append(f)
        signals = np.array(signals)
        freqzs = np.array(freqzs)
        return signals, freqzs

    def get_paths(self):
        res = []
        for obj in self.scene:
            res.append(obj.get_path())
        return res
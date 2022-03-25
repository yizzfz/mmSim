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
        # print(f'[{self.name}] Allowed maximum velocity is {self.max_v} m/s')

    def run(self, freqz=False, ret_snr=False):
        n_rx = len(self.radars)
        signals = []
        snr = []
        freqzs = []
        cnt = 0
        for radar in self.radars:
            signal, info = radar.reflect_motion_multi(self.scene, freqz=freqz)
            signals.append(signal)
            snr.append(info.get('snr'))
            freqzs.append(info.get('freqz'))
            cnt += 1
        #     print(f'Progress {cnt/n_rx*100:.1f}%', end='\r')
        # print('')
        signals = np.array(signals)
        if freqz:
            freqzs = np.array(freqzs)
            return signals, freqzs

        if ret_snr:
            return signals, np.mean(snr)
        return signals

    def get_paths(self):
        res = []
        for obj in self.scene:
            res.append(obj.get_path())
        return res
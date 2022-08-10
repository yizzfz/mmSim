import numpy as np
import datetime

class Simulator:
    """The main simulator class"""
    def __init__(self, radars: list[object], scene: list[object], period: float, fps: int):
        """
        Parameters:
            radars: a list of radar objects.
            scene: a list of scene objects.
            period: simulation period in seconds.
            fps: number of chirps per second.
        """
        self.T = period
        self.fps = fps
        self.radars = radars
        self.steps = self.T*self.fps            # total number of chirps to simulate
        for radar in radars:
            radar.register(self.T, self.fps)    # register the frame information for each radar
        self.scene = scene
        for obj in self.scene:
            obj.register(self.T, self.fps)      # register the frame information for each object
        self.name = f'{self.__class__.__name__}'
        self.max_v = 1e-3*fps                   # maximum allowed velocity to avoid ambigious phase
        # print(f'[{self.name}] Allowed maximum velocity is {self.max_v} m/s')

    def run(self, freqz=False, ret_snr=False, print_progress=False):
        """Run the simulation and return the simulated data
        
        Parameters:
            freqz: return ground truth frequency information or not. Not recommended with large models. 
            ret_snr: return snr information of the simulation. Require `freqz` to be disabled. 
            print_progress: print out rough simulation progress.

        Return:
            simulated radar siganl matrix of shape (n_rx, n_chirp, n_sample), with freqz or snr.
        """
        n_rx = len(self.radars)
        signals = []
        snr = []
        freqzs = []
        cnt = 0
        for radar in self.radars:   
            # simulate for each rx
            signal, info = radar.reflect_motion_multi(self.scene, freqz=freqz)
            signals.append(signal)
            snr.append(info.get('snr'))
            freqzs.append(info.get('freqz'))
            cnt += 1
            if print_progress:
                ts = datetime.datetime.now().strftime('%H:%M')
                print(f'[{ts}] Progress {cnt/n_rx*100:.1f}%', end='\r')
        if print_progress:
            ts = datetime.datetime.now().strftime('%H:%M')
            print(f'[{ts}] Simulation finished.')
        signals = np.array(signals)
        if freqz:   # if return ground truth frequency and phase
            freqzs = np.array(freqzs)
            return signals, freqzs

        if ret_snr:
            return signals, np.mean(snr)
        return signals

    def get_paths(self):
        """Return the path of all points in the scene"""
        res = []
        for obj in self.scene:
            res.append(obj.get_path())
        return res
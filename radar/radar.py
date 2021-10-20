import numpy as np
import sys

class Radar:
    cnt = 0
    def __init__(self, pos=(0, 0, 0), f1=77e9, slope=40e12, ADC_rate=15e6, chirp_time=100e-6, phase_shift=0):
        self.f1 = f1
        self.slope = slope
        self.pos = pos
        self.tx_f = f1
        self.tx_p = 0
        self.c = 3e8
        self.ADC_rate = ADC_rate
        self.chirp_time = chirp_time
        self.phase_shift = phase_shift
        self.name = f'{self.__class__.__name__}_{type(self).cnt}'
        type(self).cnt += 1
        print(f'[{self.name}] configured to {f1/1e9:.1f} GHz to {(f1+slope*chirp_time)/1e9:.1f} GHz')
        # self.wavelength = self.c/self.f1

    def distance_to(self, pos):
        dis = np.linalg.norm(np.array(self.pos)-np.array(pos))
        return dis

    def IF(self, t, tof):
        return np.cos(2*np.pi*self.slope*tof*t+2*np.pi*self.f1*tof-np.pi*self.slope*tof*tof+self.phase_shift)

    def freqz(self, tof):
        return self.slope*tof, (2*np.pi*self.f1*tof-np.pi*self.slope*tof*tof+self.phase_shift)%(2*np.pi)

    def reflect(self, obj, sum=False):
        '''Take (n, 3) obj, return one signal and frequency response'''
        if isinstance(obj, np.ndarray):
            n_obj = obj.shape[0]
            pos = obj
        else:
            n_obj = obj.get_size()
            pos = obj.get_pos()
        n_sample = int(self.chirp_time * self.ADC_rate)
        signal = np.zeros((n_obj, n_sample))
        freq = np.zeros((n_obj, 2))
        for i, pos_i in enumerate(pos):
            dis = self.distance_to(pos_i)
            tof = 2*dis/self.c
            # freq = tof*self.slope
            t = np.arange(0, self.chirp_time, 1/self.ADC_rate)
            signal[i] = self.IF(t, tof)
            freq[i] = self.freqz(tof)
        if sum:
            return np.sum(signal, axis=0)/n_obj, freq
        return signal, freq

    def reflect_motion(self, obj, sum=False):
        '''Take an obj (that has a motion of certain steps), return the simulation signal and the frequency response

        Parameters:
            obj: Scene class with (n, 3) pos
            sum: return one single signal, or n signals for every pos

        Return:
            signals: (steps, signal_len) if sum==True, (steps, n_obj, signal_len) otherwise
            freqzs: (steps, n_obj, 2), frequency and phase of each pos at each step
        '''
        path = obj.get_path()
        n_obj = obj.get_size()
        n_sample = int(self.chirp_time * self.ADC_rate)
        if sum:
            signals = np.zeros((self.steps, n_sample))
        else:
            signals = np.zeros((self.steps, n_obj, n_sample))
        freqzs = np.zeros((self.steps, n_obj, 2))
        for i in range(self.steps):
            pos = path[i]
            signals[i], freqzs[i] = self.reflect(pos, sum=sum)
        return signals, freqzs

    def reflect_motion_multi(self, objs: list):
        '''Take a list of objs, return the simulation signal and the frequency response

        Parameters:
            [obj]: list of Scene class with (n, 3) pos

        Return:
            signals: (steps, signal_len)
            freqzs: (steps, n_obj, 2), frequency and phase of each pos at each step
        '''
        n_total_objs = np.sum([o.get_size() for o in objs])
        n_sample = int(self.chirp_time * self.ADC_rate)
        signal = np.zeros((self.steps, n_sample))
        freqzs = np.zeros((self.steps, n_total_objs, 2))
        cnt = 0
        for obj in objs:
            obj_path = obj.get_path()
            n_obj = obj.get_size()
            for i in range(self.steps):
                pos = obj_path[i]
                s, f = self.reflect(pos, sum=True)
                signal[i] += s
                freqzs[i, cnt:cnt+n_obj] = f
            cnt += n_obj
        signal = signal / len(objs)
        return signal, freqzs

    def register(self, time, fps):
        if fps > 1/self.chirp_time:
            print('Err: radar scanning rate is too high')
            sys.exit(1)
        self.T = time
        self.fps = fps
        self.steps = time*fps
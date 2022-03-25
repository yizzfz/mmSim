import numpy as np
import sys
from collections.abc import Sequence
from .rx_config import RxConfig

class Radar:
    cnt = 0
    def __init__(self, Tx_pos=(0, 0, 0), Rx_pos=None, f1=77e9, slope=40e12, ADC_rate=15e6, chirp_time=100e-6, phase_shift=0, noise=None):
        self.f1 = f1
        self.slope = slope
        self.Tx_pos = Tx_pos
        self.Rx_pos = Rx_pos
        self.tx_f = f1
        # self.tx_p = 0
        self.c = 3e8
        self.ADC_rate = ADC_rate
        self.chirp_time = chirp_time
        self.phase_shift = phase_shift
        self.rng = np.random.default_rng()

        wavelength = self.c / self.f1
        self.K = 10 ** 2.6 * 1e-3 * wavelength**2 * 1e-2**2 * chirp_time / (4*np.pi)**3 / 4e-21
        self.A = lambda d: np.sqrt(self.K/d**4)
        self.snr_db = lambda d: 10*np.log(self.K/d**4)
        self.snr = lambda d: self.K/d**4
        self.noise = noise
        if noise is not None:
            self.noise_std = (10**(noise/10)/2)**0.5    # convert dB to linear, calculate std for normal distribution
        # radar equation:
        # snr per chirp = P_tx * G_tx * G_rx * wavelength**2 * radar_cross_section * chirptime / ((4pi)**3 * kT * d**4)
        #               = (12dbm + 7db + 7db) * wavelength**2 * 1e-2**2 * chirptime / ((4pi)**3 * 4e-21 * d**4)
        #               = (10**2.6 * 1e-3) * wavelength**2 * 1e-2**2 * chirptime / ((4pi)**3 * 4e-21 * d**4)
        # http://www.ece.uah.edu/courses/material/EE619-2011/RadarRangeEquation(2)2011.pdf
        # https://e2e.ti.com/support/sensors-group/sensors/f/sensors-forum/689147/iwr1443-sensing-estimator-snr-calculation
        
        self.name = f'{self.__class__.__name__}_{type(self).cnt}'
        type(self).cnt += 1
        # print(f'[{self.name}] configured to {f1/1e9:.1f} GHz to {(f1+slope*chirp_time)/1e9:.1f} GHz')

    def distance_between(self, pos1, pos2):
        dis = np.linalg.norm(np.array(pos1)-np.array(pos2))
        return dis

    def dis_to(self, pos):
        dis1 = self.distance_between(self.Tx_pos, pos)
        if self.Rx_pos is not None:
            dis2 = self.distance_between(self.Rx_pos, pos)
        else:
            dis2 = dis1
        return dis1+dis2

    def IF(self, t, dis):
        tof = dis/self.c
        signal = np.exp(1j*(2*np.pi*self.slope*tof*t+2*np.pi*self.f1*tof-np.pi*self.slope*tof*tof+self.phase_shift))        
        amplitude = self.A(dis/2)
        return amplitude * signal

    def freqz(self, dis):
        tof = dis/self.c
        return self.slope*tof, (2*np.pi*self.f1*tof-np.pi*self.slope*tof*tof+self.phase_shift)%(2*np.pi)

    def reflect(self, obj, sum=False, freqz=True):
        '''Take (n, 3) obj, return one signal and frequency response'''
        if isinstance(obj, np.ndarray):
            n_obj = obj.shape[0]
            pos = obj
        else:
            n_obj = obj.get_size()
            pos = obj.get_pos()
        n_sample = int(self.chirp_time * self.ADC_rate)
        signal = np.zeros((n_obj, n_sample), dtype=complex)
        if freqz:
            freq = np.zeros((n_obj, 2))
        for i, pos_i in enumerate(pos):
            dis = self.dis_to(pos_i)
            t = np.arange(0, self.chirp_time, 1/self.ADC_rate)
            signal[i] = self.IF(t, dis)
            if freqz:
                freq[i] = self.freqz(dis)
        if sum:
            signal = np.sum(signal, axis=0)
        if freqz:
            return signal, freq
        return signal

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

    def reflect_motion_multi(self, objs: list, freqz=False):
        '''Take a list of objs, return the simulation signal and the frequency response

        Parameters:
            [obj]: list of Scene class with (n, 3) pos

        Return:
            signals: (steps, signal_len)
            freqzs: (steps, n_obj, 2), frequency and phase of each pos at each step
        '''
        n_total_objs = np.sum([o.get_size() for o in objs])
        n_sample = int(self.chirp_time * self.ADC_rate)
        signal = np.zeros((self.steps, n_sample), dtype=complex)
        if freqz:
            freqzs = np.zeros((self.steps, n_total_objs, 2))
        cnt = 0
        snr = np.zeros((self.steps))
        for obj in objs:
            obj_path = obj.get_path()
            n_obj = obj.get_size()
            for i in range(self.steps):
                pos = obj_path[i]
                res = self.reflect(pos, sum=True, freqz=freqz)
                if freqz:
                    signal[i] += res[0]
                    freqzs[i, cnt:cnt+n_obj] = res[1]
                else:
                    signal[i] += res
                snr[i] = 10*np.log10(self.signal_power(signal[i]))
                if self.noise is not None:
                    # signal_power = np.mean(np.abs(signal)**2)
                    # print(signal_power)
                    # noise_power = signal_power*(1/(10**(self.snr/10)))
                    # # noise_power = 10**((10*np.log(signal_power) - self.snr)/10)       same 
                    # noise_power = 10**(self.noise/10)
                    # noise_std = np.sqrt(noise_power/2)
                    noise = self.rng.normal(0, self.noise_std, n_sample) + 1j*self.rng.normal(0, self.noise_std, n_sample)
                    snr[i] = snr[i] - 10*np.log10(self.signal_power(noise))
                    signal[i] = signal[i] + noise
            cnt += n_obj
        snr = np.mean(snr)
        info = {}
        if freqz:
            info['freqz'] = freqzs
        info['snr'] = snr
        return signal, info

    def register(self, time, fps):
        if fps > 1/self.chirp_time:
            print('Err: radar scanning rate is too high')
            sys.exit(1)
        self.T = time
        self.fps = fps
        self.steps = int(time*fps)

    def signal_power(self, X):
        return np.mean(np.abs(X)**2)

class RadarArray:
    def __init__(self, Tx_pos, layout, f1=77e9, slope=40e12, ADC_rate=15e6, chirp_time=100e-6, noise=None):
        space = 3e8/f1/2    # wavelength / 2
        self.rxcfg = RxConfig(layout)
        self.radars = []
        rx = self.rxcfg.rx*space
        for pos in rx:
            self.radars.append(Radar(Tx_pos=Tx_pos, Rx_pos=pos, f1=f1, slope=slope, ADC_rate=ADC_rate, chirp_time=chirp_time, noise=noise))

    def __len__(self):
        return len(self.radars)

    def __getitem__(self, i):
        return self.radars[i]

    def __iter__(self):
        for r in self.radars:
            yield r

    def get_rxconfig(self):
        return self.rxcfg
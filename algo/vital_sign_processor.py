import numpy as np
from collections import deque
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from algo import phase_tracking, get_aaed_phase, phase_difference, lowpass
from algo import wavelet_v1 as wavelet
from skimage.feature import peak_local_max
from scipy.signal import find_peaks


class VitalSignProcessor:
    def __init__(self, config, bin_change_rate, tracking_win, smoothing_win, downsample=1):
        """config must include: fps"""
        self.config = config
        self.tracking_win = tracking_win
        self.smoothing_win = smoothing_win
        self.bin_change_rate = bin_change_rate
        self.last_path = None
        self.downsample = downsample

    def generate_phase_signal(self, fft_mags, fft_phases, init=None, debug=False, ret_path=False):
        if fft_mags.shape != fft_phases.shape or len(fft_mags.shape) != 2:
            raise ValueError(f'Incorrect data shape for fft_mags {fft_mags.shape}')
        fft_peaks = phase_tracking(
            fft_mags, sigma=self.bin_change_rate, win=self.tracking_win, init=init)
        fft_peaks = gaussian_filter1d(fft_peaks.astype(float), self.smoothing_win)
        # fft_peaks = np.ones(fft_mags.shape[0])*np.argmax(np.sum(fft_mags, axis=0))
        phases = get_aaed_phase(fft_phases, fft_peaks)
        phases = phases[::self.downsample]
        if debug:
            wavelet_width = 0.02
            wavelet_func = 'shan'
            fig, axs = plt.subplots(2, 3)
            axs[0][0].imshow(fft_mags.T, aspect=fft_mags.shape[0]/fft_mags.shape[1], origin='lower')
            axs[0][0].plot(fft_peaks, color='red')
            axs[0][1].plot(self.wavelet(phases, wavelet_width=wavelet_width, wavelet_func=wavelet_func))
            axs[0][2].plot(self.wavelet(phases.cumsum(),
                           wavelet_width=wavelet_width,wavelet_func=wavelet_func))
            axs[1][1].plot(phases)
            axs[1][2].plot(phases.cumsum())
            x = np.fft.fftfreq(phases.shape[0]*4, d=10/self.config['fps'])
            y = np.abs(np.fft.fft(phases*np.hamming(phases.shape[0]), phases.shape[0]*4))
            lim = np.argmax(x>4)
            axs[1][0].plot(x[:lim], y[:lim])
            plt.tight_layout()
            plt.show()
        # phases = np.unwrap(phases, period=2)
        if ret_path:
            return phases, fft_peaks
        return phases

    def generate_phase_signal_cont(self, fft_mags, fft_phases, ret_path=False):
        if fft_mags.shape != fft_phases.shape or len(fft_mags.shape) != 2:
            raise ValueError(f'Incorrect data shape for fft_mags {fft_mags.shape}')
        init = None
        if self.last_path is not None:
            if self.last_path[-1] < 15:
                self.last_path = None
            else:
                init = self.last_path[-1]

        fft_peaks = phase_tracking(
            fft_mags, sigma=self.bin_change_rate, win=self.tracking_win, init=init)
        if self.last_path is not None:
            tmp = np.concatenate((self.last_path, fft_peaks))
            fft_peaks_filtered = gaussian_filter1d(tmp.astype(float), self.smoothing_win)[self.last_path.shape[0]:]
        else:
            fft_peaks_filtered = gaussian_filter1d(fft_peaks.astype(float), self.smoothing_win)
        phases = get_aaed_phase(fft_phases, fft_peaks_filtered)
        phases = phases[::self.downsample]
        self.last_path = fft_peaks
        if ret_path:
            return phases, fft_peaks_filtered
        return phases

    def bin_tracking_cont(self, fft_mags):
        init = None
        if self.last_path is not None:
            if self.last_path[-1] < 15:
                self.last_path = None
            else:
                init = self.last_path[-1]

        fft_peaks = phase_tracking(
            fft_mags, sigma=self.bin_change_rate, win=self.tracking_win, init=init)
        self.last_path = fft_peaks
        return fft_peaks

    def get_phase_signal_from_path(self, path, fft_phases):
        path = gaussian_filter1d(path.astype(float), self.smoothing_win)
        phases = get_aaed_phase(fft_phases, path)
        phases = phases[::self.downsample]
        return phases

    def phase_diff(self, x):
        return phase_difference(x)
    
    def wavelet(self, x, wavelet_width=0.1, wavelet_func='morl', border=0.05):
        wavelet_width = self.config['fps']/self.downsample*wavelet_width
        return np.abs(wavelet(x, wavelet_width=wavelet_width, wavelet_func=wavelet_func, border=border))

    def find_peaks(self, x, border=0.05, debug=False):
        fps = self.config['fps']/self.downsample
        steps = x.shape[0]
        border = int(steps*border)

        x = lowpass(x, cutoff=0.1)
        peaks = find_peaks(x, distance=int(fps*0.35), height=0.2 *
                           x.max(), prominence=0.1)[0]
        # peaks = peak_local_max(x, min_distance=int(0.3*fps), threshold_rel=0.2, exclude_border=True)
        # peaks = np.sort(peaks.flatten())
        # peaks = CFAR_hr(x, win1=win1, win2=win2, skip=skip, start=start, end=end)
        if debug:
            plt.plot(x)
            plt.scatter(peaks, x[peaks], s=64, facecolors='none', edgecolors='r')
            plt.show()
        return peaks
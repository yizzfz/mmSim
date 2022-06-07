import numpy as np
import datetime
import matplotlib.pyplot as plt


class FFTProcessor:
    def __init__(self, config, multiplier=4, max_d=1.5):
        """config must include: fps, samples_per_chirp, ADC_rate, slope"""
        self.config = config
        self.multiplier = multiplier
        self.max_d = max_d
        n_samples = self.config['samples_per_chirp']
        ADC_rate = self.config['ADC_rate']
        slope = self.config['slope']
        self.n_fft = n_samples*multiplier
        fft_freq = np.fft.fftfreq(self.n_fft, d=1.0/ADC_rate)
        fft_freq_d = fft_freq*3e8/2/slope
        self.max_freq_i = np.argmax(fft_freq_d>self.max_d)
        self.fft_freq = fft_freq[:self.max_freq_i]
        self.win = np.hanning(n_samples)
        import pdb; pdb.set_trace()

    def compute_FFTs(self, data, split=True):
        data = data * self.win
        fft_out = np.fft.fft(data, self.n_fft)[:, :self.max_freq_i]
        if split:
            fft_mags = np.abs(fft_out)
            fft_phases = np.angle(fft_out)/np.pi
            return np.stack((fft_mags, fft_phases))
        return fft_out

    def get_fft_freq(self):
        return self.fft_freq

if __name__ == '__main__':
    config = {
        'fps': 5000,
        'samples_per_chirp': 1500,
        'ADC_rate': 9e6,
        'slope': 21e12,
    }
    FP = FFTProcessor(config)
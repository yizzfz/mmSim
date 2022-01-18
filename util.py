import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# find the highest value in an FFT, return (idx, mag, phase)
def get_FFT_peak(data):
    mag = np.abs(data)
    # mag[:20] = 0            # ignore signal too close
    phases = np.angle(data)
    phases = (phases+2*np.pi) % (2*np.pi)
    peak_idx = np.argmax(mag)
    return (peak_idx, mag[peak_idx], phases[peak_idx])


def highpass(x, order=8, cutoff=0.5):
    b, a = signal.butter(order, cutoff, 'highpass')
    y = signal.filtfilt(b, a, x)
    return y

def lowpass(x, order=8, cutoff=0.5):
    b, a = signal.butter(order, cutoff, 'lowpass')
    y = signal.filtfilt(b, a, x)
    return y
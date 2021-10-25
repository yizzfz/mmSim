import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def CFAR(fft_data, win1=2, win2=4, phase=False):
    mag = np.absolute(fft_data)
    mag = 10*np.log10(mag/np.max(mag))
    phases = np.angle(fft_data)
    phases = (phases+2*np.pi) % (2*np.pi)
    n = fft_data.shape[0]
    th = -3
    res = []

    for i in range(n):
        if mag[i] < th:
            continue

        # if the current point is a local maxima
        if np.max(mag[max(i-win2, 0):i+win2+1]) == mag[i]:
            # compare with average value of neighbours
            threshold = np.average(mag[max(i-win1, 0):i+win1+1])
            if mag[i] < threshold:
                continue
            
            # if detecting peaks of a phase-fft
            # check if this is an alias of previous peak
            # if phase:
            #     dup = False
            #     for peak, mag in res:
            #         if (abs(2*peak - i) < 2) and data[i] < mag - 5:
            #             dup = True
            #             break
            #     if dup:
            #         continue
            res.append((i, mag[i], phases[i]))
    res = sorted(res, key=lambda x:x[1], reverse=True) 
    return res

def FFT(signal, fs):
    n_samples = signal.shape[0]
    n_fft = n_samples*4
    half = int(n_fft/2)
    freq = np.fft.fftfreq(n_fft, d=1.0/fs)[:half]
    fft = np.fft.fft(signal, n_fft)[:half]
    mag = np.abs(fft)*2/n_samples
    return freq, mag

def unwrap(signal):
    S = signal.copy()
    for i in range(1, len(S)):
        if S[i] - S[i-1] > 1:
            S[i:] -= 2
        elif S[i-1] - S[i] > 1:
            S[i:] += 2
    return S 

def highpass(x, order=10, cutoff=0.5):
    b, a = signal.butter(order, cutoff, 'highpass')
    y = signal.filtfilt(b, a, x)
    return y

def wavelet(x, fps, pulse_length=0.1):
    # widths = np.arange(1, fps*pulse_length, 1)
    # cwt = signal.cwt(x, signal.ricker, widths)
    # plt.imshow(cwt, cmap='PRGn', aspect='auto', vmax=abs(cwt).max(), vmin=-abs(cwt).max())
    # plt.show()
    # import pdb; pdb.set_trace()
    widths = [int(fps*pulse_length*0.2)]
    y = signal.cwt(x, signal.ricker, widths)[0]
    border = int(y.shape[0]*0.01)
    y[:border] = y[border]
    y[-border:] = y[-border]
    return y
import numpy as np
def CFAR(fft_data, phase=False):
    mag = np.absolute(fft_data)
    mag = 10*np.log10(mag/np.max(mag))
    phases = np.angle(fft_data)
    phases = (phases+2*np.pi) % (2*np.pi)
    n = fft_data.shape[0]
    win1 = 0
    win2 = 1
    th = -6
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
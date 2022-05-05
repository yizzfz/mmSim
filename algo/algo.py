import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from collections import deque
from scipy.ndimage import convolve
import pywt
from collections import deque
from scipy.stats import trim_mean
import time
from scipy.ndimage import median_filter
import cProfile
from scipy.signal import find_peaks

# classic CFAR
def CFAR(data, th=-6, win1=2, win2=4, FFT=True, norm=True):
    if norm:
        mag = np.absolute(data)
        mag = 10*np.log10(mag/np.max(mag))
    else:
        mag = data
    if FFT:
        phases = np.angle(data)
        phases = (phases+2*np.pi) % (2*np.pi)

    n = data.shape[0]
    res = []

    for i in range(n):
        if norm and mag[i] < th:
            continue

        # if the current point is a local maxima
        if np.max(mag[max(i-win2, 0):i+win2+1]) == mag[i]:
            # compare with average value of neighbours
            threshold = np.average(mag[max(i-win1, 0):i+win1+1])
            if mag[i] > threshold and not np.isclose(mag[i], threshold, rtol=1e-2, atol=1e-4):
                if FFT:       
                    res.append((i, mag[i], phases[i]))
                else:
                    res.append((i, mag[i]))
    res = sorted(res, key=lambda x:x[1], reverse=True)
    return res


def CFAR_CA(data, th=None, win=8, guard=4, debug=False, min_db=None, min_rel=None):
    """CFAR Cell Averaging

    Parameters:
        data: 1d array, should be in dB scale
        th: interpreted as false alarm rate if < 1, a noise threshold in dB otherwise
        win: length of train cells at one side
        guard: length of guard cells at one side
        debug: plot threshold

    Return:
        A list of peak indices
    """
    assert guard < win and "guard cells must be less than train cells"
    data_n = data
    data_n[data_n<0] = 0
    N = win*2-guard*2
    res = []
    sum_win = convolve(data_n, np.ones(win*2+1,dtype=int), mode='wrap')
    sum_guard = convolve(data_n, np.ones(guard*2+1,dtype=int), mode='wrap')
    noise = (sum_win - sum_guard) / N
    if th is None:
        th = noise
    elif th < 1:      # interpret th as false alarm rate as traditional CFAR
        alpha = N*(th**(-1/N)-1)
        th = noise * alpha
    else:           # interpret th as a noise threshold in dB
        th = noise + th
    
    res = np.where(data_n > th)[0]

    if min_db is not None:
        maxpower = np.max(data_n)
        minpower = maxpower+min_db
        th[th < minpower] = minpower
    elif min_rel is not None:
        maxpower = np.max(data_n)
        minpower = maxpower * min_rel
        th[th < minpower] = minpower

    if debug:
        plt.plot(data_n, label='data')
        plt.plot(th, label='th')
        plt.legend()
        plt.show()
    return res


def CFAR_A(data, th=None, win=8, guard=4, debug=False, mode='GO', rank=None, min_db=None, min_rel=None):
    """CFAR Greatest of / Smallest of / Order statistic

    Parameters:
        data: 1d array, should be in dB scale
        th: interpreted as false alarm rate if < 1, a noise threshold in dB otherwise
        win: length of train cells at one side
        guard: length of guard cells at one side
        debug: plot threshold
        mode: GO or SO or OS or CA
        rank: which element to use in OS mode

    Return:
        A list of peak indices
    """
    assert guard < win and "guard cells must be less than train cells"
    data_n = data
    data_n[data_n<0] = 0
    N = win-guard
    res = []
    if mode == 'GO':
        func = lambda x, y, N: np.max((np.sum(x), np.sum(y)))/N
    elif mode == 'SO':
        func = lambda x, y, N: np.min((np.sum(x), np.sum(y)))/N
    elif mode == 'OS':
        if not rank:
            rank = N
        func = lambda x, y, N: np.sort(np.concatenate((x, y)))[::-1][N]
    elif mode == 'CA':
        func = lambda x, y, N: (np.sum(x) + np.sum(y))/N/2
    else:
        raise ValueError(f'Unsupported mode {mode}')

    n = data.shape[0]
    noise = data_n.copy()
    data_n = np.tile(data_n, 7)
    ind = np.arange(n*3, n*4)

    for i in ind:
        noise_left = data_n[i-win:i-guard]
        noise_right = data_n[i+guard+1:i+win+1]
        try:
            noise[i-n*3] = func(noise_left, noise_right, N)
        except Exception as e:
            print(e)
            print(mode, win, guard)
            raise ValueError

    if th is None:
        th = noise
    elif th < 1:        # interpret th as false alarm rate as traditional CFAR
        alpha = N*(th**(-1/N)-1)
        th = noise * alpha
    else:               # interpret th as a noise threshold in dB
        th = noise + th
    
    if min_db is not None:
        maxpower = np.max(data_n)
        minpower = maxpower+min_db
        th[th<minpower] = minpower
    elif min_rel is not None:
        maxpower = np.max(data_n)
        minpower = maxpower * min_rel
        th[th < minpower] = minpower

    res = np.where(data > th)[0]
    if debug:
        plt.plot(data, label='data')
        plt.plot(th, label='th')
        plt.legend()
        plt.show()
    return res

# CFAR adapted for searching heart pulses in a phase signal
def CFAR_hr(data, th=-6, win1=2, win2=4, start=0, end=None, skip=0, sort=False):
    mag = np.absolute(data)
    if np.max(mag) == 0:
        return []
    mag = 10*np.log10(mag/np.max(mag))
    if not end:
        end = data.shape[0]
    res = []
    i = start
    while i < end:
        if mag[i] > th:
            # if the current point is a local maxima
            if np.max(mag[max(i-win2, 0):i+win2+1]) == mag[i]:
                # compare with average value of neighbours
                threshold = np.average(mag[max(i-win1, 0):i+win1+1])
                if mag[i] > threshold and not np.isclose(mag[i], threshold, rtol=1e-2, atol=1e-4):
                    res.append((i, mag[i]))
                    i += skip
        i += 1
    if sort:
        res = sorted(res, key=lambda x:x[1], reverse=True)
    return res

# wavelet transform
def wavelet(x, fps, pulse_length=0.1, norm=False, border=0.05):
    # widths = np.arange(1, fps*pulse_length, 1)
    # cwt = signal.cwt(x, signal.ricker, widths)
    # plt.imshow(cwt, cmap='PRGn', aspect='auto', vmax=abs(cwt).max(), vmin=-abs(cwt).max())
    # plt.show()
    widths = [int(fps*pulse_length*0.2)]
    y = signal.cwt(x, signal.ricker, widths)[0]
    border = int(y.shape[0]*border)
    if border > 0:
        # y[:border] = y[border]
        # y[-border:] = y[-border]
        y[:border] = 0
        y[-border:] = 0
    if norm:
        y = y/y.max()
    return y


def wavelet_v1(x, wavelet_width, border=0.05, wavelet_func='morl'):
    wavelet_width = [wavelet_width]
    y = pywt.cwt(x, wavelet_width, wavelet_func)[0][0]
    y = np.abs(y)
    border = int(y.shape[0]*border)
    if border > 0:
        y[:border] = 0
        y[-border:] = 0
    return y

# smooth tracked FFT bin a bit, deprecated
def path_smoothing(peaks, win=None, fps=1000, return_one=False, max_dp=None):
    res = peaks.copy()
    n_samples = peaks.shape[0]
    if win:
        for j in range(n_samples):
            res[j] = np.median(peaks[max(0, j-win):j+win])
    if max_dp:
        for j in range(1, n_samples):
            res[j] = min(res[j], res[j-1]+max_dp)
            res[j] = max(res[j], res[j-1]-max_dp)
    if return_one:
        res[:] = np.array([np.average(peaks)]*n_samples, dtype=int)
    return res

def phase_difference(p):
    r = np.zeros(p.shape)
    r[1:] = p[1:] - p[:-1]
    r = r/r.max()
    return r

def phase_tracking_v0(mag:np.ndarray, multiplier:int, win:int, start_idx=0, init=None):
    assert win > 0
    n_steps = mag.shape[0]
    n_fft = mag.shape[1]
    if not init:
        init = CFAR_hr(np.sum(mag[start_idx:start_idx+win], axis=0), start=1, skip=0, sort=True)[0][0]
        start_idx += 1

    min_score = 1e-3/((multiplier*2)**2)
    path = np.zeros((n_steps), dtype=int)

    # start from the first chirp to find a path
    if init == 0:
        print('Initial bin found to be zero, lilely to be an error')
        raise ValueError

    j = 0
    cur = int(init)
    # for each subsequent chirp 
    for j in range(start_idx, n_steps):
        # greedily find the next bin
        f = gaussian(n_fft, mu=cur, sigma=multiplier)
        f = f/f.max()
        m = np.sum(mag[max(0,j-win):j+win], axis=0)
        m = m/m.max()
        s = m*f
        s[s<min_score] = 0
        bin = np.argmax(s)
        
        if bin == 0:    # if no suitable path found
            break
        path[j] = bin
        cur = bin

    # if a suitable path has been found, return it
    if j == n_steps-1:
        path[0] = start_idx
        return path
    
    print('no bin found')
    raise ValueError


def phase_tracking_v1(mag:np.ndarray, sigma:int, win:int, start_idx=0, init=None):
    assert win > 0
    sigma = int(sigma)
    win = int(win)
    n_steps = mag.shape[0]
    n_fft = mag.shape[1]
    Q = deque(maxlen=int(win*0.1))
    if init is None:
        mag_sum = np.sum(mag[start_idx:start_idx+win], axis=0)
        peaks = CFAR_hr(mag_sum, start=1, skip=0, sort=True)
        if len(peaks) == 0:
            init = 1
        else:
            init = CFAR_hr(mag_sum, start=1, skip=0, sort=True)[0][0]
        start_idx += 1

    min_score = 1e-3/((sigma*2)**2)
    path = np.zeros((n_steps), dtype=int)

    # start from the first chirp to find a path
    if init == 0:
        print('Initial bin found to be zero, lilely to be an error')
        raise ValueError

    j = 0
    cur = int(init)
    # for each subsequent chirp 
    for j in range(start_idx, n_steps):
        # greedily find the next bin
        f = gaussian(n_fft, mu=cur, sigma=sigma)
        f = f/f.max()
        m = np.sum(mag[max(0,j-win):j+win], axis=0)
        m = m/m.max()
        s = m*f
        s[s<min_score] = 0
        bin = np.argmax(s)
        # plt.plot(f, label='gaussian')
        # plt.plot(m, label='next')
        # plt.plot(s, label='product')
        # plt.legend()
        # plt.show()
        
        if bin == 0:    # if no suitable path found
            bin = cur
        path[j] = bin
        cur = bin

    # if a suitable path has been found, return it
    if j == n_steps-1:
        path[0] = init
        return path
    
    raise ValueError('no bin found')

def phase_tracking(mag:np.ndarray, sigma:int, win:int, start_idx=0, init=None):
    assert win > 0
    sigma = int(sigma)
    win = int(win*2)       # win*2 to be consistent with previous version
    n_steps = mag.shape[0]
    n_fft = mag.shape[1]

    MA = np.zeros((mag.shape))
    MA[win:] = moving_average(mag, win, axis=0)
    MA = MA/np.max(MA)
    start_idx = max(start_idx, win)
    
    mag_sum = MA[win]
    peak = np.argmax(mag_sum)
    if init is None or np.abs(peak-init) > 10:
        init = peak
        start_idx += 1

    min_score = 1e-3/((sigma*2)**2)
    path = np.zeros((n_steps), dtype=float)
    path[:start_idx] = init

    # start from the first chirp to find a path
    if init == 0:
        print('Initial bin found to be zero, lilely to be an error')
        raise ValueError

    j = 0
    cur = init
    gaussian_template = gaussian(n_fft, mu=int((n_fft-1)/2), sigma=sigma)
    # fmax = gaussian_template.max()
    gaussian_template = gaussian_template/gaussian_template.max()
    # for each subsequent chirp
    for j in range(start_idx, n_steps):
        # greedily find the next bin
        f = gaussian_T(gaussian_template, mu=cur)
        # f = gaussian(n_fft, cur, sigma)/fmax
        m = MA[j]
        s = m*f
        s[s<min_score] = 0
        bin = np.argmax(s)
        # plt.plot(f, label='gaussian')
        # plt.plot(m, label='next')
        # plt.plot(s, label='product')
        # plt.legend()
        # plt.show()
        
        if bin == 0:    # if no suitable path found
            bin = 0
        path[j] = bin
        cur = bin
    return path

# track only one bin at mag[-1]
def phase_tracking_once(mag, multiplier, last=None):
    n_fft = mag.shape[1]
    if not last:
        return CFAR_hr(mag[-1], start=1, skip=0, sort=True)[0][0]
    min_score = 1e-3/((multiplier*2)**2)
    cur = int(last)
    f = gaussian(n_fft, mu=cur, sigma=multiplier)
    f = f/f.max()
    m = np.sum(mag, axis=0)
    m = m/m.max()
    s = m*f
    s[s<min_score] = 0
    bin = np.argmax(s)
    if bin == 0:
        print('no bin found')
        raise ValueError
    return bin

# developing version, not used, for debug only
def phase_tracking_alpha(mag, phase, fps, multiplier):
    n_steps = mag.shape[0]
    n_fft = mag.shape[1]
    init = CFAR_hr(mag[0], start=1, skip=0, sort=True)
    min_score = 1e-3/((multiplier*2)**2)
    win = int(fps*0.2)
    path = np.zeros((n_steps), dtype=int)
    # start from the first chirp to find a path
    # init = peaks_all[0]
    for start, _ in init:
        if start == 0:
            break

        cur = int(start)
        phase_cur = phase[0, cur]
        # for each subsequent chirp 
        for j in range(1, n_steps):
            # greedily find the next bin
            f = gaussian(n_fft, mu=cur, sigma=multiplier)
            f = f/f.max()
            m = np.sum(mag[max(0,j-win):j+win], axis=0)
            m = m/m.max()
            s = m*f
            s[s<min_score] = 0
            bin = np.argmax(s)
            if np.abs(bin-cur) > 1:
            # plt.ion()
            # plt.show()
            # if j >= 0.7*fps:
            #     print(cur, bin)
                # plt.cla()
                # m0 = np.sum(mag[max(0,j-win-1):j+win-1], axis=0)
                # m0 = m0/m0.max()
                # plt.plot(m0, label='Last FFT result')
                # plt.plot(mag[j], label='FFT result at t=1')
                plt.scatter(cur, f[cur], label='Last bin', s=64, color='r')
                plt.plot(f, label='Gaussian distribution around the last bin', color='r')
                plt.plot(s, label='Probablity of the new bin', color='b')
                plt.plot(m, label='Current FFT result', color='green')
                plt.scatter(bin, s[bin], label='New bin', s=64, color='b')
                plt.plot([cur, cur], [0, f[cur]], 'r--')
                plt.plot([bin, bin], [0, s[bin]], 'b--')
                plt.xlabel('FFT bin', fontsize=18)
                plt.ylabel('Normalised value', fontsize=18)
                plt.legend(fontsize=18)
                plt.tick_params(axis='both', which='major', labelsize=16)
                plt.show()
                # plt.waitforbuttonpress()

            if bin == 0:    # if no suitable path found
                break
            path[j] = bin
            cur = bin
        # if a suitable path has been found, return it
        # otherwise, go to the next start point
        if j == n_steps-1:
            path[0] = start
            return path
    print('No path found')
    raise ValueError

def moving_average(X, n, axis=1):
    """Moving average of a 2D array. axis=1: row average. axis=0: col average"""
    ss = np.cumsum(X, axis=axis)
    if axis == 1:
        ma = (ss[:, n:] - ss[:, :-n]) / n
    else:
        ma = (ss[n:] - ss[:-n]) / n
    return ma

def gaussian(n, mu=0.0, sigma=1.0):
    """Generate a gaussian distribution"""
    x = np.arange(0, n, 1)
    y = np.exp(-(x-mu)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    return y

def gaussian_T(T, mu=0):
    """Translate a gaussian function T to mu"""
    n = int((T.shape[0]-1)/2)
    d = int(mu - n)
    y = np.zeros(T.shape)
    if d > 0:
        y[d:] = T[:-d]
    elif d < 0:
        y[:d] = T[-d:]
    else:
        y = T
    return y

# generate a single-sided gaussian distribution
def single_sided_gaussian(n, direction, mu=0, sigma=1):
    x = np.arange(0, n, 1)
    y = np.exp(-(x-mu)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    if direction == 0:
        y[:mu] = 0
    else:
        y[mu+1:] = 0
    return y

# get interploated phase from a smoothed path
def _get_aaed_phase_v0(phase, bin):
    assert phase.shape[0] == bin.shape[0]
    steps = phase.shape[0]
    bins = phase.shape[1]
    res = np.zeros(steps)
    # for i in range(bins):       # this is a lot of extra processing though
    #     phase[:, i] = unwrap(phase[:, i])
    for i in range(steps):
        res[i] = np.interp(bin[i], np.arange(0, bins), phase[i])
    return res

# get interploated phase from a smoothed path
def _get_aaed_phase_v1(phase, bin):
    assert phase.shape[0] == bin.shape[0]
    steps = phase.shape[0]
    bins = phase.shape[1]
    res = np.zeros(steps)
    bin_int = bin.astype(int)
    last = 0
    bins_involved = np.unique(bin_int)
    bins_involved = np.unique((bins_involved, bins_involved+1))
    phase_u = phase.copy()
    phase_sum = np.sum(phase, axis=1)
    corrupted = phase_sum == 0
    corrupted = corrupted | np.roll(corrupted, 1) | np.roll(corrupted, -1)
    for i in bins_involved:       # this is a lot of extra processing though
        phase_u[:, i] = np.unwrap(phase_u[:, i], period=2)
    for i in range(1, steps):
        bi = bin_int[i]
        if corrupted[i]:       # corrupted packet
            for j in [bi, bi+1]:
                d = np.mean(np.diff(phase_u[max(0, last-10):last, j]))
                phase_u[i, j] = phase_u[i-1, j] + d
        else:
            last = i
            
        if bi == bins-1:
            res[i] = phase_u[i, bi]
        else:
            # p1 = np.unwrap(phase[i-1:i+1, bi], period=2)
            # p2 = np.unwrap(phase[i-1:i+1, bi+1], period=2)
            # a1 = p1[1]-p1[0]
            # a2 = p2[1]-p2[0]
            a1 = phase[i, bi] - phase[i-1, bi]
            a2 = phase[i, bi+1] - phase[i-1, bi+1]
            d = np.interp(bin[i]-bi, [0, 1], [a1, a2])
            res[i] = d + res[i-1]
            # Q.append(d)
    import pdb; pdb.set_trace()
    #     if 11500 < i < 11550:
    #         print(i, bi, bin[i], p1, a1, p2, a2, np.array(res[i], res[i-1]), d)

    # plt.plot(res[11000:12000]-res[11500], label='res')
    # plt.plot(np.unwrap(phase[11000:12000, 37*8]-phase[11500, 37*8]), label='37')
    # plt.plot(np.unwrap(phase[11000:12000, 42*8]-phase[11500, 42*8]), label='42') 
    # plt.legend()
    # plt.show()
    # import pdb; pdb.set_trace()
    return res

def get_aaed_phase(phase, bin):
    assert phase.shape[0] == bin.shape[0]
    phase = phase.copy()        # avoid modifying the array
    steps = phase.shape[0]
    bins = phase.shape[1]
    res = np.zeros(steps)
    bin_int = bin.astype(int)
    bins_involved = np.unique(bin_int)
    bins_involved = np.unique((bins_involved, bins_involved+1))

    for i in bins_involved:
        if i >= bins:
            continue
        tmp = phase[:, i]
        tmp = np.unwrap(tmp, period=2)
        tmp = np.concatenate(([0], np.diff(tmp)))
        tmp = median_filter(tmp, size=9)
        phase[:, i] = tmp

    for i in range(0, steps):
        bi = bin_int[i]
        if bi >= bins-1:
            res[i] = phase[i, bins-1]
        else:
            d = np.interp(bin[i]-bi, [0, 1], [phase[i, bi], phase[i, bi+1]])
            res[i] = d
    return res

# use DACM (extended differential and cross-multiply)
def get_aaed_phase_DACM(ffts, bin):
    assert ffts.shape[0] == bin.shape[0]
    # phase = phase.copy()        # avoid modifying the array
    steps = ffts.shape[0]
    bins = ffts.shape[1]
    phase = np.zeros((steps, bins))
    res = np.zeros(steps)
    bin_int = bin.astype(int)

    I0 = ffts.real
    Q0 = ffts.imag
    I1 = np.gradient(ffts.real, axis=0)
    Q1 = np.gradient(ffts.imag, axis=0)
    phase = ((I0*Q1-I1*Q0)/(I0**2+Q0**2))
    phase[np.isnan(phase)] = 0

    # peak = int(np.mean(bin))

    # phase1 = phase.cumsum(axis=0)[:, peak]
    # phase2 = np.unwrap(np.angle(ffts[:, peak]), period=np.pi*2)

    # import pdb
    # pdb.set_trace()

    for i in range(0, steps):
        bi = bin_int[i]
        if bi >= bins-1:
            res[i] = phase[i, bins-1]
        else:
            d = np.interp(bin[i]-bi, [0, 1], [phase[i, bi], phase[i, bi+1]])
            res[i] = d
    return res

# equivalent to numpy.unwrap
def unwrap(signal, period=2):
    S = signal.copy()
    dis = period/2
    for i in range(1, len(S)):
        if S[i] - S[i-1] > dis:
            S[i:] -= period
        elif S[i-1] - S[i] > dis:
            S[i:] += period
    return S 


def highpass(x, order=8, cutoff=0.5):
    sos = signal.butter(order, cutoff, 'highpass', output='sos')
    y = signal.sosfiltfilt(sos, x)
    return y

def lowpass(x, order=8, cutoff=0.5):
    sos = signal.butter(order, cutoff, 'lowpass', output='sos')
    y = signal.sosfiltfilt(sos, x)
    return y

def bandpass(x, order=8, cutoff=(0.2, 0.8)):
    sos = signal.butter(order, cutoff, 'bandpass', output='sos')
    y = signal.sosfiltfilt(sos, x)
    return y

def find_peaks_valleys(x):
    peaks = find_peaks(x)[0]
    valleys = find_peaks(-x)[0]
    res = np.zeros((len(peaks), 3), dtype=int)
    res[:, 0] = peaks
    if peaks[0] < valleys[0]:
        valleys = np.concatenate(([0], valleys))
    if peaks[-1] > valleys[-1]:
        valleys = np.concatenate((valleys, [len(peaks)-1]))
    res[:, 1] = valleys[0:len(peaks)]
    res[:, 2] = valleys[1:len(peaks)+1]
    return res
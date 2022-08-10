"""Some helper functions"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import convolve

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
    """Generate a gaussian distribution using a template and a centroid"""
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

def single_sided_gaussian(n, direction, mu=0, sigma=1):
    """generate a single-sided gaussian distribution"""
    x = np.arange(0, n, 1)
    y = np.exp(-(x-mu)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
    if direction == 0:
        y[:mu] = 0
    else:
        y[mu+1:] = 0
    return y

def unwrap(signal, period=2):
    """equivalent to numpy.unwrap"""
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
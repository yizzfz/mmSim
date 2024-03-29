import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from algo import CFAR_A
from .evaluation import Evaluator
from enum import Enum
import scipy
import scipy.signal
from skimage.feature import peak_local_max
from scipy.ndimage import maximum_filter, gaussian_filter
from radar import RxConfig
from util import log, config_to_id, read_csv, save_one_fig
import multiprocessing
from dataset import IFDataset, DatasetMesh
import warnings
import os
import traceback

AoA = Enum('AoA', 'conventional mvdr music fft')    # supported AoA estimation algorithms
N_METRICS = Evaluator.N_METRICS     # number of evaluation metric

class PointCloudProcessorBase:
    """Contain functions for calculating the range, velocity and angle of the objects based on IF signal.
    Input format (n_rx, chirps_per_frame, samples_per_chirp)."""
    def __init__(self, config: dict, max_d=None, max_v=None, n_aoa_fft=64, range_fft_m=1, doppler_fft_m=1, aoa_fft_flag=None, srpc_a=0, srpc_r=0, output_size=None):
        """
        Parameters:
            config: radar and scene configuration.
            max_d: maximum distance to consider.
            max_v: maximum object velocity to consider.
            n_aoa_fft: number of bins when calculating the angle-of-arrival. More bins can give better resolution. 
            range_fft_m: control the zero padding when applying the range-FFT.
            doppler_fft_m: control the zero padding when applying the Doppler-FFT.
            aoa_fft_flag: 0 for old style angle-FFT for 1443 type antennas, 1 for 2D angle-FFT, None for auto.
            srpc_a: hyperparameter of SRPC algorithm on range domain.
            srpc_r: hyperparameter of SRPC algorithm on angle domain.
            output_size: maximum number of points of the output point cloud.
        """
        self.config = config.copy()
        self.id = config_to_id(config)
        n_samples = self.config['samples_per_chirp']
        chirps_per_frame = self.config['chirps_per_frame']
        self.rx_config = self.config.get('rx_config')
        self.srpc_a = srpc_a
        self.srpc_r = srpc_r
        self.output_size = output_size
        if self.rx_config is None:
            self.rx_config = RxConfig(self.config['layout'])

        self.range_fft_m = range_fft_m
        self.doppler_fft_m = doppler_fft_m
        self.n_range_fft = n_samples * range_fft_m
        self.n_range_fft_cut = int(self.n_range_fft/2)      # cut negative freqs
        self.n_doppler_fft = chirps_per_frame * doppler_fft_m
        self.n_doppler_fft_cut = int(chirps_per_frame/2)    # cut negative freqs
        self.n_doppler_fft_boundary = 0
        self.n_aoa_fft = n_aoa_fft
        self.n_aoa_fft_cut = int(self.n_aoa_fft/2)          # cut negative freqs
        self.aoa_fft_flag = aoa_fft_flag
        if self.aoa_fft_flag is None:
            if len(self.rx_config.elevation_rx) > 1 and self.rx_config.elevation_rx[1].shape[0] > 2:
                self.aoa_fft_flag = 1       # can use a 2D FFT or two 1D FFTs
            else:
                self.aoa_fft_flag = 0       # use the old style FFT for 1443 type antenna

        if max_d:   # cap range-FFT result with a maximum distance
            ADC_rate = self.config['ADC_rate']
            fft_freq = np.fft.fftfreq(self.n_range_fft, d=1.0/ADC_rate)
            self.fft_freq_d = self.range_freq_to_dis(fft_freq)
            self.n_range_fft_cut = np.argmax(self.fft_freq_d > max_d)

        if max_v:   # cap Doppler-FFT result with a maximum velocity
            fft_freq = np.fft.fftfreq(self.n_doppler_fft, d=1)
            self.fft_freq_v = self.doppler_freq_to_vel(fft_freq)
            self.n_doppler_fft_cut = np.argmax(self.fft_freq_v > max_v)
            self.n_doppler_fft_boundary = int((self.n_doppler_fft-self.n_doppler_fft_cut*2)/2)
        # convert axis of range-FFT and Doppler-FFT to real units
        self.dis = self.range_freq_to_dis(np.arange(self.n_range_fft_cut)*(ADC_rate/self.n_range_fft))
        self.vel = self.doppler_freq_to_vel(np.arange(-self.n_doppler_fft_cut, self.n_doppler_fft_cut)*(1/self.n_doppler_fft))

        # prepare FFT parameters
        self.fft_freq_a = np.arange(-1, 1, 2/self.n_aoa_fft)
        self.angle = np.arange(-np.pi/2, np.pi/2, np.pi/self.n_aoa_fft)
        self.angle_phi = np.arcsin(self.fft_freq_a)
        self.range_win = np.hanning(n_samples)
        self.doppler_win = np.hanning(chirps_per_frame)

        # prepare steering vectors
        n_azimuth = self.rx_config.azimuth_rx.shape[0]
        self.azimuth_sv = {
            'phi': self.steering_vector_1d(n_azimuth, self.n_aoa_fft, format='phi'),
            'theta': self.steering_vector_1d(n_azimuth, self.n_aoa_fft, format='theta')
        }
        gs = [8]
        while gs[-1] < n_aoa_fft/4:
            gs.append(gs[-1]*4)
        gs.append(n_aoa_fft)
        # prepare 2d subgrid and fullgrid steering vectors
        self.twod_sv_grid = self.steering_vector_2d_grid(self.rx_config.rx, gs, format='phi')
        self.twod_sv = self.twod_sv_grid[-1]
        # self.twod_sv = self.steering_vector_2d(self.rx_config.rx, self.n_aoa_fft, format='phi')
        # assert np.allclose(self.azimuth_sv['phi'], self.twod_sv[:n_azimuth, self.n_aoa_fft_cut])

    def range_fft(self, X):
        """Return a range-FFT matrix [n_rx, n_chirps, n_range_fft]
        """
        range_fft = np.fft.fft(X*self.range_win, self.n_range_fft)
        return range_fft

    def doppler_fft(self, X):
        """Doppler FFT.

        Parameters:
            X: [n_rx, n_chirps, ...] 3D array.

        Return:
            [n_rx, n_doppler, ...] 3D array.
        """
        doppler_fft = np.fft.fftshift(np.fft.fft(np.transpose(X, (0, 2, 1))*self.doppler_win, self.n_doppler_fft), axes=2)
        if self.n_doppler_fft_boundary > 0:
            doppler_fft = doppler_fft[:, :, self.n_doppler_fft_boundary:-self.n_doppler_fft_boundary]
        doppler_fft = np.transpose(doppler_fft, (0, 2, 1))
        return doppler_fft

    def cfar(self, X, th=None, win=8, guard=2, mode='OS', min_db=None, min_rel=None, debug=False):
        return CFAR_A(X, th=th, win=win, guard=guard, mode=mode, min_db=min_db, min_rel=min_rel, debug=debug)

    def cfar2d(self, X, th=None, win=8, guard=2, lim=None, debug=False, mode='OS', min_db=None, min_rel=None):
        """2D CFAR detection. 

        Parameters:
            data: a 2D array, e.g. [n_doppler, n_range]
            win: number of neighbour cells to calculate the noise power (each side), including the length of guard cells.
            guard: number of guard cells (each side).
            lim: (optional) if the second dimension is range, then only consider within < lim. 
            debug: draw X and peaks.
            mode: OS (order statistic), CA (cell averaging), GO (greatest of), SO (smallest of).
            min_db: if given and X is in log scale, a peak will only be reported if it is higher than (X.max + min_db).
            min_rel: if given, a peak will only be reported if it is higher than (X.max * min_rel)

        Return:
            [n, 3] array, contains n detected peaks and their indices in the input array, and power.
        """
        if not lim:
            lim = X.shape[1]
        else:
            lim = np.min(X.shape[1], lim)

        bitmask = np.zeros(X.shape[0], dtype=bool)
        res = []
        for k in range(lim):
            cfar1 = self.cfar(X[:, k], th=th, win=win, guard=guard, mode=mode, min_db=min_db, min_rel=min_rel)   # cfar in doppler domain for each range bin
            bitmask[cfar1] = 1

        detection_list = np.where(bitmask == 1)[0]
        for k in detection_list:
            cfar2 = self.cfar(X[k], th=th, win=win, guard=guard, mode=mode, min_db=min_db, min_rel=min_rel)  # cfar in range domain for each doppler bin
            for i in cfar2:
                res.append((k, i, X[k, i]))    # save result with (doppler idx, range idx, power)
        res = np.asarray(res)
        if debug:
            Y = np.zeros(X.shape)
            for i, j, _ in res.astype(int):
                Y[i, j] = 1
            _, axs = plt.subplots(1, 2, figsize=(12, 5))
            axs[0].pcolormesh(X)
            axs[1].pcolormesh(Y)
            plt.show()
        return res

    def find_peaks(self, X, th=0.25, n_peaks=1):
        """Find peaks in a signal `X`. Applies SRPC.
        
        Parameters:
            X: 1D signal.
            th: threshold in relative to the peak.
            n_peaks: number of peaks to return.
        """
        th = np.max(X)*th
        peaks, vals = scipy.signal.find_peaks(X, height=th)
        heights = vals['peak_heights']
        order = np.argsort(heights)[::-1]
        peaks_sorted = peaks[order]   # sort to height descending
        heights_sroted = heights[order]
        if n_peaks is not None:
            peaks_sorted = peaks_sorted[:n_peaks]
            heights_sroted = heights_sroted[:n_peaks]
        
        if self.srpc_a == 0:
            return peaks_sorted
        
        n_peaks_s = int(n_peaks*self.srpc_a)
        lo = np.max(X)*0.95
        # if len(peaks_sorted) == 1:
        #     lo = np.max(X)*0.95
        # else:
        #     lo = heights_sroted[-1]

        idx = np.where(X>lo)[0]
        peaks_s = np.random.permutation(idx)[:n_peaks_s]
        peaks_s = np.unique(np.concatenate((peaks_sorted, peaks_s)))
        return peaks_s

    def find_peaks_2d(self, X, th=0.25, n_peaks=1, final=True):
        """Find peaks in a 2D signal `X`. Applies SRPC.
        
        Parameters:
            X: 2D signal.
            th: threshold in relative to the peak.
            n_peaks: number of peaks to return.
            final: False to disable SRPC or when using in the middle of subgrid search. 
        """
        peaks = peak_local_max(X, threshold_rel=th, num_peaks=n_peaks)
        if len(peaks) == 0:
            return peaks

        if not final or self.srpc_a == 0:
            return peaks
        # if len(peaks) == 1:
        #     lo = X[peaks[-1, 0], peaks[-1, 1]] * 0.9
        # else:
        #     lo = X[peaks[-1, 0], peaks[-1, 1]]
        lo = np.max(X)*0.95
        n_peaks_s = int(n_peaks*self.srpc_a)
        candidate = np.where(X>lo)
        n_idx = candidate[0].shape[0]
        if n_idx == 0:
            return peaks
        idx = np.random.permutation(n_idx)[:n_peaks_s]
        # idx = np.random.choice(n_idx, n_peaks_s, replace=False)
        peaks_s = np.stack((candidate[0][idx], candidate[1][idx]), axis=1)
        # peaks_s = np.asarray([(candidate[0][i], candidate[1][i]) for i in idx])
        peaks_s = np.concatenate((peaks_s, peaks))
        peaks_s = np.unique(peaks_s, axis=0)
        return peaks_s

    def threshold(self, X, th=0.5, db=None, debug=False):
        """Filter a 2D array with a threshold"""
        if db:
            th = X.max() - db
        else:
            th = X.max() * th
        rows, cols = np.where(X > th)
        res = np.stack((rows, cols, X[rows, cols]), axis=1)
        res = np.asarray(res)
        if debug:
            Y = np.zeros(X.shape)
            for i, j, _ in res:
                Y[i, j] = 1
            _, axs = plt.subplots(1, 2, figsize=(12, 5))
            axs[0].pcolormesh(X)
            axs[1].pcolormesh(Y)
            plt.show()
        return res

    def aoa_fft_1d(self, all_rx, detection, rd_spectrum=None):
        """Uses an azimuth FFT first, then an elevation FFT or another azimuth FFT on the second pair of antennas. Suitable for 1843 type antennas."""
        res = []
        azimuth = all_rx[self.rx_config.azimuth_rx]
        elevation = all_rx[self.rx_config.elevation_rx[self.aoa_fft_flag]]
        # azimuth fft
        azimuth_fft = np.fft.fft(azimuth, self.n_aoa_fft)
        azimuth_fft = np.fft.fftshift(azimuth_fft)
        azimuth_fft_mag = np.abs(azimuth_fft)
        # find azimuth angles
        p1s = self.find_peaks(azimuth_fft_mag, th=0.5)
        v = self.vel[detection[0]]
        d = self.dis[detection[1]]

        # elevation FFT
        elevation_fft = np.fft.fft(elevation, self.n_aoa_fft)
        elevation_fft = np.fft.fftshift(elevation_fft)
        phase_offset = self.rx_config.phase_offset
        # find elevation angles for each azimuth angle
        for p1 in p1s:
            wx = self.fft_freq_a[p1]
            if self.aoa_fft_flag == 0:          # use phase for elevation
                wz = np.angle(azimuth_fft[p1]*(elevation_fft[p1].conj())*np.exp(1j*phase_offset*wx*np.pi))/np.pi
                x, y, z = self.xyz_estimate(d, wx, wz)
                res.append((x, y, z, v))
            else:                               # use another fft
                elevation_fft_mag = np.flip(np.abs(elevation_fft))      # flip so that object from top has a positive phase
                p2s = self.find_peaks(elevation_fft_mag, th=0.5)
                for p2 in p2s:
                    wz = self.fft_freq_a[p2]
                    if self.srpc_r:
                        pts = self.xyz_estimate_srpc(detection, wx, wz, rd_spectrum)
                        pts = np.concatenate((pts, np.ones((pts.shape[0], 1))*v), axis=1)
                        res.append(pts)
                    else:
                        x, y, z = self.xyz_estimate(d, wx, wz)
                        res.append((x, y, z, v))
        res = np.concatenate(res).reshape((-1, 4))
        return res

    def aoa_fft_2d(self, all_rx, detection, rd_spectrum=None):
        """Uses an 2D angle-FFT to calculate AoA"""
        res = []
        v = self.vel[detection[0]]
        d = self.dis[detection[1]]

        all_rx_2d = self.rx_config.prepare_2d_fft_data(all_rx)
        # 2D angle FFT
        afft = np.fft.fft2(all_rx_2d, s=(self.n_aoa_fft, self.n_aoa_fft))
        afft = np.fft.fftshift(afft)
        afft = np.abs(afft)
        # find peaks
        peaks = self.find_peaks_2d(afft, th=0.75, n_peaks=2)
        for wz, wx in peaks:
            if self.srpc_r:
                pts = self.xyz_estimate_srpc(detection, self.fft_freq_a[wx], self.fft_freq_a[wz], rd_spectrum)
                pts = np.concatenate((pts, np.ones((pts.shape[0], 1))*v), axis=1)
                res.append(pts)
            else:     
                x, y, z = self.xyz_estimate(d, self.fft_freq_a[wx], self.fft_freq_a[wz])
                res.append((x, y, z, v))
        res = np.concatenate(res).reshape((-1, 4))
        return res

    def aoa_fft_per_point(self, X, detection_list, return_velocity=False, npass=None, rd_spectrum=None):
        """Performs AoA FFT given FFT data matrix and CFAR detection list, return a point cloud

        Parameters:
            X: [n_range_fft, n_doppler_fft/n_chirps] array, first dimension is range, second dimension is Doppler.
            detection_list: [n, 3] detection list returned from the 2D CFAR detection.

        Return:
            [n, 4] array that has the object's x-y-z coordinates and velocity.
        """
        n_objs = detection_list.shape[0]
        res = []
        if npass is None:
            npass = 1 if self.aoa_fft_flag == 1 else 2
        if npass == 1 and self.aoa_fft_flag == 0:
            warnings.warn("Warning: calling 2D FFT but receiver row <= 2, forcing to be 1D FFT")
            npass = 2

        if npass == 1:
            func = self.aoa_fft_2d      # 2d FFT
        else:
            func = self.aoa_fft_1d      # old style FFT

        for i in range(n_objs):
            all_rx = X[:, detection_list[i, 0], detection_list[i, 1]]
            res.append(func(all_rx, detection_list[i], rd_spectrum))
        res = np.concatenate(res)
        res = res[~np.isnan(res).any(axis=1)]
        if return_velocity:
            return res
        return res[:, :3]

    def aoa_fft_azimuth(self, X, per_range=False):
        """Performs azimuth AoA FFT given range FFT data matrix, 
        return an angle power spectrum, summed over all chirps.

        Parameters:
            X: [n_rx, n_chirp, n_range_fft] 3D array.
            per_range: default False, calculate angle power spectrum per range.

        Return:
            [n_angles] 1D array or [n_range, n_angles] if per_range is True.
        """
        azimuth = X[self.rx_config.azimuth_rx, :, :self.n_range_fft_cut]
        azimuth = np.transpose(azimuth, (2, 1, 0))
        azimuth_fft = np.fft.fft(azimuth, self.n_aoa_fft)
        azimuth_fft = np.fft.fftshift(azimuth_fft, axes=2)
        azimuth_fft_mag = np.abs(azimuth_fft)

        if per_range:
            res = np.sum(azimuth_fft_mag, axis=1)
        else:
            res = np.sum(azimuth_fft_mag, axis=(0, 1))
        return res

    def bf_raw(self, cov, steering_vec, method, n_sample=1, cov_inv=None, nullspace=None):
        """Beamforming algorithms for AoA estimation, for developing purpose."""
        res = np.zeros((self.n_aoa_fft))
        if cov_inv is None:
            cov_inv = self.mat_inv(cov)
            if cov_inv is None:
                return res

        if method == AoA.music and nullspace is None:
            nullspace = self.nullspace(cov, n_sample)
            if nullspace is None:
                return res

        # for speed optimization later we can use
        # 1/np.abs(np.array(sv.H*cov_inv).T* np.array(sv)).sum(axis=0)
        for a in range(self.n_aoa_fft):
            s = steering_vec[:, a]
            if method == AoA.conventional:
                res[a] = np.abs(s.H * cov * s)
            elif method == AoA.mvdr:
                res[a] = np.abs(1/(s.H * cov_inv * s))
            elif method == AoA.music:
                res[a] = np.abs(1/(s.H * nullspace * s))
        # equivalent fft form:
        # np.mean(np.square(np.abs(np.fft.fftshift(np.fft.fft(azimuth[:, :, d].T, self.n_aoa_fft), axes=1))), axis=0)
        return res

    def bf(self, cov, steering_vec, method, n_sample=1, cov_inv=None, nullspace=None):
        """Beamforming algorithms for AoA estimation, optimized for speed"""
        res = np.zeros((self.n_aoa_fft))
        if cov_inv is None:
            cov_inv = self.mat_inv(cov)
            if cov_inv is None:
                return res

        if method == AoA.music and nullspace is None:
            nullspace = self.nullspace(cov, n_sample)
            if nullspace is None:
                return res

        # xy = 8x8, a = 512
        if method == AoA.conventional:
            res = np.abs(
                np.einsum('ax,xy,ya->a', np.conj(steering_vec.T), cov, steering_vec))
        elif method == AoA.mvdr:
            res = np.abs(1/np.einsum('ax,xy,ya->a',
                         np.conj(steering_vec.T), cov_inv, steering_vec))
        elif method == AoA.music:
            res = np.abs(1/np.einsum('ax,xy,ya->a',
                         np.conj(steering_vec.T), nullspace, steering_vec))
        return res

    def bf_2d(self, cov, steering_vec, method, n_sample=1, cov_inv=None, nullspace=None):
        """Beamforming algorithms for AoA estimation, optimized for speed"""
        if cov_inv is None:
            cov_inv = self.mat_inv(cov)
            if cov_inv is None:
                return

        sv = steering_vec
        svc = np.conj(steering_vec)
        if method == AoA.music:
            if nullspace is None:
                nullspace = self.nullspace(cov, n_sample)
                if nullspace is None:
                    return
        if method == AoA.conventional:
            res = np.abs(np.einsum('xae,xy,yae->ae', svc, cov, sv))
        elif method == AoA.mvdr:
            res = np.abs(1/np.einsum('xae,xy,yae->ae', svc, cov_inv, sv))
        elif method == AoA.music:
            res = np.abs(1/np.einsum('xae,xy,yae->ae', svc, nullspace, sv))
        return res

    def bf_2d_grid(self, cov, method, n_sample=1):
        """2D BF using sparser grids then denser grids"""
        sv_grid = self.twod_sv_grid

        eigenvalues, _ = np.linalg.eigh(cov)         # auto sorted in ascending order
        eigenvalues = np.flip(eigenvalues)                      # reverse to descending order
        n_obj = self.estimate_n_source(eigenvalues, n_sample)
        if n_obj == 0:
            return
        cov_inv = self.mat_inv(cov)
        if cov_inv is None:
            return
        nullspace = self.nullspace(cov, n_sample)

        # perform BF at each grid
        peaks = []
        for i, sv in enumerate(sv_grid):
            if i == 0:  # 1st grid
                bf = self.bf_2d(cov, sv, method, 1, cov_inv, nullspace)
                last_gsize = sv.shape[1]
            else:       # define denser grid around peaks
                bf = np.zeros(sv.shape[1:])
                cur_gsize = sv.shape[1]
                ratio = cur_gsize/last_gsize
                rng = int(cur_gsize/last_gsize)+1
                for y, x in peaks:
                    y = int((y+0.5)*ratio)
                    x = int((x+0.5)*ratio)
                    top = max(0, y-rng)
                    bot = min(cur_gsize, y+rng)
                    left = max(0, x-rng)
                    right = min(cur_gsize, x+rng)
                    bf_new = self.bf_2d(cov, sv[:, top:bot, left:right], method, 1, cov_inv, nullspace)
                    bf[top:bot, left:right] = np.maximum(bf[top:bot, left:right], bf_new)
                last_gsize = cur_gsize
            if i == len(sv_grid) - 1:   # final grid
                break
            peaks = self.find_peaks_2d(bf, th=0.25, n_peaks=n_obj, final=False)
            if len(peaks) == 0:
                return
        return bf

    def bf_per_range_azimuth(self, X, method=AoA.conventional, format='phi', norm=True, return_list=False):
        """Performs azimuth beamforming on each range bin, return a range-Azimuth heatmap.

        Parameters:
            X:[n_rx, n_chirp, n_range_fft] 3D array.
            method: conventional, mvdr, or music.

        Returns:
            [n_range_fft, n_angles] 2D array, the range-Azimuth heatmap.
        """
        azimuth = X[self.rx_config.azimuth_rx]
        cov = self.covariance_matrix_per_range(azimuth, dl=True)
        sv = self.azimuth_sv[format]
        res = np.zeros((self.n_range_fft_cut, self.n_aoa_fft))
        if return_list:
            detection_list = []
        for d in range(self.n_range_fft_cut):
            res[d] = self.bf(cov[d], sv, method, azimuth.shape[1])
            if norm:
                res[d] = res[d] * d**4      # to compensate power loss due to further distances
                if method == AoA.music:
                    res[d] = np.log10(res[d]+1)     # music power spectrum has some extremely high values that has to be cut down to log scale
            if return_list:
                eigenvalues, _ = np.linalg.eigh(cov[d])         # auto sorted in ascending order
                eigenvalues = np.flip(eigenvalues)                      # reverse to descending order
                n_obj = self.estimate_n_source(eigenvalues, X.shape[1])
                if n_obj == 0:
                    continue
                peaks = self.find_peaks(res[d])[:n_obj]
                for p in peaks:
                    detection_list.append((d, p))
        if return_list:
            return np.array(detection_list)
        return res

    def bf_per_range_2d(self, X, method=AoA.conventional):
        """Performs 2d beamforming on each range bin, return a point cloud.

        Parameters:
            X:[n_rx, n_chirp, n_range_fft] 3D array.
            method: conventional, mvdr, or music.

        Returns:
            [n, 3] detected point cloud.
        """
        # sv = self.twod_sv
        res = []
        tmp = []
        for i in range(X.shape[2]):
            # print(f'Progress {i/X.shape[2]*100:.1f}%', end='\r')
            all_rx = X[:, :, i]           # (12, 50)
            cov = self.covariance_matrix(all_rx)
            eigenvalues, _ = np.linalg.eigh(cov)         # auto sorted in ascending order
            eigenvalues = np.flip(eigenvalues)                      # reverse to descending order
            n_obj = self.estimate_n_source(eigenvalues, X.shape[1])
            if n_obj == 0:
                continue
            # tmp.append(n_obj)

            # bf = self.bf_2d(cov, sv, method)
            bf = self.bf_2d_grid(cov, method, n_sample=all_rx.shape[1])
            if bf is None:
                continue

            peaks = self.find_peaks_2d(bf, th=0.25, n_peaks=n_obj)
            d = self.dis[i]
            for wz, wx in peaks:
                x, y, z = self.xyz_estimate(d, self.fft_freq_a[wx], self.fft_freq_a[wz])
                res.append((x, y, z))
        res = np.asarray(res).reshape((-1, 3))
        res = res[~np.isnan(res).any(axis=1)]       # remove points with nan coordinates
        # print(f'Beamforming finished with {res.shape[0]} points.')
        # print(tmp)
        return res

    def bf_per_point(self, X, detection_list, method=AoA.conventional, return_velocity=False, rd_spectrum=None):
        """Performs beamforming given a dection list, return their x-y-z coordinates.
        Format has to be phi.

        Parameters:
            X: [n_rx, n_doppler, n_range] 3D array.
            detection_list: [n, 2] detection list returned from the 2D CFAR detection.
            method: conventional, mvdr, or music.
            return_velocity: boolean, return (x,y,z,v) if True, (x,y,z) otherwise.

        Returns:
            [n, 3] or [n, 4] array that has the object's x-y-z coordinates (and velocity).
        """
        n_objs = detection_list.shape[0]
        # azimuth_sv = self.azimuth_sv['phi']
        azimuth_sv = self.twod_sv[self.rx_config.azimuth_rx, self.n_aoa_fft_cut]
        res = []
        for i in range(n_objs):
            all_rx = X[:, detection_list[i, 0], detection_list[i, 1]]           # (12)
            azimuth = all_rx[self.rx_config.azimuth_rx]                         # (8)

            cov_azimuth = self.covariance_matrix(azimuth)
            cov_elevation = self.covariance_matrix(all_rx)

            # azimuth
            # assert np.allclose(azimuth_bf, self.bf_raw(cov_azimuth, azimuth_sv, method))
            # azimuth_bf = self.bf(cov, azimuth_sv, method, cov_inv=cov_inv, nullspace=nullspace)
            eigenvalues, _ = np.linalg.eigh(cov_elevation)         # auto sorted in ascending order
            eigenvalues = np.flip(eigenvalues)                      # reverse to descending order
            n_obj = self.estimate_n_source(eigenvalues, X.shape[1])
            if n_obj == 0:
                continue

            azimuth_bf = self.bf(cov_azimuth, azimuth_sv, method, n_sample=self.n_doppler_fft)
            wxs = self.find_peaks(azimuth_bf, n_peaks=n_obj)
            # wxs = self.fft_freq_a[wxs]
            if len(wxs) == 0:
                continue

            v = self.vel[detection_list[i, 0]]
            d = self.dis[detection_list[i, 1]]

            # elevation
            # cov_all = self.covariance_matrix(all_rx, dl=True)
            for wx in wxs:
                # sve = self.steering_vector_1d_elevation(self.rx_config.rx, wx, n_angles)
                elevation_sv = self.twod_sv[:, :, wx]
                elevation_bf = self.bf(cov_elevation, elevation_sv, method, n_sample=self.n_doppler_fft)
                # assert np.allclose(elevation_bf, self.bf_raw(cov_elevation, elevation_sv, method))
                wzs = self.find_peaks(elevation_bf, n_peaks=n_obj)
                # wzs = self.fft_freq_a[peaks]
                for wz in wzs:
                    if self.srpc_r:
                        pts = self.xyz_estimate_srpc(detection_list[i], self.fft_freq_a[wx], self.fft_freq_a[wz], rd_spectrum)
                        pts = np.concatenate((pts, np.ones((pts.shape[0], 1))*v), axis=1)
                        res.append(pts)
                    else:     
                        x, y, z = self.xyz_estimate(d, self.fft_freq_a[wx], self.fft_freq_a[wz])
                        res.append((x, y, z, v))
        res = np.concatenate(res).reshape((-1, 4))
        res = res[~np.isnan(res).any(axis=1)]       # remove points with nan coordinates
        # print(f'Beamforming finished with {res.shape[0]} points.')
        if return_velocity:
            return res
        return res[:, :3]

    def bf_per_point_2d(self, X, detection_list, method=AoA.conventional, return_velocity=False, rd_spectrum=None):
        """Performs 2D beamforming given a dection list, return their x-y-z coordinates.
        Format has to be phi.

        Parameters:
            X: [n_rx, n_doppler, n_range] 3D array, the last dimension is chirps or Doppler-fft.
            detection_list: [n, 2] detection list returned from the 2D CFAR detection.
            method: conventional, mvdr, or music.
            return_velocity: boolean, return (x,y,z,v) if True, (x,y,z) otherwise.

        Returns:
            [n, 3] or [n, 4] array that has the object's x-y-z coordinates (and optionally velocity).
        """
        n_objs = detection_list.shape[0]
        res = []
        for i in range(n_objs):
            # print(f'Progress {i/n_objs*100:.1f}%', end='\r')
            all_rx = X[:, detection_list[i, 0], detection_list[i, 1]]           # (12)

            cov = self.covariance_matrix(all_rx)
            eigenvalues, _ = np.linalg.eigh(cov)         # auto sorted in ascending order
            eigenvalues = np.flip(eigenvalues)                      # reverse to descending order
            n_obj = self.estimate_n_source(eigenvalues, X.shape[1])
            if n_obj == 0:
                continue
            bf = self.bf_2d_grid(cov, method)
            # bf = self.bf_2d(cov, self.twod_sv, method)
            if bf is None:
                continue

            peaks = self.find_peaks_2d(bf, th=0.25, n_peaks=n_obj)
            v = self.vel[detection_list[i, 0]]
            d = self.dis[detection_list[i, 1]]
            for wz, wx in peaks:
                if self.srpc_r:
                    pts = self.xyz_estimate_srpc(detection_list[i], self.fft_freq_a[wx], self.fft_freq_a[wz], rd_spectrum)
                    pts = np.concatenate((pts, np.ones((pts.shape[0], 1))*v), axis=1)
                    res.append(pts)
                else:     
                    x, y, z = self.xyz_estimate(d, self.fft_freq_a[wx], self.fft_freq_a[wz])
                    res.append((x, y, z, v))
        res = np.concatenate(res).reshape((-1, 4))
        res = res[~np.isnan(res).any(axis=1)]       # remove points with nan coordinates
        # print(f'Beamforming finished with {res.shape[0]} points.')
        if return_velocity:
            return res
        return res[:, :3]

    def bf_per_point_elevation(self, X, detection_list, method=AoA.conventional):
        """Performs elevation beamforming given a dection list in range-azimuth domain, 
        return their x-y-z coordinates. Used for non-Doppler DPC.
        Format has to be phi.

        Parameters:
            X: [n_rx, n_chirp, n_range_fft] 3D array.
            detection_list: [n, 2] detection list, (range, wx), returned from the 2D CFAR detection.
            method: conventional, mvdr, or music.

        Returns:
            [n, 3] array that has the object's x-y-z coordinates.
        """
        n_objs = detection_list.shape[0]
        sv = self.twod_sv
        res = []
        # tmp = []
        for i in range(n_objs):
            all_rx = X[:, :, detection_list[i, 0]]           # (12, 50)
            wx = detection_list[i, 1]
            cov = self.covariance_matrix(all_rx)

            sv = self.twod_sv[:, :, wx]
            bf = self.bf(cov, sv, method, n_sample=X.shape[1])
            if bf is None:
                continue

            eigenvalues, _ = np.linalg.eigh(cov)         # auto sorted in ascending order
            eigenvalues = np.flip(eigenvalues)                      # reverse to descending order
            n_obj = self.estimate_n_source(eigenvalues, X.shape[1])
            # tmp.append(n_obj)
            if n_obj == 0:
                continue

            peaks = self.find_peaks(bf, n_peaks=n_obj)
            d = self.dis[detection_list[i, 0]]
            for wz in peaks:
                x, y, z = self.xyz_estimate(d, self.fft_freq_a[wx], self.fft_freq_a[wz])
                res.append((x, y, z))
        res = np.asarray(res).reshape((-1, 3))
        res = res[~np.isnan(res).any(axis=1)]       # remove points with nan coordinates
        # print(f'Beamforming finished with {res.shape[0]} points.')
        # print(tmp)
        return res

    def resolution_versus_rx(self, data, wx=None):
        """Compare the output resolution when using different numbers of receivers"""
        r1 = np.arange(0, 8)
        r2 = np.arange(0, 16)
        r3 = np.arange(0, 24)
        r4 = np.arange(0, 32)
        cnt = 0
        for r in [r1, r2, r3, r4]:
            x = data[r]
            c = self.covariance_matrix(x)
            if wx is not None:
                s = self.twod_sv[r, :, wx]
            else:
                s = self.twod_sv[r, self.n_aoa_fft_cut]
            r = self.bf(c, s, AoA.mvdr)
            plt.plot(r/r.max(), label=str(cnt))
            cnt += 1
        plt.legend()
        plt.show()

    def bf_azimuth_per_frame(self, X, method=AoA.conventional, format='phi'):
        """Performs azimuth beamforming on the entire frame, return an angle power spectrum heatmap.
        For algorithm developing purpose only.

        Parameters:
            X: [n_rx, n_chirp, n_range_fft] 3D array
            method: conventional, mvdr, or music

        Returns:
            [n_angles] 1D array, angle power spectrum
        """
        azimuth = X[self.rx_config.azimuth_rx]
        cov = self.covariance_matrix_per_frame(azimuth, dl=True)
        sv = self.azimuth_sv[format]
        res = self.bf(cov, sv, method, azimuth.shape[-1])
        return res

    def covariance_matrix_per_range(self, X, dl=True):
        """Calculate the covariance matrix per range bin.

        Parameters:
            X: [n_rx, n_chirp, n_range] 3D array, output from the range-FFT.
            dl: Perform diagonal loading or not. Required for MVDR.

        Returns:
            [n_range, n_rx, n_rx], one covariance matrix for each range bin.
        """
        n_rx, n_chirp, _ = X.shape
        cov = np.zeros((self.n_range_fft_cut, n_rx, n_rx), dtype=complex)
        for d in range(self.n_range_fft_cut):
            cov[d] = self.covariance_matrix(X[:, :, d], dl=dl)
        return cov

    def covariance_matrix_per_frame(self, X, dl=True):
        """Calculate only one covariance matrix of the input. For algorithm developing purpose only. 

        Parameters:
            data: [n_rx, n_chirp, n_range] 3D array, output from the range-FFT.
            dl: Perform diagonal loading or not. Required for MVDR.

        Returns:
            [n_rx, n_rx], one covariance matrix accounting all range bins.
        """
        n_rx, n_chirp, _ = X.shape
        cov = np.zeros((n_rx, n_rx), dtype=complex)
        for i in range(n_chirp):
            x = X[:, i, :]
            cov = cov + self.covariance_matrix(x, dl=dl)       # np.matmul(x, np.conj(x.T))/(n_samples-1)
        cov = cov / n_chirp
        self.mat_inv(cov)
        if dl:
            cov = self.diagonal_loading(cov)
        return cov

    def covariance_matrix(self, X, dl=True, fba=True):
        """Calculate only one covariance matrix of a generic input. 

        Parameters:
            data: [n_rx, ...] any input, will be reshaped to [n_rx, -1].
            dl: Perform diagonal loading or not. Required for MVDR.
            fba: Apply Forward-Backward Averaging.

        Returns:
            [n_rx, n_rx] nomalized covariance matrix.
        """
        X = X.reshape((X.shape[0], -1))
        if X.shape[1] > 1:
            X -= X.mean(axis=1)[:, None]
        cov = X @ np.conj(X.T)
        if fba:
            Ex = np.fliplr(np.eye(X.shape[0]))
            cov_bw = Ex @ np.conj(cov) @ Ex
            cov = (cov + cov_bw)/2
        cov /= max(1, X.shape[1]-1)
        if dl:
            cov = self.diagonal_loading(cov)
        return cov

    def estimate_n_source(self, eigenvalues, s, debug=False):
        """Minimum description length: estimate the number of data sources given the eigenvalues of the data covariance matrix. 
        
        Parameters:
            eigenvalues: a 1D vector of eigenvalues in ascending order.
            s: number of samples used for calculating the data covariance matrix.

        Return:
            A integer, the estimated number of data sources.
        """
        n = eigenvalues.shape[0]
        s = min(s, 2)
        eigenvalues = eigenvalues/eigenvalues.min()
        aic = np.zeros((n-1))
        mdl = np.zeros((n-1))
        for k in range(n-1):
            t1 = np.product(eigenvalues[k:])**(1/(n-k))
            t2 = np.sum(eigenvalues[k:])/(n-k)
            L = -s*(n-k)*np.log(t1/t2)
            P = k*(2*n-k)
            aic[k] = L + P
            mdl[k] = L + 0.5*P*np.log(s)
        if debug:
            plt.plot(aic, label='aic')
            plt.plot(mdl, label='mdl')
            plt.legend()
            plt.show()
        return np.argmin(mdl)

    def mat_inv(self, X):
        """Get inverse matrix"""
        try:
            return np.linalg.inv(X)
        except np.linalg.LinAlgError as err:
            # warnings.warn('[Warning] Singular covariance matrix detected. ')
            return None

    def nullspace(self, cov, n_sample):
        """Get null space for MUSIC algorithm"""
        eigenvalues, eigenvectors = np.linalg.eigh(cov)         # auto sorted in ascending order
        eigenvalues = np.flip(eigenvalues)                      # reverse to descending order
        eigenvectors = np.flip(eigenvectors, axis=1)
        n_obj = self.estimate_n_source(eigenvalues, n_sample)
        if n_obj == 0:
            # warnings.warn('[Warning] zero data source detected')
            return None
        nullvectors = eigenvectors[:, n_obj:]
        nullspace = nullvectors @ np.conj(nullvectors.T)
        return nullspace

    def diagonal_loading(self, cov):
        """Diagonal loading algorithm on a covariance matrix.
        """
        n = cov.shape[0]
        tr = np.trace(cov)/n
        cov = cov + np.identity(n)*tr*0.05
        return cov

    def steering_vector_1d(self, n_rx: int, n_angle: int, theta_range=[-np.pi/2, np.pi/2], format='phi'):
        """Generate a 1D steering vector for a ULA.

        Parameters:
            n_rx: number of rx.
            n_angle: resolution = theta_range/n_angle
            theta_range: default -90 to 90
            format: 'theta' or 'phi'. phi(a) = sin(a)

        Returns:
            [n_rx, n_angle] matrix
        """
        # print(f'Generating {n_angle} steering vectors ({format}).')
        tmin, tmax = theta_range
        if format == 'theta':
            tres = (tmax-tmin)/n_angle
            theta = np.sin(np.arange(tmin, tmax, tres))
        elif format == 'phi':
            tmin, tmax = np.sin(tmin), np.sin(tmax)
            tres = (tmax-tmin)/n_angle
            theta = np.arange(tmin, tmax, tres)
        rx = np.arange(n_rx)
        vec = np.exp(1j*np.outer(rx, np.pi*theta))
        return vec

    def steering_vector_1d_elevation(self, rx, wx, n_angle: int, theta_range=[-np.pi/2, np.pi/2]):
        """Generate a 1D elevation steering vector for a Rx array, given wx. Format has to be phi.

        Parameters:
            n_rx: [n_rx, 3] rx layout.
            n_angle: resolution = theta_range/n_angle
            theta_range: default -90 to 90

        Returns:
            [n_rx, n_angle] matrix
        """
        # print(f'Generating {n_angle} elevation steering vectors.')
        tmin, tmax = theta_range
        tmin, tmax = np.sin(tmin), np.sin(tmax)
        tres = (tmax-tmin)/n_angle
        theta = np.arange(tmin, tmax, tres)
        xs = -rx[:, 0]
        zs = -rx[:, 2]
        v1 = np.exp(1j*np.outer(xs, np.pi*wx))
        v2 = np.exp(1j*np.outer(zs, np.pi*theta))
        res = v1*v2
        return res

    def steering_vector_2d(self, rx, n_angle: int, theta_range=[-np.pi/2, np.pi/2], format='phi'):
        """Generate a 2D steering vector for a 2D ULA.

        Parameters:
            n_rx: number of rx.
            n_angle: resolution = theta_range/n_angle
            theta_range: default -90 to 90
            format: 'theta' or 'phi'. phi(a) = cos(e)sin(a), phi(e) = sin(e)

        Returns:
            [n_rx, n_elevation, n_azimuth] matrix.
            res[:, i, j] = self.steering_vector_1d_elevation(.., angle[j], .., 'phi')[:, i]  
        """
        tmin, tmax = theta_range
        if format == 'theta':
            tres = (tmax-tmin)/n_angle
            angles = np.arange(tmin, tmax, tres)
            X, Y = np.meshgrid(angles, angles)
            azimuth = np.cos(Y)*np.sin(X)
            elevation = np.sin(Y)                                               # 2 x (n_angle, n_angle)
        elif format == 'phi':
            tmin, tmax = np.sin(tmin), np.sin(tmax)
            tres = (tmax-tmin)/n_angle
            angles = np.arange(tmin, tmax, tres)
            azimuth, elevation = np.meshgrid(angles, angles)                # 2 x (n_angle, n_angle)
        xs = -rx[:, 0]
        zs = -rx[:, 2]

        # azimuth_vec[:, any, i] = self.steering_vector_1d(..)[:, i]
        azimuth_vec = np.exp(1j*np.einsum('i,jk->ijk', xs, np.pi*azimuth))      # (n_rx, n_angle, n_angle)
        elevation_vec = np.exp(1j*np.einsum('i,jk->ijk', zs, np.pi*elevation))  # (n_rx, n_angle, n_angle)

        # res[:, i, j] = self.steering_vector_1d_elevation(.., angle[j])[:, i]
        # assert np.allclose(res[:, ii, jj], self.steering_vector_1d_elevation(rx, angles[jj], n_aoa_fft)[:, ii])
        res = azimuth_vec * elevation_vec
        return res

    def steering_vector_2d_grid(self, rx, n_angles: list[int], theta_range=[-np.pi/2, np.pi/2], format='phi'):
        """Generate a 2D steering vector grid for a 2D ULA.

        Parameters:
            n_rx: number of rx.
            n_angles: number of angles of each gird, resolution = theta_range/n_angle
            theta_range: default -90 to 90
            format: 'theta' or 'phi'. phi(a) = cos(e)sin(a), phi(e) = sin(e)

        Returns:
            [n_level, n_rx, n_elevation, n_azimuth] matrix.
            res[:, i, j] = self.steering_vector_1d_elevation(.., angle[j], .., 'phi')[:, i]
        """
        res = []
        for n_angle in n_angles:
            sv = self.steering_vector_2d(rx, n_angle, theta_range=theta_range, format=format)
            res.append(sv)
        return res

    def xyz_estimate(self, d, wx, wz):
        """Estimate the x-y-z coordinates based on phase and distance

        Parameters:
            d: distance of the point.
            wx: azimuth phase difference between two receivers.
            wz: elevation phase difference between two receivers.
        """
        x = d*wx
        z = d*wz
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = np.sqrt(d**2-x**2-z**2)
        return x, y, z

    def xyz_estimate_srpc(self, detection, wx, wz, rd_spectrum):
        """Estimate the x-y-z coordinates with SRPC"""
        vi, di, pi = detection
        dleft = max(0, di-self.range_fft_m)
        dright = min(rd_spectrum.shape[1], di+self.range_fft_m+1)
        n = self.range_fft_m*20
        c = np.linspace(self.dis[dleft], self.dis[dright], n)
        p = np.interp(np.linspace(dleft, dright, n), np.arange(dleft, dright), rd_spectrum[vi, dleft:dright])
        p = p/p.sum()
        ds = np.unique(np.append(np.random.choice(c, int(pi*self.srpc_r), p=p), self.dis[di]))
        res = []
        for d in ds:
            res.append(self.xyz_estimate(d, wx, wz))
        return np.asarray(res)

    def range_freq_to_dis(self, x):
        slope = self.config['slope']
        x = x*(3e8/2/slope)
        return x

    def doppler_freq_to_vel(self, x):
        fps = self.config['fps']
        wavelength = 3e8/77e9
        # x = x*wavelength/(4*np.pi*(1/fps))/2*np.pi
        x = x*wavelength*fps/2
        return x


class PointCloudProcessor(PointCloudProcessorBase):
    """To process the simulated data and construct a point cloud. Input format (n_rx, chirps_per_frame, samples_per_chirp)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def search_task(self):
        """Prepare search for the best peak detection algorithm"""
        tasks = []
        win = 80
        guard = 64
        thr = list(np.arange(2, 16, 1))
        for th in thr:
            tasks.append(f'ca-{th:.3f}-{win}-{guard}')
        return tasks

        # for i in range(30, 60, 4):
        #     tasks.append(f'th-{i}')
        # for win in range(72, 90, 4):
        #     for guard in range(win-32, win-14, 4):
        #         thr = [i for i in np.arange(8, 16, 1)]
        #         for th in thr:
        #             tasks.append(f'ca-{th:.3f}-{win}-{guard}')
        # return tasks

        # Backup
        # tasks = []
        # for i in range(30, 60, 4):
        #     tasks.append(f'th-{i}')
        # for win in range(64, 160, 16):
        #     for guard in range(int(win/2), win-8, 8):
        #         thr = [i for i in np.arange(12, 32, 2)]
        #         for th in thr:
        #             tasks.append(f'go-{th:.3f}-{win}-{guard}')
        #             tasks.append(f'so-{th:.3f}-{win}-{guard}')
        #             tasks.append(f'os-{th:.3f}-{win}-{guard}')
        #             tasks.append(f'ca-{th:.3f}-{win}-{guard}')
        # return tasks

    def search(self, x, y, method, doppler: bool, npass=None, callback=None, prefix=None, tasks=None):
        """Search for the best peak detection algorithm on a given dataset
        
        Parameters:
            x: simulated IF signal, (n_data, n_trial, n_rx, n_chirp, n_samples).
            y: ground truth point cloud, (n_data, n_trial, n_points, 3).
            method: one of AoA method.
            doppler: bool, use doppler fft or not.
            npass: the npass parameter of bf methods.
            callback: callback function to be called when search finishes.
            prefix: a name prefix of this search.
            tasks: candidate algorithms to search.
            e.g. 'th-10' = threshold(x, db=30); 'go-0.4-8-3-0.5' = cfar2d(x, mode='GO', th=0.4, win=8, guard=3, min_rel=0.5) 
        """
        logname = f'search/'
        if prefix:
            logname = logname + prefix + '-'
        logname = logname + f'{self.id}-{method}-{npass}'
        if doppler:
            logname = logname + '-doppler'
        f = open(f'{logname}.csv', 'w')
        if tasks is None:
            tasks = self.search_task()
        log(f'Searching {len(tasks)} tasks, saving to {logname}.csv')
        res = np.zeros((len(tasks), N_METRICS))
        t1 = datetime.datetime.now()
        for i, t in enumerate(tasks):
            res[i] = self.test(x, y, method, doppler, npass, t, n_trial=1)
            t2 = datetime.datetime.now()
            eta = ((t2-t1)/(i+1)*len(tasks)+t1).strftime('%m.%d-%H:%M')
            print(f'Progress {100*(i+1)/len(tasks):.2f}% [ETA {eta}]', end='\r')
            string = f'{t}, ' + ', '.join(f'{k:.4f}' for k in res[i]) + '\n'
            f.write(string)
        f.close()
        t3 = datetime.datetime.now()
        best = np.max(res[:, 0])
        best_spec = tasks[np.argmax(res[:, 0])]
        message = f'Search finished {len(tasks)} tasks in {t3-t1}, saved to {logname}.csv, best IoU {best:.4f} with {best_spec}'
        log(message)
        if callback is not None:
            callback(message)

    def test(self, x, y, method, doppler: bool, npass=None, peak_2d=None, n_trial=None, progress=False, ret_npoints=False):
        """ Evaluate an algorithm on a dataset. 

        Parameters:
            x: simulated IF signal, (n_data, n_trial, n_rx, n_chirp, n_samples).
            y: ground truth point cloud, (n_data, n_trial, n_points, 3).
            method: one of AoA method.
            doppler: bool, use doppler fft or not.
            npass: the npass parameter of bf methods.
            peak_2d: parameters of peak detection algorithms. 
            e.g. 'th-10' = threshold(x, db=30); 'go-0.4-8-3-0.5' = cfar2d(x, mode='GO', th=0.4, win=8, guard=3, min_rel=0.5) 
        
        Return: (mIoU, mean precision, mean Sensitivity)
        """
        # log('Test start')
        n_data = x.shape[1]
        if n_trial is None:
            n_trial = x.shape[0]
        if n_trial > x.shape[0]:
            raise ValueError("n_trial larger than dataset")
        if doppler:
            func = self.generate_point_cloud_with_doppler
        else:
            func = self.generate_point_cloud
        res = np.zeros((n_trial, n_data, N_METRICS))
        n_points = np.zeros((n_trial, n_data))
        t1 = datetime.datetime.now()
        cnt = 0
        total = n_data * n_trial
        for i in range(n_trial):
            for j in range(n_data):
                pc = func(x[i, j], method=method, npass=npass, peak_2d=peak_2d)
                n_points[i, j] = pc.shape[0]
                res[i, j] = Evaluator(pc, y[i, j]).run(print_result=False, norm=True)
                cnt += 1
                if progress:
                    t2 = datetime.datetime.now()
                    eta = ((t2-t1)/(cnt)*total+t1).strftime('%m.%d-%H:%M')
                    print(f'Progress {100*cnt/total:.2f}% [ETA {eta}]', end='\r')
        n_points = np.mean(n_points, axis=(0, 1))
        res = np.mean(res, axis=(0, 1))
        # log(f'Test complete, mean IoU {res[0]:.2f}')
        if ret_npoints:
            return res, n_points
        return res

    def parse_peak_detection(self, name):
        """Parse a peak detection string into the right function. 
        e.g. 'th-10' = threshold(x, db=30); 'go-0.4-8-3-0.5' = cfar2d(x, mode='GO', th=0.4, win=8, guard=3, min_rel=0.5) 
        """
        name = name.lower()
        try:
            strs = name.split('-')
            if strs[0] == 'th':
                func = self.threshold
                args = {'db': float(strs[1])}
            elif strs[0] in ['ca', 'so', 'go', 'os']:
                func = self.cfar2d
                args = {
                    'mode': strs[0].upper(),
                    'th': float(strs[1]),
                    'win': int(strs[2]),
                    'guard': int(strs[3]),
                }
                if len(strs) > 4:
                    args['min_rel'] = float(strs[4])
            return func, args
        except Exception as e:
            print(e)
            print('invalid peak detection specification')

    def generate_point_cloud_with_doppler(self, data, method=AoA.fft, npass=2, debug=False, peak_2d=None):
        """Generate a point cloud using the TI OOB demo algorithm

        Parameters:
            data: IF signal.
            method: AoA estimation algorithm to use.
            npass: one pass algorithm (azimuth and elevation together) or two pass algorithm (azimuth then elevation).
            debug: visualize intermediate result for debugging.
            peak_2d: 2D peak detection algorithm to use.

        Return:
            (n, 3) point cloud.
        """
        # range Doppler FFT
        range_fft = self.range_fft(data)
        range_fft = range_fft[:, :, :self.n_range_fft_cut]
        doppler_fft = self.doppler_fft(range_fft)
        doppler_fft_sum = 20*np.log10(np.sum(np.abs(doppler_fft), axis=0))   # sum over all rx and convert to dB

        if peak_2d is None:
            peak_2d_func = self.threshold
            peak_2d_args = {'db': 30}
        else:
            peak_2d_func, peak_2d_args = self.parse_peak_detection(peak_2d)
        # peak detection on the range-Doppler spectrum
        peaks = peak_2d_func(doppler_fft_sum, **peak_2d_args)
        peaks[:, 2] = 10**((peaks[:, 2]/peaks[:, 2].min()-1))
        peaks = np.asarray(peaks, dtype=int)
        if debug:
            print(f'Detected {peaks.shape[0]} points from FFT')
            doppler_fft_peaks = np.zeros(doppler_fft_sum.shape)
            for d, r, _ in peaks.astype(int):
                doppler_fft_peaks[d, r] = 1
            _, axs = plt.subplots(1, 2, figsize=(12, 5))
            self.plot_range_doppler_heatmap(doppler_fft_sum, axs[0])
            self.plot_range_doppler_heatmap(doppler_fft_peaks, axs[1])
            plt.show()
        # AoA estimation for each range-Doppler peak
        if method == AoA.fft:
            point_cloud = self.aoa_fft_per_point(doppler_fft, peaks, npass=npass, rd_spectrum=doppler_fft_sum)
        else:
            if npass == 1:
                func = self.bf_per_point_2d
            elif npass == 2:
                func = self.bf_per_point
            else:
                raise ValueError(f"npass value {npass} incorrect")
            point_cloud = func(doppler_fft, peaks, method=method, rd_spectrum=doppler_fft_sum)
        if self.output_size is not None:
            point_cloud = point_cloud[np.random.choice(point_cloud.shape[0], self.output_size, replace=False)]
        elif self.srpc_a or self.srpc_r:
            target = 1/(np.mean(peaks[:, 2])*self.srpc_r+1) * 1/(self.srpc_a+1)
            target = int(point_cloud.shape[0] * target)
            i = np.random.choice(point_cloud.shape[0], target, replace=False)
            point_cloud = point_cloud[i]
        return point_cloud

    def generate_point_cloud(self, data, method=AoA.conventional, npass=2, batch_size=None, debug=False, peak_2d=None):
        """Generate a point cloud without Doppler FFT
        Parameters:
            data: IF signal.
            method: AoA estimation algorithm to use.
            npass: one pass algorithm (azimuth and elevation together) or two pass algorithm (azimuth then elevation).
            batch_size: process chirps in batch, default use all chirps. 
            debug: visualize intermediate result for debugging.
            peak_2d: 2D peak detection algorithm to use.

        Return:
            (n, 3) point cloud.
        """
        range_fft = self.range_fft(data)
        range_fft = range_fft[:, :, :self.n_range_fft_cut]
        n_chirp = range_fft.shape[1]
        if peak_2d is None:
            peak_2d_func = self.cfar2d
            peak_2d_args = {'mode': 'GO', 'th': 0.37}
        else:
            peak_2d_func, peak_2d_args = self.parse_peak_detection(peak_2d)

        point_cloud_all = []
        if batch_size is None:
            batch_size = n_chirp
        # n_batch = int(np.ceil(n_chirp/batch_size))
        n_batch = 1
        for j in range(n_batch):
            X = range_fft[:, j*batch_size:(j+1)*batch_size]
            if method == AoA.fft:
                # print('Method set to FFT, using the standard doppler approach')
                assert ValueError and "FFT without doppler, this should not be used"
                # doppler_fft = self.doppler_fft(range_fft)
                # doppler_fft_sum = 20*np.log10(np.sum(np.abs(doppler_fft), axis=0))   # sum over all rx and convert to dB
                # peaks = self.threshold(doppler_fft_sum, db=30)
                # point_cloud = self.aoa_fft_per_point(doppler_fft, peaks)
            elif npass == 1:
                point_cloud = self.bf_per_range_2d(X, method=method)
            elif npass == 2:
                peaks = self.bf_per_range_azimuth(X, method, norm=True, return_list=True)
                # print(f'Detected {peaks.shape[0]} points from range-azimuth heatmap')
                point_cloud = self.bf_per_point_elevation(X, peaks, method=method)
            elif npass == 3:
                range_azimuth = self.bf_per_range_azimuth(X, method, norm=True)
                range_azimuth = range_azimuth/range_azimuth.max()*128
                peaks = peak_2d_func(range_azimuth, **peak_2d_args)

                if debug:
                    print(f'Detected {peaks.shape[0]} points from range-azimuth heatmap')
                    range_azimuth_peaks = np.zeros(range_azimuth.shape)
                    for d, r, _ in peaks:
                        range_azimuth_peaks[d, r] = 1
                    _, axs = plt.subplots(1, 2, figsize=(12, 5))
                    axs[0].pcolormesh(range_azimuth)
                    axs[1].pcolormesh(range_azimuth_peaks)
                    plt.show()
                if peaks.shape[0] > 2048:
                    # print(f'Too many points {peaks.shape[0]} from peak_2d, capping at 2048')
                    peaks = peaks[np.random.choice(peaks.shape[0], 2048, replace=False)]
                    assert peaks.shape[0] == 2048
                point_cloud = self.bf_per_point_elevation(X, peaks, method=method)
                if self.output_size is not None:
                    point_cloud = point_cloud[np.random.choice(point_cloud.shape[0], self.output_size, replace=False)]
            else:
                raise ValueError(f"npass value {npass} incorrect")
            point_cloud_all.append(point_cloud)
        res = np.concatenate(point_cloud_all)
        return res

    def generate_azimuth_heatmap(self, data, method=AoA.conventional, format='phi'):
        """Generate a 1D azimuth heatmap
        
        Parameters:
            data: IF signal.
            method: AoA estimation algorithm to use.
            format: `phi` or `theta`
        """
        range_fft = self.range_fft(data)

        # per frame
        if method == AoA.fft:
            azimuth_spectrum = self.aoa_fft_azimuth(range_fft)
        else:
            azimuth_spectrum = self.bf_azimuth_per_frame(range_fft, method=method, format=format)

        # azimuth_spectrum = np.log2(azimuth_spectrum)
        azimuth_spectrum = azimuth_spectrum/azimuth_spectrum.max()

        x = self.angle_phi if format == 'phi' else self.angle
        x = x/np.pi*180
        plt.plot(x, azimuth_spectrum)
        plt.show()

    def generate_range_azimuth_heatmap(self, data, method=AoA.conventional, projection='raw', format='phi'):
        """Generate a 2D range-azimuth heatmap
        
        Parameters:
            data: IF signal.
            method: AoA estimation algorithm to use.
            projection: `raw`, `polar` or `cartesian`
            format: `phi` or `theta`
        """
        range_fft = self.range_fft(data)
        # per bin
        if method == AoA.fft:
            range_azimuth = self.aoa_fft_azimuth(range_fft, per_range=True)
            if format == 'theta':
                warnings.warn('Warning: Forcing FFT result to be in phi.')
                format = 'phi'
        else:
            range_azimuth = self.bf_per_range_azimuth(range_fft, method=method, format=format)
        self.plot_range_azimuth_heatmap(range_azimuth, projection, format=format)

    def generate_range_doppler_heatmap(self, data):
        """generate a 2D range doppler heatmap from IF signal.
        """
        range_fft = self.range_fft(data)
        range_fft = range_fft[:, :, :self.n_range_fft_cut]
        doppler_fft = self.doppler_fft(range_fft)
        doppler_fft_sum = np.abs(np.sum(doppler_fft, axis=0))   # sum over all rx
        doppler_fft_sum /= doppler_fft_sum.max()
        self.plot_range_doppler_heatmap(doppler_fft_sum)

    def plot_range_doppler_heatmap(self, doppler_fft_mags, ax=None):
        """Plot a 2D range doppler heatmap, using range-Doppler-fft output
        """
        if not ax:
            fig = plt.figure()
            pc = plt.pcolormesh(self.dis, self.vel, doppler_fft_mags[:, :self.n_range_fft_cut])
            cb = fig.colorbar(pc)
            cb.ax.tick_params(labelsize=16)
            plt.show()
        else:
            ax.pcolormesh(self.dis, self.vel, doppler_fft_mags[:, :self.n_range_fft_cut])

    def plot_range_azimuth_heatmap(self, range_azimuth, projection='raw', format='phi'):
        """Plot a 2D range-azimuth heatmap
        
        Parameters:
            range_azimuth: range-azimuth heatmap.
            projection: `raw`, `polar` or `cartesian`
            format: `phi` or `theta`
        """
        range_azimuth = range_azimuth[:self.n_range_fft_cut]
        # range_azimuth = np.log2(range_azimuth)
        range_azimuth = range_azimuth/range_azimuth.max()
        print(f'Plotting range-azimuth heatmap in {format} domain.')
        if format == 'phi':
            azimuth = self.angle_phi
        else:
            azimuth = self.angle

        bg = plt.cm.get_cmap(plt.rcParams["image.cmap"])(0)

        if projection == 'raw':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.pcolormesh(azimuth/np.pi*180, self.dis, range_azimuth)
            plt.show()
        elif projection == 'polar':
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='polar')
            pc = ax.pcolormesh(azimuth, self.dis, range_azimuth)
            ax.set_thetamin(-75)
            ax.set_thetamax(75)
            ax.set_theta_direction(-1)
            ax.grid(True, alpha=0.2)
            ax.set_theta_zero_location('N')
            ax.set_facecolor(bg)
            # cb = fig.colorbar(pc)
            # cb.ax.tick_params(labelsize=16)
            plt.show()
        elif projection == 'cartesian':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            dis, angle = np.meshgrid(self.dis, azimuth)
            xs = dis*np.cos(angle)
            ys = dis*np.sin(angle)
            ax.pcolormesh(ys, xs, range_azimuth.T)
            ax.set_facecolor(bg)
            ax.set_ylim(bottom=0)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            plt.show()


class PointCloudParameterSearcher(PointCloudProcessor):
    """Mutithreaded module to search for best 2D peak detection algorithm"""
    def __init__(self, dataset, *args, thread=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.thread = 8 if not thread else thread

    def search_mp(self, method, doppler: bool, npass=None, callback=None, prefix=None, tasks=None):
        """Multithreaded version of the search function"""
        tasks = []
        logname = f'search/'
        if prefix:
            logname = logname + prefix + '-'
        logname = logname + f'{self.id}-{method}-{npass}'
        if doppler:
            logname = logname + '-doppler'

        if os.path.exists(f'{logname}.csv'):
            log(f'Existing result {logname}.csv found')
            return
        if tasks is None:
            tasks = self.search_task()
        log(f'Searching {len(tasks)} tasks using {self.thread} threads, saving to {logname}.csv')
        self.res = np.zeros((len(tasks), N_METRICS))
        self.cnt = 0
        self.total = len(tasks)
        self.t1 = datetime.datetime.now()

        pool = multiprocessing.Pool(self.thread)
        for i, t in enumerate(tasks):
            pool.apply_async(self.test_wrapper, args=(i, method, doppler, npass, t, 2, ),
                             callback=self.callback_succ, error_callback=self.callback_err)

        pool.close()
        pool.join()

        t3 = datetime.datetime.now()

        res = self.res
        f = open(f'{logname}.csv', 'w')
        for i in range(self.total):
            string = f'{tasks[i]}, ' + ', '.join(f'{k:.4f}' for k in res[i]) + '\n'
            f.write(string)
        f.close()

        best = np.max(res[:, 0])
        best_spec = tasks[np.argmax(res[:, 0])]
        message = f'Search finished {len(tasks)} tasks in {t3-self.t1}, saved to {logname}.csv, best IoU {best:.4f} with {best_spec}'
        log(message)
        if callback is not None:
            callback(message)
        del self.res

    def test_wrapper(self, i, *args):
        dataset = IFDataset(self.dataset, self.config)
        train_x, train_y = dataset.load(train=True)
        return (i, self.test(train_x, train_y, *args))

    def callback_succ(self, res):
        self.cnt += 1
        i, res = res
        self.res[i] = res
        t1 = self.t1
        t2 = datetime.datetime.now()
        eta = ((t2-t1)/(self.cnt)*self.total+t1).strftime('%m.%d-%H:%M')
        print(f'Progress {100*(self.cnt)/self.total:.2f}% [ETA {eta}]', end='\r')

    def callback_err(self, e):
        print(e)
        # print(traceback.format_exc())


class PointCloudTester(PointCloudParameterSearcher):
    """Mutithreaded module to evaluate a 2D peak detection algorithm"""
    def __init__(self, dataset, *args, thread=None, **kwargs):
        super().__init__(dataset, *args, thread=thread, **kwargs)

    def make_fig(self, i, method, doppler: bool, npass, peak_2d, prefix=None, postfix=''):
        """Plot point cloud for all data in a dataset"""
        taskname = ''
        if prefix:
            taskname = taskname + prefix + '-'
        taskname = taskname + f'{self.id}-{method}-{npass}'
        if doppler:
            taskname = taskname + '-doppler'

        if peak_2d == 'auto':
            csvname = f'search/{taskname}.csv'
            peak_2d = read_csv(csvname)

        figname = f'test/figs/{taskname}{postfix}'
        dataset = IFDataset(self.dataset, self.config)
        x, y = dataset.load(train=False)
        data = x[i, 0]
        dataset_mesh = DatasetMesh('FAUST')
        mesh = dataset_mesh[i]
        if doppler:
            func = self.generate_point_cloud_with_doppler
        else:
            func = self.generate_point_cloud
        pc = func(data, method=method, npass=npass, peak_2d=peak_2d)
        save_one_fig(figname, pc, mesh, view='front')
        save_one_fig(figname, pc, mesh, view='left')

    def test_mp(self, method, doppler: bool, npass, peak_2d, n_trial=None, prefix=None, callback=None, postfix=''):
        """Evaluate point cloud construction algorithm for all data in a dataset"""
        taskname = ''
        if prefix:
            taskname = taskname + prefix + '-'
        taskname = taskname + f'{self.id}-{method}-{npass}'
        if doppler:
            taskname = taskname + '-doppler'

        if peak_2d == 'auto':
            csvname = f'search/{taskname}.csv'
            peak_2d = read_csv(csvname)

        logname = f'test/{taskname}{postfix}'
        dataset = IFDataset(self.dataset, self.config)
        x, y = dataset.load(train=False)

        if n_trial is None:
            n_trial = x.shape[0]
        if n_trial > x.shape[0]:
            raise ValueError("n_trial larger than dataset")
        n_data = x.shape[1]
        self.total = n_trial * n_data

        log(f'Testing {n_data} data {n_trial} times using {self.thread} threads, saving to {logname}.npy')
        self.res = np.zeros((n_trial, n_data, N_METRICS))
        self.n_points = np.zeros((n_trial, n_data))
        self.cnt = 0
        self.t1 = datetime.datetime.now()

        pool = multiprocessing.Pool(self.thread)
        for i in range(n_trial):
            for j in range(n_data):
                pool.apply_async(self.test_wrapper, args=(i, j, method, doppler, npass, peak_2d, n_trial, ),
                                 callback=self.callback_succ, error_callback=self.callback_err)
        pool.close()
        pool.join()

        t3 = datetime.datetime.now()
        np.save(f'{logname}.npy', self.res)
        np.save(f'{logname}-npoints.npy', self.n_points)
        avg = np.mean(self.res, axis=(0, 1))

        message = f'Test finished in {t3-self.t1}, saved to {logname}.npy, res {avg}'
        log(message)
        if callback is not None:
            callback(message)
        del self.res

    def test_each(self, x, y, method, doppler, npass, peak_2d, n_trial):
        if doppler:
            func = self.generate_point_cloud_with_doppler
        else:
            func = self.generate_point_cloud
        pc = func(x, method=method, npass=npass, peak_2d=peak_2d)
        res = Evaluator(pc, y).run(print_result=False, norm=True)
        return res, pc.shape[0]

    def test_wrapper(self, i, j, *args):
        dataset = IFDataset(self.dataset, self.config)
        x, y = dataset.load(train=False)
        return (i, j, self.test_each(x[i, j], y[i, j], *args))

    def callback_succ(self, res):
        self.cnt += 1
        i, j, (res, n_points) = res
        self.res[i, j] = res
        self.n_points[i, j] = n_points
        t1 = self.t1
        t2 = datetime.datetime.now()
        eta = ((t2-t1)/(self.cnt)*self.total+t1).strftime('%m.%d-%H:%M')
        print(f'Progress {100*(self.cnt)/self.total:.2f}% [ETA {eta}]', end='\r')
import numpy as np
import sys
import datetime
from dataset import Dataset, IFDataset
import open3d as o3d
np.set_printoptions(precision=2, suppress=True, linewidth=200)

def main():
    n_frames = 1
    chirps_per_frame = 50
    steps = int(n_frames*chirps_per_frame)
    fps = 1000
    chirp_time = 100e-6
    ADC_rate = 15e6
    slope = 40e12
    noise = 20
    layout = '4x4'
    n_samples = 512
    distance = 2
    motion = 'Line'
    velocity = (0, 0, 0)

    config = dict({
        'ADC_rate': ADC_rate,
        'chirp_time': chirp_time,
        'slope': slope,
        'fps': fps,                    # fps stands for chirps per second regardless frame
        'n_frames': n_frames,
        'chirps_per_frame': chirps_per_frame,
        'steps': steps,                 # steps stands for all chirps over simulation period    
        'simulation_period': steps/fps,
        'samples_per_chirp': int(chirp_time*ADC_rate),
        'layout': layout,
        'noise': noise,
        'n_samples': n_samples,
        'distance': distance,
        'motion': motion,
        'velocity': velocity
    })

    dataset = IFDataset('FAUST', config, location='d:/datasets/mmSim/')
    dataset.construct(split=None)

    
if __name__ == "__main__":
    main()

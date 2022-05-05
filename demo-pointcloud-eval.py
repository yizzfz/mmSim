import numpy as np
from util import *
from algo import *
import matplotlib.pyplot as plt
import sys
import datetime
from motion import Line
from radar import Radar, RadarArray, rx_config
from simulator import Simulator
from scene import Point
from dataset import Dataset, DatasetMesh
import open3d as o3d
np.set_printoptions(precision=2, suppress=True, linewidth=200)

def main():
    ts = datetime.datetime.now().strftime('%H:%M')
    print(f'[{ts}] System start.')

    # configure the radar and the scene
    ADC_rate = 15e6
    chirp_time = 100e-6
    slope = 40e12
    fps = 1000   # fps stands for chirps per second regardless frame
    simulation_period = 0.05
    samples_per_chirp = int(chirp_time*ADC_rate)
    steps = int(fps*simulation_period)  # steps stands for all chirps over simulation period
    chirps_per_frame = 50
    noise = 20
    n_frames = int(steps/chirps_per_frame)
    frame_time = simulation_period / n_frames
    layout = '1443'

    # one radar at the origin, with a receving antenna array
    radars = RadarArray((0, 0, 0), layout=layout, ADC_rate=ADC_rate, chirp_time=chirp_time, slope=slope, noise=noise)
    rx_config = radars.get_rxconfig()
    n_rx = rx_config.n_rx
    config = dict({
        'ADC_rate': ADC_rate,
        'chirp_time': chirp_time,
        'slope': slope,
        'fps': fps,
        'simulation_period': simulation_period,
        'steps': steps,
        'samples_per_chirp': samples_per_chirp,
        'chirps_per_frame': chirps_per_frame,
        'frame_time': frame_time,
        'n_frames': n_frames,
        'layout': layout,
        'noise': noise,
    })

    motion = Line((0, 0.5, 0))

    dataset = Dataset('FAUST', n_samples=256, distance=2, train=True, location='d:/datasets/tmp/').dataset
    ref = DatasetMesh('FAUST', distance=2, train=True, location='d:/datasets/tmp/')
    ts = datetime.datetime.now().strftime('%H:%M')
    print(f'[{ts}] Dataset loaded.')
    
    i = 0   # try the first one
    datapoints = dataset[i].pos.numpy()
    mesh = ref[i]
    scene = [Point(datapoints, motion=motion)]
    
    # put them into the simulator
    simulator = Simulator(radars, scene, simulation_period, fps)
    ts = datetime.datetime.now().strftime('%H:%M')
    print(f'[{ts}] Simulation start ...')

    data = simulator.run()
    data = data.reshape((n_rx, n_frames, chirps_per_frame, samples_per_chirp))
    data = data[:, 0]   # take frame one
    ts = datetime.datetime.now().strftime('%H:%M')
    print(f'[{ts}] Simulation finished. Processing data...')
    pcp = PointCloudProcessor(config, max_d=5, range_fft_m=4, n_aoa_fft=512)
    res = pcp.generate_point_cloud_with_doppler(data, method=AoA.mvdr, npass=2, debug=False)
    plot_point_cloud_o3d(res, mesh=mesh, video=False)
    Evaluator(res, datapoints).run(print_result=True)

    
if __name__ == "__main__":
    main()

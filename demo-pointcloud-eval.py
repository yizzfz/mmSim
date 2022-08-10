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
    ADC_rate = 15e6                                 # 15 MHz
    chirp_time = 100e-6                             # 100 us
    slope = 40e12                                   # 40 MHz/us
    fps = 1000                                      # 1000 chirps per frame
    simulation_period = 0.05                        # simulate 0.05 seconds
    samples_per_chirp = int(chirp_time*ADC_rate)    # 1500 samples per chirp
    steps = int(fps*simulation_period)              # 50 chirps in total
    chirps_per_frame = 50                           # 50 chirps per frame
    noise = 20                                      # Gaussian white noise 20 dB
    n_frames = int(steps/chirps_per_frame)          # 1 frame
    frame_time = simulation_period / n_frames       # frame time 0.05s
    layout = '1443'                                 # simulate iwr1443 radar antenna layout

    # one radar at the origin, with a receving antenna array
    radars = RadarArray((0, 0, 0), layout=layout, ADC_rate=ADC_rate, chirp_time=chirp_time, slope=slope, noise=noise)
    rx_config = radars.get_rxconfig()
    n_rx = rx_config.n_rx

    # put all configuration as a dict
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

    # assume the person is moving away at 0.5 m/s
    motion = Line((0, 0.5, 0))
    # construct FAUST point cloud dataset
    dataset = Dataset('FAUST', n_samples=256, distance=2, train=True, location='d:/datasets/mmSim/').dataset
    # construct FAUST mesh dataset
    ref = DatasetMesh('FAUST', distance=2, train=True, location='d:/datasets/mmSim/')
    ts = datetime.datetime.now().strftime('%H:%M')
    print(f'[{ts}] Dataset loaded.')
    
    i = 0   # load the first human model in FAUST
    datapoints = dataset[i].pos.numpy()     # get point cloud as a (n,3) array
    mesh = ref[i]   # get ground truth mesh model
    scene = [Point(datapoints, motion=motion)]  # construct the scene using the point cloud
    
    # put them into the simulator
    simulator = Simulator(radars, scene, simulation_period, fps)
    ts = datetime.datetime.now().strftime('%H:%M')
    print(f'[{ts}] Simulation start ...')

    # get simulation data
    data = simulator.run()
    data = data.reshape((n_rx, n_frames, chirps_per_frame, samples_per_chirp))
    data = data[:, 0]   # take the first frame
    ts = datetime.datetime.now().strftime('%H:%M')
    print(f'[{ts}] Simulation finished. Processing data...')

    # construct a point cloud
    pcp = PointCloudProcessor(config, max_d=5, range_fft_m=4, n_aoa_fft=512)
    res = pcp.generate_point_cloud_with_doppler(data, method=AoA.mvdr, npass=2, debug=False)

    # visualize the point cloud and the ground truth
    plot_point_cloud_o3d(res, mesh=mesh, video=False)

    # evaluate the accuracy of the point cloud
    Evaluator(res, datapoints).run(print_result=True)

    
if __name__ == "__main__":
    main()

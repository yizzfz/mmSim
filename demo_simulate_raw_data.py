import numpy as np
import scipy, scipy.io
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import sys
import datetime
from motion import Pulse, MotionList, Line
from radar import Radar, RadarArray
from simulator import Simulator
from scene import Circle, Point
np.set_printoptions(precision=4, suppress=True)

def main():
    ts = datetime.datetime.now().strftime('%H:%M')
    print(f'[{ts}] System Start')

    # configure the radar and the scene
    ADC_rate = 15e6
    chirp_time = 100e-6
    slope = 40e12
    fps = 1000   # fps stands for chirps per second regardless frame
    simulation_period = 0.05
    samples_per_chirp = int(chirp_time*ADC_rate)
    steps = int(fps*simulation_period)  # steps stands for all chirps over simulation period
    n_rx = 1
    chirps_per_frame = 50
    n_frames = int(steps/chirps_per_frame)
    frame_time = simulation_period / n_frames

    # one radars at the origin
    radars = RadarArray((0, 0, 0), layout='1443', ADC_rate=ADC_rate, chirp_time=chirp_time, slope=slope)
    n_rx = len(radars)
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
        'n_rx': n_rx
    })
 
    points = [(1,2,0), (-1,2,0), (1,-2,0), (1,2,1), (-1,2,1), (1,-2,1), (2,1,0), (-2,1,0),
(1,3,0), (-1,3,0), (1,-3,0), (1,3,1), (-1,3,1), (1,-3,1), (3,1,0), (-3,1,0),
(1,4,0), (-1,4,0), (1,-4,0), (1,4,1), (-1,4,1), (1,-4,1), (4,1,0), (-4,1,0),
(1,5,0), (-1,5,0), (1,-5,0), (1,5,1), (-1,5,1), (1,-5,1), (5,1,0), (-5,1,0),
(1,6,0), (-1,6,0), (1,-6,0), (1,6,1), (-1,6,1), (1,-6,1), (6,1,0), (-6,1,0),
(2,2,0), (-2,2,0), (2,-2,0), (2,2,1), (-2,2,1), (2,-2,1), (2,1,2), (-2,1,2),
(2,3,0), (-2,3,0), (2,-3,0), (2,3,1), (-2,3,1), (2,-3,1), (3,1,3), (-3,1,3),
(2,4,0), (-2,4,0), (2,-4,0), (2,4,1), (-2,4,1), (2,-4,1), (4,2,0), (-4,2,0),
(2,5,0), (-2,5,0), (2,-5,0), (2,5,1), (-2,5,1), (2,-5,1), (5,2,0), (-5,2,0),
(2,6,0), (-2,6,0), (2,-6,0), (2,6,1), (-2,6,1), (2,-6,1), (6,2,0), (-6,2,0),
(3,2,0), (-3,2,0), (3,-2,0), (3,2,1), (-3,2,1), (3,-2,1), (4,3,0), (-4,3,0),
(3,3,0), (-3,3,0), (3,-3,0), (3,3,1), (-3,3,1), (3,-3,1), (5,3,0), (-5,3,0),
(3,4,0), (-3,4,0), (3,-4,0), (3,4,1), (-3,4,1), (3,-4,1), (6,3,0), (-6,3,0),
(3,5,0), (-3,5,0), (3,-5,0), (3,5,1), (-3,5,1), (3,-5,1), (6,4,0), (-6,4,0),
(3,6,0), (-3,6,0), (3,-6,0), (3,6,1), (-3,6,1), (3,-6,1), (5,4,0), (-5,4,0),]

    scene = [
        Point(points, motion=Line((0, -0.5, 0))),
        ]

    # put them into the simulator
    simulator = Simulator(radars, scene, simulation_period, fps)
    ts = datetime.datetime.now().strftime('%H:%M')
    print(f'[{ts}] Simulation Start ...')

    data, freqzs = simulator.run()
    data = data.reshape((n_rx, -1))
    np.save('simulation.npy', data)

    
if __name__ == "__main__":
    main()

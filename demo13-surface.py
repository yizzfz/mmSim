'''Demo ver13, surface
'''
from radar import Radar
from scene import Point, Circle
from motion import Line, Oscillation, MotionList, Pulse
from simulator import Simulator
from util import CFAR, FFT, unwrap, wavelet
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import plotly
import datetime
import pickle

def main():
    # configure the radar and the scene
    ADC_rate = 15e6
    chirp_time = 100e-6
    slope = 40e12
    fps = 1000
    simulation_period = 10
    steps = fps*simulation_period
    phase_shift = 0.5
 
    # one radars at the origin
    radars = [
        Radar((0, 0, 0), ADC_rate=ADC_rate, chirp_time=chirp_time, slope=slope, phase_shift=0),
        ]
    n_radars = len(radars)
    rates = [15/60, 80/60]
    breath_rate = Oscillation((0, 0.012, 0), rates[0])
    heart_rate = Pulse((0, 0.0002, 0), rates[1])
    scene = [
        Circle((0, 1, 0), 0.6, step=0.005, motion=MotionList([
            heart_rate,
            breath_rate,
            Oscillation((0, 0.01, 0), 0.5),
            Oscillation((0, 0.002, 0), 0.6),
            # Oscillation((0, 0.15, 0), 0.8),
            Oscillation((0, 0.08, 0), 0.3),
            # Line((0, 0.4, 0)),
            Oscillation((0, 2, 0), 0.1),
        ])),
        ]


    # put them into the simulator
    simulator = Simulator(radars, scene, simulation_period, fps)
    heart_path = heart_rate.get_path()[:, 1]
    ts = datetime.datetime.now().strftime('%H:%M')
    print(f'[{ts}] Simulation Started...')

    # define parameters for FFT
    n_samples = int(ADC_rate*chirp_time)
    n_fft = n_samples*10
    half = int(n_fft/2)
    fft_freq_d = np.fft.fftfreq(n_fft, d=1.0/ADC_rate)[:half]

    # get simulation data
    # output dim: 
    #   signals: (n_radars, n_steps, signal_len)
    #   freqzs: (n_radars, n_steps, n_objs, (freq, phase))
    signals, freqzs = simulator.run()
    freqzs[:,:,:,1] = freqzs[:,:,:,1]/np.pi
    fft_freqzs = np.zeros((n_radars, steps, 1, 2))

    ts = datetime.datetime.now().strftime('%H:%M')
    print(f'[{ts}] Simulation finished. Processing data...')

    # process simulation data
    for n in range(len(radars)):
        for i in range(steps):
            signal = signals[n, i]
            fft = np.fft.fft(signal, n_fft)[:half]
            peaks = CFAR(fft)
            peaks = np.array([(fft_freq_d[i], phase/np.pi) for i, _, phase in peaks])
            fft_freqzs[n, i] = peaks[0]
        # fft_freqzs = np.array(fft_freqzs)   # assuming FFT always report the same number of objs
    P1 = fft_freqzs[0, :, 0, 1]
    P1_unwrapped = unwrap(P1)
    res = wavelet(P1_unwrapped, fps)
    y_hi = np.max([P1_unwrapped])
    y_lo = np.min([P1_unwrapped])
    ts = datetime.datetime.now().strftime('%H:%M')
    print(f'[{ts}] Data processed. Preparing graph...')

    # visualization
    fig = plt.figure()

    # plot the phase
    ax1 = plt.subplot(222)
    t1 = np.arange(0, simulation_period, 1/fps)
    ax1.plot(t1, P1, label='Measured')
    # ax1.plot(t1, freqzs[0, :, 0, 1], label='Real')
    # ax1.legend()
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Phase in pi')
    ax1.set_title('Phase of the IF signal')

    ax1 = plt.subplot(223)
    t1 = np.arange(0, simulation_period, 1/fps)
    line1, = ax1.plot(t1, P1_unwrapped)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Phase in pi')
    # ax1.set_ylim([y_lo, y_hi])
    # ax1.set_xlim([0, simulation_period])
    ax1.set_title('Unwrapped phase of the IF signal')

    ax3 = plt.subplot(224)
    line3, = ax3.plot(t1, res)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Result')
    # ax3.set_ylim([0, 1])
    # ax3.set_xlim([0, simulation_period])
    ax3.set_title('Wavelet')
    ax_s1 = ax3.twinx()
    ax_s1.set_ylabel('Real Heart Rate')
    ax_s1.plot(t1, heart_path, color='red')
    ax_s1.tick_params(axis='y', labelcolor='tab:red')
    ax_s1.set_ylim([-0.001, 0.001])

    # the motion of the object(s) in the scene
    ax_s = plt.subplot(221)
    path = simulator.get_paths()[0]
    path = path[:, 0, 1]
    line_s, = ax_s.plot(t1, path)
    # ax_s.set_xlim([0, simulation_period])
    # ax_s.set_ylim([np.min(path), np.max(path)])
    ax_s.set_xlabel('Time (s)')
    ax_s.set_ylabel('Y (m)')
    ax_s.set_title('Scene')
    ax_s.tick_params(axis='y', labelcolor='tab:blue')
    # txt1 = ax_s.text(ax_s.get_xlim()[0], ax_s.get_ylim()[1], f'Simulation time: 0 s', horizontalalignment='left', verticalalignment='top')


    # set up animation
    plt.tight_layout()
    # def animate(i):
    #     line1.set_xdata(t1[:i])
    #     line1.set_ydata(P1_unwrapped[:i])

    #     line2.set_xdata(t1[:i])
    #     line2.set_ydata(res1[:i])

    #     line3.set_xdata(t1[:i])
    #     line3.set_ydata(res2[:i])

    #     line_s.set_xdata(t1[:i])
    #     line_s.set_ydata(path[:i])

    #     txt1.set_text((f'Simulation time: {i/fps:.2f} s'))
    #     return line1, line2

    # fig.set_size_inches(16, 9)
    # ani = animation.FuncAnimation(fig, animate, steps, interval=20, blit=False, save_count=50)
    # ani.save(f'{Path(__file__).stem}.mp4', fps=fps, dpi=200)


    plt.show()
    ts = datetime.datetime.now().strftime('%m%d%H%M')
    with open(f'{ts}.pkl', 'wb') as f:
        pickle.dump(signals, f)
        pickle.dump(freqzs, f)
        print(f'Simulation data saved to {ts}.pkl')

    # plotly_fig = plotly.tools.mpl_to_plotly(fig)
    # plotly.io.write_html(plotly_fig, file=f'{Path(__file__).stem}.html')


if __name__ == "__main__":
    main()
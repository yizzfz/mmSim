'''Demo showing how to run the simulator
'''
from radar import Radar
from scene import Point
from motion import Oscillation, MotionList, Pulse
from simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

def main():
    # configure the radar and the scene
    ADC_rate = 15e6     # 15 MHz
    chirp_time = 100e-6 # 100 us
    slope = 40e12       # 40 MHz/us
    fps = 50            # 50 chirps per second
    simulation_period = 20  # simulate 20 seconds
    steps = fps*simulation_period   # total number of chirps
 
    # two radars at the origin
    radars = [
        Radar((0, 0, 0), ADC_rate=ADC_rate, chirp_time=chirp_time, slope=slope, phase_shift=0),
        Radar((0, 0, 0), ADC_rate=ADC_rate, chirp_time=chirp_time, slope=slope, phase_shift=0.5*np.pi),
        ]
    n_radars = len(radars)

    # define one point at (0, 1, 0), oscillating and pulsing
    scene = [
        Point((0, 1, 0), motion=MotionList([
            Oscillation((0, 0.008, 0), 0.5), 
            Pulse((0, 0.0002, 0), 1),
        ])),
        ]

    # put them into the simulator
    simulator = Simulator(radars, scene, simulation_period, fps)
    print('Simulation ready.')

    # define parameters for FFT
    n_samples = int(ADC_rate*chirp_time)
    n_fft = n_samples*10
    half = int(n_fft/2)
    fft_freq_d = np.fft.fftfreq(n_fft, d=1.0/ADC_rate)[:half]

    # get simulation data
    # output dim: 
    #   signals: (n_radars, n_steps, signal_len)
    #   freqzs: (n_radars, n_steps, n_objs, (freq, phase))
    signals, freqzs = simulator.run(freqz=True)
    freqzs[:,:,:,1] = freqzs[:,:,:,1]/np.pi
    fft_freqzs = np.zeros((n_radars, steps, 1, 2))
    ffts = np.zeros((n_radars, steps, half))
    print('Simulation finished. Processing data...')

    # process simulation data
    for n in range(len(radars)):
        for i in range(steps):
            signal = signals[n, i]
            fft = np.fft.fft(signal, n_fft)[:half]
            ffts[n, i] = np.abs(fft)
            p = np.argmax(fft)
            fft_freqzs[n, i] = fft_freq_d[p], np.angle(fft[p]/np.pi)
    fft_freqzs[:, :, :, 1][fft_freqzs[:, :, :, 1]<0] += 2
    print('Data processed. Preparing graph...')

    # visualization (showing data for radar 1 only)
    fig = plt.figure()

    # fig 1 shows the IF signal in time domain
    ax1 = plt.subplot(321)
    t = np.arange(0, chirp_time, 1/ADC_rate)
    line, = ax1.plot(t, signals[0, 0].real)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Signal Amplitude')
    ax1.set_title('IF signal in time domain')
    ax1.set_ylim([-100, 100])
    ax1.set_xlim([0, chirp_time])

    # fig 2 shows the FFT of the IF siganl
    ax2 = plt.subplot(323)
    line2, = ax2.plot(fft_freq_d, ffts[0, 0])
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('FFT response')
    ax2.set_title('FFT of the IF signal')
    ax2.set_xlim([0, 600e3])
    # ax2.set_ylim([0, 1.2])

    # fig 3 plot the phase
    ax3 = plt.subplot(325)
    t1 = np.arange(0, simulation_period, 1/fps)
    line3, = ax3.plot([], [], label='Ground Truth')
    line4, = ax3.plot([], [], label='Measured')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Phase in pi')
    ax3.set_ylim([0, 2])
    ax3.set_xlim([0, simulation_period])
    ax3.legend(loc='upper right')
    ax3.set_title('Phase of the IF signal')

    # the motion of the object(s) in the scene
    ax5 = plt.subplot(122)
    path = simulator.get_paths()[0]
    path = path[:,0, (0, 1)]
    sc3 = ax5.scatter([], [], s=40)
    ax5.set_xlim([-0.001, 0.001])
    ax5.set_ylim([1, 1.01])
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.set_title('Scene')
    txt1 = ax5.text(ax5.get_xlim()[0], ax5.get_ylim()[1], f'Simulation time: 0 s', horizontalalignment='left', verticalalignment='top')


    # set up animation
    plt.tight_layout()
    def animate(i):
        line.set_ydata(signals[0, i].real)  # update the data.
        line2.set_ydata(ffts[0, i])

        line3.set_xdata(t1[:i])
        line3.set_ydata(freqzs[0, :i, 0, 1])
        line4.set_xdata(t1[:i])
        line4.set_ydata(fft_freqzs[0, :i, 0, 1])
        sc3.set_offsets(path[i])
        
        txt1.set_text((f'Simulation time: {i/fps:.2f} s'))
        return line, line2, line3, line4, sc3

    fig.set_size_inches(16, 9)
    ani = animation.FuncAnimation(fig, animate, steps, interval=20, blit=False, save_count=50)
    # ani.save(f'{Path(__file__).stem}.mp4', fps=fps, dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
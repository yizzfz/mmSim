import numpy as np
from .motionbase import MotionBase
import sys

class Oscillation(MotionBase):
    """A sinusoid motion"""
    def __init__(self, vector, freq, **kwargs):
        """
        Parameters:
            name: name of the motion.
            delay: start the motion after `delay` seconds.
            vector: a vector of size 3, defines the direction and amplitude of the motion.
            freq: frequency of the oscillation.
        """
        super().__init__(**kwargs)
        self.vector = np.array(vector)
        self.freq = freq

    def make_path(self):
        t = np.arange(0, self.t, 1/self.fps)
        A = (np.cos(2*np.pi*self.freq*t+np.pi)+1)/2
        self.path = np.zeros((self.steps, 3))
        for i in range(self.steps):
            self.path[i] = self.vector*A[i]

class Pulse(Oscillation):
    """A periodical pulse motion"""
    def __init__(self, vector, freq, pulse_length=0.1, **kwargs):
        """
        Parameters:
            name: name of the motion.
            delay: start the motion after `delay` seconds.
            vector: a vector of size 3, defines the direction and amplitude of the pulse.
            freq: frequency of the pulse.
            pulse_length: length of pulse in seconds.
        """        
        super().__init__(vector, freq, **kwargs)
        self.pulse_length = pulse_length

    def make_path(self):
        motion_length = int(self.pulse_length * self.fps)   # how many samples a pulse would take
        pattern = self.make_pattern(motion_length) 
        n_pulse = int(self.t*self.freq)                     # how many pulses to make based on frequency
        interval = int(1/self.freq*self.fps)                # how many samples (a pulse + reset) would take
        if interval < motion_length:
            print(f'[{self.name}] Error, frequency too high or pulse too big')
            sys.exit(1)
        self.path = np.zeros((self.steps, 3))
        for i in range(n_pulse):
            j = i*interval
            self.path[j:j+motion_length] = pattern * self.vector

    # make a two sided pulse, pulse shape hardcoded as below
    # default pulse length is 0.1s
    # peak top = 1.8x peak bottom
    # 0 to max : max to min : min to 0 = 6:5:4
    def make_pattern(self, length):
        P = np.zeros(length)
        p1 = int(np.round(length*6/15))
        p2 = int(np.round(length*5/15))
        p3 = int(np.round(length*4/15))
        P[:p1] = np.arange(0, 1, 1/p1)
        P[p1:p1+p2] = np.arange(1, -0.55, -1.55/p2)
        P[p1+p2:] = np.arange(-0.55, 0, 0.55/p3)
        return np.repeat(P, 3).reshape((length, 3))

def make_heart_pulse(total_length, pulse_length):
    T = np.zeros(total_length)
    P = np.zeros(pulse_length)
    p1 = int(np.round(pulse_length*6/15))
    p2 = int(np.round(pulse_length*5/15))
    p3 = int(np.round(pulse_length*4/15))
    P[:p1] = np.arange(0, 1, 1/p1)
    P[p1:p1+p2] = np.arange(1, -0.55, -1.55/p2)
    P[p1+p2:] = np.arange(-0.55, 0, 0.55/p3)
    border = int((total_length-pulse_length)/2)
    T[border:-border] = P
    return T
    
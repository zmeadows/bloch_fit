import sys
sys.path.append('/Users/zac/Research/muon_g2_2015/nmr/bloch_fit')

import matplotlib.pyplot as plt
from pulse import NMRPulseFiltered
import util as u
import numpy as np
import math

p = NMRPulseFiltered(u.get_pulse_path())

d_hz = p.raw_freqs[1] - p.raw_freqs[0]
ei = (p.w_ref * 8 / (2*math.pi)) / d_hz

plt.figure(facecolor=u.LIGHTGREY)
plt.plot(p.raw_freqs[:ei], np.abs(p.raw_fft)[:ei], color=u.BLUE)
plt.plot(p.raw_freqs[:ei], np.abs(p.filter_fft)[:ei], color=u.RED, alpha=0.7)
plt.grid()
plt.gca().set_axis_bgcolor(u.GREY)
plt.yscale('log')
plt.title("FILTERED PULSE FFT")
plt.xlabel("Frequency (Hz)")
plt.ylim(np.min(0.99*np.abs(p.raw_fft)), 1.1*np.max(np.abs(p.raw_fft)))

plt.show()

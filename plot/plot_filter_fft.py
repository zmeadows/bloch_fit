import matplotlib.pyplot as plt
from pulse import NMRPulseFiltered
import util as u
import numpy as np

p = NMRPulseFiltered(u.get_pulse_path())

plt.figure(facecolor=u.LIGHTGREY)
plt.plot(p.raw_freqs, np.abs(p.raw_fft), color=u.BLUE)
plt.plot(p.raw_freqs, np.abs(p.filter_fft), color=u.RED, alpha=0.7)
plt.grid()
plt.gca().set_axis_bgcolor(u.GREY)
plt.yscale('log')
plt.title("FILTERED PULSE FFT")
plt.xlabel("Frequency (Hz)")

plt.show()

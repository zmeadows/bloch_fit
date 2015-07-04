import matplotlib.pyplot as plt
from pulse import NMRPulseFFT
import util as u
import numpy as np

p = NMRPulseFFT(u.get_pulse_path(), fft_time_cut = 35e-3)

print p.fft_freq

plt.figure(facecolor=u.LIGHTGREY)
plt.plot(p.raw_freqs, np.abs(p.raw_fft), color=u.BLUE)
plt.plot(p.fft_fit_freqs, p.fft_fit, color=u.RED, alpha = 0.7)
plt.grid()
plt.gca().set_axis_bgcolor(u.GREY)
plt.yscale('log')
plt.title("RAW PULSE FFT")
plt.xlabel("Frequency (Hz)")

plt.show()

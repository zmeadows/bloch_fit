import matplotlib.pyplot as plt
from pulse import NMRPulseFullFit
import util as u
import math
import numpy as np

p = NMRPulseFullFit(u.get_pulse_path(), full_fit_use_filter = False,
                    full_fit_cut = 0.0, full_fit_stop = 20e-3,
                    w_ref = 2*math.pi*9681, time_since_pi2_pulse = 132e-6)

print p.fit_freq
print p.fit_rms, p.fit_chisq


plt.figure(facecolor=u.LIGHTGREY)
plt.plot(p.raw_times, p.raw_signal, color=u.BLUE, label="raw signal")
plt.plot(p.best_fit_times, p.best_fit, color=u.GREEN, label="fit")
plt.plot(p.best_fit_times, p.fit_residuals, color=u.RED, label="residuals")
plt.legend(loc="lower right")
plt.grid()
plt.gca().set_axis_bgcolor(u.GREY)
plt.xlabel("Time (s)")
plt.ylabel("PHASE (rad)")
plt.xlim(0.0, p.best_fit_times[-1])

plt.figure(facecolor=u.LIGHTGREY)
plt.plot(p.raw_freqs, np.abs(p.raw_fft), color=u.BLUE, label="raw signal")
plt.plot(p.raw_freqs, np.abs(p.fit_fft), color=u.GREEN, label="fit")
plt.plot(p.raw_freqs, np.abs(p.fit_residuals_fft), color=u.RED, label="residuals")
plt.legend(loc="lower right")
plt.grid()
plt.yscale("log")
plt.gca().set_axis_bgcolor(u.GREY)
plt.xlabel("Frequency (Hz)")
plt.xlim(0.0, 8*p.w_ref/(2*math.pi))

plt.figure(facecolor=u.LIGHTGREY)
plt.plot(p.best_fit_times, p.freq_fit, color="#4f4f4f")
plt.grid()
plt.gca().set_axis_bgcolor(u.GREY)
plt.title("FREQUENCY VS. TIME FIT")
plt.xlabel("Time (s)")
plt.ylabel("FREQUENCY (rad)")
plt.xlim(0.0, p.best_fit_times[-1])

plt.show()


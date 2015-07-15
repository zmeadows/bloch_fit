import sys
sys.path.append('/Users/zac/Research/muon_g2_2015/nmr/bloch_fit')

import matplotlib.pyplot as plt
from pulse import NMRPulseFullFit
import util as u
import math
import numpy as np

p = NMRPulseFullFit(u.get_pulse_path(), full_fit_use_filter = False,
                    full_fit_stop = 20e-3, w_ref = 2*math.pi*9685, init_signal_cut = 2.5e-4,
                    time_since_pi2_pulse = 132e-6, debug=True, fit_harmonics=True)

plt.figure(facecolor="#ffffff")
plt.plot(p.raw_times, p.raw_signal, color="#3f3f3f", label="raw signal")
plt.legend(loc="lower right", prop={'size':20})
plt.grid()
plt.gca().set_axis_bgcolor(u.GREY)
plt.xlabel("Time (sec)", fontsize=18)
plt.ylabel("Voltage (V)", fontsize=18)
plt.xlim(0.0, p.best_fit_times[-1])

plt.figure(facecolor="#ffffff")
plt.plot(p.raw_times, p.raw_signal, color="#3f3f3f", label="raw signal")
plt.plot(p.best_fit_times, p.best_fit, color="#ff0000", label="fit")
plt.plot(p.best_fit_times, p.fit_residuals, color=u.BLUE, label="residuals")
plt.legend(loc="lower right", prop={'size':20})
plt.grid()
plt.gca().set_axis_bgcolor(u.GREY)
plt.xlabel("Time (sec)", fontsize=18)
plt.ylabel("Voltage (V)", fontsize=18)
plt.xlim(0.0, p.best_fit_times[-1])

plt.figure(facecolor="#ffffff")
plt.plot(p.raw_freqs, np.abs(p.raw_fft), color="#3f3f3f", label="raw signal")
plt.legend(loc="lower right", prop={'size':20})
plt.grid()
plt.yscale("log")
plt.gca().set_axis_bgcolor(u.GREY)
plt.xlabel("Frequency (Hz)", fontsize=18)
plt.ylim(1e-1, 1e5)
plt.xlim(0.0, 10*p.w_ref/(2*math.pi))

plt.figure(facecolor="#ffffff")
plt.plot(p.raw_freqs, np.abs(p.raw_fft), color="#3f3f3f", label="raw signal")
plt.plot(p.raw_freqs, np.abs(p.fit_fft), color="#ff0000", label="fit")
plt.plot(p.raw_freqs, np.abs(p.fit_residuals_fft), color=u.BLUE, label="residuals", alpha=0.5)
plt.legend(loc="lower right", prop={'size':20})
plt.grid()
plt.yscale("log")
plt.gca().set_axis_bgcolor(u.GREY)
plt.xlabel("Frequency (Hz)", fontsize=18)
plt.ylim(1e-1, 1e5)
plt.xlim(0.0, 10*p.w_ref/(2*math.pi))

plt.show()


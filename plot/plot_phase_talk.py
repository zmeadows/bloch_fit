import sys
sys.path.append('/Users/zac/Research/muon_g2_2015/nmr/bloch_fit')

import matplotlib.pyplot as plt
from pulse import NMRPulsePhaseFit
import util as u
import math

p = NMRPulsePhaseFit(u.get_pulse_path(),
                     zc_use_filter = True,
                     init_signal_cut = 4.0e-4,
                     zc_stop = 20e-3,
                     debug=True,
                     w_ref = 2*math.pi*9685)

plt.figure(facecolor="#ffffff")
plt.plot(p.phase_times, (p.phase_data - p.w_ref*p.phase_times) - p.phase_fit.best_values['d'], linestyle="None", marker=".", markeredgecolor = "#4f4f4f")
#plt.plot(p.phase_times, (p.phase_fit.best_fit - p.w_ref*p.phase_times) - p.phase_fit.best_values['d'], color="#ff0000", linewidth = 2.0)
plt.grid()
plt.gca().set_axis_bgcolor(u.LIGHTGREY)
plt.ylim(-0.6, 0.05)
plt.xlim(0, 20e-3)
plt.xlabel("Time (s)")
plt.ylabel("PHASE (rad)")

plt.show()


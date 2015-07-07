import sys
sys.path.append('/Users/zac/Research/muon_g2_2015/nmr/bloch_fit')

import matplotlib.pyplot as plt
from pulse import NMRPulsePhaseFit
import util as u
import math
import numpy as np

p = NMRPulsePhaseFit(u.get_pulse_path(),
                     phase_fit_use_filter = True,
                     init_signal_cut = 4.0e-4,
                     phase_fit_stop = 20e-3,
                     debug=True,
                     w_ref = 2*math.pi*9685)

plt.figure(facecolor=u.LIGHTGREY)
plt.plot(p.zcs, np.zeros(p.zcs.size), linestyle="None", marker=".", markeredgecolor = "#ff0000", markerfacecolor="#ff0000", markersize=20)
plt.plot(p.raw_times, p.raw_signal, color="#4f4f4f")
plt.grid()
plt.gca().set_axis_bgcolor(u.GREY)
plt.title("PHASE VS. TIME FIT")
plt.xlabel("Time (s)")
plt.ylabel("PHASE (rad)")

plt.show()


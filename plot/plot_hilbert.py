import sys
sys.path.append('/Users/zac/Research/muon_g2_2015/nmr/bloch_fit')

import matplotlib.pyplot as plt
from pulse import NMRHilbertFit
import util as u
import math
import numpy as np

p = NMRHilbertFit(u.get_pulse_path(),
                     init_signal_cut = 4.0e-4,
                     debug=True,
                     hilbert_stop = 20e-3,
                     hilbert_cut = 2.5e-4,
                     w_ref = 2*math.pi*9685)

offset = p.hilbert_phase_nonlinear[0]

plt.figure(facecolor="#ffffff")
plt.plot(p.hilbert_times, p.hilbert_phase_nonlinear - offset, color=u.BLUE, linestyle="None", marker=".", markevery=20)
plt.plot(p.hilbert_times, p.hilbert_fit_nonlinear - offset, color="#ff0000", linewidth = 3.0)
plt.grid()
plt.gca().set_axis_bgcolor(u.LIGHTGREY)
plt.ylim(-0.7, 0.1)
plt.xlim(0, p.hilbert_times[-1])
plt.xlabel("Time (s)", fontsize=16)
plt.ylabel("Phase (rad)", fontsize=16)

plt.show()


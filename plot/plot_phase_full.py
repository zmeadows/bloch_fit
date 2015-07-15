import sys
sys.path.append('/Users/zac/Research/muon_g2_2015/nmr/bloch_fit')

import matplotlib.pyplot as plt
from pulse import NMRPulseFullFit, NMRPulsePhaseFit
import util as u
import math

data_path = u.get_pulse_path()

pf = NMRPulseFullFit(data_path, full_fit_use_filter = False,
                    full_fit_stop = 12e-3, w_ref = 2*math.pi*9685, init_signal_cut = 2.5e-4,
                    time_since_pi2_pulse = 132e-6, debug=True, fit_harmonics=True)


pp = NMRPulsePhaseFit(data_path,
                     zc_use_filter = True,
                     init_signal_cut = 4.0e-4,
                     zc_stop = 12e-3,
                     debug=True,
                     w_ref = 2*math.pi*9685)

plt.figure(facecolor="#ffffff")
plt.plot(pp.phase_times, pp.phase_freq_vs_time/(2*math.pi), color=u.BLUE, linewidth = 2.0, label="phase")
plt.plot(pf.best_fit_times, pf.freq_fit, color="#4f4f4f", label="full")
plt.grid()
plt.gca().set_axis_bgcolor(u.LIGHTGREY)
#plt.ylim(-0.6, 0.05)
#plt.xlim(0, 20e-3)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

plt.show()


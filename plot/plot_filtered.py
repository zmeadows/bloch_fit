import sys
sys.path.append('/Users/zac/Research/muon_g2_2015/nmr/bloch_fit')

import matplotlib.pyplot as plt
from pulse import NMRPulseFiltered
import util as u

p = NMRPulseFiltered(u.get_pulse_path())

plt.figure(facecolor=u.LIGHTGREY)
plt.plot(p.filter_times, p.filter_signal, color=u.BLUE)
plt.grid()
plt.gca().set_axis_bgcolor(u.GREY)
plt.title("FILTERED PULSE SIGNAL")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")

plt.show()

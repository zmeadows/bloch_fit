import sys
sys.path.append('/Users/zac/Research/muon_g2_2015/nmr/bloch_fit')

import matplotlib.pyplot as plt
from pulse import NMRPulse
import util as u

p = NMRPulse(u.get_pulse_path())

plt.figure(facecolor=u.LIGHTGREY)
plt.plot(p.raw_times, p.raw_signal, color=u.BLUE)

plt.grid()
plt.gca().set_axis_bgcolor(u.GREY)
plt.xlim(p.raw_times[0], p.raw_times[-1])
plt.ylim(-0.8,0.8)
plt.title("RAW PULSE SIGNAL")
plt.ylabel("Voltage (V)")
plt.xlabel("Time (s)")

plt.show()

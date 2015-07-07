import sys
sys.path.append('/Users/zac/Research/muon_g2_2015/nmr/bloch_fit')

import matplotlib.pyplot as plt
import numpy as np

files = [ i + ".dat" for i in ["17e-3"] ]

results = [ np.loadtxt(i) for i in files ]
pulse_nums = results[0][:,1]
ff_freqs = results[0][:,2]
p_freqs = results[0][:,6]
pf_freqs = results[0][:,8]

plt.figure(facecolor="#efefef")
plt.plot(pulse_nums, pf_freqs - ff_freqs, color="blue", linestyle="None", marker=".", label="PF - FF")
plt.plot(pulse_nums, p_freqs - ff_freqs, color="red", linestyle="None", marker=".", label="P - FF")
plt.grid()
plt.gca().set_axis_bgcolor("#bfbfbf")
plt.ylim(-5,5)
plt.xlabel("Pulse Number")
plt.ylabel("deviation from full fit frequency")
plt.legend(loc="lower right")

plt.show()


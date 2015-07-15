import sys
sys.path.append('/Users/zac/Research/muon_g2_2015/nmr/bloch_fit')

import matplotlib.pyplot as plt
import util as u
import numpy as np
import math

times = np.linspace(0, 60e-3, 50000)
signal = np.exp(-times / 10e-3) * np.cos(2*math.pi*10e3 * times)

plt.figure(facecolor="#ffffff")
plt.plot(times * 1e3, signal, color="#4f4f4f")

plt.grid()
plt.gca().set_axis_bgcolor(u.GREY)
plt.ylabel("Voltage (V)", fontsize=17)
plt.xlabel("Time (ms)", fontsize=17)
plt.xlim(0,20)
plt.show()

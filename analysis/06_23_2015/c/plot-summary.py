import numpy as np
import matplotlib.pyplot as plt
import sys

dur_ms, freqs, stddev = np.loadtxt(sys.argv[1], delimiter=' ', usecols=(0, 1, 2), unpack=True)

plt.figure(facecolor="#e8e8e8")
plt.gca().set_axis_bgcolor("#d7d7d7")
plt.grid()
plt.xlim(3,12)
#plt.ylim(21382.5,21383.75)
plt.errorbar(dur_ms, freqs - 10011, yerr=stddev/2, linestyle="None")
plt.plot(dur_ms, freqs - 10011, linestyle="None", marker='.', color="#ef0506", markersize=10.0)
plt.xlabel("fit duration (ms)")
plt.ylabel("deviation from 10011 Hz")
plt.title("PULSE FIT TIME REGION COMPARISON")

plt.show()



import numpy as np
import matplotlib.pyplot as plt
import sys

dur_ms, freqs, stddev = np.loadtxt(sys.argv[1], delimiter=' ', usecols=(0, 1, 2), unpack=True)

plt.figure(facecolor="#e8e8e8")
plt.gca().set_axis_bgcolor("#d7d7d7")
plt.grid()
plt.xlim(3,11)
#plt.ylim(21382.5,21383.75)
plt.errorbar(dur_ms, freqs - 21383, yerr=stddev, linestyle="None")
plt.plot(dur_ms, freqs - 21383, linestyle="None", marker='.', color="#ef0506", markersize=10.0)
plt.xlabel("fit duration (ms)")
plt.ylabel("deviation from 21383 Hz")
plt.title("PULSE FIT TIME REGION COMPARISON")

plt.show()



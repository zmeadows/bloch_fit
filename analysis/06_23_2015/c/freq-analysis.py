import numpy as np
import sys

freqs = np.loadtxt(sys.argv[1], delimiter=' ', usecols=[0], unpack=True)

print sys.argv[1], str(np.mean(freqs)), str(np.std(freqs))


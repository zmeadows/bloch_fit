import numpy as np
import sys

freqs, chis, rmss = np.loadtxt(sys.argv[1], delimiter=' ', usecols=(0, 5, 6), unpack=True)

print sys.argv[1], str(np.mean(freqs)), str(np.std(freqs)), str(np.mean(chis)), str(np.mean(rmss))


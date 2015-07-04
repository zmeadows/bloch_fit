import sys
sys.path.append('/Users/zac/Research/muon_g2_2015/nmr/bloch_fit')

from pulse import NMRPulseFullFit, NMRPulsePhaseFit
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import math

fit_duration = float(sys.argv[1])
output_file = sys.argv[1] + ".dat"

file_codes = [ (i,j) for i in range(8,9) for j in range(1,100) if (j % 5 != 0)]

num_cores = multiprocessing.cpu_count()

def get_freq(run_num, pulse_num):
    base_dir = "/Users/zac/Research/muon_g2_2015/nmr/candella-lab-pulse-data/06_29_15/run-"
    filepath = base_dir + str(run_num) + "/" + str(pulse_num) + ".bin"

    full_fit_unfiltered = NMRPulseFullFit(filepath, full_fit_use_filter = False,
                         full_fit_cut = 2.5e-4, full_fit_stop = fit_duration,
                         time_since_pi2_pulse = 132e-6, w_ref = 2*math.pi*9685.5)

    phase_fit_unfiltered = NMRPulsePhaseFit(filepath, phase_fit_use_filter = False,
                          phase_fit_stop = fit_duration, time_since_pi2_pulse = 132e-6,
                          w_ref = 2*math.pi*9685.5)

    phase_fit_filtered = NMRPulsePhaseFit(filepath, phase_fit_use_filter = True,
                          phase_fit_stop = fit_duration, time_since_pi2_pulse = 132e-6,
                          w_ref = 2*math.pi*9685.5)

    return run_num, pulse_num, \
           full_fit_unfiltered.fit_freq, \
           full_fit_unfiltered.fit_chisq, \
           full_fit_unfiltered.fit_rms, \
           full_fit_unfiltered.std_dev, \
           phase_fit_unfiltered.phase_freq, \
           phase_fit_unfiltered.phase_freq - full_fit_unfiltered.fit_freq, \
           phase_fit_filtered.phase_freq, \
           phase_fit_filtered.phase_freq - full_fit_unfiltered.fit_freq

results = Parallel(n_jobs=num_cores - 1, verbose=50)(delayed(get_freq)(i,j) for (i,j) in file_codes)
results = np.asarray(results)

info_header = "FIT DURATION: " + str(fit_duration) + "\n"
column_header = "RUN\t PULSE\t FULL FIT FREQ\t FULL FIT CHISQ\t FULL FIT RMS\t NOISE\t\t PHASE FREQ\t PHASE - FULL\t PHASE-FILT FREQ\t PHASE-FILT - FULL"
np.savetxt(output_file, results, fmt='%d\t %d\t %.3f\t %.5f\t %.4e\t %.4e\t %.5f\t %.5f\t %.5f\t\t %.5f\t', header=info_header + column_header)

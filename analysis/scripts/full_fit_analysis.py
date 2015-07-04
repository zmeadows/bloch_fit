from pulse import NMRPulseFullFit
from joblib import Parallel, delayed
import multiprocessing
import os.path

cut = float(raw_input("full fit duration: "))
output_file = raw_input("output file: ")


file_codes = [ (i,j) for i in range(8,9) for j in range(1,5) ]

num_cores = multiprocessing.cpu_count()

def get_freq(run_num, pulse_num):
    base_dir = "/Users/zac/Research/muon_g2_2015/nmr/candella-lab-pulse-data/06_22_15/run-"
    filepath = base_dir + str(run_num) + "/" + str(pulse_num) + ".dat"
    if os.path.isfile(filepath):
        p = NMRPulseFullFit(filepath, full_fit_use_filter = True, full_fit_cut = cut)
        return run_num, pulse_num, p.fit_freq, p.fit_chisq, p.fit_rms

results = Parallel(n_jobs=num_cores - 1, verbose=50)(delayed(get_freq)(i,j) for (i,j) in file_codes)

text_file = open(output_file, "w")
for i in results:
    text_file.write(str(i) + "\n")
text_file.close()

from pulse import NMRPulsePhaseFit
from joblib import Parallel, delayed
import multiprocessing
import os.path

cut = float(raw_input("phase fit duration: "))
output_file = raw_input("output file: ")

files = []
for i in range(8,9):
    for j in range(1,16):
        files.append("/Users/zac/Research/muon_g2_2015/nmr/candella-lab-pulse-data/06_22_15/run-" + str(i) + "/" + str(j) + ".dat")

num_cores = multiprocessing.cpu_count()

def get_freq(filepath):
    if os.path.isfile(filepath):
        p = NMRPulsePhaseFit(filepath, phase_fit_use_filter = True, phase_fit_cut = cut)
        return p.phase_freq

results = Parallel(n_jobs=num_cores - 1, verbose=50)(delayed(get_freq)(i) for i in files)

text_file = open(output_file, "w")
for i in results:
    text_file.write(str(i) + "\n")
text_file.close()

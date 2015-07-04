import numpy as np

files = [ i + ".dat" for i in ["10e-3", "12e-3", "14e-3", "16e-3", "18e-3", "20e-3", "22e-3"] ]

results = [ np.loadtxt(i) for i in files ]

summary_file = open("summary.dat", "w")
for data in results:
    summary_file.write(str(np.mean(data[:,2])))
    summary_file.write(" ")
    summary_file.write(str(np.std(data[:,2])))
    summary_file.write(" ")
    summary_file.write(str(np.mean(data[:,3])))
    summary_file.write(" ")
    summary_file.write(str(np.mean(data[:,6])))
    summary_file.write(" ")
    summary_file.write(str(np.std(data[:,6])))
    summary_file.write(" ")
    summary_file.write(str(np.mean(data[:,7])))
    summary_file.write(" ")
    summary_file.write(str(np.mean(data[:,10])))
    summary_file.write(" ")
    summary_file.write(str(np.std(data[:,10])))
    summary_file.write(" ")
    summary_file.write(str(np.mean(data[:,11])))
    summary_file.write(" ")
    summary_file.write(str(np.std(data[:,11])))
    summary_file.write(" ")
    summary_file.write(str(np.mean(data[:,12])))
    summary_file.write(" ")
    summary_file.write(str(np.std(data[:,10] - data[:,2])))
    summary_file.write(" ")
    summary_file.write(str(np.std(data[:,11] - data[:,3])))
    summary_file.write("\n")



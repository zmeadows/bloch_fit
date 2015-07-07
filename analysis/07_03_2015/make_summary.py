import numpy as np

files = [ i + ".dat" for i in ["17e-3"] ]

results = [ np.loadtxt(i) for i in files ]

summary_file = open("summary.dat", "w")
summary_file.write("FIT DURATION\tFF FREQ\t\tFF STD\t\tFF CHISQ\tP FREQ\t\tP FREQ STD\tP - FF FREQ\tP - FF STD\tPF FREQ\t\tPF FREQ STD\tPF - FF\t\tPF - FF STD \n")
i = 0
for data in results:
    summary_file.write(files[i][:-4])
    i += 1
    summary_file.write("\t\t")
    summary_file.write('%.4f' % np.mean(data[:,2]))
    summary_file.write("\t")
    summary_file.write('%.4f' % np.std(data[:,2]))
    summary_file.write("\t\t")
    summary_file.write('%.4f' % np.mean(data[:,3]))
    summary_file.write("\t\t")
    summary_file.write('%.4f' % np.mean(data[:,6]))
    summary_file.write("\t")
    summary_file.write('%.4f' % np.std(data[:,6]))
    summary_file.write("\t\t")
    summary_file.write('%.4f' % np.mean(data[:,7]))
    summary_file.write("\t\t")
    summary_file.write('%.4f' % np.std(data[:,7]))
    summary_file.write("\t\t")
    summary_file.write('%.4f' % np.mean(data[:,8]))
    summary_file.write("\t")
    summary_file.write('%.4f' % np.std(data[:,8]))
    summary_file.write("\t\t")
    summary_file.write('%.4f' % np.mean(data[:,9]))
    summary_file.write("\t\t")
    summary_file.write('%.4f' % np.std(data[:,9]))
    summary_file.write("\n")



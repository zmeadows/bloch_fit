import numpy as np

results = np.loadtxt("17e-3.dat")

summary_file = open("summary.dat", "w")
summary_file.write("P FREQ\t\tP STD\t\tPF FREQ\t\tPF STD\t\tPF - P\t\tPF - P STD\n")
i = 0
summary_file.write('%.4f' % np.mean(results[:,2]))
summary_file.write("\t")
summary_file.write('%.4f' % np.std(results[:,2]))
summary_file.write("\t\t")
summary_file.write('%.4f' % np.mean(results[:,3]))
summary_file.write("\t")
summary_file.write('%.4f' % np.std(results[:,3]))
summary_file.write("\t\t")
summary_file.write('%.4f' % np.mean(results[:,4]))
summary_file.write("\t\t")
summary_file.write('%.4f' % np.std(results[:,4]))
summary_file.write("\n")



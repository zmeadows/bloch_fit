import numpy as np
import matplotlib.pyplot as plt

sf = np.loadtxt("summary.dat")

times = [10e-3, 12e-3, 14e-3, 16e-3, 18e-3, 20e-3, 22e-3]

full_fit_freqs = 9685 - sf[:,0]
full_fit_std = sf[:,1]/2

full_fit_freqs_filt = 9685 - sf[:,3]
full_fit_std_filt = sf[:,4]/2

phase_fit_freqs = 9685 - sf[:,6]
phase_fit_std = sf[:,7]/2

phase_fit_freqs_filt = 9685 - sf[:,8]
phase_fit_std_filt = sf[:,9]/2

plt.figure(facecolor="#E8E8E8")
plt.plot(times, full_fit_freqs, color = "green", linestyle="None", marker=".", markersize=10.0)
plt.errorbar(times, full_fit_freqs, yerr=full_fit_std, linestyle="None", color="green", alpha=0.5)
plt.grid()
plt.gca().set_axis_bgcolor("#d7d7d7")
plt.title("FULL PULSE FIT UNFILTERED")
plt.xlabel("Time (s)")
plt.ylabel("deviation from 9865 Hz")
plt.xlim(8e-3, 24e-3)

plt.figure(facecolor="#E8E8E8")
plt.plot(times, full_fit_freqs_filt, color = "blue", linestyle="None", marker=".", markersize=10.0)
plt.errorbar(times, full_fit_freqs_filt, yerr=full_fit_std, linestyle="None", color="blue", alpha=0.5)
plt.grid()
plt.gca().set_axis_bgcolor("#d7d7d7")
plt.title("FULL PULSE FIT FILTERED")
plt.xlabel("Time (s)")
plt.ylabel("deviation from 9865 Hz")
plt.xlim(8e-3, 24e-3)


plt.figure(facecolor="#E8E8E8")
plt.plot(times[:-1], phase_fit_freqs[:-1], color = "red", linestyle="None", marker=".", markersize=10.0)
plt.errorbar(times[:-1], phase_fit_freqs[:-1], yerr=full_fit_std_filt[:-1], linestyle="None", color="red", alpha=0.5)
plt.grid()
plt.gca().set_axis_bgcolor("#d7d7d7")
plt.title("PHASE FIT UNFILTERED")
plt.xlabel("Time (s)")
plt.ylabel("deviation from 9865 Hz")
plt.xlim(8e-3, 24e-3)


plt.figure(facecolor="#E8E8E8")
plt.plot(times[:-1], phase_fit_freqs_filt[:-1], color = "purple", linestyle="None", marker=".", markersize=10.0)
plt.errorbar(times[:-1], phase_fit_freqs_filt[:-1], yerr=phase_fit_std_filt[:-1], linestyle="None", color="purple", alpha=0.5)
plt.grid()
plt.gca().set_axis_bgcolor("#d7d7d7")
plt.title("PHASE FIT FILTERED")
plt.xlabel("Time (s)")
plt.ylabel("deviation from 9865 Hz")
plt.xlim(8e-3, 24e-3)

plt.figure(facecolor="#E8E8E8")
plt.plot(times, full_fit_freqs, color = "green", linestyle="None", marker=".", markersize=10.0)
plt.errorbar(times, full_fit_freqs, yerr=full_fit_std, linestyle="None", color="green", alpha=0.5)
plt.plot(times, full_fit_freqs_filt, color = "blue", linestyle="None", marker=".", markersize=10.0)
plt.errorbar(times, full_fit_freqs_filt, yerr=full_fit_std, linestyle="None", color="blue", alpha=0.5)
plt.plot(times[:-1], phase_fit_freqs[:-1], color = "red", linestyle="None", marker=".", markersize=10.0)
plt.errorbar(times[:-1], phase_fit_freqs[:-1], yerr=full_fit_std_filt[:-1], linestyle="None", color="red", alpha=0.5)
plt.plot(times[:-1], phase_fit_freqs_filt[:-1], color = "purple", linestyle="None", marker=".", markersize=10.0)
plt.errorbar(times[:-1], phase_fit_freqs_filt[:-1], yerr=phase_fit_std_filt[:-1], linestyle="None", color="purple", alpha=0.5)
plt.grid()
plt.gca().set_axis_bgcolor("#d7d7d7")
plt.title("FREQUENCY DETERMINATION METHODS")
plt.xlabel("Time (s)")
plt.ylabel("deviation from 9865 Hz")
plt.xlim(8e-3, 24e-3)

plt.show()

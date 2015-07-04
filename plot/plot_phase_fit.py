import matplotlib.pyplot as plt
from pulse import NMRPulsePhaseFit
import util as u

p = NMRPulsePhaseFit(u.get_pulse_path(), phase_fit_use_filter = True, phase_fit_stop = 22e-3)

plt.figure(facecolor=u.LIGHTGREY)
plt.plot(p.phase_times, p.phase_data - p.w_ref*p.phase_times, linestyle="None", marker=".", markeredgecolor = "#4f4f4f")
plt.plot(p.phase_times, p.phase_fit.best_fit - p.w_ref*p.phase_times, color=u.RED)
plt.grid()
plt.gca().set_axis_bgcolor(u.GREY)
plt.title("PHASE VS. TIME FIT")
plt.xlabel("Time (s)")
plt.ylabel("PHASE (rad)")

plt.show()


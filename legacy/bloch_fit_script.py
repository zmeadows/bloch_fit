import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import lmfit

#######################
### CONSTANTS/FLAGS ###
#######################

PRINT_DIAGNOSTICS      = False
PLOT_FIT_RESULTS       = False
PLOT_BASIC             = False

RAW_FIT                = False
FILTER_FIT             = False
PHASE_FIT              = True

FIT_START              = 0.0
FIT_END                = 6.0e-3

SAMPLING_PERIOD        = 1e-7
TIME_SINCE_PI2_PULSE   = 0.0

FILTER_EDGE_SLICE_TIME = 1.5e-4
RAW_SIGNAL_CUT         = 20e-3

RAW_STD_DEV            = 2.3e-3
IF = 10011

assert(RAW_SIGNAL_CUT > FIT_END)

#########################
### UTILITY FUNCTIONS ###
#########################

def rms(arr):
    assert(arr.size > 1)
    return math.sqrt(np.mean(np.square(arr)))

def red_chi_sq(data, fit, sigma):
    assert(data.size == fit.size)
    ress = np.sum((fit - data)**2)
    chi_sq = ress / sigma**2
    return chi_sq / data.size

############################
### LOAD PULSE FROM FILE ###
############################

def adc_to_voltage(adc_value):
    return (adc_value - 32599.9) / 30465.9

def load_pulse_to_arrays(filename):
    time_indices, signal_data = np.loadtxt(filename, delimiter=' ', usecols=(0, 1), unpack=True)
    assert(time_indices.size > 0)
    assert(time_indices.size == signal_data.size)
    scaled_data = np.vectorize(adc_to_voltage)(signal_data)
    shifted_scaled_data = scaled_data - np.average(scaled_data[3*scaled_data.size/4:])
    return (time_indices*SAMPLING_PERIOD) + TIME_SINCE_PI2_PULSE, shifted_scaled_data

dat_times, dat_signal = load_pulse_to_arrays(sys.argv[1])

#########################
### FOURIER TRANSFORM ###
#########################

# cut out noisy tail, for fourier transform
raw_signal = dat_signal[: RAW_SIGNAL_CUT / SAMPLING_PERIOD]

num_fft_samples = raw_signal.size * 5 # for zero-padding

raw_fft = np.fft.rfft(raw_signal, num_fft_samples)
raw_freqs = np.fft.rfftfreq(num_fft_samples, SAMPLING_PERIOD)

##############
### FILTER ###
##############

def step_filter(freq):
    return 1 / ( np.exp( (freq - 40e3) / 8e3 ) + 1 )

filter_fft = np.multiply(step_filter(raw_freqs), raw_fft)
filter_signal = np.fft.irfft(filter_fft)[: raw_signal.size]

############
### CUTS ###
############

rsi = FIT_START / SAMPLING_PERIOD
rei = FIT_END / SAMPLING_PERIOD

fsi =  FIT_START / SAMPLING_PERIOD + FILTER_EDGE_SLICE_TIME / SAMPLING_PERIOD
fei =  FIT_END / SAMPLING_PERIOD - FILTER_EDGE_SLICE_TIME / SAMPLING_PERIOD

raw_signal = raw_signal[rsi:rei]
raw_times = dat_times[rsi:rei]

filter_signal = filter_signal[fsi:fei]
filter_times = raw_times[fsi:fei]

#####################
### PULSE FITTING ###
#####################

def omega(t, w, dw_1, dw_2, dw_3, dw_4):
    ts = t / RAW_SIGNAL_CUT
    return w*(1 + dw_1 * ts**4 + dw_2 * ts**6 + dw_3 * ts**8 + dw_4 * ts**10)

def harmonics2_func(t, A_0 = 0.85,   w_0 = IF,   phi_0 = 0,
                 A_1 = -0.002, w_1 = 2*IF, phi_1 = 0,
                 A_2 = -0.001,  w_2 = 3*IF, phi_2 = 0,
                 dw_1 = -0.3, dw_2 = 2.0, dw_3 = -3.0, dw_4 = 0.001):
    return A_0*np.cos(2 * math.pi * g(t, w_0, dw_1, dw_2, dw_3, dw_4)*t + phi_0) \
           + A_1*np.cos(2*math.pi * g(t, w_1, dw_1, dw_2, dw_3, dw_4)*t + phi_1) \
           + A_2*np.cos(2*math.pi * g(t, w_2, dw_1, dw_2, dw_3, dw_4)*t + phi_2)

def harmonics3_func(t, A_0 = 0.5,   w_0 = IF,   phi_0 = 0,
                 A_1 = -5e-2, w_1 = 2*IF, phi_1 = 0,
                 A_2 = 1e-2,  w_2 = 3*IF, phi_2 = 0,
                 A_3 = -2e-2, w_3 = 4*IF, phi_3 = 0,
                 dw_1 = 1.0e-3,  dw_2 = 300.0):
    return A_0*np.cos(2 * math.pi * g(t, w_0, dw_1, dw_2)*t + phi_0) \
           + A_1*np.cos(2*math.pi * g(t, w_1, dw_1, dw_2)*t + phi_1) \
           + A_2*np.cos(2*math.pi * g(t, w_2, dw_1, dw_2)*t + phi_2) \
           + A_3*np.cos(2*math.pi * g(t, w_3, dw_1, dw_2)*t + phi_3)

def harmonics4_func(t, A_0 = 0.5,   w_0 = IF,   phi_0 = 0,
                 A_1 = -5e-2, w_1 = 2*IF, phi_1 = 0,
                 A_2 = 1e-2,  w_2 = 3*IF, phi_2 = 0,
                 A_3 = -2e-2, w_3 = 4*IF, phi_3 = 0,
                 A_4 = -5e-3, w_4 = 5*IF, phi_4 = 0,
                 dw_1 = 0.01528475,  dw_2 = -0.03984211, dw_3 = 0.02873211):
    return A_0*np.cos(2 * math.pi * g(t, w_0, dw_1, dw_2, dw_3)*t + phi_0) \
           + A_1*np.cos(2*math.pi * g(t, w_1, dw_1, dw_2, dw_3)*t + phi_1) \
           + A_2*np.cos(2*math.pi * g(t, w_2, dw_1, dw_2, dw_3)*t + phi_2) \
           + A_3*np.cos(2*math.pi * g(t, w_3, dw_1, dw_2, dw_3)*t + phi_3) \
           + A_4*np.cos(2*math.pi * g(t, w_4, dw_1, dw_2, dw_3)*t + phi_4)


def harmonics4_no_evol_func(t, A_0 = 0.5,   w_0 = IF,   phi_0 = 0,
                 A_1 = -5e-2, w_1 = 2*IF, phi_1 = 0,
                 A_2 = 1e-2,  w_2 = 3*IF, phi_2 = 0,
                 A_3 = -2e-2, w_3 = 4*IF, phi_3 = 0,
                 A_4 = -5e-3, w_4 = 5*IF, phi_4 = 0):
    return A_0*np.cos(2 * math.pi * w_0*t + phi_0) \
           + A_1*np.cos(2*math.pi * w_1*t + phi_1) \
           + A_2*np.cos(2*math.pi * w_2*t + phi_2) \
           + A_3*np.cos(2*math.pi * w_3*t + phi_3) \
           + A_4*np.cos(2*math.pi * w_4*t + phi_4)

def harmonics5_func(t, A_0 = 0.5,   w_0 = IF,   phi_0 = 0,
                 A_1 = -5e-2, w_1 = 2*IF, phi_1 = 0,
                 A_2 = 1e-2,  w_2 = 3*IF, phi_2 = 0,
                 A_3 = -2e-2, w_3 = 4*IF, phi_3 = 0,
                 A_4 = -5e-3, w_4 = 5*IF, phi_4 = 0,
                 A_5 = -5e-3, w_5 = 6*IF, phi_5 = 0,
                 dw_1 = 100.0,  dw_2 = 260.0):
    return A_0*np.cos(2 * math.pi * g(t, w_0, dw_1, dw_2)*t + phi_0) \
           + A_1*np.cos(2*math.pi * g(t, w_1, dw_1, dw_2)*t + phi_1) \
           + A_2*np.cos(2*math.pi * g(t, w_2, dw_1, dw_2)*t + phi_2) \
           + A_3*np.cos(2*math.pi * g(t, w_3, dw_1, dw_2)*t + phi_3) \
           + A_4*np.cos(2*math.pi * g(t, w_4, dw_1, dw_2)*t + phi_4) \
           + A_5*np.cos(2*math.pi * g(t, w_5, dw_1, dw_2)*t + phi_5)

def harmonics6_func(t, A_0 = 0.5,   w_0 = IF,   phi_0 = 0,
                 A_1 = -5e-2, w_1 = 2*IF, phi_1 = 0,
                 A_2 = 1e-2,  w_2 = 3*IF, phi_2 = 0,
                 A_3 = -2e-2, w_3 = 4*IF, phi_3 = 0,
                 A_4 = -5e-3, w_4 = 5*IF, phi_4 = 0,
                 A_5 = -5e-3, w_5 = 6*IF, phi_5 = 0,
                 A_6 = -5e-4, w_6 = 7*IF, phi_6 = 0,
                 dw_1 = 15.0,  dw_2 = 260.0):
    return A_0*np.cos(2 * math.pi * g(t, w_0, dw_1, dw_2)*t + phi_0) \
           + A_1*np.cos(2*math.pi * g(t, w_1, dw_1, dw_2)*t + phi_1) \
           + A_2*np.cos(2*math.pi * g(t, w_2, dw_1, dw_2)*t + phi_2) \
           + A_3*np.cos(2*math.pi * g(t, w_3, dw_1, dw_2)*t + phi_3) \
           + A_4*np.cos(2*math.pi * g(t, w_4, dw_1, dw_2)*t + phi_4) \
           + A_5*np.cos(2*math.pi * g(t, w_5, dw_1, dw_2)*t + phi_5) \
           + A_6*np.cos(2*math.pi * g(t, w_6, dw_1, dw_2)*t + phi_6)


def harmonics7_func(t, A_0 = 0.5,   w_0 = IF,   phi_0 = 0,
                 A_1 = -5e-2, w_1 = 2*IF, phi_1 = 0,
                 A_2 = 1e-2,  w_2 = 3*IF, phi_2 = 0,
                 A_3 = -2e-2, w_3 = 4*IF, phi_3 = 0,
                 A_4 = -5e-3, w_4 = 5*IF, phi_4 = 0,
                 A_5 = -5e-3, w_5 = 6*IF, phi_5 = 0,
                 A_6 = -5e-4, w_6 = 7*IF, phi_6 = 0,
                 A_7 = -1e-4, w_7 = 8*IF, phi_7 = 0,
                 dw_1 = 0.01528475,  dw_2 = -0.03984211, dw_3 = 0.02873211):
    return A_0*np.cos(2 * math.pi * g(t, w_0, dw_1, dw_2, dw_3)*t + phi_0) \
           + A_1*np.cos(2*math.pi * g(t, w_1, dw_1, dw_2, dw_3)*t + phi_1) \
           + A_2*np.cos(2*math.pi * g(t, w_2, dw_1, dw_2, dw_3)*t + phi_2) \
           + A_3*np.cos(2*math.pi * g(t, w_3, dw_1, dw_2, dw_3)*t + phi_3) \
           + A_4*np.cos(2*math.pi * g(t, w_4, dw_1, dw_2, dw_3)*t + phi_4) \
           + A_5*np.cos(2*math.pi * g(t, w_5, dw_1, dw_2, dw_3)*t + phi_5) \
           + A_6*np.cos(2*math.pi * g(t, w_6, dw_1, dw_2, dw_3)*t + phi_6) \
           + A_7*np.cos(2*math.pi * g(t, w_7, dw_1, dw_2, dw_3)*t + phi_7)

def harmonics0_func(t, A_0 = 0.5, w_0 = IF, phi_0 = 0, dw_1 = -0.3, dw_2 = 2.0, dw_3 = -3.0, dw_4 = 0.001):
    return A_0*np.cos(2 * math.pi * g(t, w_0, dw_1, dw_2, dw_3, dw_4)*t + phi_0)

def harmonics0_no_evol_func(t, A_0 = 0.5, w_0 = IF, phi_0 = 0):
    return A_0*np.cos(2 * math.pi * w_0 * t + phi_0)

def envelope3_func(t, e_1 = 100, e_2 = 55e3, e_3 = -2.0e6):
    return np.exp(-e_1*t - e_2*t**2 - e_3*t**3)

def envelope2_func(t, e_1 = 166, e_2 = 21e3):
    return np.exp(-e_1*t - e_2*t**2)

def offset_func(t, c = 0.0): return c

envelope2          = lmfit.Model(envelope2_func)
envelope3          = lmfit.Model(envelope3_func)
harmonics0         = lmfit.Model(harmonics0_func)
harmonics0_no_evol = lmfit.Model(harmonics0_no_evol_func)
harmonics2         = lmfit.Model(harmonics2_func)
harmonics3         = lmfit.Model(harmonics3_func)
harmonics4         = lmfit.Model(harmonics4_func)
harmonics4_no_evol = lmfit.Model(harmonics4_no_evol_func)
harmonics5         = lmfit.Model(harmonics5_func)
harmonics6         = lmfit.Model(harmonics6_func)
harmonics7         = lmfit.Model(harmonics7_func)
offset             = lmfit.Model(offset_func)

def pulse_fit(times, signal, start_time, end_time, env_model, harm_model):
    i1 = start_time / SAMPLING_PERIOD
    i2 = end_time/SAMPLING_PERIOD
    pulse_model = env_model * harm_model + offset
    fit = pulse_model.fit(signal[i1:i2], t=times[i1:i2])
    c = red_chi_sq(signal[i1:i2], fit.best_fit, RAW_STD_DEV)
    return (fit, c)

def gen_freq_vs_time(times, signal):
    ts = []
    fs = []
    for i in np.arange(0.5e-3, 13e-3, 0.25e-3):
        fit, c = pulse_fit(dat_times, dat_signal, i-0.5e-3, i, envelope2, harmonics0_no_evol)
        ts.append(i + 0.25e-3)
        fs.append(fit.best_values['w_0'])
        print "fit ", i, fit.best_values['w_0'], c
    return ts, fs

if RAW_FIT:
    raw_model = envelope3 * harmonics0 + offset
    raw_fit_result = raw_model.fit(raw_signal, t=raw_times)
    c = red_chi_sq(raw_signal, raw_fit_result.best_fit, 2.4e-3)

    raw_fit_fft = np.fft.rfft(raw_fit_result.best_fit, num_fft_samples)
    raw_fit_residuals_fft = np.fft.rfft(raw_fit_result.best_fit - raw_signal, num_fft_samples)

if PRINT_DIAGNOSTICS and RAW_FIT:
    print "placeholder"

####################################
### ZERO-CROSSING/PHASE ANALYSIS ###
####################################

if RAW_FIT:
    w_ref = raw_fit_result.best_values['w_0']
else:
    w_ref = IF

def two_point_zc(tarr, yarr):
    assert(tarr.size == yarr.size)
    zcs = []

    for i in range(1,tarr.size):
        y1 = yarr[i-1]
        y2 = yarr[i]
        if y1 * y2 < 0:
            t1 = tarr[i-1]
            t2 = tarr[i]
            m = (y2 - y1) / (t2 - t1)
            zcs.append(t1 - y1/m)

    return np.array(zcs)

def phase_vs_time(tarr, yarr, w_ref):
    zcs = two_point_zc(tarr,yarr)
    ps = np.zeros(zcs.size)

    T = 1.0/w_ref
    for i in range(1,len(zcs)):
        ref_time = zcs[0] + i * 0.5 * T
        ps_test = -(2*math.pi*(zcs[i] - ref_time)/T)
        if (abs(ps_test - ps[i-1]) >  0.2 * math.pi):
            while( abs(ps_test - ps[i-1]) > 0.3 ):
                ps_test = ps_test - np.sign(ps_test)*math.pi
            ps[i] = ps_test
        else:
            ps[i] = ps_test

    return zcs, ps

if PHASE_FIT:
    raw_zcs, raw_ps = phase_vs_time(filter_times, filter_signal, w_ref)
    ref_zcs = np.fromfunction(lambda i: raw_zcs[0] + 0.5*i/w_ref, [raw_zcs.size])
    def phase_func(t, a = 0.0, b = 0.0, c = 0.0, d = 0.0, e = 0.0):#, f = 0.0):#, q = 0.0, r = 0.0, s=0.0):
        t2 = t/RAW_SIGNAL_CUT
        return a + b*t2 + c*t2**4 + d*t2**5 + e*t2**6 # + q*t2**11 + r*t2**13 + s*t2**15
    phase_model = lmfit.Model(phase_func)
    phase_fit = phase_model.fit(raw_ps, t=raw_zcs)
    print (phase_fit.best_values['b']/RAW_SIGNAL_CUT)/(2*math.pi) + IF, phase_fit.best_values['a'], phase_fit.best_values['b'], phase_fit.best_values['c'], phase_fit.best_values['d'], phase_fit.best_values['e']

################
### PLOTTING ###
################

DARKGREY  = "#9f9f9f"
GREY      = "#D7D7D7"
LIGHTGREY = "#E8E8E8"
RED       = "#990000"
GREEN     = "#3BA03B"
BLUE      = "#5959FF"

if PLOT_FIT_RESULTS and RAW_FIT:
    plt.figure(facecolor=LIGHTGREY)
    plt.plot(raw_times, raw_signal, color=DARKGREY, marker=".", label='raw', markerfacecolor = "#000000", markevery=5)
    plt.plot(raw_times, raw_fit_result.best_fit, color=GREEN, label='fit')
    plt.plot(raw_times, raw_fit_result.best_fit - raw_signal, color=RED, label='fit')
    plt.legend(loc='lower right')
    plt.grid()
    plt.gca().set_axis_bgcolor(GREY)
    plt.xlim(raw_times[0], raw_times[-1])
    plt.ylim(-0.8,0.8)
    plt.title("RAW/FILTERED SIGNAL")
    plt.ylabel("Voltage (V)")
    plt.xlabel("Time (s)")

    plt.figure(facecolor=LIGHTGREY)
    plt.gca().set_axis_bgcolor(GREY)
    plt.grid()
    plt.yscale('log')
    plt.title("RAW/FIT FOURIER TRANSFORMS")
    plt.xlabel("Frequency (Hz)")
    plt.plot(raw_freqs, np.abs(raw_fft), color=BLUE, label='raw')
    plt.plot(raw_freqs, np.abs(raw_fit_fft), color=RED, label='fit', alpha=0.8)
    plt.plot(raw_freqs, np.abs(raw_fit_residuals_fft), color=GREEN, label='residuals', alpha=0.8)
    plt.xlim(0, 250000)
    plt.ylim(1e-4,1e4)
    plt.legend(loc='upper right')

    plt.figure(facecolor=LIGHTGREY)
    plt.plot(raw_times, raw_fit_result.best_fit - raw_signal, color=RED, linewidth=1.0, alpha = 0.7)
    plt.gca().set_axis_bgcolor(GREY)
    plt.grid()
    plt.ylim(-10e-3,10e-3)
    plt.xlim(raw_times[0], raw_times[-1])
    plt.title("RESIDUALS WITH FREQUENCY EVOLUTION")
    plt.ylabel("Voltage (V)")
    plt.xlabel("Time (s)")

if PLOT_FIT_RESULTS and PHASE_FIT:
    plt.figure(facecolor=LIGHTGREY)
    plt.plot(raw_times, raw_signal, color=DARKGREY, label='raw signal')
    plt.plot(raw_zcs, np.zeros(raw_zcs.size), linestyle="None", marker=".", label='raw zero crossings', markerfacecolor = RED)
    plt.plot(ref_zcs, np.zeros(ref_zcs.size), linestyle="None", marker=".", label='ref zero crossings', markerfacecolor = GREEN)
    plt.legend(loc='lower right')
    plt.grid()
    plt.gca().set_axis_bgcolor(GREY)
    plt.xlim(raw_times[0], raw_times[-1])
    plt.ylim(-0.8,0.8)
    plt.title("ZERO-CROSSING ANALYSIS")
    plt.ylabel("Voltage (V)")
    plt.xlabel("Time (s)")

    plt.figure(facecolor=LIGHTGREY)
    plt.plot(raw_zcs, raw_ps, marker=".", markeredgecolor="#3f3f3f", linestyle="None")
    plt.plot(raw_zcs, phase_fit.best_fit, color=RED)
    plt.gca().set_axis_bgcolor(GREY)
    plt.grid()
    plt.xlim(0.0, raw_zcs[-1])
    plt.title("PHASE")
    plt.xlabel("Time (s)")

    plt.figure(facecolor=LIGHTGREY)
    plt.plot(range(raw_zcs.size), raw_zcs, color="#4f4f4f", linewidth=1.0)
    plt.gca().set_axis_bgcolor(GREY)
    plt.grid()
    #plt.xlim(0.0, raw_zcs[-1])
    plt.title("ZERO_CROSSINGS")
    plt.ylabel("Time (s)")

#
#     us, ue = 7.2*times.size/8, 7.4*times.size/8
#
#     plt.axes([0.7, 0.65, 0.15, 0.2], axisbg=GREY)
#     plt.grid()
#     plt.locator_params(nbins=7)
#     plt.plot(times[us:ue], signal[us:ue], color=BLUE)
#     plt.plot(times[us:ue], best_fit_raw[us:ue], color=GREEN)
#     plt.plot(times[us:ue], residuals_raw[us:ue], color=RED)
#     plt.xlim(times[us], times[ue])
#     plt.ylabel("Voltage (V)")
#     plt.xlabel("Time (s)")
#
#     plt.figure(facecolor=LIGHTGREY)
#     plt.plot(times, residuals_raw, color=RED, linewidth=1.0, alpha = 0.7)
#     plt.gca().set_axis_bgcolor(GREY)
#     plt.grid()
#     plt.xlim(times[0], times[-1])
#     plt.title("RESIDUALS WITH FREQUENCY EVOLUTION")
#     plt.ylabel("Voltage (V)")
#     plt.xlabel("Time (s)")
#
#     plt.figure(facecolor=LIGHTGREY)
#     plt.gca().set_axis_bgcolor(GREY)
#     plt.grid()
#     plt.yscale('log')
#     plt.title("Fourier Transforms")
#     plt.xlabel("Frequency (Hz)")
#     plt.plot(fft_freqs, np.abs(fft_signal), color=BLUE, label='signal')
#     plt.plot(fft_freqs, np.abs(fft_raw_fit), color=RED, label='fit', alpha=0.8)
#     plt.plot(fft_freqs, np.abs(fft_residuals_raw), color=GREEN, label='residuals')
#     plt.legend(loc='upper right')
#
#     plt.figure(facecolor=LIGHTGREY)
#     plt.plot(times, g(times, popt_raw[8], popt_raw[13], popt_raw[14]) - popt_raw[8])
#     plt.gca().set_axis_bgcolor(GREY)
#     plt.grid()
#     plt.xlim(times[0], times[-1])
#     plt.title("FREQUENCY DEVIATION vs. TIME")
#     plt.ylabel("Frequency (Hz)")
#     plt.xlabel("Time (s)")
#
# #     plt.figure(facecolor=LIGHTGREY)
# #     plt.plot(times, signal, color=BLUE, label='signal')
# #     plt.plot(times, best_fit_no_freq_evolution, color=GREEN, label='fit')
# #     plt.plot(times, signal - best_fit_no_freq_evolution, color=RED, label='residuals')
# #     plt.legend(loc='lower right')
# #     plt.gca().set_axis_bgcolor(GREY)
# #     plt.grid()
# #     plt.xlim(times[0], times[-1])
# #     plt.ylim(-0.5,0.5)
# #     plt.title("BEST FIT WITHOUT FREQUENCY EVOLUTION")
# #     plt.ylabel("Voltage (V)")
# #     plt.xlabel("Time (s)")
# #
# #     plt.axes([0.5, 0.45, 0.15, 0.2], axisbg=GREY)
# #     plt.grid()
# #     plt.plot(times[us:ue], signal[us:ue], color=BLUE)
# #     plt.plot(times[us:ue], best_fit_no_freq_evolution[us:ue], color=GREEN)
# #     plt.plot(times[us:ue], signal[us:ue] - best_fit_no_freq_evolution[us:ue], color=RED)
# #     plt.xlim(times[us], times[ue])
# #     plt.ylabel("Voltage (V)")
# #     plt.xlabel("Time (s)")
# #
# #     plt.figure(facecolor=LIGHTGREY)
# #     plt.plot(times, signal - best_fit_no_freq_evolution, color=RED, linewidth=2.0)
# #     plt.gca().set_axis_bgcolor(GREY)
# #     plt.grid()
# #     plt.xlim(times[0], times[-1])
# #     plt.title("RESIDUALS WITHOUT FREQUENCY EVOLUTION")
# #     plt.ylabel("Voltage (V)")
# #     plt.xlabel("Time (s)")
# #     plt.text(0.0081,0.0145, "RMS = " + str(np.sqrt(np.average((signal-best_fit_no_freq_evolution)**2)))[:10], fontsize=20)
# #
# #     plt.show()
#
# # if PRINT_DIAGNOSTICS:
# #     print( math.sqrt(np.average((step_filtered_signal-best_fit)**2)))
# #
# #
# #     print( "A_0: " + '%.6e'%(popt[0]) + " err: " + '%.6e'%(perr[0]))
# #     print( "A_2: " + '%.6e'%(popt[1]) + " err: " + '%.6e'%(perr[1]))
# #     print( "A_3: " + '%.6e'%(popt[2]) + " err: " + '%.6e'%(perr[2]))
# #     print( "A_4: " + '%.6e'%(popt[3]) + " err: " + '%.6e'%(perr[3]))
# #     print( "A_5: " + '%.6e'%(popt[4]) + " err: " + '%.6e'%(perr[4]))
# #     print( "e_1: " + '%.6e'%(popt[5]) + " err: " + '%.6e'%(perr[5]))
# #     print( "e_2: " + '%.6e'%(popt[6]) + " err: " + '%.6e'%(perr[6]))
# #     print( "w_0: " + '%.6e'%(popt[7]) + " err: " + '%.6e'%(perr[7]))
# #     print( "w_2: " + '%.6e'%(popt[8]) + " err: " + '%.6e'%(perr[8]))
# #     print( "w_3: " + '%.6e'%(popt[9]) + " err: " + '%.6e'%(perr[9]))
# #     print( "w_4: " + '%.6e'%(popt[10]) + " err: " + '%.6e'%(perr[10]))
# #     print( "w_5: " + '%.6e'%(popt[11]) + " err: " + '%.6e'%(perr[11]))
# #     print( "pw_1: " + '%.6e'%(popt[12]) + " err: " + '%.6e'%(perr[12]))
# #     print( "pw_2: " + '%.6e'%(popt[13]) + " err: " + '%.6e'%(perr[13]))
# #     print( "phi_0: " + '%.6e'%(popt[14]) + " err: " + '%.6e'%(perr[14]))
# #     print( "phi_2: " + '%.6e'%(popt[15]) + " err: " + '%.6e'%(perr[15]))
# #     print( "phi_3: " + '%.6e'%(popt[16]) + " err: " + '%.6e'%(perr[16]))
# #     print( "phi_4: " + '%.6e'%(popt[17]) + " err: " + '%.6e'%(perr[17]))
# #     print( "phi_5: " + '%.6e'%(popt[18]) + " err: " + '%.6e'%(perr[18]))
# #     #print( "offset: " + '%.6e'%(popt[19]) + " err: " + '%.6e'%(perr[19]))
# #
# #     print np.sum( ((best_fit - step_filtered_signal)**2) / ( ((1.3e-3)**2) * (best_fit.size - 20)))
#
#
# if PLOT_RESULTS and FIT_FILTER:
#      plt.figure(facecolor=LIGHTGREY)
#      plt.plot(times, signal, color=BLUE, label='signal')
#      #plt.plot(times, best_fit_raw, color=GREEN, label='fit')
#      #plt.plot(times, step_filtered_signal - best_fit, color=RED, label='residuals')
#      plt.plot(times, step_filtered_signal, color='purple', label='filtered')
#      plt.legend(loc='lower right')
#      plt.gca().set_axis_bgcolor(GREY)
#      plt.grid()
#      #plt.xlim(times[0], times[-1])
#      #plt.ylim(-0.5,0.5)
#      #plt.title("BEST FIT WITH FREQUENCY EVOLUTION")
#      plt.ylabel("Voltage (V)")
#      plt.xlabel("Time (s)")
#
#
if PLOT_FIT_RESULTS: plt.show()

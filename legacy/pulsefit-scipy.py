import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from scipy.optimize import curve_fit

#################
### CONSTANTS ###
#################

PRINT_DIAGNOSTICS    = True
PLOT_RESULTS         = False

FIT_RAW              = True
FIT_FILTER           = False

PRINT_FREQ_LINE      = False

FIT_DURATION         = 6e-3

SAMPLING_PERIOD      = 1e-7
TIME_SINCE_PI2_PULSE = 0.0

CUT_TIME = 0

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

def two_point_frequencies(tarr, yarr):
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

    freq_hz = []
    freq_ts = []
    for i in range(1,len(zcs)):
        half_period = zcs[i] - zcs[i-1]
        freq_hz.append(0.5 / half_period)
        freq_ts.append( 0.5 * (zcs[i] + zcs[i-1]))
    return freq_ts, freq_hz


############################
### LOAD PULSE FROM FILE ###
############################

def adc_to_voltage(adc_value):
    return (adc_value - 32599.9) / 30465.9

def load_pulse_to_array(filename):
    raw_time_indices, raw_data = np.loadtxt(filename, delimiter=' ', usecols=(0, 1), unpack=True)
    scaled_data = np.vectorize(adc_to_voltage)(raw_data)
    shifted_scaled_data = scaled_data - np.average(scaled_data[3*scaled_data.size/4:])
    return (raw_time_indices*SAMPLING_PERIOD) + TIME_SINCE_PI2_PULSE, shifted_scaled_data

times, signal   = load_pulse_to_array(sys.argv[1])
NUM_RAW_SAMPLES = signal.size
RAW_STD_DEV     = rms(signal[5*signal.size/6:])

#########################
### FOURIER TRANSFORM ###
#########################

num_fft_samples = signal.size * 4

fft_signal = np.fft.rfft(signal, num_fft_samples)
fft_freqs = np.fft.rfftfreq(num_fft_samples, SAMPLING_PERIOD)

##############
### FILTER ###
##############

def step_filter(freq):
    return 1 / ( np.exp( (freq - 30e3) / 8e3 ) + 1 )

step_filter_fft = np.multiply(step_filter(fft_freqs), fft_signal)
step_filtered_signal = np.fft.irfft(step_filter_fft)[:NUM_RAW_SAMPLES]

############
### CUTS ###
############

si = CUT_TIME / SAMPLING_PERIOD
ei = FIT_DURATION / SAMPLING_PERIOD

FILTER_STD_DEV = rms(step_filtered_signal[5*NUM_RAW_SAMPLES/6:NUM_RAW_SAMPLES - si])

signal = signal[si:ei]
times = times[si:ei]
step_filtered_signal = step_filtered_signal[si:ei]

assert(signal.size == times.size and step_filtered_signal.size == signal.size)

#####################
### PULSE FITTING ###
#####################

def g(t, w_0, pw_1, pw_2):
    '''frequency fit function'''
    return w_0 * (1 + pw_1*t**2*np.sin(pw_2*t)**2)

if FIT_RAW:
    def raw_pulse_func(t, A_0, A_2, A_3, A_4, A_5, e_1, e_2, e_3,
                       w_0, w_2, w_3, w_4, w_5, pw_1, pw_2,
                       phi_0, phi_2, phi_3, phi_4, phi_5, offset):
        # if (abs(w_0 - 21381) > 200 or abs(w_2 - 2*w_0) > 1000 or abs(w_3 - 3*w_0) > 1000
        #         or abs(w_4 - 4*w_0) > 1000 or abs(w_5 - 5*w_0) > 1000):
        #     return 99999
        envelope = np.exp(-e_1 * t -e_2 * t**2 - e_3*t**3) # * np.sinc(e_3*t)
        harmonics = A_0*np.cos(2*math.pi * g(t, w_0, pw_1, pw_2) * t + phi_0) \
                        + A_2*np.cos(2*math.pi * g(t, w_2, pw_1, pw_2)*t + phi_2) \
                        + A_3*np.cos(2*math.pi * g(t, w_3, pw_1, pw_2)*t + phi_3) \
                        + A_4*np.cos(2*math.pi * g(t, w_4, pw_1, pw_2)*t + phi_4) \
                        + A_5*np.cos(2*math.pi * g(t, w_5, pw_1, pw_2)*t + phi_5)
        return envelope * harmonics + offset

    p0_raw = [ 0.5,     # A_0
               -4e-2,   # A_2
               2e-2,    # A_3
               -2e-3,   # A_4
               -8e-3,   # A_5
               1.0,   # e_1
               1.0,     # e_2
               1.0,     # e_3
               21381,   # w_0
               2*21381,   # w_2
               3*21381,   # w_3
               4*21381,   # w_4
               5*21381,  # w_5
               12.3,     # pw_1
               270,     # pw_2
               0.0,     # phi_0
               0.0,     # phi_2
               0.0,     # phi_3
               0.0,     # phi_4
               0.0,     # phi_5
               0.0      # offset
             ]
    popt_raw, pcov_raw = curve_fit(raw_pulse_func, times, signal, p0_raw, ftol=1.5e-7, xtol=1.5e-7, maxfev=5000, absolute_sigma = RAW_STD_DEV)
    perr_raw = np.sqrt(np.diag(pcov_raw))
    best_fit_raw = raw_pulse_func(times, *popt_raw)
    residuals_raw = best_fit_raw - signal

    fft_raw_fit = np.fft.rfft(best_fit_raw, num_fft_samples)
    fft_residuals_raw = np.fft.rfft(residuals_raw, num_fft_samples)

if FIT_FILTER:
    def filter_pulse_func(t, A_0, A_2, A_3, A_4, A_5, e_1, e_2, e_3,
                       w_0, w_2, w_3, w_4, w_5, pw_1, pw_2,
                       phi_0, phi_2, phi_3, phi_4, phi_5, offset):
        envelope = np.exp(-e_1 * t -e_2 * t**2 - e_3*t**3)
        harmonics = A_0*np.cos(2*math.pi * g(t, w_0, pw_1, pw_2) * t + phi_0) \
                        + A_2*np.cos(2*math.pi * g(t, w_2, pw_1, pw_2)*t + phi_2) \
                        + A_3*np.cos(2*math.pi * g(t, w_3, pw_1, pw_2)*t + phi_3) \
                        + A_4*np.cos(2*math.pi * g(t, w_4, pw_1, pw_2)*t + phi_4) \
                        + A_5*np.cos(2*math.pi * g(t, w_5, pw_1, pw_2)*t + phi_5)
        return envelope * harmonics + offset


    p0_filter = [ 0.5,     # A_0
               -4e-4,   # A_2
               2e-6,    # A_3
               -2e-6,   # A_4
               -8e-7,   # A_5
               1.5e2,   # e_1
               3e4,     # e_2
               300,     # e_3
               21362,   # w_0
               42761,   # w_2
               64000,   # w_3
               85514,   # w_4
               128336,  # w_5
               1.0e4,     # pw_1
               2.0,     # pw_2
               0.0,     # phi_0
               0.0,     # phi_2
               0.0,     # phi_3
               0.0,     # phi_4
               0.0,     # phi_5
               0.0      # offset
             ]

    popt_filter, pcov_filter = curve_fit(filter_pulse_func, times, step_filtered_signal, p0_filter, ftol=1.5e-6, xtol=1.5e-6, maxfev=5000, absolute_sigma = FILTER_STD_DEV)
    perr_filter = np.sqrt(np.diag(pcov_filter))
    best_fit_filter = filter_pulse_func(times, *popt_filter)
    residuals_filter = best_fit_filter - step_filtered_signal

    fft_filter_fit = np.fft.rfft(best_fit_filter, num_fft_samples)
    fft_residuals_fit = np.fft.rfft(residuals_filter, num_fft_samples)

###################
### DIAGNOSTICS ###
###################

if PRINT_DIAGNOSTICS and FIT_FILTER:
    print "#########################"
    print "### STEP FILTERED FIT ###"
    print "#########################"
    print "NOISE: " + str(FILTER_STD_DEV)
    print "RMS: " + str(rms(residuals_filter))
    print "FREQUENCY: " + str(popt_filter[8])
    print "FREQUENCY ERROR: " + str(perr_filter[8])
    print "CHI-SQ/NDF: " + str(red_chi_sq(step_filtered_signal, best_fit_filter, FILTER_STD_DEV))
    print "\n"

if PRINT_DIAGNOSTICS and FIT_RAW:
    print "###############"
    print "### RAW FIT ###"
    print "###############"
    print "NOISE: " + str(RAW_STD_DEV)
    print "RMS: " + str(rms(residuals_raw))
    print "FREQUENCY: " + str(popt_raw[8])
    print "FREQUENCY ERROR: " + str(perr_raw[8])
    print "CHI-SQ/NDF: " + str(red_chi_sq(signal, best_fit_raw, RAW_STD_DEV))
    print "\n"

################
### PLOTTING ###
################

GREY      = "#D7D7D7"
LIGHTGREY = "#E8E8E8"
RED       = "#990000"
GREEN     = "#3BA03B"
BLUE      = "#5959FF"

if PLOT_RESULTS and FIT_RAW:
    plt.figure(facecolor=LIGHTGREY)
    plt.plot(times, signal, color=BLUE, label='signal')
    plt.plot(times, best_fit_raw, color=GREEN, label='fit')
    plt.plot(times, residuals_raw, color=RED, label='residuals')
    plt.legend(loc='lower right')
    plt.grid()
    plt.gca().set_axis_bgcolor(GREY)
    plt.xlim(times[0], times[-1])
    plt.ylim(-0.5,0.5)
    plt.title("RAW SIGNAL BEST FIT")
    plt.ylabel("Voltage (V)")
    plt.xlabel("Time (s)")

    us, ue = 7.2*times.size/8, 7.4*times.size/8

    plt.axes([0.7, 0.65, 0.15, 0.2], axisbg=GREY)
    plt.grid()
    plt.locator_params(nbins=7)
    plt.plot(times[us:ue], signal[us:ue], color=BLUE)
    plt.plot(times[us:ue], best_fit_raw[us:ue], color=GREEN)
    plt.plot(times[us:ue], residuals_raw[us:ue], color=RED)
    plt.xlim(times[us], times[ue])
    plt.ylabel("Voltage (V)")
    plt.xlabel("Time (s)")

    plt.figure(facecolor=LIGHTGREY)
    plt.plot(times, residuals_raw, color=RED, linewidth=1.0, alpha = 0.7)
    plt.gca().set_axis_bgcolor(GREY)
    plt.grid()
    plt.xlim(times[0], times[-1])
    plt.title("RESIDUALS WITH FREQUENCY EVOLUTION")
    plt.ylabel("Voltage (V)")
    plt.xlabel("Time (s)")

    plt.figure(facecolor=LIGHTGREY)
    plt.gca().set_axis_bgcolor(GREY)
    plt.grid()
    plt.yscale('log')
    plt.title("Fourier Transforms")
    plt.xlabel("Frequency (Hz)")
    plt.plot(fft_freqs, np.abs(fft_signal), color=BLUE, label='signal')
    plt.plot(fft_freqs, np.abs(fft_raw_fit), color=RED, label='fit', alpha=0.8)
    plt.plot(fft_freqs, np.abs(fft_residuals_raw), color=GREEN, label='residuals')
    plt.legend(loc='upper right')

    plt.figure(facecolor=LIGHTGREY)
    plt.plot(times, g(times, popt_raw[8], popt_raw[13], popt_raw[14]) - popt_raw[8])
    plt.gca().set_axis_bgcolor(GREY)
    plt.grid()
    plt.xlim(times[0], times[-1])
    plt.title("FREQUENCY DEVIATION vs. TIME")
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")

#     plt.figure(facecolor=LIGHTGREY)
#     plt.plot(times, signal, color=BLUE, label='signal')
#     plt.plot(times, best_fit_no_freq_evolution, color=GREEN, label='fit')
#     plt.plot(times, signal - best_fit_no_freq_evolution, color=RED, label='residuals')
#     plt.legend(loc='lower right')
#     plt.gca().set_axis_bgcolor(GREY)
#     plt.grid()
#     plt.xlim(times[0], times[-1])
#     plt.ylim(-0.5,0.5)
#     plt.title("BEST FIT WITHOUT FREQUENCY EVOLUTION")
#     plt.ylabel("Voltage (V)")
#     plt.xlabel("Time (s)")
#
#     plt.axes([0.5, 0.45, 0.15, 0.2], axisbg=GREY)
#     plt.grid()
#     plt.plot(times[us:ue], signal[us:ue], color=BLUE)
#     plt.plot(times[us:ue], best_fit_no_freq_evolution[us:ue], color=GREEN)
#     plt.plot(times[us:ue], signal[us:ue] - best_fit_no_freq_evolution[us:ue], color=RED)
#     plt.xlim(times[us], times[ue])
#     plt.ylabel("Voltage (V)")
#     plt.xlabel("Time (s)")
#
#     plt.figure(facecolor=LIGHTGREY)
#     plt.plot(times, signal - best_fit_no_freq_evolution, color=RED, linewidth=2.0)
#     plt.gca().set_axis_bgcolor(GREY)
#     plt.grid()
#     plt.xlim(times[0], times[-1])
#     plt.title("RESIDUALS WITHOUT FREQUENCY EVOLUTION")
#     plt.ylabel("Voltage (V)")
#     plt.xlabel("Time (s)")
#     plt.text(0.0081,0.0145, "RMS = " + str(np.sqrt(np.average((signal-best_fit_no_freq_evolution)**2)))[:10], fontsize=20)
#
#     plt.show()

# if PRINT_DIAGNOSTICS:
#     print( math.sqrt(np.average((step_filtered_signal-best_fit)**2)))
#
#
#     print( "A_0: " + '%.6e'%(popt[0]) + " err: " + '%.6e'%(perr[0]))
#     print( "A_2: " + '%.6e'%(popt[1]) + " err: " + '%.6e'%(perr[1]))
#     print( "A_3: " + '%.6e'%(popt[2]) + " err: " + '%.6e'%(perr[2]))
#     print( "A_4: " + '%.6e'%(popt[3]) + " err: " + '%.6e'%(perr[3]))
#     print( "A_5: " + '%.6e'%(popt[4]) + " err: " + '%.6e'%(perr[4]))
#     print( "e_1: " + '%.6e'%(popt[5]) + " err: " + '%.6e'%(perr[5]))
#     print( "e_2: " + '%.6e'%(popt[6]) + " err: " + '%.6e'%(perr[6]))
#     print( "w_0: " + '%.6e'%(popt[7]) + " err: " + '%.6e'%(perr[7]))
#     print( "w_2: " + '%.6e'%(popt[8]) + " err: " + '%.6e'%(perr[8]))
#     print( "w_3: " + '%.6e'%(popt[9]) + " err: " + '%.6e'%(perr[9]))
#     print( "w_4: " + '%.6e'%(popt[10]) + " err: " + '%.6e'%(perr[10]))
#     print( "w_5: " + '%.6e'%(popt[11]) + " err: " + '%.6e'%(perr[11]))
#     print( "pw_1: " + '%.6e'%(popt[12]) + " err: " + '%.6e'%(perr[12]))
#     print( "pw_2: " + '%.6e'%(popt[13]) + " err: " + '%.6e'%(perr[13]))
#     print( "phi_0: " + '%.6e'%(popt[14]) + " err: " + '%.6e'%(perr[14]))
#     print( "phi_2: " + '%.6e'%(popt[15]) + " err: " + '%.6e'%(perr[15]))
#     print( "phi_3: " + '%.6e'%(popt[16]) + " err: " + '%.6e'%(perr[16]))
#     print( "phi_4: " + '%.6e'%(popt[17]) + " err: " + '%.6e'%(perr[17]))
#     print( "phi_5: " + '%.6e'%(popt[18]) + " err: " + '%.6e'%(perr[18]))
#     #print( "offset: " + '%.6e'%(popt[19]) + " err: " + '%.6e'%(perr[19]))
#
#     print np.sum( ((best_fit - step_filtered_signal)**2) / ( ((1.3e-3)**2) * (best_fit.size - 20)))


if PLOT_RESULTS and FIT_FILTER:
     plt.figure(facecolor=LIGHTGREY)
     plt.plot(times, signal, color=BLUE, label='signal')
     #plt.plot(times, best_fit_raw, color=GREEN, label='fit')
     #plt.plot(times, step_filtered_signal - best_fit, color=RED, label='residuals')
     plt.plot(times, step_filtered_signal, color='purple', label='filtered')
     plt.legend(loc='lower right')
     plt.gca().set_axis_bgcolor(GREY)
     plt.grid()
     #plt.xlim(times[0], times[-1])
     #plt.ylim(-0.5,0.5)
     #plt.title("BEST FIT WITH FREQUENCY EVOLUTION")
     plt.ylabel("Voltage (V)")
     plt.xlabel("Time (s)")


if PLOT_RESULTS: plt.show()

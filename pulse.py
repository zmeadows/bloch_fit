import math
import numpy as np
import lmfit
from scipy import interpolate
from scipy.optimize import minimize

#########################
### UTILITY FUNCTIONS ###
#########################

def adc_to_voltage(adc_value):
    return (adc_value - 32599.9) / 30465.9

def rms(arr):
    assert(arr.size > 1)
    return math.sqrt(np.mean(np.square(arr)))

def red_chi_sq(data, fit, sigma):
    assert(data.size == fit.size)
    ress = np.sum((fit - data)**2)
    chi_sq = ress / sigma**2
    return chi_sq / data.size

def two_point_zc(tarr, yarr):
    assert(tarr.size == yarr.size)
    zcs = []
    for i in range(1,tarr.size):
        last_index = 0
        y1 = yarr[i-1]
        y2 = yarr[i]
        if y1 * y2 < 0:
            t1 = tarr[i-1]
            t2 = tarr[i]
            m = (y2 - y1) / (t2 - t1)
            t_zc = t1 - y1/m
            if (i - last_index > 20):
                zcs.append(t_zc)
                last_index = i
    return np.array(zcs)

def linear_zc(tarr, yarr, w_ref, sampling_period):
    assert(tarr.size == yarr.size)
    zcs = []
    T = (2*math.pi) / w_ref
    jump = (T/4) / sampling_period

    def simp_line(t, m = 0.0, b = 0.0):
        return m*t + b

    i = jump/8 + 1
    while (i < tarr.size):
        y1 = yarr[i-1]
        y2 = yarr[i]
        if y1 * y2 < 0:
            t1 = tarr[i-1]
            t2 = tarr[i]
            m_guess = (y2 - y1) / (t2 - t1)
            b_guess = - m_guess * (t1 + t2)/2
            model = lmfit.Model(simp_line)
            model.set_param_hint('m', value = m_guess)
            model.set_param_hint('b', value = b_guess)
            pars = model.make_params()
            fit = model.fit(yarr[i-jump/8:i+jump/8], pars, t=tarr[i-jump/8:i+jump/8])
            m = fit.best_values['m']
            b = fit.best_values['b']
            zcs.append(-b/m)
            i += jump
        else:
            i += 1

    return np.array(zcs)

##################################
### PULSE FIT/ANALYSIS CLASSES ###
##################################

class NMRPulse(object):
    def __init__(self, filepath, sampling_period = 1.0e-7, w_ref = 2*math.pi*9.685e3,
                 time_since_pi2_pulse = 0.0, signal_duration = 25e-3, **kwargs):
        super(NMRPulse, self).__init__(**kwargs)

        self.sampling_period = sampling_period
        self.w_ref = w_ref
        self.signal_duration = signal_duration

        if (filepath[-4:] == ".bin"):
            signal_data = np.fromfile(filepath, dtype=np.dtype('u2'))
            time_indices = np.arange(signal_data.size)
        elif (filepath[-4:] == ".dat"):
            time_indices, signal_data = np.loadtxt(filepath, delimiter=' ', usecols=(0, 1), unpack=True)
        else:
            print "INVALID DATA FILE EXTENSION."

        assert(time_indices.size > 0 and signal_data.size > 0)
        assert(time_indices.size == signal_data.size)
        self.num_samples = time_indices.size

        scaled_data = np.vectorize(adc_to_voltage)(signal_data)
        shifted_scaled_data = scaled_data - np.average(scaled_data[3*scaled_data.size/4:])

        self.raw_signal = shifted_scaled_data
        self.raw_times = time_since_pi2_pulse + sampling_period * time_indices

        self.std_dev = rms(self.raw_signal[7*self.num_samples/8:self.num_samples])

class NMRPulseFFT(NMRPulse):
    def __init__(self, filepath, fft_time_cut = None, **kwargs):
        super(NMRPulseFFT, self).__init__(filepath, **kwargs)
        if fft_time_cut:
            fft_cut_index = int(fft_time_cut/self.sampling_period)
        else:
            fft_cut_index = self.raw_signal.size
        self.num_fft_samples = fft_cut_index * 10

        self.raw_fft = np.fft.rfft(self.raw_signal[:fft_cut_index], self.num_fft_samples)
        self.raw_freqs = np.fft.rfftfreq(self.num_fft_samples, self.sampling_period)

        delta_hz = self.raw_freqs[1] - self.raw_freqs[0]
        freq_guess = self.w_ref / (2*math.pi)
        start_peak_ind = (freq_guess - 1000)/delta_hz
        end_peak_ind = (freq_guess + 1000)/delta_hz

        f = interpolate.interp1d(self.raw_freqs[start_peak_ind:end_peak_ind],
                np.abs(self.raw_fft)[start_peak_ind:end_peak_ind])

        self.fft_freq = minimize(lambda h: -1 * f(h), [freq_guess], bounds=[(freq_guess-70,freq_guess+70)], tol=1.0e-3 ).x[0]
        self.fft_fit_freqs = np.linspace(self.raw_freqs[start_peak_ind] + 100, self.raw_freqs[end_peak_ind] - 100, 4 * (end_peak_ind - start_peak_ind))
        self.fft_fit = np.vectorize(f)(self.fft_fit_freqs)

class NMRPulseFiltered(NMRPulseFFT):
    def __init__(self, filepath, filter_type = "step", filter_time_cut = 2e-4, **kwargs):
        super(NMRPulseFiltered, self).__init__(filepath, **kwargs)

        def step_filter(freq):
            hz_ref = self.w_ref / (2 * math.pi)
            cutoff_freq = 2 * hz_ref
            scale = hz_ref / 10
            if (freq < 20 * hz_ref):
                return 1 / ( np.exp( (freq - cutoff_freq) / scale ) + 1 )
            else: return 0

        filter_cut_index = filter_time_cut / self.sampling_period

        self.filter = np.vectorize(step_filter)(self.raw_freqs)
        self.filter_fft = np.multiply(self.filter, self.raw_fft)
        self.filter_signal = np.fft.irfft(self.filter_fft)[filter_cut_index : self.raw_signal.size]
        self.filter_times = self.raw_times[filter_cut_index:]
        self.filter_std_dev = rms(self.filter_signal[7*self.num_samples/8:self.num_samples])

class NMRPulseFullFit(NMRPulseFiltered):
    def __init__(self, filepath, full_fit_use_filter = False, full_fit_stop = 6e-3, full_fit_cut = 0.0, **kwargs):
        super(NMRPulseFullFit, self).__init__(filepath, **kwargs)

        ##########################################
        ### VARIOUS POSSIBLE FITTING FUNCTIONS ###
        ##########################################


        def g(t, w, dw_1, dw_2):
            ts = t / self.signal_duration
            return w*(1 + dw_1*ts + dw_2*ts**2)
            #return w*(1 + dw_1*ts + dw_2*ts**2 + dw_4 * ts**3 + dw_4 * ts**4)
            #return w*(1 + dw_1 * ts**4 + dw_2 * ts**6 + dw_3 * ts**8 + dw_4 * ts**10)

        def harmonics2_func(t, A_0 = 0.85,   w_0 = self.w_ref,   phi_0 = 0,
                         A_1 = 0.0, w_1 = 2*self.w_ref, phi_1 = 0,
                         A_2 = 0.0,  w_2 = 3*self.w_ref, phi_2 = 0,
                         dw_1 = 0.0, dw_2 = 0.0):
            return A_0*np.cos( g(t, w_0, dw_1, dw_2)*t + phi_0) \
                   + A_1*np.cos( g(t, w_1, dw_1, dw_2)*t + phi_1) \
                   + A_2*np.cos( g(t, w_2, dw_1, dw_2)*t + phi_2)

        def harmonics3_func(t, A_0 = 0.5,   w_0 = self.w_ref,   phi_0 = 0,
                         A_1 = -5e-2, w_1 = 2*self.w_ref, phi_1 = 0,
                         A_2 = 1e-2,  w_2 = 3*self.w_ref, phi_2 = 0,
                         A_3 = -2e-2, w_3 = 4*self.w_ref, phi_3 = 0,
                         dw_1 = 1.0e-3,  dw_2 = 300.0):
            return A_0*np.cos(2 * math.pi * g(t, w_0, dw_1, dw_2)*t + phi_0) \
                   + A_1*np.cos(2*math.pi * g(t, w_1, dw_1, dw_2)*t + phi_1) \
                   + A_2*np.cos(2*math.pi * g(t, w_2, dw_1, dw_2)*t + phi_2) \
                   + A_3*np.cos(2*math.pi * g(t, w_3, dw_1, dw_2)*t + phi_3)

        def harmonics4_func(t, A_0 = 0.5,   w_0 = self.w_ref,   phi_0 = 0,
                         A_1 = -5e-2, w_1 = 2*self.w_ref, phi_1 = 0,
                         A_2 = 1e-2,  w_2 = 3*self.w_ref, phi_2 = 0,
                         A_3 = -2e-2, w_3 = 4*self.w_ref, phi_3 = 0,
                         A_4 = -5e-3, w_4 = 5*self.w_ref, phi_4 = 0,
                         dw_1 = 0.0,  dw_2 = 0.0):
            return A_0*np.cos( g(t, w_0, dw_1, dw_2)*t + phi_0) \
                   + A_1*np.cos( g(t, w_1, dw_1, dw_2)*t + phi_1) \
                   + A_2*np.cos( g(t, w_2, dw_1, dw_2)*t + phi_2) \
                   + A_3*np.cos( g(t, w_3, dw_1, dw_2)*t + phi_3) \
                   + A_4*np.cos( g(t, w_4, dw_1, dw_2)*t + phi_4)


        def harmonics4_no_evol_func(t, A_0 = 0.5,   w_0 = self.w_ref,   phi_0 = 0,
                         A_1 = -5e-2, w_1 = 2*self.w_ref, phi_1 = 0,
                         A_2 = 1e-2,  w_2 = 3*self.w_ref, phi_2 = 0,
                         A_3 = -2e-2, w_3 = 4*self.w_ref, phi_3 = 0,
                         A_4 = -5e-3, w_4 = 5*self.w_ref, phi_4 = 0):
            return A_0*np.cos(2 * math.pi * w_0*t + phi_0) \
                   + A_1*np.cos(2*math.pi * w_1*t + phi_1) \
                   + A_2*np.cos(2*math.pi * w_2*t + phi_2) \
                   + A_3*np.cos(2*math.pi * w_3*t + phi_3) \
                   + A_4*np.cos(2*math.pi * w_4*t + phi_4)

        def harmonics5_func(t, A_0 = 0.5,   w_0 = self.w_ref,   phi_0 = 0,
                         A_1 = -5e-2, w_1 = 2*self.w_ref, phi_1 = 0,
                         A_2 = 1e-2,  w_2 = 3*self.w_ref, phi_2 = 0,
                         A_3 = -2e-2, w_3 = 4*self.w_ref, phi_3 = 0,
                         A_4 = -5e-3, w_4 = 5*self.w_ref, phi_4 = 0,
                         A_5 = -5e-3, w_5 = 6*self.w_ref, phi_5 = 0,
                         dw_1 = 100.0,  dw_2 = 260.0):
            return A_0*np.cos(2 * math.pi * g(t, w_0, dw_1, dw_2)*t + phi_0) \
                   + A_1*np.cos(2*math.pi * g(t, w_1, dw_1, dw_2)*t + phi_1) \
                   + A_2*np.cos(2*math.pi * g(t, w_2, dw_1, dw_2)*t + phi_2) \
                   + A_3*np.cos(2*math.pi * g(t, w_3, dw_1, dw_2)*t + phi_3) \
                   + A_4*np.cos(2*math.pi * g(t, w_4, dw_1, dw_2)*t + phi_4) \
                   + A_5*np.cos(2*math.pi * g(t, w_5, dw_1, dw_2)*t + phi_5)

        def harmonics6_func(t, A_0 = 0.5,   w_0 = self.w_ref,   phi_0 = 0,
                         A_1 = -5e-2, w_1 = 2*self.w_ref, phi_1 = 0,
                         A_2 = 1e-2,  w_2 = 3*self.w_ref, phi_2 = 0,
                         A_3 = -2e-2, w_3 = 4*self.w_ref, phi_3 = 0,
                         A_4 = -5e-3, w_4 = 5*self.w_ref, phi_4 = 0,
                         A_5 = -5e-3, w_5 = 6*self.w_ref, phi_5 = 0,
                         A_6 = -5e-4, w_6 = 7*self.w_ref, phi_6 = 0,
                         dw_1 = 15.0,  dw_2 = 260.0):
            return A_0*np.cos(2 * math.pi * g(t, w_0, dw_1, dw_2)*t + phi_0) \
                   + A_1*np.cos(2*math.pi * g(t, w_1, dw_1, dw_2)*t + phi_1) \
                   + A_2*np.cos(2*math.pi * g(t, w_2, dw_1, dw_2)*t + phi_2) \
                   + A_3*np.cos(2*math.pi * g(t, w_3, dw_1, dw_2)*t + phi_3) \
                   + A_4*np.cos(2*math.pi * g(t, w_4, dw_1, dw_2)*t + phi_4) \
                   + A_5*np.cos(2*math.pi * g(t, w_5, dw_1, dw_2)*t + phi_5) \
                   + A_6*np.cos(2*math.pi * g(t, w_6, dw_1, dw_2)*t + phi_6)


        def harmonics7_func(t, A_0 = 0.5,   w_0 = self.w_ref,   phi_0 = 0,
                         A_1 = -5e-2, w_1 = 2*self.w_ref, phi_1 = 0,
                         A_2 = 1e-2,  w_2 = 3*self.w_ref, phi_2 = 0,
                         A_3 = -2e-2, w_3 = 4*self.w_ref, phi_3 = 0,
                         A_4 = -5e-3, w_4 = 5*self.w_ref, phi_4 = 0,
                         A_5 = -5e-3, w_5 = 6*self.w_ref, phi_5 = 0,
                         A_6 = -5e-4, w_6 = 7*self.w_ref, phi_6 = 0,
                         A_7 = -1e-4, w_7 = 8*self.w_ref, phi_7 = 0,
                         dw_1 = 0.01528475,  dw_2 = -0.03984211, dw_3 = 0.02873211):
            return A_0*np.cos(2 * math.pi * g(t, w_0, dw_1, dw_2, dw_3)*t + phi_0) \
                   + A_1*np.cos(2*math.pi * g(t, w_1, dw_1, dw_2, dw_3)*t + phi_1) \
                   + A_2*np.cos(2*math.pi * g(t, w_2, dw_1, dw_2, dw_3)*t + phi_2) \
                   + A_3*np.cos(2*math.pi * g(t, w_3, dw_1, dw_2, dw_3)*t + phi_3) \
                   + A_4*np.cos(2*math.pi * g(t, w_4, dw_1, dw_2, dw_3)*t + phi_4) \
                   + A_5*np.cos(2*math.pi * g(t, w_5, dw_1, dw_2, dw_3)*t + phi_5) \
                   + A_6*np.cos(2*math.pi * g(t, w_6, dw_1, dw_2, dw_3)*t + phi_6) \
                   + A_7*np.cos(2*math.pi * g(t, w_7, dw_1, dw_2, dw_3)*t + phi_7)

        def harmonics0_func(t, A_0 = 0.75, w_0 = self.w_ref, phi_0 = 0, dw_1 = 0.0, dw_2 = 0.0):
            return A_0*np.cos(g(t, w_0, dw_1, dw_2)*t + phi_0)

        def harmonics0_no_evol_func(t, A_0 = 0.5, w_0 = self.w_ref, phi_0 = 0):
            return A_0*np.cos(w_0 * t + phi_0)

        def envelope3_func(t, e_1 = 4.5, e_2 = -0.25, e_3 = 2.0):
            t2 = t / self.signal_duration
            return np.exp(-e_1*t2 - e_2*t2**2 - e_3*t2**3)

        def envelope2_func(t, e_1 = 166, e_2 = 21e3):
            return np.exp(-e_1*t - e_2*t**2)

        def offset_func(t, c = 0.0): return c

        pulse_model = lmfit.Model(envelope3_func) * lmfit.Model(harmonics2_func) + lmfit.Model(offset_func)

        si = full_fit_cut / self.sampling_period
        ei = full_fit_stop / self.sampling_period
        if full_fit_use_filter:
            cut_signal = self.filter_signal[:ei]
            cut_times = self.filter_times[:ei]
        else:
            cut_signal = self.raw_signal[si:ei]
            cut_times = self.raw_times[si:ei]

        self.fit_data = pulse_model.fit(cut_signal, t=cut_times)
        self.best_fit = self.fit_data.best_fit
        self.best_fit_times = cut_times
        self.fit_fft = np.fft.rfft(self.best_fit, self.num_fft_samples)

        self.fit_residuals = self.best_fit - cut_signal
        self.fit_residuals_fft = np.fft.rfft(self.fit_residuals, self.num_fft_samples)
        if full_fit_use_filter:
            self.fit_chisq = red_chi_sq(cut_signal, self.best_fit, self.filter_std_dev)
        else:
            self.fit_chisq = red_chi_sq(cut_signal, self.best_fit, self.std_dev)

        self.fit_rms = rms(self.fit_residuals)
        self.fit_freq = self.fit_data.best_values['w_0'] / (2*math.pi)

        self.freq_fit = (g(self.best_fit_times, self.fit_data.best_values['w_0'],
                self.fit_data.best_values['dw_1'],
                self.fit_data.best_values['dw_2'])/(2*math.pi))

class NMRPulsePhaseFit(NMRPulseFiltered):
    def __init__(self, filepath, phase_fit_use_filter = False, phase_fit_stop = 6e-3, **kwargs):
        super(NMRPulsePhaseFit, self).__init__(filepath, **kwargs)

        cut_index = phase_fit_stop/self.sampling_period

        if phase_fit_use_filter:
            cut_signal = self.filter_signal[:cut_index]
            cut_times = self.filter_times[:cut_index]
        else:
            cut_signal = self.raw_signal[:cut_index]
            cut_times = self.raw_times[:cut_index]

        zcs = linear_zc(cut_times, cut_signal, self.w_ref, self.sampling_period)

        zc_hz = np.zeros(zcs.size - 1)
        for i in range(zcs.size - 1):
            t1 = zcs[i]
            t2 = zcs[i+1]
            half_period = t2 - t1
            zc_hz[i] = 1 / (2 * half_period)

        self.zc_freq = np.mean(zc_hz)

        self.phase_data = np.fromfunction(lambda i: i*math.pi, [zcs.size])
        self.phase_times = zcs

        def phase_func(t, a = 0.0, b = 0.0, c = 0.0, h=0.0):
            t2 = t/self.signal_duration
            return self.w_ref*(1 + a + b*t2 + c*t2**2)*t + h

        phase_model = lmfit.Model(phase_func)
        self.phase_fit = phase_model.fit(self.phase_data, t=self.phase_times)
        self.phase_freq = self.w_ref*(1 + self.phase_fit.best_values['a'])/(2*math.pi)

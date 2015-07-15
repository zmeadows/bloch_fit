import math
import numpy as np
import lmfit
from scipy import interpolate
from scipy.optimize import minimize
from scipy.signal import hilbert
import time

VERSION = "0.4.4"
VERSION_STRING = "[BLOCH FIT v" + VERSION + "]"

class NMRPulse(object):
    ''' A base class that holds the raw sample data.
        All fitting/analysis classes inherit from this.
    '''
    def __init__(self, filepath, sampling_period = 1.0e-7, w_ref = 2*math.pi*10e3,
                 time_since_pi2_pulse = 0.0, signal_duration = 30e-3, init_signal_cut = 0.0,
                 debug = False, **kwargs):
        ''' PARAMS:
            filepath -- path to pulse data file.

            sampling_period -- the sampling period of the data

            w_ref -- the (rough) expected frequency of the pulse

            time_since_pi2_pulse -- time between the end of the pi/2 rf pulse and the
                                    start of the sampled data

            signal_duration -- rough estimate of the duration of the pulse

            init_signal_cut -- amount of data to cut from the start of the pulse
                               to eliminate switch/ADC noise
        '''
        super(NMRPulse, self).__init__(**kwargs)

        self.data_file       = filepath
        self.sampling_period = sampling_period
        self.w_ref           = w_ref
        self.T_ref           = 2*math.pi / w_ref
        self.signal_duration = signal_duration
        self.debug           = debug

        ###############################
        ### LOAD RAW ADC PULSE DATA ###
        ###############################
        self.debug_print("LOADING DATA FILE...")
        self.debug_print("FILEPATH: " + filepath)
        start_time = time.clock()

        if (filepath[-4:] == ".bin"):
            signal_data = np.fromfile(filepath, dtype=np.dtype('u2'))
            time_indices = np.arange(signal_data.size)

        elif (filepath[-4:] == ".dat"):
            time_indices, signal_data = np.loadtxt(filepath, delimiter=' ',
                                                   usecols=(0, 1), unpack=True)
        else:
            print "INVALID/UNKNOWN DATA FILE EXTENSION."
            exit(1)

        self.debug_print("LOADED " + str(time_indices.size) + " SAMPLES.")

        assert(time_indices.size > 0 and signal_data.size > 0)
        assert(time_indices.size == signal_data.size)

        ###################
        ### CUT/CONVERT ###
        ###################
        signal_data  = signal_data[init_signal_cut/self.sampling_period:]
        time_indices = time_indices[init_signal_cut/self.sampling_period:]

        self.num_samples = time_indices.size

        def adc_to_voltage(adc_value):
            ''' convert the ADC values to a voltage, then correct for the offset '''
            return (adc_value - 32599.9) / 30465.9 # values from Flay, don't change!

        scaled_data = np.vectorize(adc_to_voltage)(signal_data)
        offset_estimate = np.average(scaled_data[3*scaled_data.size/4:])
        self.debug_print("OFFSET ESTIMATE: " + str(offset_estimate) + " V")
        shifted_scaled_data = scaled_data - offset_estimate

        self.raw_signal = shifted_scaled_data
        self.raw_times = time_since_pi2_pulse + sampling_period * time_indices

        # calculate the inherent noise of the data by taking the RMS of the tail
        self.std_dev = rms(self.raw_signal[7*self.num_samples/8:self.num_samples])
        self.debug_print("NOISE ESTIMATE: " + str(self.std_dev) + " V")

        end_time = time.clock()
        self.debug_print("FINISHED. (t = " + str(end_time - start_time) + " sec.)\n")

    def debug_print(self, msg):
        ''' print a debug message with bloch fit version prefixed '''
        if self.debug:
            print VERSION_STRING, msg

class NMRPulseFFT(NMRPulse):
    ''' Adds a fourier transform to the raw pulse class. '''
    def __init__(self, filepath, fit_fft_peak = False, **kwargs):
        ''' PARAMS:
            filepath -- path to pulse data file.

            fit_fft_peak -- if True, uses a peak finding and interpolation algorithm to
                            fit a spline to the peak of the fourier transform.
        '''
        super(NMRPulseFFT, self).__init__(filepath, **kwargs)

        #############################
        ### CUT AND CALCULATE FFT ###
        #############################

        fft_cut_index = int(self.signal_duration/self.sampling_period)

        # increase the number of fft samples for 'zero padding'
        self.num_fft_samples = fft_cut_index * 6

        self.debug_print("CALCULATING FOURIER TRANSFORM...")
        start_time = time.clock()
        self.raw_fft = np.fft.rfft(self.raw_signal[:fft_cut_index], self.num_fft_samples)
        self.raw_freqs = np.fft.rfftfreq(self.num_fft_samples, self.sampling_period)
        end_time = time.clock()
        self.debug_print("FINISHED. (t = " + str(end_time - start_time) + " sec.)\n")

        ###############################
        ### FIND FFT PEAK FREQUENCY ###
        ###############################

        # TODO: make this more rigorous
        if fit_fft_peak:
            self.debug_print("LOCATING FFT PEAK...")
            start_time = time.clock()
            delta_hz = self.raw_freqs[1] - self.raw_freqs[0]
            freq_guess = self.w_ref / (2*math.pi)
            start_peak_ind = (freq_guess - 1000)/delta_hz
            end_peak_ind = (freq_guess + 1000)/delta_hz

            f = interpolate.interp1d(self.raw_freqs[start_peak_ind:end_peak_ind],
                    np.abs(self.raw_fft)[start_peak_ind:end_peak_ind])

            self.fft_freq = minimize(lambda h: -1 * f(h), [freq_guess],
                                     bounds=[(freq_guess-70,freq_guess+70)],
                                     tol=1.0e-3 ).x[0]

            self.fft_fit_freqs = np.linspace(self.raw_freqs[start_peak_ind] + 100,
                                             self.raw_freqs[end_peak_ind] - 100,
                                             4 * (end_peak_ind - start_peak_ind))
            self.fft_fit = np.vectorize(f)(self.fft_fit_freqs)
            end_time = time.clock()
            self.debug_print("FINISHED. (t = " + str(end_time - start_time) + " sec.)\n")

class NMRPulseFiltered(NMRPulseFFT):
    ''' Filters the pulse with a user chosen filter shape, possibly to be used in fitting
        classes derived from this class.
    '''
    def __init__(self, filepath, filter_type = "STEP", filter_time_cut = 1e-4, **kwargs):
        ''' PARAMS:
            filepath -- path to pulse data file.

            filter_type -- the type of filter to apply to the data, right now the only option
                           is "step"

            filter_time_cut -- An amount of time to cut off the start/end of the signal
                               AFTER filtering, to account for the effects of the filter
        '''
        super(NMRPulseFiltered, self).__init__(filepath, **kwargs)

        self.debug_print("FILTERING SIGNAL...")
        self.debug_print("FILTER TYPE: " + filter_type)
        start_time = time.clock()

        def step_filter(freq):
            hz_ref = self.w_ref / (2 * math.pi)
            cutoff_freq = 1.75 * hz_ref
            scale = hz_ref / 16
            if (freq < 20 * hz_ref):
                return 1 / ( np.exp( (freq - cutoff_freq) / scale ) + 1 )
            else: return 0

        filter_cut_index = filter_time_cut / self.sampling_period

        self.filter = np.vectorize(step_filter)(self.raw_freqs)
        self.filter_fft = np.multiply(self.filter, self.raw_fft)
        self.filter_signal = np.fft.irfft(self.filter_fft)[filter_cut_index : self.raw_signal.size]
        self.filter_times = self.raw_times[filter_cut_index:]
        self.filter_std_dev = rms(self.filter_signal[7*self.num_samples/8:self.num_samples])

        end_time = time.clock()
        self.debug_print("FINISHED. (t = " + str(end_time - start_time) + " sec.)\n")

class NMRPulseFullFit(NMRPulseFiltered):
    ''' Fits the entire pulse in order to find the frequency '''
    def __init__(self, filepath, full_fit_use_filter = False, full_fit_stop = 10e-3,
                 fit_harmonics = True, env_deg = 3, amplitude_guess = 0.75, **kwargs):
        ''' PARAMS:
            filepath -- path to pulse data file.

            full_fit_use_filter -- if True, the signal will be filtered before the fitting
                                   routine.

            full_fit_stop -- The duration of the signal to apply the fit to.

            fit_harmonics -- If True, the fitting routine will include the first seven mixer
                             harmonics in the fit.

            env_deg -- Determines which envelope to use in the fit. (1,2 or 3)

            amplitude_guess -- An initial guess of the t=0 signal amplitude which helps the
                               fit converge faster.
        '''
        super(NMRPulseFullFit, self).__init__(filepath, **kwargs)

        def g(t, w, dw_1, dw_2):
            ''' Frequency fit function '''
            t2 = t / self.signal_duration
            return w*(1 + dw_1*t2 + dw_2*t2**2)

        def envelope3_func(t, e_1 = 0.0, e_2 = 0.0, e_3 = 0.0):
            t2 = t / self.signal_duration
            return np.exp(-e_1*t2 - e_2*t2**2 - e_3*t2**3)

        def envelope2_func(t, e_1 = 0.0, e_2 = 0.0):
            t2 = t / self.signal_duration
            return np.exp(-e_1*t2 - e_2*t2**2)

        def envelope1_func(t, e_1 = 0.0):
            t2 = t / self.signal_duration
            return np.exp(-e_1*t2)

        def envelope_sinc_func(t, e_1 = 0.0, e_2 = 0.0, e_3 = 0.0, e_4 = 1.0):
            t2 = t / self.signal_duration
            return np.sinc(e_4*t2)*np.exp(-e_1*t2 - e_2*t2**2 - e_3*t2**3)

        def offset_func(t, c = 0.0): return c

        def harmonics0_func(t, A_0 = amplitude_guess, w_0 = self.w_ref, phi_0 = 0,
                            dw_1 = 0.0, dw_2 = 0.0):
            return A_0*np.cos(g(t, w_0, dw_1, dw_2)*t + phi_0)

        def harmonics7_func(t, A_0 = amplitude_guess, w_0 = self.w_ref, phi_0 = 0,
                            A_1 = 0.0,  phi_1 = 0,
                            A_2 = 0.0,  phi_2 = 0,
                            A_3 = 0.0,  phi_3 = 0,
                            A_4 = 0.0,  phi_4 = 0,
                            A_5 = 0.0,  phi_5 = 0,
                            A_6 = 0.0,  phi_6 = 0,
                            A_7 = 0.0,  phi_7 = 0,
                            dw_1 = 0.0,  dw_2 = 0.0):
            return   A_0*np.cos( g(t, w_0, dw_1, dw_2)*t + phi_0) \
                   + A_1*np.cos( g(t, 2*w_0, dw_1, dw_2)*t + phi_1) \
                   + A_2*np.cos( g(t, 3*w_0, dw_1, dw_2)*t + phi_2) \
                   + A_3*np.cos( g(t, 4*w_0, dw_1, dw_2)*t + phi_3) \
                   + A_4*np.cos( g(t, 5*w_0, dw_1, dw_2)*t + phi_4) \
                   + A_5*np.cos( g(t, 6*w_0, dw_1, dw_2)*t + phi_5) \
                   + A_6*np.cos( g(t, 7*w_0, dw_1, dw_2)*t + phi_6) \
                   + A_7*np.cos( g(t, 8*w_0, dw_1, dw_2)*t + phi_7)

        if env_deg == 1:
            envelope_model = lmfit.Model(envelope1_func)
        elif env_deg == 2:
            envelope_model = lmfit.Model(envelope2_func)
        elif env_deg == 3:
            envelope_model = lmfit.Model(envelope3_func)
        else:
            print("invalid env_deg parameter passed, must be 1, 2, or 3")
            exit(1)

        if fit_harmonics:
            harmonics_model = lmfit.Model(harmonics7_func)
        else:
            harmonics_model = lmfit.Model(harmonics0_func)

        pulse_model = envelope_model * harmonics_model + lmfit.Model(offset_func)

        ei = full_fit_stop / self.sampling_period
        if full_fit_use_filter:
            cut_signal = self.filter_signal[:ei]
            cut_times = self.filter_times[:ei]
        else:
            cut_signal = self.raw_signal[:ei]
            cut_times = self.raw_times[:ei]

        if self.debug: print "FITTING FULL PULSE..."
        self.fit_data = pulse_model.fit(cut_signal, t=cut_times)
        if self.debug: print "FINISHED.\n"

        self.best_fit = self.fit_data.best_fit
        self.best_fit_times = cut_times

        if self.debug: print "CALCULATING FIT FOURIER TRANSFORM..."
        self.fit_fft = np.fft.rfft(self.best_fit, self.num_fft_samples)
        if self.debug: print "FINISHED.\n"

        self.fit_residuals = self.best_fit - cut_signal
        self.fit_residuals_fft = np.fft.rfft(self.fit_residuals, self.num_fft_samples)
        if full_fit_use_filter:
            self.fit_chisq = red_chi_sq(cut_signal, self.best_fit, self.filter_std_dev)
        else:
            self.fit_chisq = red_chi_sq(cut_signal, self.best_fit, self.std_dev)

        self.fit_rms = rms(self.fit_residuals)
        self.fit_freq = self.fit_data.best_values['w_0'] / (2*math.pi)

        if self.debug:
            print "FIT FREQUENCY: ", self.fit_freq
            print "FIT RMS: ", self.fit_rms
            print "FIT CHISQ: ", self.fit_chisq

        self.freq_fit = (g(self.best_fit_times, self.fit_data.best_values['w_0'],
                           self.fit_data.best_values['dw_1'],
                           self.fit_data.best_values['dw_2'])/(2*math.pi))

# TODO: see if offset can be more accurately determined
# TODO: make this more rigorous (make 'jump' stuff more predictable over
# different freqs)
class NMRPulseZCAnalysis(NMRPulseFiltered):
    ''' Class used to analyze the zero crossings of the pulse. '''
    def __init__(self, filepath, zc_use_filter = False, zc_stop = 6e-3, **kwargs):
        ''' PARAMS:
            filepath -- path to pulse data file.

            zc_fit_use_filter -- if True, the signal will be filtered before the zero crossings
                                 locations are determined.

            zc_stop -- The duration over which to look for zero crossings.
        '''
        super(NMRPulseZCAnalysis, self).__init__(filepath, **kwargs)
        cut_index = zc_stop/self.sampling_period

        if zc_use_filter:
            cut_signal = self.filter_signal[:cut_index]
            cut_times = self.filter_times[:cut_index]
        else:
            cut_signal = self.raw_signal[:cut_index]
            cut_times = self.raw_times[:cut_index]

        zcs = []
        T = (2*math.pi) / self.w_ref
        jump = (T/4) / self.sampling_period

        def simp_line(t, m, b):
            return m*t + b

        self.debug_print("CALCULATING ZERO CROSSINGS...")
        start_time = time.clock()

        i = jump/8 + 1
        # TODO: check that abs(y2-y1) < some number based on amplitude of
        # signal, in order to identify ADC glitches
        while (i < cut_times.size):
            y1 = cut_signal[i-1]
            y2 = cut_signal[i]
            if y1 * y2 < 0:
                t1 = cut_times[i-1]
                t2 = cut_times[i]

                m_guess = (y2 - y1) / (t2 - t1)
                b_guess = - m_guess * (t1 + t2)/2
                model = lmfit.Model(simp_line)
                model.set_param_hint('m', value = m_guess)
                model.set_param_hint('b', value = b_guess)
                pars = model.make_params()

                fit = model.fit(cut_signal[i-jump/8:i+jump/8], pars,
                                t=cut_times[i-jump/8:i+jump/8])

                m = fit.best_values['m']
                b = fit.best_values['b']

                zcs.append(-b/m)
                i += jump
            else:
                i += 1

        self.zero_crossings = np.array(zcs)

        half_periods = []
        for i in range(self.zero_crossings.size - 1):
            half_periods.append(self.zero_crossings[i + 1] - self.zero_crossings[i])
        self.zc_freq = 0.5/np.mean(half_periods)

        end_time = time.clock()
        self.debug_print("FOUND " + str(self.zero_crossings.size) + " ZERO CROSSINGS.")
        self.debug_print("MEAN FREQUENCY " + str(self.zc_freq) + " Hz.")
        self.debug_print("FINISHED. (t = " + str(end_time - start_time) + " sec.)\n")

class NMRPulsePhaseFit(NMRPulseZCAnalysis):
    ''' Determine zero crossing locations and fit the phase of the pulse
        in order to calculate the mean frequency (Cowan/Flowers method)
    '''
    def __init__(self, filepath, **kwargs):
        ''' PARAMS:
            filepath -- path to pulse data file.
        '''
        super(NMRPulsePhaseFit, self).__init__(filepath, **kwargs)

        self.phase_data = np.fromfunction(lambda i: i*math.pi, [self.zero_crossings.size])
        self.phase_times = self.zero_crossings

        ###############################
        ### CALCULATE UNCERTAINTIES ###
        ###############################

        phase_uncertainties = np.zeros(self.zero_crossings.size)
        for i in range(self.zero_crossings.size):
            zc_index_est = self.zero_crossings[i] / self.sampling_period
            next_peak_index = zc_index_est + (self.T_ref / 4) / self.sampling_period
            low_search = next_peak_index - (self.T_ref / 10) / self.sampling_period
            high_search = next_peak_index + (self.T_ref / 10) / self.sampling_period
            peak_height = np.max(np.abs(self.raw_signal[low_search:high_search]))
            phase_uncertainties[i] = self.std_dev / peak_height

        #################
        ### PHASE FIT ###
        #################

        def phase_func(t, a = 0.0, b = 0.0, c = 0.0, d=0.0):
            t2 = t/self.signal_duration
            return self.w_ref*(1 + a + b*t2 + c*t2**2)*t + d

        phase_model = lmfit.Model(phase_func)

        self.debug_print("FITTING PHASE...")
        self.phase_fit = phase_model.fit(self.phase_data, t=self.phase_times,
                                         weights=1/phase_uncertainties)
        self.phase_freq = self.w_ref*(1 + self.phase_fit.best_values['a'])/(2*math.pi)
        self.debug_print("PHASE FREQ: " + str(self.phase_freq))
        self.debug_print("FINISHED.\n")

        def phase_freq_func(t):
                t2 = t/self.signal_duration
                a = self.phase_fit.best_values['a']
                b = self.phase_fit.best_values['b']
                c = self.phase_fit.best_values['c']
                return self.w_ref*(1 + a + 2*b*t2 + 3*c*t2**2)

        self.phase_freq_vs_time = np.vectorize(phase_freq_func)(self.phase_times)

class NMRHilbertFit(NMRPulseFiltered):
    ''' Use a Hilbert transform to calculate frequency vs. time. '''
    def __init__(self, filepath, hilbert_use_filter = False, hilbert_stop = 10e-3, hilbert_cut = 25e-4, **kwargs):
        ''' PARAMS:
            filepath -- path to pulse data file.
        '''
        super(NMRHilbertFit, self).__init__(filepath, **kwargs)

        stop_index = hilbert_stop / self.sampling_period
        self.hilbert_times = self.raw_times[:stop_index]

        if hilbert_use_filter:
            cut_signal = self.filter_signal[:stop_index]
        else:
            cut_signal = self.raw_signal[:stop_index]

        self.hilbert_transform = hilbert(cut_signal)
        cut_index = hilbert_cut / self.sampling_period
        self.hilbert_times = self.hilbert_times[cut_index:]
        self.hilbert_amp = np.abs(self.hilbert_transform)[cut_index: -cut_index]

        self.hilbert_phase = np.angle(self.hilbert_transform)

        self.hilbert_phase = self.hilbert_phase[cut_index:self.hilbert_phase.size - cut_index]
        self.hilbert_times = self.hilbert_times[:self.hilbert_times.size - cut_index]

        self.hilbert_phase = np.unwrap(self.hilbert_phase)

        def phase_func(t, a = 0.0, b = 0.0, c = 0.0, d=0.0):
            t2 = t/self.signal_duration
            return self.w_ref*(1 + a + b*t2 + c*t2**2)*t + d

        phase_model = lmfit.Model(phase_func)
        self.hilbert_fit = phase_model.fit(self.hilbert_phase, t=self.hilbert_times,
                                           weights = self.hilbert_amp)
        self.hilbert_freq = self.w_ref*(1 + self.hilbert_fit.best_values['a'])/(2*math.pi)
        self.debug_print(self.hilbert_freq)

        self.hilbert_fit_nonlinear = phase_func(self.hilbert_times, a = 0,
                                                  b = self.hilbert_fit.best_values['b'],
                                                  c = self.hilbert_fit.best_values['c'], d = self.hilbert_fit.best_values['d']) - self.w_ref*self.hilbert_times
        self.hilbert_phase_nonlinear = self.hilbert_phase - self.w_ref*self.hilbert_times


def rms(arr):
    assert(arr.size > 1)
    return math.sqrt(np.mean(np.square(arr)))

def red_chi_sq(data, fit, sigma):
    assert(data.size == fit.size)
    ress = np.sum((fit - data)**2)
    chi_sq = ress / sigma**2
    return chi_sq / data.size

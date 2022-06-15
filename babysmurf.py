import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

def make_measured_signal(tvec,sig,squid,noise=None,reset_rate=1.e4,nphi0=4,fsamp=2.4e6):
   """
   Take the detector signal and produce a modulated resonator frequency. 
   This does not yet include resonator frequency estimation effects from SMuRF. 

   Args:
   tvec: time vector of signal. units = seconds
   sig: signal vector, of same size as time vector. units = radians
   squid: single period of SQUID curve, units = Hz (of resonance frequency) vs x-axis of 0 to 2pi. 
   noise: noise vector, of same size as MODULATED signal. units = ?; defaults None
   reset_rate: flux ramp sawtooth reset rate. units = Hz, defaults 10kHz
   nphi0: number of phi0 per ramp, need not be integer. units = none, defaults 4
   fsamp = per-channel sample rate of SMuRF. units = Hz, defualts 2.4MHz

   Returns:
   meas (delta(t)/reset_rate * frame_size) x 1: SQUID modulated signal. units = Hz, frame_size = fsamp / reset_rate
   tt (length(tvec) * frame_size) x 1: time vector of meas; this is sampled at fsamp. units = Hz
   ph (delta(t) * reset_rate) x 1: resampled signal at one point per flux ramp frame. mostly useful for plotting. 
   """

   framesize = int(fsamp / reset_rate)
   freqnorm = nphi0 * reset_rate / fsamp # normalized phi0 frequency
   nframes = np.floor((tvec[-1] - tvec[0]) / (1/reset_rate)) # enough frames to fully sample the signal at the reset rate

   obs_n = np.zeros((int(framesize * nframes), 1)) # preallocate the observed signal
   # plt.plot(tvec,sig); plt.show()
   # plt.plot(tvec); plt.plot(np.arange(nframes)/reset_rate,'.'); plt.show()
   sig_sampled = np.interp(np.arange(nframes)/reset_rate,np.ravel(tvec),np.ravel(sig)) # resample the signal to guarantee the right cadence

   # construct the SQUID signal
   # interpolate SQUID curve to the correct number of points
   squidint = np.interp(np.linspace(0,1,int(framesize/nphi0),endpoint=False),np.linspace(0,1,int(len(squid)),endpoint=False),squid)
   sigmod = np.matlib.repmat(np.reshape(squidint,(int(framesize/nphi0),1)),nphi0,1)

   for ii in np.arange(nframes):
     ii = int(ii)
     poff = sig_sampled[ii] # get the single point of phase that offsets the entire frame
     fracperiod = poff / (2*np.pi) # convert this phase offset to fraction of a period
     sigmod2 = np.roll(sigmod, 1*int(np.round(fracperiod/freqnorm))) # shift the SQUID curve by the desired fraction of a period
     obs_n[framesize*ii:framesize*(ii+1),:] = np.reshape(sigmod2,(framesize,1))


   # fill everything back in
   meas = obs_n
   tt = np.arange(len(meas)) / fsamp
   ph = sig_sampled

   return meas, tt, ph


def make_squid_curve(lda,amp,npts=600):
   """
   Make a SQUID curve given a lambda and amplitude. Returns a single period. 
   Some work will have to be done to resample this into the correct dimensions.
   This should be thought of as returning a frequency (in Hz) vs an x-axis of 0 to 2pi Phi0.
   
   Args:
   lda: SQUID curve lambda, we usually prefer chips that have lambda in the neighborhood of 0.33. Higher values are less sinusoidal. 
   amp: amplitude of the curve, units = Hz
   npts: number of points, defaults 600

   Returns:
   squid: a single period of the SQUID curve. May need to resample to get it to work with other functions. units = Hz
   """
   framesize = int(npts)
   freqnorm = 1 / framesize
   sigsin = np.sin(2*np.pi*freqnorm*(np.arange(framesize)))
   sigmod = (lda*np.transpose(sigsin)) / (1 + lda*np.transpose(sigsin))
   
   sigrange = np.max(sigmod) - np.min(sigmod)
   squid = (sigmod - np.mean(sigmod)) * amp/sigrange # center and scale

   return squid


def add_noise(sig,noise_spec,fsamp=2.4e6):
   """
   Add some noise on top of a signal before passing to demodulation. 
   This should be thought of as a frequency noise, so SQUID + resonator effects but not detector effects (those go straight into signal!)
   
   Args:
   sig: input signal to make noisy
   noise_spec: single-sided noise power spectral density to sample from; (n_points x 2), first column is frequency points and second column is a noise psd in Hz/rtHz
   fsamp: sample rate of sig, units = Hz, defaults 2.4MHz

   Returns: 
   noisy_sig: same size as input signal, but with added noise. Still in units of resonance frequency [Hz]
   """

   noise = np.zeros(np.shape(sig)) # set up as the same size as sig

   # TO DO FILL THIS IN

   noisy_sig = noise + sig
   return noisy_sig 

def lms_fit_demod(measurement,reset_rate=10.e3,nphi0=4,gain=1/32,blank=[0,1],fsamp=2.4e6,nharm=3):
   """
   The main SMuRF demodulation loop. Currently assumes "measurement" is perfect estimate of resonator frequency. 
   Args:
   measurement: the measured signal, in (nframes * fsamp / reset_rate) x 1. Unit = Hz
   reset_rate: flux ramp sawtooth reset rate. Units = Hz, defaults 10kHz
   nphi0: number of phi0 per flux ramp, unitless, defaults 4
   gain: feedback gain parameter, unitless, defaults 1/32. I need to figure out the normalization on this thing. 
   blank: start and end fractions of the flux ramp frame to use for phase estimation. unitless, defaults [0,1]
   fsamp: sample rate, units = Hz, defaults 2.4MHz
   nharm: number of harmonics to use in estimation, defaults 3. Currently SMuRF supports nharm <= 3; future iterations could use more.


   Returns:
   demod_sig: (nframes x 1), sampled at the reset rate. The demodulated phase of the first harmonic, units = radians
   phases: (length(measurement) x nharm), sampled at fsamp. The phases of each harmonic estimate in real time, units = radians
   # do I want to return yhat?
   errs: (same size as measurement), sampled at fsamp rate. The errors between the tracked frequency and estimate, units = Hz. 
   """
   framesize = int(fsamp / reset_rate) # number of time samples per flux ramp frame
   nframes = int(np.shape(measurement)[0] // framesize) # number of full flux ramp frames in the measurement

   H = make_obs_mat(reset_rate,nphi0,fsamp,nharm) # construct observation coefficient matrix

   alpha = np.ones((2*nharm+1,)) # initialize coefficient vector alpha
   yhat = np.zeros((framesize*nframes,1)) # initialize frequency estimate vector yhat

   phases = np.zeros((framesize*nframes,nharm)) # initialize phase estimates of harmonics
   errs = np.zeros((framesize*nframes,1)) # also initialize error vector

   for ii in range(framesize*nframes):
      blankmin = np.round(framesize*blank[0]) # get where in frame to start
      blankmax = np.round(framesize*blank[1]) # ...and where to stop

      rowno = np.mod(ii+framesize,framesize) # figure out where in the frame we are
      h = H[rowno,:] # get the harmonic components of this frame
      yhat[ii] = np.matmul(h,alpha) # estimate is based on existing alpha
      # this is SMuRF paper Eqs. 13 and 14

      if np.logical_or(rowno < blankmin, rowno > blankmax):
         alpha = alpha # in the blankoff region, do not update the coefficients
      else:
         # get the error
         errs[ii] = measurement[ii] - yhat[ii]
         # update the coefficients for the next point (SMuRF paper eq. 16)
         alpha = alpha + gain*errs[ii]*np.transpose(h) / (np.sum(np.power(h,2))) # h^2 is a normalization factor

      # save the phase estimate for each harmonic
      for jj in range(nharm):
         phases[ii,jj] = np.arctan2(alpha[2*jj+1],alpha[jj])

   demod = get_frame_averages(phases[:,0],reset_rate,fsamp)

   return demod, phases, errs 

def make_obs_mat(reset_rate=10.e3,nphi0=4,fsamp=2.4e6,nharm=3):
   """
   construct the observation matrix H for the SMuRF coefficient estimation
   this is currently SMuRF paper eq. 10

   Args:
   reset_rate: flux ramp sawtooth reset rate. Units = Hz, defaults 10kHz
   nphi0: number of phi0 per flux ramp, unitless, defaults 4
   fsamp: sample rate, units = Hz, defaults 2.4MHz
   nharm: number of harmonics to use in estimation, defaults 3. Currently SMuRF supports nharm <= 3; future iterations could use more.

   Returns:
   H: (M x 2N+1) matrix, where M is number of samples in a flux ramp frame (fsamp / reset_rate) and N is the number of harmonics used in the estimation. 
   """
   freqnorm = nphi0 * reset_rate / fsamp # normalized frequency, this is phi0 per sample
   framesize = fsamp / reset_rate # number of time samples per flux ramp frame

   H = np.zeros((int(framesize),2*nharm+1)) # construct observation matrix, per SMuRF paper equation 10
   for ii in range(nharm):
     H[:,2*ii] = np.cos(2*np.pi*(ii+1)*freqnorm*np.arange(framesize))
     H[:,2*ii+1] = np.sin(2*np.pi*(ii+1)*freqnorm*np.arange(framesize))

   H[:,2*nharm] = np.ones((int(framesize),)) # add constant term

   return H

def get_frame_averages(phase,reset_rate=10.e3,fsamp=2.4e6):
   """
   Average over flux ramp frames to return one sample per frame

   Args:
   phase: the per-harmonic phase (probably of the principal harmonic) as returned by lms_demod, units=radians. Implicit assumption that this starts at the beginning of a flux ramp frame. 
   reset_rate: flux ramp reset rate, units = Hz, defaults 10kHz
   fsamp: per channel sample rate, units = Hz, defaults 2.4MHz

   Returns:
   avg: demodulated phase averaged over the flux ramp periods, so at the reset_rate. Still in units of radians. 
   """
   # do this by averaging method to avoid for loops
   framesize = fsamp / reset_rate # number of samples per flux ramp frame
   nframes = len(phase)//framesize
   phase_reshaped = np.reshape(phase,(-1,int(nframes),int(framesize)))
   avg = np.transpose(np.sum(phase_reshaped,-1) / framesize)

   return avg

def filter_and_downsample(sig,filt,fsamp_new=200.,fr_rate=10.e3):
   """
   Filter and downsample a signal. 
   The SMuRF data streamer does this before passing off to ocs, gcp, etc. 
   Other data products are available for debugging only, but usually we only 
   write this streamed data to disk. 

   Args:
   sig: signal to filter and downsample. Probably in units of radians. 
   filt: filter coefficients [b,a], to be passed to scipy.filtfilt. 
   fsamp_new: new sample frequency to downsample to, units=Hz, defaults 200Hz
   fr_rate: flux ramp frequency (that data came out of the baseband processor at), units=Hz, defaults 10kHz

   Returns:
   new_sig: filtered and downsampled signal, this is probably what is written to disk
   """
   new_sig = sig
   return new_sig

def make_res_s21(f0,deltaf,q,asym=0,npts=100):
   """
   Make a resonator S21 (amplitude, phase) to interact with. 
   Note that this resonator is not guaranteed to be physical. 

   Args:
   f0: center frequency, units = Hz
   deltaf: bandwidth, units = Hz
   q: quality factor, unitless
   asym: asymmetry, for probing effects of the asymmetry, unitless, defaults 0
   npts: number of points to return, defaults 100 (~1kHz sampling for a 100kHz wide resonator)

   Returns (all npts x 1):
   fvec:  frequency points
   amp: S21 amplitude
   ph: S21 phase
   """

   return fvec,amp,ph

def estimate_eta(fvec,res_s21,offset=30.e3):
   """
   Estimate eta calibration angle per Eq. 3 of the SMuRF paper. 
   This estimation is critical to the SMuRF estimate of frequency!!!
   This step is done in the setup_notches step of pysmurf.

   Args:
   fvec: frequency points the S21 is sampled at, units=Hz
   res_s21: complex transmission (decide if I want this...)
   f0: center frequency, units=Hz
   offset: offset from center at which to sample eta, units=Hz, defaults 30kHz

   Returns:
   eta_mag,eta_phase: complex coefficients by which to rotate and scale resonator circle to perform frequency error estimates
   """

   return eta_mag,eta_phase

def estimate_frequency(fvec,res_s21,eta,probe_f):
   """
   Estimate frequency error per Eq. 4 of the SMuRF paper. 
   Assume that the fvec,res_s21 are the TRUE state of the resonance

   Args:
   fvec: frequency points the S21 is sampled at, units=Hz
   res_s21: complex transmission
   eta: calibration factor, as estimated from estimate_eta
   probe_f: frequency of the resonator probe tone, units=Hz. Currently assuming a perfectly spectrally clean probe tone. 
   """
   return fhat

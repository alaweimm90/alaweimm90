# signal_analysis
**Module:** `Python/Signal_Processing/signal_analysis.py`
## Overview
Signal Analysis and Processing
==============================
Comprehensive signal processing toolkit for scientific computing applications.
Includes time-domain analysis, frequency-domain analysis, filtering, and
advanced signal processing techniques.
Author: Berkeley SciComp Team
Date: 2024
## Constants
- **`BERKELEY_BLUE`**
- **`CALIFORNIA_GOLD`**
## Functions
### `demo_signal_processing()`
Demonstrate signal processing capabilities.
**Source:** [Line 612](Python/Signal_Processing/signal_analysis.py#L612)
## Classes
### `SignalProcessor`
Core signal processing operations for scientific applications.
Provides comprehensive tools for signal analysis, filtering,
spectral analysis, and feature extraction.
#### Methods
##### `__init__(self, sampling_rate)`
Initialize signal processor.
Args:
sampling_rate: Sampling frequency in Hz
**Source:** [Line 33](Python/Signal_Processing/signal_analysis.py#L33)
##### `generate_signal(self, signal_type, duration, frequency, amplitude, phase, noise_level)`
Generate various types of signals for testing and analysis.
Args:
signal_type: Type of signal ('sine', 'cosine', 'square', 'sawtooth', 'chirp', 'multi_sine')
duration: Signal duration in seconds
frequency: Signal frequency (Hz) or list of frequencies for multi-sine
amplitude: Signal amplitude or list of amplitudes
phase: Phase shift in radians or list of phases
noise_level: Gaussian noise standard deviation
Returns:
Time array and signal array
**Source:** [Line 43](Python/Signal_Processing/signal_analysis.py#L43)
##### `compute_fft(self, signal_data, window, normalize)`
Compute Fast Fourier Transform of signal.
Args:
signal_data: Input signal
window: Window function ('hann', 'hamming', 'blackman', etc.)
normalize: Whether to normalize the FFT
Returns:
Frequency array and FFT magnitude
**Source:** [Line 94](Python/Signal_Processing/signal_analysis.py#L94)
##### `compute_spectrogram(self, signal_data, window_size, overlap, window)`
Compute spectrogram using Short-Time Fourier Transform.
Args:
signal_data: Input signal
window_size: Size of the window for STFT
overlap: Overlap fraction between windows (0-1)
window: Window function name
Returns:
Time array, frequency array, and spectrogram magnitude
**Source:** [Line 132](Python/Signal_Processing/signal_analysis.py#L132)
##### `design_filter(self, filter_type, cutoff, order, ripple, attenuation)`
Design digital filters.
Args:
filter_type: 'lowpass', 'highpass', 'bandpass', 'bandstop'
cutoff: Cutoff frequency/frequencies (normalized to Nyquist)
order: Filter order
ripple: Passband ripple for Chebyshev filters (dB)
attenuation: Stopband attenuation for Chebyshev Type II (dB)
Returns:
Filter coefficients (b, a)
**Source:** [Line 157](Python/Signal_Processing/signal_analysis.py#L157)
##### `apply_filter(self, signal_data, b, a, method)`
Apply digital filter to signal.
Args:
signal_data: Input signal
b: Numerator coefficients
a: Denominator coefficients
method: 'filter' for causal, 'filtfilt' for zero-phase
Returns:
Filtered signal
**Source:** [Line 192](Python/Signal_Processing/signal_analysis.py#L192)
##### `detect_peaks(self, signal_data, height, threshold, distance, prominence)`
Detect peaks in signal.
Args:
signal_data: Input signal
height: Minimum peak height
threshold: Minimum difference to neighbors
distance: Minimum distance between peaks (samples)
prominence: Minimum peak prominence
Returns:
Dictionary with peak indices and properties
**Source:** [Line 217](Python/Signal_Processing/signal_analysis.py#L217)
##### `compute_envelope(self, signal_data, method)`
Compute signal envelope.
Args:
signal_data: Input signal
method: 'hilbert' for Hilbert transform, 'peak' for peak detection
Returns:
Signal envelope
**Source:** [Line 247](Python/Signal_Processing/signal_analysis.py#L247)
##### `compute_correlation(self, signal1, signal2, mode, normalize)`
Compute cross-correlation between two signals.
Args:
signal1: First signal
signal2: Second signal
mode: 'full', 'valid', or 'same'
normalize: Whether to normalize correlation
Returns:
Cross-correlation array
**Source:** [Line 279](Python/Signal_Processing/signal_analysis.py#L279)
##### `compute_coherence(self, signal1, signal2, window_size, overlap)`
Compute coherence between two signals.
Args:
signal1: First signal
signal2: Second signal
window_size: Size of window for spectral estimation
overlap: Overlap fraction
Returns:
Frequency array and coherence values
**Source:** [Line 303](Python/Signal_Processing/signal_analysis.py#L303)
##### `compute_snr(self, signal_data, noise_data)`
Compute Signal-to-Noise Ratio in dB.
Args:
signal_data: Clean signal
noise_data: Noise component
Returns:
SNR in dB
**Source:** [Line 326](Python/Signal_Processing/signal_analysis.py#L326)
##### `extract_features(self, signal_data)`
Extract statistical and spectral features from signal.
Args:
signal_data: Input signal
Returns:
Dictionary of features
**Source:** [Line 347](Python/Signal_Processing/signal_analysis.py#L347)
##### `resample_signal(self, signal_data, original_rate, target_rate)`
Resample signal to different sampling rate.
Args:
signal_data: Input signal
original_rate: Original sampling rate
target_rate: Target sampling rate
Returns:
Resampled signal
**Source:** [Line 395](Python/Signal_Processing/signal_analysis.py#L395)
##### `denoise_signal(self, signal_data, method)`
Denoise signal using various methods.
Args:
signal_data: Noisy signal
method: Denoising method ('wavelet', 'median', 'savgol')
**kwargs: Method-specific parameters
Returns:
Denoised signal
**Source:** [Line 420](Python/Signal_Processing/signal_analysis.py#L420)
**Class Source:** [Line 25](Python/Signal_Processing/signal_analysis.py#L25)
### `AdaptiveFilter`
Adaptive filtering algorithms for signal processing.
Implements LMS, NLMS, RLS and other adaptive filtering techniques.
#### Methods
##### `__init__(self, filter_order, step_size)`
Initialize adaptive filter.
Args:
filter_order: Order of the adaptive filter
step_size: Learning rate for adaptation
**Source:** [Line 480](Python/Signal_Processing/signal_analysis.py#L480)
##### `lms_filter(self, input_signal, desired_signal)`
Least Mean Squares (LMS) adaptive filter.
Args:
input_signal: Input signal
desired_signal: Desired/reference signal
Returns:
Filtered output and error signal
**Source:** [Line 492](Python/Signal_Processing/signal_analysis.py#L492)
##### `nlms_filter(self, input_signal, desired_signal, regularization)`
Normalized Least Mean Squares (NLMS) adaptive filter.
Args:
input_signal: Input signal
desired_signal: Desired/reference signal
regularization: Small constant to avoid division by zero
Returns:
Filtered output and error signal
**Source:** [Line 526](Python/Signal_Processing/signal_analysis.py#L526)
##### `rls_filter(self, input_signal, desired_signal, forgetting_factor)`
Recursive Least Squares (RLS) adaptive filter.
Args:
input_signal: Input signal
desired_signal: Desired/reference signal
forgetting_factor: Exponential weighting factor (0 < λ ≤ 1)
Returns:
Filtered output and error signal
**Source:** [Line 565](Python/Signal_Processing/signal_analysis.py#L565)
**Class Source:** [Line 473](Python/Signal_Processing/signal_analysis.py#L473)
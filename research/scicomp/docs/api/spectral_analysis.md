# spectral_analysis
**Module:** `Python/Signal_Processing/spectral_analysis.py`
## Overview
Spectral Analysis Module
========================
Advanced spectral analysis techniques for signal processing including
power spectral density, spectrograms, wavelets, and time-frequency analysis.
Author: Berkeley SciComp Team
Date: 2024
## Functions
### `demo_spectral_analysis()`
Demonstrate spectral analysis capabilities.
**Source:** [Line 555](Python/Signal_Processing/spectral_analysis.py#L555)
## Classes
### `SpectralAnalyzer`
Advanced spectral analysis tools for scientific signal processing.
Provides comprehensive spectral analysis including FFT, power spectral
density, time-frequency analysis, and wavelet transforms.
#### Methods
##### `__init__(self, sampling_rate)`
Initialize spectral analyzer.
Args:
sampling_rate: Sampling frequency in Hz
**Source:** [Line 35](Python/Signal_Processing/spectral_analysis.py#L35)
##### `compute_power_spectrum(self, signal_data, method, window, nperseg, noverlap, scaling)`
Compute power spectral density or power spectrum.
Args:
signal_data: Input signal
method: 'periodogram', 'welch', 'multitaper'
window: Window function for spectral estimation
nperseg: Length of each segment for Welch's method
noverlap: Number of points to overlap between segments
scaling: 'density' for PSD, 'spectrum' for power spectrum
Returns:
Frequency array and power spectral density/spectrum
**Source:** [Line 45](Python/Signal_Processing/spectral_analysis.py#L45)
##### `compute_cross_spectrum(self, signal1, signal2, window, nperseg, noverlap)`
Compute cross power spectral density between two signals.
Args:
signal1: First signal
signal2: Second signal
window: Window function
nperseg: Length of each segment
noverlap: Number of overlapping points
Returns:
Frequency array and cross PSD
**Source:** [Line 130](Python/Signal_Processing/spectral_analysis.py#L130)
##### `compute_transfer_function(self, input_signal, output_signal, window, nperseg, noverlap)`
Estimate transfer function between input and output signals.
Args:
input_signal: Input signal
output_signal: Output signal
window: Window function
nperseg: Segment length
noverlap: Overlap length
Returns:
Frequency array, transfer function magnitude, and phase
**Source:** [Line 157](Python/Signal_Processing/spectral_analysis.py#L157)
##### `compute_cepstrum(self, signal_data, type)`
Compute cepstrum of signal.
Args:
signal_data: Input signal
type: 'real' for real cepstrum, 'complex' for complex cepstrum
Returns:
Quefrency array and cepstrum
**Source:** [Line 194](Python/Signal_Processing/spectral_analysis.py#L194)
##### `compute_mel_spectrogram(self, signal_data, n_mels, n_fft, hop_length, window, fmin, fmax)`
Compute mel-scaled spectrogram.
Args:
signal_data: Input signal
n_mels: Number of mel bands
n_fft: FFT size
hop_length: Number of samples between frames
window: Window function
fmin: Minimum frequency
fmax: Maximum frequency (defaults to Nyquist)
Returns:
Time array, mel frequencies, mel spectrogram
**Source:** [Line 231](Python/Signal_Processing/spectral_analysis.py#L231)
##### `_hz_to_mel(self, frequencies)`
Convert frequency in Hz to mel scale.
**Source:** [Line 277](Python/Signal_Processing/spectral_analysis.py#L277)
##### `_mel_to_hz(self, mels)`
Convert mel scale to frequency in Hz.
**Source:** [Line 281](Python/Signal_Processing/spectral_analysis.py#L281)
##### `_create_mel_filterbank(self, n_mels, n_fft, sample_rate, fmin, fmax)`
Create mel filterbank matrix.
**Source:** [Line 285](Python/Signal_Processing/spectral_analysis.py#L285)
##### `compute_wavelet_transform(self, signal_data, wavelet, level, mode)`
Compute discrete wavelet transform.
Args:
signal_data: Input signal
wavelet: Wavelet name (e.g., 'db4', 'sym5', 'coif3')
level: Decomposition level (None for maximum)
mode: Signal extension mode
Returns:
Dictionary with approximation and detail coefficients
**Source:** [Line 317](Python/Signal_Processing/spectral_analysis.py#L317)
##### `compute_cwt(self, signal_data, scales, wavelet)`
Compute continuous wavelet transform.
Args:
signal_data: Input signal
scales: Scale array (default: logarithmic from 1 to 128)
wavelet: Wavelet name for CWT
Returns:
Time array, scale array, CWT coefficients
**Source:** [Line 353](Python/Signal_Processing/spectral_analysis.py#L353)
##### `compute_hilbert_huang(self, signal_data, n_imfs)`
Compute Hilbert-Huang transform (simplified EMD).
Args:
signal_data: Input signal
n_imfs: Number of Intrinsic Mode Functions to extract
Returns:
Dictionary with IMFs and instantaneous frequencies
**Source:** [Line 382](Python/Signal_Processing/spectral_analysis.py#L382)
##### `_extract_imf(self, signal_data, max_iterations)`
Extract one Intrinsic Mode Function using sifting.
Simplified EMD sifting process.
**Source:** [Line 431](Python/Signal_Processing/spectral_analysis.py#L431)
##### `compute_spectral_features(self, signal_data, n_bands)`
Extract comprehensive spectral features.
Args:
signal_data: Input signal
n_bands: Number of frequency bands for band-wise features
Returns:
Dictionary of spectral features
**Source:** [Line 470](Python/Signal_Processing/spectral_analysis.py#L470)
**Class Source:** [Line 27](Python/Signal_Processing/spectral_analysis.py#L27)
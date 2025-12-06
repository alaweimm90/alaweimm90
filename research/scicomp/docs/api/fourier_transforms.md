# fourier_transforms
**Module:** `Python/Signal_Processing/core/fourier_transforms.py`
## Overview
Fourier transform implementations and spectral analysis tools.
This module provides comprehensive Fourier analysis capabilities including
FFT, spectral analysis, and frequency domain processing.
Classes:
FFT: Fast Fourier Transform implementation
SpectralAnalysis: Spectral analysis tools
Author: Berkeley SciComp Team
Date: 2025
## Classes
### `FFT`
Fast Fourier Transform implementation.
#### Methods
##### `__init__(self)`
Initialize FFT class.
**Source:** [Line 23](Python/Signal_Processing/core/fourier_transforms.py#L23)
##### `compute_fft(self, signal, sample_rate)`
Compute FFT of signal.
Args:
signal: Input signal
sample_rate: Sampling rate
Returns:
Tuple of (frequencies, spectrum)
**Source:** [Line 27](Python/Signal_Processing/core/fourier_transforms.py#L27)
##### `compute_ifft(self, spectrum)`
Compute inverse FFT.
Args:
spectrum: Frequency domain signal
Returns:
Time domain signal
**Source:** [Line 50](Python/Signal_Processing/core/fourier_transforms.py#L50)
**Class Source:** [Line 20](Python/Signal_Processing/core/fourier_transforms.py#L20)
### `SpectralAnalysis`
Spectral analysis tools.
#### Methods
##### `__init__(self)`
Initialize spectral analysis.
**Source:** [Line 66](Python/Signal_Processing/core/fourier_transforms.py#L66)
##### `power_spectrum(self, signal, sample_rate)`
Compute power spectral density.
Args:
signal: Input signal
sample_rate: Sampling rate
Returns:
Tuple of (frequencies, power spectrum)
**Source:** [Line 70](Python/Signal_Processing/core/fourier_transforms.py#L70)
##### `spectrogram(self, signal, window_size, overlap, sample_rate)`
Compute spectrogram.
Args:
signal: Input signal
window_size: Size of analysis window
overlap: Overlap fraction between windows
sample_rate: Sampling rate
Returns:
Tuple of (time, frequency, spectrogram)
**Source:** [Line 91](Python/Signal_Processing/core/fourier_transforms.py#L91)
**Class Source:** [Line 63](Python/Signal_Processing/core/fourier_transforms.py#L63)
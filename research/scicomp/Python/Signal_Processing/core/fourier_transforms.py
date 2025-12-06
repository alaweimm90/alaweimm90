"""FFT and spectral analysis tools."""
import numpy as np
from typing import Tuple, Optional, Union
import warnings
class FFT:
    """Fast Fourier Transform implementation."""
    def __init__(self):
        """Initialize FFT class."""
        pass
    def compute_fft(self,
                   signal: np.ndarray,
                   sample_rate: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FFT of signal.
        Args:
            signal: Input signal
            sample_rate: Sampling rate
        Returns:
            Tuple of (frequencies, spectrum)
        """
        N = len(signal)
        # Compute FFT
        spectrum = np.fft.fft(signal)
        # Generate frequency array
        frequencies = np.fft.fftfreq(N, d=1/sample_rate)
        return frequencies, spectrum
    def compute_ifft(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Compute inverse FFT.
        Args:
            spectrum: Frequency domain signal
        Returns:
            Time domain signal
        """
        return np.fft.ifft(spectrum)
class SpectralAnalysis:
    """Spectral analysis tools."""
    def __init__(self):
        """Initialize spectral analysis."""
        pass
    def power_spectrum(self,
                      signal: np.ndarray,
                      sample_rate: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectral density.
        Args:
            signal: Input signal
            sample_rate: Sampling rate
        Returns:
            Tuple of (frequencies, power spectrum)
        """
        fft = FFT()
        frequencies, spectrum = fft.compute_fft(signal, sample_rate)
        # Compute power spectrum
        power = np.abs(spectrum)**2 / len(signal)
        return frequencies, power
    def spectrogram(self,
                   signal: np.ndarray,
                   window_size: int,
                   overlap: float = 0.5,
                   sample_rate: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram.
        Args:
            signal: Input signal
            window_size: Size of analysis window
            overlap: Overlap fraction between windows
            sample_rate: Sampling rate
        Returns:
            Tuple of (time, frequency, spectrogram)
        """
        N = len(signal)
        step = int(window_size * (1 - overlap))
        # Number of windows
        n_windows = (N - window_size) // step + 1
        # Initialize arrays
        times = np.arange(n_windows) * step / sample_rate
        frequencies = np.fft.fftfreq(window_size, d=1/sample_rate)
        spec = np.zeros((len(frequencies), n_windows), dtype=complex)
        # Compute STFT
        fft = FFT()
        for i in range(n_windows):
            start = i * step
            end = start + window_size
            window = signal[start:end]
            # Apply window function
            window = window * np.hanning(window_size)
            # Compute FFT
            _, spectrum = fft.compute_fft(window, sample_rate)
            spec[:, i] = spectrum
        return times, frequencies, spec
"""
Signal Analysis and Processing
==============================
Comprehensive signal processing toolkit for scientific computing applications.
Includes time-domain analysis, frequency-domain analysis, filtering, and
advanced signal processing techniques.
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
import scipy.signal as signal
import scipy.fft as fft
from scipy import stats
from typing import Tuple, Optional, Union, List, Dict, Any
import warnings
# Berkeley colors
BERKELEY_BLUE = '#003262'
CALIFORNIA_GOLD = '#FDB515'
class SignalProcessor:
    """
    Core signal processing operations for scientific applications.
    Provides comprehensive tools for signal analysis, filtering,
    spectral analysis, and feature extraction.
    """
    def __init__(self, sampling_rate: float = 1.0):
        """
        Initialize signal processor.
        Args:
            sampling_rate: Sampling frequency in Hz
        """
        self.sampling_rate = sampling_rate
        self.nyquist_freq = sampling_rate / 2
    def generate_signal(self, signal_type: str, duration: float,
                       frequency: Union[float, List[float]] = 1.0,
                       amplitude: Union[float, List[float]] = 1.0,
                       phase: Union[float, List[float]] = 0.0,
                       noise_level: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
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
        """
        t = np.linspace(0, duration, int(duration * self.sampling_rate), endpoint=False)
        if signal_type == 'sine':
            signal_data = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        elif signal_type == 'cosine':
            signal_data = amplitude * np.cos(2 * np.pi * frequency * t + phase)
        elif signal_type == 'square':
            signal_data = amplitude * signal.square(2 * np.pi * frequency * t + phase)
        elif signal_type == 'sawtooth':
            signal_data = amplitude * signal.sawtooth(2 * np.pi * frequency * t + phase)
        elif signal_type == 'chirp':
            f0 = frequency if isinstance(frequency, (int, float)) else frequency[0]
            f1 = frequency if isinstance(frequency, (int, float)) else frequency[-1]
            signal_data = amplitude * signal.chirp(t, f0, duration, f1)
        elif signal_type == 'multi_sine':
            signal_data = np.zeros_like(t)
            freqs = frequency if isinstance(frequency, list) else [frequency]
            amps = amplitude if isinstance(amplitude, list) else [amplitude] * len(freqs)
            phases = phase if isinstance(phase, list) else [phase] * len(freqs)
            for f, a, p in zip(freqs, amps, phases):
                signal_data += a * np.sin(2 * np.pi * f * t + p)
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")
        # Add noise if specified
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, len(signal_data))
            signal_data += noise
        return t, signal_data
    def compute_fft(self, signal_data: np.ndarray,
                    window: Optional[str] = None,
                    normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Fast Fourier Transform of signal.
        Args:
            signal_data: Input signal
            window: Window function ('hann', 'hamming', 'blackman', etc.)
            normalize: Whether to normalize the FFT
        Returns:
            Frequency array and FFT magnitude
        """
        N = len(signal_data)
        # Apply window if specified
        if window:
            window_func = signal.get_window(window, N)
            signal_windowed = signal_data * window_func
        else:
            signal_windowed = signal_data
        # Compute FFT
        fft_vals = fft.fft(signal_windowed)
        fft_freq = fft.fftfreq(N, 1/self.sampling_rate)
        # Take positive frequencies only
        positive_freq_idx = fft_freq >= 0
        fft_freq = fft_freq[positive_freq_idx]
        fft_mag = np.abs(fft_vals[positive_freq_idx])
        # Normalize if requested
        if normalize:
            fft_mag = 2 * fft_mag / N
        return fft_freq, fft_mag
    def compute_spectrogram(self, signal_data: np.ndarray,
                           window_size: int = 256,
                           overlap: float = 0.5,
                           window: str = 'hann') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute spectrogram using Short-Time Fourier Transform.
        Args:
            signal_data: Input signal
            window_size: Size of the window for STFT
            overlap: Overlap fraction between windows (0-1)
            window: Window function name
        Returns:
            Time array, frequency array, and spectrogram magnitude
        """
        nperseg = window_size
        noverlap = int(window_size * overlap)
        f, t, Sxx = signal.spectrogram(signal_data, self.sampling_rate,
                                       window=window, nperseg=nperseg,
                                       noverlap=noverlap, scaling='spectrum')
        return t, f, np.abs(Sxx)
    def design_filter(self, filter_type: str, cutoff: Union[float, List[float]],
                     order: int = 5, ripple: Optional[float] = None,
                     attenuation: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Design digital filters.
        Args:
            filter_type: 'lowpass', 'highpass', 'bandpass', 'bandstop'
            cutoff: Cutoff frequency/frequencies (normalized to Nyquist)
            order: Filter order
            ripple: Passband ripple for Chebyshev filters (dB)
            attenuation: Stopband attenuation for Chebyshev Type II (dB)
        Returns:
            Filter coefficients (b, a)
        """
        # Normalize cutoff frequencies
        if isinstance(cutoff, list):
            cutoff_norm = [f / self.nyquist_freq for f in cutoff]
        else:
            cutoff_norm = cutoff / self.nyquist_freq
        # Design filter based on specifications
        if ripple is not None:
            # Chebyshev Type I
            b, a = signal.cheby1(order, ripple, cutoff_norm, filter_type, analog=False)
        elif attenuation is not None:
            # Chebyshev Type II
            b, a = signal.cheby2(order, attenuation, cutoff_norm, filter_type, analog=False)
        else:
            # Butterworth
            b, a = signal.butter(order, cutoff_norm, filter_type, analog=False)
        return b, a
    def apply_filter(self, signal_data: np.ndarray, b: np.ndarray, a: np.ndarray,
                    method: str = 'filtfilt') -> np.ndarray:
        """
        Apply digital filter to signal.
        Args:
            signal_data: Input signal
            b: Numerator coefficients
            a: Denominator coefficients
            method: 'filter' for causal, 'filtfilt' for zero-phase
        Returns:
            Filtered signal
        """
        if method == 'filtfilt':
            # Zero-phase filtering
            filtered = signal.filtfilt(b, a, signal_data)
        elif method == 'filter':
            # Causal filtering
            filtered = signal.lfilter(b, a, signal_data)
        else:
            raise ValueError(f"Unknown filtering method: {method}")
        return filtered
    def detect_peaks(self, signal_data: np.ndarray,
                    height: Optional[float] = None,
                    threshold: Optional[float] = None,
                    distance: Optional[int] = None,
                    prominence: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Detect peaks in signal.
        Args:
            signal_data: Input signal
            height: Minimum peak height
            threshold: Minimum difference to neighbors
            distance: Minimum distance between peaks (samples)
            prominence: Minimum peak prominence
        Returns:
            Dictionary with peak indices and properties
        """
        peaks, properties = signal.find_peaks(signal_data,
                                             height=height,
                                             threshold=threshold,
                                             distance=distance,
                                             prominence=prominence)
        return {
            'indices': peaks,
            'values': signal_data[peaks],
            'properties': properties
        }
    def compute_envelope(self, signal_data: np.ndarray,
                        method: str = 'hilbert') -> np.ndarray:
        """
        Compute signal envelope.
        Args:
            signal_data: Input signal
            method: 'hilbert' for Hilbert transform, 'peak' for peak detection
        Returns:
            Signal envelope
        """
        if method == 'hilbert':
            # Hilbert transform method
            analytic_signal = signal.hilbert(signal_data)
            envelope = np.abs(analytic_signal)
        elif method == 'peak':
            # Peak detection method
            # Find peaks and troughs
            peaks = signal.find_peaks(signal_data)[0]
            troughs = signal.find_peaks(-signal_data)[0]
            # Interpolate between peaks
            if len(peaks) > 1:
                envelope = np.interp(np.arange(len(signal_data)), peaks, signal_data[peaks])
            else:
                envelope = np.ones_like(signal_data) * np.max(np.abs(signal_data))
        else:
            raise ValueError(f"Unknown envelope method: {method}")
        return envelope
    def compute_correlation(self, signal1: np.ndarray, signal2: np.ndarray,
                          mode: str = 'full', normalize: bool = True) -> np.ndarray:
        """
        Compute cross-correlation between two signals.
        Args:
            signal1: First signal
            signal2: Second signal
            mode: 'full', 'valid', or 'same'
            normalize: Whether to normalize correlation
        Returns:
            Cross-correlation array
        """
        correlation = signal.correlate(signal1, signal2, mode=mode)
        if normalize:
            # Normalize to [-1, 1] range
            norm_factor = np.sqrt(np.sum(signal1**2) * np.sum(signal2**2))
            if norm_factor > 0:
                correlation = correlation / norm_factor
        return correlation
    def compute_coherence(self, signal1: np.ndarray, signal2: np.ndarray,
                         window_size: int = 256,
                         overlap: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute coherence between two signals.
        Args:
            signal1: First signal
            signal2: Second signal
            window_size: Size of window for spectral estimation
            overlap: Overlap fraction
        Returns:
            Frequency array and coherence values
        """
        nperseg = window_size
        noverlap = int(window_size * overlap)
        f, Cxy = signal.coherence(signal1, signal2, self.sampling_rate,
                                 nperseg=nperseg, noverlap=noverlap)
        return f, Cxy
    def compute_snr(self, signal_data: np.ndarray, noise_data: np.ndarray) -> float:
        """
        Compute Signal-to-Noise Ratio in dB.
        Args:
            signal_data: Clean signal
            noise_data: Noise component
        Returns:
            SNR in dB
        """
        signal_power = np.mean(signal_data**2)
        noise_power = np.mean(noise_data**2)
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = np.inf
        return snr_db
    def extract_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """
        Extract statistical and spectral features from signal.
        Args:
            signal_data: Input signal
        Returns:
            Dictionary of features
        """
        # Time-domain features
        features = {
            'mean': np.mean(signal_data),
            'std': np.std(signal_data),
            'variance': np.var(signal_data),
            'rms': np.sqrt(np.mean(signal_data**2)),
            'peak_to_peak': np.ptp(signal_data),
            'crest_factor': np.max(np.abs(signal_data)) / np.sqrt(np.mean(signal_data**2)),
            'skewness': stats.skew(signal_data),
            'kurtosis': stats.kurtosis(signal_data),
            'zero_crossings': np.sum(np.diff(np.sign(signal_data)) != 0) / 2
        }
        # Frequency-domain features
        freq, fft_mag = self.compute_fft(signal_data)
        # Spectral centroid
        if np.sum(fft_mag) > 0:
            features['spectral_centroid'] = np.sum(freq * fft_mag) / np.sum(fft_mag)
        else:
            features['spectral_centroid'] = 0
        # Spectral bandwidth
        if np.sum(fft_mag) > 0:
            mean_freq = features['spectral_centroid']
            features['spectral_bandwidth'] = np.sqrt(
                np.sum(((freq - mean_freq)**2) * fft_mag) / np.sum(fft_mag)
            )
        else:
            features['spectral_bandwidth'] = 0
        # Dominant frequency
        dominant_idx = np.argmax(fft_mag)
        features['dominant_frequency'] = freq[dominant_idx]
        features['dominant_magnitude'] = fft_mag[dominant_idx]
        return features
    def resample_signal(self, signal_data: np.ndarray,
                       original_rate: float,
                       target_rate: float) -> np.ndarray:
        """
        Resample signal to different sampling rate.
        Args:
            signal_data: Input signal
            original_rate: Original sampling rate
            target_rate: Target sampling rate
        Returns:
            Resampled signal
        """
        # Calculate resampling ratio
        ratio = target_rate / original_rate
        # Calculate new length
        new_length = int(len(signal_data) * ratio)
        # Resample using scipy
        resampled = signal.resample(signal_data, new_length)
        return resampled
    def denoise_signal(self, signal_data: np.ndarray,
                      method: str = 'wavelet',
                      **kwargs) -> np.ndarray:
        """
        Denoise signal using various methods.
        Args:
            signal_data: Noisy signal
            method: Denoising method ('wavelet', 'median', 'savgol')
            **kwargs: Method-specific parameters
        Returns:
            Denoised signal
        """
        if method == 'median':
            # Median filter
            kernel_size = kwargs.get('kernel_size', 5)
            denoised = signal.medfilt(signal_data, kernel_size)
        elif method == 'savgol':
            # Savitzky-Golay filter
            window_length = kwargs.get('window_length', 51)
            polyorder = kwargs.get('polyorder', 3)
            denoised = signal.savgol_filter(signal_data, window_length, polyorder)
        elif method == 'wavelet':
            # Wavelet denoising (simple thresholding)
            import pywt
            wavelet = kwargs.get('wavelet', 'db4')
            level = kwargs.get('level', 5)
            # Decompose
            coeffs = pywt.wavedec(signal_data, wavelet, level=level)
            # Threshold coefficients
            threshold = kwargs.get('threshold', 0.04)
            coeffs_thresh = list(coeffs)
            coeffs_thresh[1:] = [pywt.threshold(c, threshold*np.max(c)) for c in coeffs_thresh[1:]]
            # Reconstruct
            denoised = pywt.waverec(coeffs_thresh, wavelet)
            # Ensure same length
            if len(denoised) > len(signal_data):
                denoised = denoised[:len(signal_data)]
            elif len(denoised) < len(signal_data):
                denoised = np.pad(denoised, (0, len(signal_data) - len(denoised)))
        else:
            raise ValueError(f"Unknown denoising method: {method}")
        return denoised
class AdaptiveFilter:
    """
    Adaptive filtering algorithms for signal processing.
    Implements LMS, NLMS, RLS and other adaptive filtering techniques.
    """
    def __init__(self, filter_order: int, step_size: float = 0.01):
        """
        Initialize adaptive filter.
        Args:
            filter_order: Order of the adaptive filter
            step_size: Learning rate for adaptation
        """
        self.filter_order = filter_order
        self.step_size = step_size
        self.weights = np.zeros(filter_order)
    def lms_filter(self, input_signal: np.ndarray,
                   desired_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Least Mean Squares (LMS) adaptive filter.
        Args:
            input_signal: Input signal
            desired_signal: Desired/reference signal
        Returns:
            Filtered output and error signal
        """
        n_samples = len(input_signal)
        output = np.zeros(n_samples)
        error = np.zeros(n_samples)
        # Pad input for convolution
        padded_input = np.pad(input_signal, (self.filter_order-1, 0), mode='constant')
        for i in range(n_samples):
            # Get input vector
            x = padded_input[i:i+self.filter_order][::-1]
            # Compute output
            output[i] = np.dot(self.weights, x)
            # Compute error
            error[i] = desired_signal[i] - output[i]
            # Update weights (LMS rule)
            self.weights += self.step_size * error[i] * x
        return output, error
    def nlms_filter(self, input_signal: np.ndarray,
                    desired_signal: np.ndarray,
                    regularization: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalized Least Mean Squares (NLMS) adaptive filter.
        Args:
            input_signal: Input signal
            desired_signal: Desired/reference signal
            regularization: Small constant to avoid division by zero
        Returns:
            Filtered output and error signal
        """
        n_samples = len(input_signal)
        output = np.zeros(n_samples)
        error = np.zeros(n_samples)
        # Pad input for convolution
        padded_input = np.pad(input_signal, (self.filter_order-1, 0), mode='constant')
        for i in range(n_samples):
            # Get input vector
            x = padded_input[i:i+self.filter_order][::-1]
            # Compute output
            output[i] = np.dot(self.weights, x)
            # Compute error
            error[i] = desired_signal[i] - output[i]
            # Normalized step size
            norm_factor = np.dot(x, x) + regularization
            # Update weights (NLMS rule)
            self.weights += (self.step_size / norm_factor) * error[i] * x
        return output, error
    def rls_filter(self, input_signal: np.ndarray,
                   desired_signal: np.ndarray,
                   forgetting_factor: float = 0.99) -> Tuple[np.ndarray, np.ndarray]:
        """
        Recursive Least Squares (RLS) adaptive filter.
        Args:
            input_signal: Input signal
            desired_signal: Desired/reference signal
            forgetting_factor: Exponential weighting factor (0 < λ ≤ 1)
        Returns:
            Filtered output and error signal
        """
        n_samples = len(input_signal)
        output = np.zeros(n_samples)
        error = np.zeros(n_samples)
        # Initialize inverse correlation matrix
        delta = 1.0
        P = np.eye(self.filter_order) / delta
        # Pad input for convolution
        padded_input = np.pad(input_signal, (self.filter_order-1, 0), mode='constant')
        for i in range(n_samples):
            # Get input vector
            x = padded_input[i:i+self.filter_order][::-1].reshape(-1, 1)
            # Compute output
            output[i] = np.dot(self.weights, x.flatten())
            # Compute error
            error[i] = desired_signal[i] - output[i]
            # Compute gain vector
            k = P @ x / (forgetting_factor + x.T @ P @ x)
            # Update weights
            self.weights += (k.flatten() * error[i])
            # Update inverse correlation matrix
            P = (P - k @ x.T @ P) / forgetting_factor
        return output, error
def demo_signal_processing():
    """Demonstrate signal processing capabilities."""
    import matplotlib.pyplot as plt
    print("Signal Processing Demo")
    print("=====================")
    # Create signal processor
    fs = 1000  # Sampling rate
    processor = SignalProcessor(sampling_rate=fs)
    # Generate test signal with multiple components
    duration = 2.0
    t, clean_signal = processor.generate_signal('multi_sine', duration,
                                               frequency=[50, 120, 200],
                                               amplitude=[1.0, 0.5, 0.3])
    # Add noise
    _, noisy_signal = processor.generate_signal('multi_sine', duration,
                                               frequency=[50, 120, 200],
                                               amplitude=[1.0, 0.5, 0.3],
                                               noise_level=0.5)
    # Compute FFT
    freq, fft_mag = processor.compute_fft(noisy_signal, window='hann')
    # Design and apply filter
    b, a = processor.design_filter('bandpass', [40, 150], order=4)
    filtered_signal = processor.apply_filter(noisy_signal, b, a)
    # Detect peaks
    peaks = processor.detect_peaks(filtered_signal, height=0.5, distance=50)
    # Extract features
    features = processor.extract_features(filtered_signal)
    # Create plots
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    # Original and noisy signals
    axes[0].plot(t, clean_signal, 'b-', label='Clean', alpha=0.7)
    axes[0].plot(t, noisy_signal, 'r-', label='Noisy', alpha=0.5)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Original and Noisy Signals')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    # FFT
    axes[1].plot(freq[:len(freq)//2], fft_mag[:len(freq)//2])
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title('Frequency Spectrum')
    axes[1].grid(True, alpha=0.3)
    # Filtered signal with peaks
    axes[2].plot(t, filtered_signal, 'g-', label='Filtered')
    axes[2].plot(t[peaks['indices']], peaks['values'], 'ro', label='Peaks')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude')
    axes[2].set_title('Filtered Signal with Peak Detection')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    # Spectrogram
    t_spec, f_spec, Sxx = processor.compute_spectrogram(noisy_signal)
    im = axes[3].pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-10))
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Frequency (Hz)')
    axes[3].set_title('Spectrogram')
    plt.colorbar(im, ax=axes[3], label='Power (dB)')
    plt.tight_layout()
    # Print features
    print("\nExtracted Features:")
    print("-" * 40)
    for key, value in features.items():
        print(f"{key:20s}: {value:10.4f}")
    print("\nDemo completed successfully!")
    return fig
if __name__ == "__main__":
    demo_signal_processing()
    import matplotlib.pyplot as plt
    plt.show()
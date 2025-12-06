"""
Spectral Analysis Module
========================
Advanced spectral analysis techniques for signal processing including
power spectral density, spectrograms, wavelets, and time-frequency analysis.
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
import scipy.signal as signal
import scipy.fft as fft
from scipy import interpolate
from typing import Tuple, Optional, Union, Dict, Any
import warnings
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    warnings.warn("PyWavelets not available. Wavelet features will be limited.")
class SpectralAnalyzer:
    """
    Advanced spectral analysis tools for scientific signal processing.
    Provides comprehensive spectral analysis including FFT, power spectral
    density, time-frequency analysis, and wavelet transforms.
    """
    def __init__(self, sampling_rate: float = 1.0):
        """
        Initialize spectral analyzer.
        Args:
            sampling_rate: Sampling frequency in Hz
        """
        self.sampling_rate = sampling_rate
        self.nyquist_freq = sampling_rate / 2
    def compute_power_spectrum(self, signal_data: np.ndarray,
                              method: str = 'periodogram',
                              window: str = 'hann',
                              nperseg: Optional[int] = None,
                              noverlap: Optional[int] = None,
                              scaling: str = 'density') -> Tuple[np.ndarray, np.ndarray]:
        """
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
        """
        if method == 'periodogram':
            f, Pxx = signal.periodogram(signal_data, self.sampling_rate,
                                       window=window, scaling=scaling)
        elif method == 'welch':
            if nperseg is None:
                nperseg = min(256, len(signal_data))
            if noverlap is None:
                noverlap = nperseg // 2
            f, Pxx = signal.welch(signal_data, self.sampling_rate,
                                 window=window, nperseg=nperseg,
                                 noverlap=noverlap, scaling=scaling)
        elif method == 'multitaper':
            # Multitaper method using DPSS windows
            from scipy.signal.windows import dpss
            if nperseg is None:
                nperseg = min(256, len(signal_data))
            # Time-bandwidth product
            NW = 4
            # Number of tapers
            K = 2 * NW - 1
            # Generate DPSS tapers
            tapers, eigenvalues = dpss(nperseg, NW, K, return_ratios=True)
            # Compute spectrum for each taper
            n_freqs = nperseg // 2 + 1
            Pxx_mt = np.zeros(n_freqs)
            # Process signal in segments
            n_segments = (len(signal_data) - nperseg) // (nperseg - (noverlap or 0)) + 1
            for i in range(n_segments):
                start = i * (nperseg - (noverlap or 0))
                segment = signal_data[start:start + nperseg]
                for k in range(K):
                    # Apply taper
                    tapered = segment * tapers[k]
                    # Compute FFT
                    fft_vals = fft.rfft(tapered)
                    # Add to power estimate
                    Pxx_mt += np.abs(fft_vals)**2 * eigenvalues[k]
            # Average over segments and tapers
            Pxx_mt /= (n_segments * K)
            # Convert to proper scaling
            if scaling == 'density':
                Pxx_mt /= self.sampling_rate
            else:
                Pxx_mt *= 2 / nperseg**2
            f = fft.rfftfreq(nperseg, 1/self.sampling_rate)
            Pxx = Pxx_mt
        else:
            raise ValueError(f"Unknown method: {method}")
        return f, Pxx
    def compute_cross_spectrum(self, signal1: np.ndarray, signal2: np.ndarray,
                              window: str = 'hann',
                              nperseg: Optional[int] = None,
                              noverlap: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cross power spectral density between two signals.
        Args:
            signal1: First signal
            signal2: Second signal
            window: Window function
            nperseg: Length of each segment
            noverlap: Number of overlapping points
        Returns:
            Frequency array and cross PSD
        """
        if nperseg is None:
            nperseg = min(256, len(signal1))
        if noverlap is None:
            noverlap = nperseg // 2
        f, Pxy = signal.csd(signal1, signal2, self.sampling_rate,
                           window=window, nperseg=nperseg, noverlap=noverlap)
        return f, Pxy
    def compute_transfer_function(self, input_signal: np.ndarray,
                                 output_signal: np.ndarray,
                                 window: str = 'hann',
                                 nperseg: Optional[int] = None,
                                 noverlap: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate transfer function between input and output signals.
        Args:
            input_signal: Input signal
            output_signal: Output signal
            window: Window function
            nperseg: Segment length
            noverlap: Overlap length
        Returns:
            Frequency array, transfer function magnitude, and phase
        """
        if nperseg is None:
            nperseg = min(256, len(input_signal))
        if noverlap is None:
            noverlap = nperseg // 2
        # Compute cross-spectral density and input power spectral density
        f, Pxy = signal.csd(input_signal, output_signal, self.sampling_rate,
                           window=window, nperseg=nperseg, noverlap=noverlap)
        _, Pxx = signal.welch(input_signal, self.sampling_rate,
                             window=window, nperseg=nperseg, noverlap=noverlap)
        # Transfer function H(f) = Pxy(f) / Pxx(f)
        H = Pxy / (Pxx + 1e-12)  # Add small value to avoid division by zero
        magnitude = np.abs(H)
        phase = np.angle(H)
        return f, magnitude, phase
    def compute_cepstrum(self, signal_data: np.ndarray,
                        type: str = 'real') -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cepstrum of signal.
        Args:
            signal_data: Input signal
            type: 'real' for real cepstrum, 'complex' for complex cepstrum
        Returns:
            Quefrency array and cepstrum
        """
        # Compute FFT
        spectrum = fft.fft(signal_data)
        if type == 'real':
            # Real cepstrum
            log_spectrum = np.log(np.abs(spectrum) + 1e-12)
            cepstrum = fft.ifft(log_spectrum).real
        elif type == 'complex':
            # Complex cepstrum
            # Unwrap phase for minimum phase assumption
            magnitude = np.abs(spectrum)
            phase = np.unwrap(np.angle(spectrum))
            log_spectrum = np.log(magnitude + 1e-12) + 1j * phase
            cepstrum = fft.ifft(log_spectrum)
        else:
            raise ValueError(f"Unknown cepstrum type: {type}")
        # Quefrency (time-like axis for cepstrum)
        quefrency = np.arange(len(cepstrum)) / self.sampling_rate
        return quefrency, cepstrum
    def compute_mel_spectrogram(self, signal_data: np.ndarray,
                               n_mels: int = 128,
                               n_fft: int = 2048,
                               hop_length: Optional[int] = None,
                               window: str = 'hann',
                               fmin: float = 0.0,
                               fmax: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
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
        """
        if hop_length is None:
            hop_length = n_fft // 4
        if fmax is None:
            fmax = self.nyquist_freq
        # Compute STFT
        f_stft, t_stft, Zxx = signal.stft(signal_data, self.sampling_rate,
                                          window=window, nperseg=n_fft,
                                          noverlap=n_fft-hop_length)
        # Create mel filterbank
        mel_filters = self._create_mel_filterbank(n_mels, n_fft,
                                                  self.sampling_rate,
                                                  fmin, fmax)
        # Apply mel filterbank
        power_spectrum = np.abs(Zxx)**2
        mel_spectrogram = np.dot(mel_filters, power_spectrum[:n_fft//2+1, :])
        # Mel frequency axis
        mel_freqs = self._hz_to_mel(np.linspace(fmin, fmax, n_mels))
        return t_stft, mel_freqs, mel_spectrogram
    def _hz_to_mel(self, frequencies: np.ndarray) -> np.ndarray:
        """Convert frequency in Hz to mel scale."""
        return 2595 * np.log10(1 + frequencies / 700)
    def _mel_to_hz(self, mels: np.ndarray) -> np.ndarray:
        """Convert mel scale to frequency in Hz."""
        return 700 * (10**(mels / 2595) - 1)
    def _create_mel_filterbank(self, n_mels: int, n_fft: int,
                              sample_rate: float, fmin: float,
                              fmax: float) -> np.ndarray:
        """Create mel filterbank matrix."""
        # Mel points
        mel_min = self._hz_to_mel(fmin)
        mel_max = self._hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        # FFT bin frequencies
        fft_freqs = fft.rfftfreq(n_fft, 1/sample_rate)
        # Create filterbank
        filterbank = np.zeros((n_mels, len(fft_freqs)))
        for i in range(n_mels):
            # Triangle filter
            left = hz_points[i]
            center = hz_points[i + 1]
            right = hz_points[i + 2]
            # Rising edge
            rising = (fft_freqs >= left) & (fft_freqs <= center)
            filterbank[i, rising] = (fft_freqs[rising] - left) / (center - left)
            # Falling edge
            falling = (fft_freqs >= center) & (fft_freqs <= right)
            filterbank[i, falling] = (right - fft_freqs[falling]) / (right - center)
        return filterbank
    def compute_wavelet_transform(self, signal_data: np.ndarray,
                                 wavelet: str = 'db4',
                                 level: Optional[int] = None,
                                 mode: str = 'symmetric') -> Dict[str, np.ndarray]:
        """
        Compute discrete wavelet transform.
        Args:
            signal_data: Input signal
            wavelet: Wavelet name (e.g., 'db4', 'sym5', 'coif3')
            level: Decomposition level (None for maximum)
            mode: Signal extension mode
        Returns:
            Dictionary with approximation and detail coefficients
        """
        if not PYWT_AVAILABLE:
            raise ImportError("PyWavelets is required for wavelet transforms")
        # Determine maximum decomposition level if not specified
        if level is None:
            level = pywt.dwt_max_level(len(signal_data), wavelet)
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(signal_data, wavelet, mode=mode, level=level)
        # Organize coefficients
        result = {
            'approximation': coeffs[0],
            'details': coeffs[1:],
            'wavelet': wavelet,
            'level': level
        }
        return result
    def compute_cwt(self, signal_data: np.ndarray,
                   scales: Optional[np.ndarray] = None,
                   wavelet: str = 'morl') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute continuous wavelet transform.
        Args:
            signal_data: Input signal
            scales: Scale array (default: logarithmic from 1 to 128)
            wavelet: Wavelet name for CWT
        Returns:
            Time array, scale array, CWT coefficients
        """
        if not PYWT_AVAILABLE:
            raise ImportError("PyWavelets is required for wavelet transforms")
        if scales is None:
            scales = np.logspace(0, 7, num=100, base=2)
        # Compute CWT
        coeffs, frequencies = pywt.cwt(signal_data, scales, wavelet,
                                       sampling_period=1/self.sampling_rate)
        # Time array
        time = np.arange(len(signal_data)) / self.sampling_rate
        return time, frequencies, np.abs(coeffs)
    def compute_hilbert_huang(self, signal_data: np.ndarray,
                             n_imfs: int = 5) -> Dict[str, np.ndarray]:
        """
        Compute Hilbert-Huang transform (simplified EMD).
        Args:
            signal_data: Input signal
            n_imfs: Number of Intrinsic Mode Functions to extract
        Returns:
            Dictionary with IMFs and instantaneous frequencies
        """
        # Simplified Empirical Mode Decomposition
        imfs = []
        residual = signal_data.copy()
        for _ in range(n_imfs):
            imf = self._extract_imf(residual)
            imfs.append(imf)
            residual = residual - imf
            # Stop if residual is too small
            if np.std(residual) < 0.01 * np.std(signal_data):
                break
        # Compute instantaneous frequencies for each IMF
        inst_freqs = []
        inst_amps = []
        for imf in imfs:
            # Hilbert transform
            analytic = signal.hilbert(imf)
            inst_amp = np.abs(analytic)
            inst_phase = np.unwrap(np.angle(analytic))
            # Instantaneous frequency
            inst_freq = np.diff(inst_phase) * self.sampling_rate / (2 * np.pi)
            inst_freq = np.append(inst_freq, inst_freq[-1])  # Pad to same length
            inst_freqs.append(inst_freq)
            inst_amps.append(inst_amp)
        return {
            'imfs': np.array(imfs),
            'instantaneous_frequencies': np.array(inst_freqs),
            'instantaneous_amplitudes': np.array(inst_amps),
            'residual': residual
        }
    def _extract_imf(self, signal_data: np.ndarray,
                    max_iterations: int = 100) -> np.ndarray:
        """
        Extract one Intrinsic Mode Function using sifting.
        Simplified EMD sifting process.
        """
        imf = signal_data.copy()
        for _ in range(max_iterations):
            # Find extrema
            peaks, _ = signal.find_peaks(imf)
            troughs, _ = signal.find_peaks(-imf)
            # Check for sufficient extrema
            if len(peaks) < 2 or len(troughs) < 2:
                break
            # Create envelopes
            t = np.arange(len(imf))
            # Upper envelope
            upper_env = np.interp(t, peaks, imf[peaks])
            # Lower envelope
            lower_env = np.interp(t, troughs, imf[troughs])
            # Mean of envelopes
            mean_env = (upper_env + lower_env) / 2
            # Update IMF
            prev_imf = imf.copy()
            imf = imf - mean_env
            # Check convergence
            if np.sum((imf - prev_imf)**2) / np.sum(imf**2) < 0.001:
                break
        return imf
    def compute_spectral_features(self, signal_data: np.ndarray,
                                 n_bands: int = 10) -> Dict[str, Union[float, np.ndarray]]:
        """
        Extract comprehensive spectral features.
        Args:
            signal_data: Input signal
            n_bands: Number of frequency bands for band-wise features
        Returns:
            Dictionary of spectral features
        """
        # Compute power spectrum
        f, Pxx = self.compute_power_spectrum(signal_data, method='welch')
        # Normalize for probability distribution
        Pxx_norm = Pxx / np.sum(Pxx)
        features = {}
        # Spectral centroid
        features['spectral_centroid'] = np.sum(f * Pxx_norm)
        # Spectral spread
        centroid = features['spectral_centroid']
        features['spectral_spread'] = np.sqrt(np.sum(((f - centroid)**2) * Pxx_norm))
        # Spectral skewness
        spread = features['spectral_spread']
        if spread > 0:
            features['spectral_skewness'] = np.sum(((f - centroid)**3) * Pxx_norm) / spread**3
        else:
            features['spectral_skewness'] = 0
        # Spectral kurtosis
        if spread > 0:
            features['spectral_kurtosis'] = np.sum(((f - centroid)**4) * Pxx_norm) / spread**4
        else:
            features['spectral_kurtosis'] = 0
        # Spectral entropy
        Pxx_norm_safe = Pxx_norm[Pxx_norm > 0]
        features['spectral_entropy'] = -np.sum(Pxx_norm_safe * np.log2(Pxx_norm_safe))
        # Spectral rolloff
        cumsum = np.cumsum(Pxx)
        rolloff_threshold = 0.85 * cumsum[-1]
        rolloff_idx = np.where(cumsum >= rolloff_threshold)[0]
        if len(rolloff_idx) > 0:
            features['spectral_rolloff'] = f[rolloff_idx[0]]
        else:
            features['spectral_rolloff'] = f[-1]
        # Spectral flux
        if len(signal_data) > len(f):
            # Compute spectrum for shifted signal
            shifted_signal = np.roll(signal_data, int(self.sampling_rate * 0.01))
            _, Pxx_shifted = self.compute_power_spectrum(shifted_signal, method='welch')
            features['spectral_flux'] = np.sum((np.sqrt(Pxx) - np.sqrt(Pxx_shifted))**2)
        else:
            features['spectral_flux'] = 0
        # Band-wise energy
        band_edges = np.linspace(f[0], f[-1], n_bands + 1)
        band_energies = []
        for i in range(n_bands):
            band_mask = (f >= band_edges[i]) & (f < band_edges[i+1])
            band_energy = np.sum(Pxx[band_mask])
            band_energies.append(band_energy)
        features['band_energies'] = np.array(band_energies)
        features['band_energy_ratios'] = band_energies / np.sum(band_energies)
        # Spectral flatness
        geometric_mean = np.exp(np.mean(np.log(Pxx + 1e-12)))
        arithmetic_mean = np.mean(Pxx)
        features['spectral_flatness'] = geometric_mean / (arithmetic_mean + 1e-12)
        # Spectral crest factor
        features['spectral_crest'] = np.max(Pxx) / (arithmetic_mean + 1e-12)
        return features
def demo_spectral_analysis():
    """Demonstrate spectral analysis capabilities."""
    import matplotlib.pyplot as plt
    print("Spectral Analysis Demo")
    print("=====================")
    # Create analyzer
    fs = 1000  # Sampling rate
    analyzer = SpectralAnalyzer(sampling_rate=fs)
    # Generate test signal
    t = np.linspace(0, 2, 2*fs, endpoint=False)
    # Chirp signal
    f0, f1 = 10, 100
    signal_data = signal.chirp(t, f0, 2, f1, method='linear')
    # Add some noise
    signal_data += 0.1 * np.random.randn(len(signal_data))
    # Compute various spectral representations
    f_psd, Pxx = analyzer.compute_power_spectrum(signal_data, method='welch')
    # Mel spectrogram
    t_mel, f_mel, mel_spec = analyzer.compute_mel_spectrogram(signal_data, n_mels=64)
    # Extract spectral features
    features = analyzer.compute_spectral_features(signal_data)
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # Time domain signal
    axes[0, 0].plot(t[:500], signal_data[:500])
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('Chirp Signal')
    axes[0, 0].grid(True, alpha=0.3)
    # Power spectral density
    axes[0, 1].semilogy(f_psd, Pxx)
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('PSD')
    axes[0, 1].set_title('Power Spectral Density')
    axes[0, 1].grid(True, alpha=0.3)
    # Mel spectrogram
    im = axes[1, 0].pcolormesh(t_mel, f_mel, 10*np.log10(mel_spec + 1e-10))
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Mel Frequency')
    axes[1, 0].set_title('Mel Spectrogram')
    plt.colorbar(im, ax=axes[1, 0])
    # Band energies
    axes[1, 1].bar(range(len(features['band_energies'])),
                   features['band_energies'])
    axes[1, 1].set_xlabel('Frequency Band')
    axes[1, 1].set_ylabel('Energy')
    axes[1, 1].set_title('Band-wise Energy Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    # Print features
    print("\nSpectral Features:")
    print("-" * 40)
    for key, value in features.items():
        if isinstance(value, (int, float)):
            print(f"{key:20s}: {value:10.4f}")
    print("\nDemo completed successfully!")
    return fig
if __name__ == "__main__":
    demo_spectral_analysis()
    import matplotlib.pyplot as plt
    plt.show()
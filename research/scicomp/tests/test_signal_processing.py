"""
Test Suite for Signal Processing Module
========================================
Comprehensive tests for signal processing functionality.
Author: Berkeley SciComp Team
Date: 2024
"""
import pytest
import numpy as np
import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Python.Signal_Processing import SignalProcessor, SpectralAnalyzer, AdaptiveFilter
class TestSignalProcessor:
    """Test SignalProcessor class."""
    @pytest.fixture
    def processor(self):
        """Create SignalProcessor instance."""
        return SignalProcessor(sampling_rate=1000)
    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor.sampling_rate == 1000
        assert processor.nyquist_freq == 500
    def test_signal_generation(self, processor):
        """Test signal generation methods."""
        duration = 1.0
        # Test sine wave
        t, signal = processor.generate_signal('sine', duration, frequency=10)
        assert len(t) == len(signal)
        assert len(signal) == int(duration * processor.sampling_rate)
        # Test chirp signal
        t, chirp = processor.generate_signal('chirp', duration, frequency=[10, 100])
        assert len(chirp) == int(duration * processor.sampling_rate)
        # Test multi-sine
        t, multi = processor.generate_signal('multi_sine', duration,
                                            frequency=[10, 20, 30])
        assert len(multi) == int(duration * processor.sampling_rate)
    def test_fft_computation(self, processor):
        """Test FFT computation."""
        # Generate test signal
        t, signal = processor.generate_signal('sine', 1.0, frequency=50)
        # Compute FFT
        freq, mag = processor.compute_fft(signal)
        # Check frequency resolution
        assert len(freq) == len(mag)
        assert freq[1] - freq[0] == pytest.approx(1.0, rel=1e-3)
        # Check peak at expected frequency
        peak_idx = np.argmax(mag[1:]) + 1  # Skip DC
        peak_freq = freq[peak_idx]
        assert peak_freq == pytest.approx(50, rel=0.1)
    def test_filter_design(self, processor):
        """Test filter design."""
        # Lowpass filter
        b_lp, a_lp = processor.design_filter('lowpass', 100, order=4)
        assert len(b_lp) == 5  # Order + 1
        assert len(a_lp) == 5
        # Bandpass filter
        b_bp, a_bp = processor.design_filter('bandpass', [50, 200], order=4)
        assert len(b_bp) > 0
        assert len(a_bp) > 0
    def test_filter_application(self, processor):
        """Test filter application."""
        # Generate noisy signal
        t, clean = processor.generate_signal('sine', 1.0, frequency=50)
        t, noisy = processor.generate_signal('sine', 1.0, frequency=50,
                                            noise_level=0.5)
        # Design and apply filter
        b, a = processor.design_filter('bandpass', [40, 60], order=4)
        filtered = processor.apply_filter(noisy, b, a)
        # Filtered should be closer to clean than noisy
        error_noisy = np.mean((clean - noisy)**2)
        error_filtered = np.mean((clean - filtered)**2)
        assert error_filtered < error_noisy
    def test_peak_detection(self, processor):
        """Test peak detection."""
        # Generate signal with known peaks
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 5 * t)  # 5 peaks in 1 second
        peaks = processor.detect_peaks(signal, height=0.5, distance=100)
        assert 'indices' in peaks
        assert 'values' in peaks
        assert len(peaks['indices']) >= 4  # Should detect at least 4 peaks
    def test_envelope_computation(self, processor):
        """Test envelope computation."""
        # Generate modulated signal
        t, carrier = processor.generate_signal('sine', 1.0, frequency=100)
        t, modulator = processor.generate_signal('sine', 1.0, frequency=5)
        signal = carrier * (1 + 0.5 * modulator)
        envelope = processor.compute_envelope(signal, method='hilbert')
        assert len(envelope) == len(signal)
        assert np.all(envelope >= 0)
    def test_snr_computation(self, processor):
        """Test SNR computation."""
        # Generate clean and noisy signals
        t, clean = processor.generate_signal('sine', 1.0, frequency=50)
        noise = 0.1 * np.random.randn(len(clean))
        noisy = clean + noise
        snr = processor.compute_snr(clean, noise)
        assert snr > 0  # Signal should be stronger than noise
        assert snr < 50  # Reasonable SNR range
    def test_feature_extraction(self, processor):
        """Test feature extraction."""
        t, signal = processor.generate_signal('multi_sine', 1.0,
                                             frequency=[50, 100, 150])
        features = processor.extract_features(signal)
        # Check required features
        required_features = ['mean', 'std', 'rms', 'spectral_centroid',
                           'dominant_frequency', 'zero_crossings']
        for feature in required_features:
            assert feature in features
        # Check feature values
        assert features['dominant_frequency'] > 0
        assert features['spectral_centroid'] > 0
        assert features['rms'] > 0
class TestSpectralAnalyzer:
    """Test SpectralAnalyzer class."""
    @pytest.fixture
    def analyzer(self):
        """Create SpectralAnalyzer instance."""
        return SpectralAnalyzer(sampling_rate=1000)
    def test_power_spectrum(self, analyzer):
        """Test power spectrum computation."""
        # Generate test signal
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
        # Compute PSD
        f, Pxx = analyzer.compute_power_spectrum(signal, method='welch')
        assert len(f) == len(Pxx)
        assert np.all(Pxx >= 0)  # Power should be non-negative
        # Check peaks at expected frequencies
        peaks_idx = np.argsort(Pxx)[-2:]  # Two highest peaks
        peak_freqs = sorted(f[peaks_idx])
        assert peak_freqs[0] == pytest.approx(50, rel=0.1)
        assert peak_freqs[1] == pytest.approx(120, rel=0.1)
    def test_cross_spectrum(self, analyzer):
        """Test cross spectrum computation."""
        t = np.linspace(0, 1, 1000)
        signal1 = np.sin(2 * np.pi * 50 * t)
        signal2 = np.sin(2 * np.pi * 50 * t + np.pi/4)  # Phase shifted
        f, Pxy = analyzer.compute_cross_spectrum(signal1, signal2)
        assert len(f) == len(Pxy)
        assert Pxy.dtype == np.complex128
    def test_transfer_function(self, analyzer):
        """Test transfer function estimation."""
        t = np.linspace(0, 1, 1000)
        input_signal = np.random.randn(1000)
        # Simple system: output = 0.5 * input with delay
        output_signal = 0.5 * np.roll(input_signal, 10)
        f, mag, phase = analyzer.compute_transfer_function(input_signal, output_signal)
        assert len(f) == len(mag) == len(phase)
        assert np.mean(mag) == pytest.approx(0.5, rel=0.2)
    def test_cepstrum(self, analyzer):
        """Test cepstrum computation."""
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 50 * t)
        quefrency, cepstrum = analyzer.compute_cepstrum(signal, type='real')
        assert len(quefrency) == len(cepstrum)
        assert cepstrum.dtype in [np.float64, np.complex128]
    def test_spectral_features(self, analyzer):
        """Test spectral feature extraction."""
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 150 * t)
        features = analyzer.compute_spectral_features(signal, n_bands=10)
        # Check required features
        required = ['spectral_centroid', 'spectral_spread', 'spectral_entropy',
                   'spectral_rolloff', 'band_energies']
        for feat in required:
            assert feat in features
        # Check feature validity
        assert features['spectral_centroid'] > 0
        assert features['spectral_spread'] > 0
        assert 0 <= features['spectral_entropy'] <= 10
        assert len(features['band_energies']) == 10
class TestAdaptiveFilter:
    """Test AdaptiveFilter class."""
    def test_lms_filter(self):
        """Test LMS adaptive filter."""
        filter = AdaptiveFilter(filter_order=32, step_size=0.01)
        # Generate reference and noisy signals
        n = 500
        reference = np.sin(2 * np.pi * 0.05 * np.arange(n))
        noise = 0.5 * np.random.randn(n)
        noisy = reference + noise
        # Apply LMS filter
        output, error = filter.lms_filter(noisy, reference)
        assert len(output) == len(reference)
        assert len(error) == len(reference)
        # Check convergence (final error should be smaller)
        initial_error = np.mean(error[:50]**2)
        final_error = np.mean(error[-50:]**2)
        assert final_error < initial_error
    def test_nlms_filter(self):
        """Test NLMS adaptive filter."""
        filter = AdaptiveFilter(filter_order=32, step_size=0.5)
        # Generate signals
        n = 500
        reference = np.sin(2 * np.pi * 0.05 * np.arange(n))
        noise = 0.5 * np.random.randn(n)
        noisy = reference + noise
        # Apply NLMS filter
        output, error = filter.nlms_filter(noisy, reference)
        assert len(output) == len(reference)
        assert len(error) == len(reference)
        # NLMS should converge
        final_error = np.mean(error[-50:]**2)
        assert final_error < 0.5
    def test_rls_filter(self):
        """Test RLS adaptive filter."""
        filter = AdaptiveFilter(filter_order=16, step_size=0.01)
        # Generate signals
        n = 300
        reference = np.sin(2 * np.pi * 0.05 * np.arange(n))
        noise = 0.3 * np.random.randn(n)
        noisy = reference + noise
        # Apply RLS filter
        output, error = filter.rls_filter(noisy, reference, forgetting_factor=0.99)
        assert len(output) == len(reference)
        assert len(error) == len(reference)
        # RLS should converge quickly
        final_error = np.mean(error[-30:]**2)
        assert final_error < 0.3
@pytest.mark.integration
class TestIntegration:
    """Integration tests for signal processing pipeline."""
    def test_complete_pipeline(self):
        """Test complete signal processing pipeline."""
        # Initialize components
        processor = SignalProcessor(sampling_rate=1000)
        analyzer = SpectralAnalyzer(sampling_rate=1000)
        # Generate complex signal
        t, signal = processor.generate_signal('multi_sine', 2.0,
                                             frequency=[50, 120, 200],
                                             amplitude=[1.0, 0.5, 0.3],
                                             noise_level=0.2)
        # Process signal
        freq, mag = processor.compute_fft(signal, window='hann')
        b, a = processor.design_filter('bandpass', [40, 150], order=4)
        filtered = processor.apply_filter(signal, b, a)
        # Analyze
        f_psd, Pxx = analyzer.compute_power_spectrum(filtered)
        features = processor.extract_features(filtered)
        spectral_features = analyzer.compute_spectral_features(filtered)
        # Verify results
        assert len(filtered) == len(signal)
        assert features['dominant_frequency'] > 40
        assert features['dominant_frequency'] < 150
        assert spectral_features['spectral_centroid'] > 0
    def test_real_world_scenario(self):
        """Test with realistic signal processing scenario."""
        processor = SignalProcessor(sampling_rate=8000)  # Audio rate
        # Simulate audio signal with noise
        duration = 0.5
        t, voice = processor.generate_signal('multi_sine', duration,
                                            frequency=[200, 400, 600, 800],
                                            amplitude=[1, 0.8, 0.6, 0.4])
        t, noise = processor.generate_signal('noise', duration, amplitude=0.1)
        noisy_audio = voice + noise
        # Process: bandpass filter for voice range
        b, a = processor.design_filter('bandpass', [100, 1000], order=6)
        cleaned = processor.apply_filter(noisy_audio, b, a)
        # Compute SNR improvement
        snr_before = processor.compute_snr(voice, noisy_audio - voice)
        snr_after = processor.compute_snr(voice, cleaned - voice)
        assert snr_after > snr_before  # Should improve SNR
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
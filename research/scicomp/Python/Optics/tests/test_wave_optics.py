"""Tests for wave optics module.
This module contains comprehensive tests for wave optics functionality
including wave propagation, diffraction, interference, and Gaussian beams.
Author: Berkeley SciComp Team
Date: 2024
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import warnings
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Optics.wave_optics import (
    PlaneWave, SphericalWave, GaussianBeam, WaveOptics,
    propagate_fresnel, calculate_diffraction, analyze_interference,
    fresnel_number, rayleigh_range
)
class TestWaveOptics:
    """Test wave optics base class."""
    def setup_method(self):
        """Setup test fixtures."""
        self.wavelength = 633e-9  # HeNe laser
        self.medium_index = 1.0
        self.tolerance = 1e-10
    def test_wave_optics_initialization(self):
        """Test WaveOptics initialization."""
        wave_optics = WaveOptics(self.wavelength, self.medium_index)
        assert wave_optics.wavelength_vacuum == self.wavelength
        assert wave_optics.medium_index == self.medium_index
        assert wave_optics.wavelength == self.wavelength / self.medium_index
        assert wave_optics.k == 2 * np.pi / wave_optics.wavelength
    def test_validation(self):
        """Test input validation."""
        wave_optics = WaveOptics(self.wavelength)
        # Valid inputs should not raise errors
        wave_optics.validate_input(1.0, 2.0, np.array([1, 2, 3]))
        # Invalid inputs should raise errors
        with pytest.raises(ValueError):
            wave_optics.validate_input(np.inf)
        with pytest.raises(ValueError):
            wave_optics.validate_input(np.array([1, np.nan, 3]))
class TestPlaneWave:
    """Test plane wave functionality."""
    def setup_method(self):
        """Setup test fixtures."""
        self.wavelength = 589e-9  # Sodium D-line
        self.amplitude = 1.0
        self.phase = 0.0
        self.direction = np.array([0, 0, 1])  # +z direction
    def test_plane_wave_initialization(self):
        """Test plane wave initialization."""
        wave = PlaneWave(self.wavelength, self.amplitude, self.phase, self.direction)
        assert wave.wavelength_vacuum == self.wavelength
        assert wave.amplitude == self.amplitude
        assert wave.phase == self.phase
        assert_allclose(wave.direction, self.direction)
    def test_field_at_point(self):
        """Test field calculation at specific points."""
        wave = PlaneWave(self.wavelength, self.amplitude, self.phase, self.direction)
        # Test at origin
        r_origin = np.array([0, 0, 0])
        field_origin = wave.field_at_point(r_origin, t=0)
        expected_origin = self.amplitude * np.exp(1j * self.phase)
        assert_allclose(field_origin, expected_origin, rtol=1e-12)
        # Test at different positions
        r_test = np.array([0, 0, self.wavelength])
        field_test = wave.field_at_point(r_test, t=0)
        # Should have phase shift of 2π (one wavelength)
        expected_test = self.amplitude * np.exp(1j * (self.phase + 2*np.pi))
        assert_allclose(field_test, expected_test, rtol=1e-12)
    def test_intensity_at_point(self):
        """Test intensity calculation."""
        wave = PlaneWave(self.wavelength, self.amplitude, self.phase, self.direction)
        r = np.array([0, 0, 0])
        intensity = wave.intensity_at_point(r)
        expected_intensity = self.amplitude**2
        assert_allclose(intensity, expected_intensity, rtol=1e-12)
    def test_propagation(self):
        """Test wave propagation."""
        wave = PlaneWave(self.wavelength, self.amplitude, self.phase, self.direction)
        distance = self.wavelength  # One wavelength
        propagated_wave = wave.propagate_distance(distance)
        # Phase should change by 2π
        expected_phase = self.phase + 2*np.pi
        assert_allclose(propagated_wave.phase, expected_phase, rtol=1e-12)
        # Other properties should remain the same
        assert propagated_wave.amplitude == self.amplitude
        assert propagated_wave.wavelength_vacuum == self.wavelength
        assert_allclose(propagated_wave.direction, self.direction)
class TestSphericalWave:
    """Test spherical wave functionality."""
    def setup_method(self):
        """Setup test fixtures."""
        self.wavelength = 633e-9
        self.source_power = 1.0  # 1 W
        self.source_position = np.array([0, 0, 0])
    def test_spherical_wave_initialization(self):
        """Test spherical wave initialization."""
        wave = SphericalWave(self.wavelength, self.source_power, self.source_position)
        assert wave.wavelength_vacuum == self.wavelength
        assert wave.source_power == self.source_power
        assert_allclose(wave.source_position, self.source_position)
    def test_field_at_point(self):
        """Test field calculation for spherical wave."""
        wave = SphericalWave(self.wavelength, self.source_power, self.source_position)
        # Test at specific distance
        distance = 1.0  # 1 meter
        r = np.array([distance, 0, 0])
        field = wave.field_at_point(r, t=0)
        # Expected amplitude decreases as 1/r
        expected_amplitude = np.sqrt(self.source_power / (4 * np.pi)) / distance
        assert_allclose(np.abs(field), expected_amplitude, rtol=1e-12)
    def test_intensity_at_point(self):
        """Test intensity calculation for spherical wave."""
        wave = SphericalWave(self.wavelength, self.source_power, self.source_position)
        distance = 1.0
        r = np.array([distance, 0, 0])
        intensity = wave.intensity_at_point(r)
        # Intensity should follow 1/r² law
        expected_intensity = self.source_power / (4 * np.pi * distance**2)
        assert_allclose(intensity, expected_intensity, rtol=1e-12)
class TestGaussianBeam:
    """Test Gaussian beam functionality."""
    def setup_method(self):
        """Setup test fixtures."""
        self.wavelength = 1064e-9  # Nd:YAG laser
        self.waist_radius = 1e-3   # 1 mm
        self.power = 1e-3          # 1 mW
    def test_gaussian_beam_initialization(self):
        """Test Gaussian beam initialization."""
        beam = GaussianBeam(self.wavelength, self.waist_radius, power=self.power)
        assert beam.wavelength_vacuum == self.wavelength
        assert beam.waist_radius == self.waist_radius
        assert beam.power == self.power
        # Check derived parameters
        expected_rayleigh_range = np.pi * self.waist_radius**2 / beam.wavelength
        assert_allclose(beam.rayleigh_range, expected_rayleigh_range, rtol=1e-12)
        expected_divergence = beam.wavelength / (np.pi * self.waist_radius)
        assert_allclose(beam.divergence_angle, expected_divergence, rtol=1e-12)
    def test_beam_radius(self):
        """Test beam radius calculation."""
        beam = GaussianBeam(self.wavelength, self.waist_radius)
        # At waist
        radius_at_waist = beam.beam_radius(0)
        assert_allclose(radius_at_waist, self.waist_radius, rtol=1e-12)
        # At Rayleigh range
        radius_at_zr = beam.beam_radius(beam.rayleigh_range)
        expected_radius = self.waist_radius * np.sqrt(2)
        assert_allclose(radius_at_zr, expected_radius, rtol=1e-12)
        # At large distance
        z_far = 10 * beam.rayleigh_range
        radius_far = beam.beam_radius(z_far)
        expected_radius_far = self.waist_radius * z_far / beam.rayleigh_range
        assert_allclose(radius_far, expected_radius_far, rtol=1e-6)
    def test_radius_of_curvature(self):
        """Test radius of curvature calculation."""
        beam = GaussianBeam(self.wavelength, self.waist_radius)
        # At waist (should be infinite)
        R_at_waist = beam.radius_of_curvature(0)
        assert np.isinf(R_at_waist)
        # At Rayleigh range
        R_at_zr = beam.radius_of_curvature(beam.rayleigh_range)
        expected_R = 2 * beam.rayleigh_range
        assert_allclose(R_at_zr, expected_R, rtol=1e-12)
    def test_gouy_phase(self):
        """Test Gouy phase calculation."""
        beam = GaussianBeam(self.wavelength, self.waist_radius)
        # At waist
        phase_at_waist = beam.gouy_phase(0)
        assert_allclose(phase_at_waist, 0, atol=1e-12)
        # At Rayleigh range
        phase_at_zr = beam.gouy_phase(beam.rayleigh_range)
        expected_phase = np.pi / 4
        assert_allclose(phase_at_zr, expected_phase, rtol=1e-12)
    def test_field_profile(self):
        """Test transverse field profile."""
        beam = GaussianBeam(self.wavelength, self.waist_radius, power=self.power)
        x = np.linspace(-3e-3, 3e-3, 100)
        y = np.array([0])
        z = 0  # At waist
        field = beam.field_profile(x, y, z)
        # Check peak intensity
        peak_intensity = np.max(np.abs(field)**2)
        expected_peak = 2 * self.power / (np.pi * self.waist_radius**2)
        assert_allclose(peak_intensity, expected_peak, rtol=1e-6)
        # Check that field is maximum at center
        center_idx = len(x) // 2
        assert np.argmax(np.abs(field[0, :])) == center_idx
    def test_intensity_profile(self):
        """Test intensity profile calculation."""
        beam = GaussianBeam(self.wavelength, self.waist_radius, power=self.power)
        x = np.linspace(-5e-3, 5e-3, 200)
        y = np.array([0])
        z = 0
        intensity = beam.intensity_profile(x, y, z)
        # Check normalization (total power)
        dx = x[1] - x[0]
        total_power = np.sum(intensity) * dx * 1e-3  # Assuming 1 mm in y
        # Note: This is approximate due to finite grid
        assert total_power > 0.5 * self.power  # Should be close to actual power
class TestDiffraction:
    """Test diffraction calculations."""
    def setup_method(self):
        """Setup test fixtures."""
        self.wavelength = 550e-9  # Green light
        self.screen_distance = 1.0  # 1 meter
    def test_single_slit_diffraction(self):
        """Test single slit diffraction pattern."""
        slit_width = 50e-6  # 50 μm
        result = calculate_diffraction(
            'single_slit', slit_width, self.wavelength,
            self.screen_distance, 0.01
        )
        assert 'position' in result
        assert 'intensity' in result
        assert 'intensity_normalized' in result
        assert result['aperture_type'] == 'single_slit'
        # Check that intensity is maximum at center
        center_idx = len(result['position']) // 2
        assert np.argmax(result['intensity_normalized']) == center_idx
        # Check first minimum position (approximately)
        first_minimum_theory = self.wavelength * self.screen_distance / slit_width
        position = result['position']
        intensity = result['intensity_normalized']
        # Find minima near theoretical position
        theory_idx = np.argmin(np.abs(position - first_minimum_theory))
        local_region = slice(max(0, theory_idx-5), min(len(intensity), theory_idx+5))
        local_min_idx = np.argmin(intensity[local_region])
        # Should be close to minimum
        assert intensity[local_region][local_min_idx] < 0.1  # Much less than maximum
    def test_circular_aperture(self):
        """Test circular aperture (Airy disk) diffraction."""
        aperture_radius = 100e-6  # 100 μm radius
        result = calculate_diffraction(
            'circular', aperture_radius, self.wavelength,
            self.screen_distance, 0.01
        )
        assert result['aperture_type'] == 'circular'
        # Check central maximum
        center_idx = len(result['position']) // 2
        assert np.argmax(result['intensity_normalized']) == center_idx
    def test_double_slit_diffraction(self):
        """Test double slit diffraction pattern."""
        slit_separation = 200e-6  # 200 μm
        result = calculate_diffraction(
            'double_slit', slit_separation, self.wavelength,
            self.screen_distance, 0.02
        )
        assert result['aperture_type'] == 'double_slit'
        # Should show interference fringes
        intensity = result['intensity_normalized']
        # Find local maxima
        maxima_count = 0
        for i in range(1, len(intensity)-1):
            if (intensity[i] > intensity[i-1] and
                intensity[i] > intensity[i+1] and
                intensity[i] > 0.5):
                maxima_count += 1
        # Should have multiple maxima due to interference
        assert maxima_count >= 3
class TestInterference:
    """Test interference calculations."""
    def setup_method(self):
        """Setup test fixtures."""
        self.wavelength = 632.8e-9  # HeNe laser
        self.screen_distance = 2.0   # 2 meters
    def test_double_slit_interference(self):
        """Test Young's double slit interference."""
        slit_separation = 150e-6  # 150 μm
        result = analyze_interference(
            self.wavelength, slit_separation, self.screen_distance
        )
        assert 'position' in result
        assert 'intensity' in result
        assert 'fringe_spacing' in result
        assert 'visibility' in result
        # Check theoretical fringe spacing
        expected_spacing = self.wavelength * self.screen_distance / slit_separation
        assert_allclose(result['fringe_spacing'], expected_spacing, rtol=1e-6)
        # Visibility should be close to 1 for ideal case
        assert result['visibility'] > 0.9
    def test_partial_coherence(self):
        """Test interference with partial coherence."""
        slit_separation = 100e-6
        coherence_length = 1e-3  # 1 mm
        result = analyze_interference(
            self.wavelength, slit_separation, self.screen_distance,
            coherence_length=coherence_length
        )
        # Visibility should be reduced compared to fully coherent case
        assert result['visibility'] < 1.0
        assert result['visibility'] > 0.1  # But not zero
class TestUtilityFunctions:
    """Test utility functions."""
    def test_fresnel_number(self):
        """Test Fresnel number calculation."""
        aperture_radius = 1e-3  # 1 mm
        wavelength = 633e-9
        distance = 1.0  # 1 meter
        fn = fresnel_number(aperture_radius, wavelength, distance)
        expected_fn = aperture_radius**2 / (wavelength * distance)
        assert_allclose(fn, expected_fn, rtol=1e-12)
    def test_rayleigh_range(self):
        """Test Rayleigh range calculation."""
        wavelength = 1064e-9
        waist_radius = 1e-3  # 1 mm
        zr = rayleigh_range(wavelength, waist_radius)
        expected_zr = np.pi * waist_radius**2 / wavelength
        assert_allclose(zr, expected_zr, rtol=1e-12)
        # Test with different medium
        medium_index = 1.5
        zr_medium = rayleigh_range(wavelength, waist_radius, medium_index)
        expected_zr_medium = np.pi * waist_radius**2 / (wavelength / medium_index)
        assert_allclose(zr_medium, expected_zr_medium, rtol=1e-12)
class TestErrorHandling:
    """Test error handling and edge cases."""
    def test_invalid_wavelength(self):
        """Test behavior with invalid wavelength."""
        with pytest.raises((ValueError, RuntimeError)):
            PlaneWave(-1e-9, 1.0)  # Negative wavelength
    def test_zero_waist_radius(self):
        """Test behavior with zero waist radius."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            GaussianBeam(633e-9, 0)  # Zero waist radius
    def test_invalid_aperture_type(self):
        """Test behavior with invalid aperture type."""
        with pytest.raises(ValueError):
            calculate_diffraction('invalid_type', 1e-3, 633e-9, 1.0, 0.01)
if __name__ == "__main__":
    pytest.main([__file__])
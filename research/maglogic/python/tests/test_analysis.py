"""
Tests for MagLogic magnetization analysis module.

This module contains comprehensive tests for magnetization analysis functionality,
including domain detection, energy calculations, and topological analysis.

Author: Meshal Alawein
Email: meshal@berkeley.edu
License: MIT
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

from maglogic.analysis.magnetization import MagnetizationAnalyzer
from maglogic.parsers import OOMMFParser


class TestMagnetizationAnalyzer:
    """Test MagnetizationAnalyzer functionality."""
    
    @pytest.fixture
    def analyzer(self):
        """Create MagnetizationAnalyzer instance."""
        return MagnetizationAnalyzer()
    
    @pytest.fixture
    def sample_magnetization_2d(self):
        """Create sample 2D magnetization data."""
        # 5x5 grid with vortex-like pattern
        size = 5
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # Create vortex pattern
        r = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        
        # Avoid singularity at center
        r = np.maximum(r, 0.1)
        
        mx = -np.sin(theta) * np.exp(-r)
        my = np.cos(theta) * np.exp(-r)
        mz = np.tanh(r - 0.5)
        
        return {
            'mx': mx,
            'my': my,
            'mz': mz,
            'magnitude': np.sqrt(mx**2 + my**2 + mz**2),
            'theta': np.arccos(np.clip(mz / np.sqrt(mx**2 + my**2 + mz**2), -1, 1)),
            'phi': np.arctan2(my, mx),
            'mx_norm': mx / np.sqrt(mx**2 + my**2 + mz**2),
            'my_norm': my / np.sqrt(mx**2 + my**2 + mz**2),
            'mz_norm': mz / np.sqrt(mx**2 + my**2 + mz**2)
        }
    
    @pytest.fixture
    def sample_coordinates_2d(self):
        """Create sample 2D coordinate data."""
        size = 5
        x = np.linspace(0, 4e-9, size)
        y = np.linspace(0, 4e-9, size)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = np.zeros_like(X)
        
        return {
            'x': X,
            'y': Y,
            'z': Z
        }
    
    @pytest.fixture
    def uniform_magnetization(self):
        """Create uniform magnetization data."""
        size = 10
        mx = np.ones((size, size))
        my = np.zeros((size, size))
        mz = np.zeros((size, size))
        
        return {
            'mx': mx,
            'my': my,
            'mz': mz,
            'magnitude': np.ones((size, size)),
            'theta': np.zeros((size, size)),
            'phi': np.zeros((size, size)),
            'mx_norm': mx,
            'my_norm': my,
            'mz_norm': mz
        }
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer.material_params is not None
        assert 'Ms' in analyzer.material_params
        assert 'A' in analyzer.material_params
        assert analyzer.oommf_parser is not None
        assert analyzer.mumax3_parser is not None
    
    def test_custom_material_params(self):
        """Test initialization with custom material parameters."""
        custom_params = {'Ms': 1e6, 'A': 2e-11, 'alpha': 0.01}
        analyzer = MagnetizationAnalyzer(material_params=custom_params)
        
        assert analyzer.material_params['Ms'] == 1e6
        assert analyzer.material_params['A'] == 2e-11
        assert analyzer.material_params['alpha'] == 0.01
    
    def test_analyze_domains(self, analyzer, sample_magnetization_2d, sample_coordinates_2d):
        """Test domain analysis."""
        result = analyzer.analyze_domains(sample_magnetization_2d, sample_coordinates_2d)
        
        # Check result structure
        assert 'num_domains' in result
        assert 'domain_labels' in result
        assert 'domain_walls' in result
        assert 'domain_statistics' in result
        assert 'average_domain_size' in result
        assert 'domain_wall_density' in result
        
        # Check that domains are detected
        assert result['num_domains'] >= 0
        assert result['domain_labels'].shape == sample_magnetization_2d['mx'].shape
        assert isinstance(result['domain_statistics'], dict)
    
    def test_analyze_uniform_domains(self, analyzer, uniform_magnetization, sample_coordinates_2d):
        """Test domain analysis on uniform magnetization."""
        # Modify coordinates to match uniform magnetization shape
        size = 10
        x = np.linspace(0, 9e-9, size)
        y = np.linspace(0, 9e-9, size)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = np.zeros_like(X)
        
        coords = {'x': X, 'y': Y, 'z': Z}
        
        result = analyzer.analyze_domains(uniform_magnetization, coords)
        
        # Uniform state should have fewer domains
        assert result['num_domains'] <= 2  # Might have one large domain
        assert result['domain_wall_density'] < 0.1  # Low domain wall density
    
    def test_calculate_energy_densities(self, analyzer, sample_magnetization_2d, sample_coordinates_2d):
        """Test energy density calculations."""
        result = analyzer.calculate_energy_densities(sample_magnetization_2d, sample_coordinates_2d)
        
        # Check result structure
        assert 'exchange_energy' in result
        assert 'demagnetization_energy' in result
        assert 'anisotropy_energy' in result
        assert 'total_energy' in result
        
        # Check energy components
        for energy_type in ['exchange_energy', 'demagnetization_energy', 'anisotropy_energy', 'total_energy']:
            energy = result[energy_type]
            assert 'density' in energy
            assert 'total' in energy
            assert 'average' in energy
            
            # Check that energy values are reasonable
            assert np.isfinite(energy['total'])
            assert np.isfinite(energy['average'])
            assert energy['density'].shape == sample_magnetization_2d['mx'].shape
    
    def test_spatial_statistics(self, analyzer, sample_magnetization_2d):
        """Test spatial statistics calculation."""
        result = analyzer.spatial_statistics(sample_magnetization_2d)
        
        # Check result structure
        assert 'field_statistics' in result
        assert 'correlations' in result
        assert 'gradients' in result
        assert 'uniformity_index' in result
        assert 'coherence_length' in result
        
        # Check field statistics
        field_stats = result['field_statistics']
        for component in ['mx_stats', 'my_stats', 'mz_stats', 'magnitude_stats']:
            assert component in field_stats
            stats = field_stats[component]
            assert 'mean' in stats
            assert 'std' in stats
            assert 'min' in stats
            assert 'max' in stats
        
        # Check correlations
        correlations = result['correlations']
        assert 'mx_my_correlation' in correlations
        assert 'mx_mz_correlation' in correlations
        assert 'my_mz_correlation' in correlations
        
        # Check that correlation values are in valid range
        for corr in correlations.values():
            assert -1 <= corr <= 1
    
    def test_analyze_topology(self, analyzer, sample_magnetization_2d):
        """Test topological analysis."""
        result = analyzer.analyze_topology(sample_magnetization_2d)
        
        # Check result structure
        assert 'vortices' in result
        assert 'total_topological_charge' in result
        assert 'topological_charge_density' in result
        assert 'skyrmions' in result
        assert 'num_topological_defects' in result
        
        # Check vortices
        vortices = result['vortices']
        assert 'positions' in vortices
        assert 'charges' in vortices
        assert len(vortices['positions']) == len(vortices['charges'])
        
        # Check topological charge
        assert np.isfinite(result['total_topological_charge'])
        
        # Check skyrmions
        skyrmions = result['skyrmions']
        assert 'skyrmions' in skyrmions
        assert 'num_skyrmions' in skyrmions
    
    def test_analyze_texture(self, analyzer, sample_magnetization_2d):
        """Test texture analysis."""
        result = analyzer.analyze_texture(sample_magnetization_2d)
        
        # Check result structure
        assert 'texture_metrics' in result
        assert 'pattern_analysis' in result
        assert 'texture_complexity' in result
        
        # Check texture metrics
        metrics = result['texture_metrics']
        assert 'roughness' in metrics
        assert 'directionality' in metrics
        assert 'complexity_index' in metrics
        
        # Check pattern analysis
        patterns = result['pattern_analysis']
        assert 'uniform_state' in patterns
        assert 'vortex_state' in patterns
        assert 'stripe_domains' in patterns
        assert 'flux_closure' in patterns
        
        # Check that texture complexity is a finite number
        assert np.isfinite(result['texture_complexity'])
    
    def test_pattern_recognition_uniform(self, analyzer, uniform_magnetization):
        """Test pattern recognition on uniform state."""
        result = analyzer.analyze_texture(uniform_magnetization)
        patterns = result['pattern_analysis']
        
        # Uniform magnetization should be detected as uniform state
        assert patterns['uniform_state'] is True
        assert patterns['vortex_state'] is False  # Should not detect vortices in uniform state
    
    def test_analyze_ovf_file(self, analyzer, tmp_path):
        """Test complete OVF file analysis."""
        # Create sample OVF file
        ovf_content = """# OOMMF OVF 2.0
# Segment count: 1
# Begin: Segment
# Begin: Header
# Title: Test analysis
# meshtype: rectangular
# meshunit: m
# xnodes: 3
# ynodes: 3
# znodes: 1
# xstepsize: 1e-9
# ystepsize: 1e-9
# zstepsize: 1e-9
# valuedim: 3
# valueunit: A/m
# End: Header
# Begin: Data Text
1.0 0.0 0.0
0.707 0.707 0.0
0.0 1.0 0.0
-0.707 0.707 0.0
-1.0 0.0 0.0
-0.707 -0.707 0.0
0.0 -1.0 0.0
0.707 -0.707 0.0
1.0 0.0 0.0
# End: Data Text
# End: Segment
"""
        
        ovf_file = tmp_path / "test_analysis.ovf"
        ovf_file.write_text(ovf_content)
        
        result = analyzer.analyze_ovf_file(ovf_file)
        
        # Check complete analysis structure
        assert 'filepath' in result
        assert 'metadata' in result
        assert 'domain_analysis' in result
        assert 'energy_analysis' in result
        assert 'spatial_analysis' in result
        assert 'topological_analysis' in result
        assert 'texture_analysis' in result
        
        # Check that filepath is recorded
        assert str(ovf_file) in result['filepath']
    
    def test_plot_magnetization_map(self, analyzer, tmp_path):
        """Test magnetization map plotting."""
        # Create sample OVF file
        ovf_content = """# OOMMF OVF 2.0
# Segment count: 1
# Begin: Segment
# Begin: Header
# Title: Test plotting
# meshtype: rectangular
# meshunit: m
# xnodes: 4
# ynodes: 4
# znodes: 1
# xstepsize: 1e-9
# ystepsize: 1e-9
# zstepsize: 1e-9
# valuedim: 3
# valueunit: A/m
# End: Header
# Begin: Data Text
1.0 0.0 0.0
0.5 0.866 0.0
-0.5 0.866 0.0
-1.0 0.0 0.0
0.866 0.5 0.0
0.0 1.0 0.0
-0.866 0.5 0.0
-0.5 -0.866 0.0
0.866 -0.5 0.0
0.0 -1.0 0.0
-0.866 -0.5 0.0
0.5 -0.866 0.0
1.0 0.0 0.0
0.5 0.866 0.0
-0.5 0.866 0.0
-1.0 0.0 0.0
# End: Data Text
# End: Segment
"""
        
        ovf_file = tmp_path / "test_plot.ovf"
        ovf_file.write_text(ovf_content)
        
        # Analyze file
        analysis_results = analyzer.analyze_ovf_file(ovf_file)
        
        # Test plotting different components
        for component in ['mx', 'my', 'mz']:
            fig = analyzer.plot_magnetization_map(analysis_results, component=component)
            
            # Check that figure was created
            assert fig is not None
            assert len(fig.axes) >= 1
            
            # Close figure to free memory
            import matplotlib.pyplot as plt
            plt.close(fig)
    
    def test_energy_calculation_physical_values(self, analyzer):
        """Test that energy calculations give physically reasonable values."""
        # Create simple test case with known properties
        size = 10
        dx = 1e-9  # 1 nm spacing
        
        # Create simple domain wall
        x = np.linspace(-5*dx, 5*dx, size)
        y = np.linspace(-5*dx, 5*dx, size)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = np.zeros_like(X)
        
        # Tanh domain wall profile
        mx = np.tanh(X / dx)  # Domain wall along y-axis
        my = np.zeros_like(mx)
        mz = np.zeros_like(mx)
        
        magnetization = {
            'mx': mx,
            'my': my, 
            'mz': mz,
            'magnitude': np.sqrt(mx**2 + my**2 + mz**2),
            'theta': np.arccos(np.clip(mz / np.sqrt(mx**2 + my**2 + mz**2), -1, 1)),
            'phi': np.arctan2(my, mx),
            'mx_norm': mx / np.sqrt(mx**2 + my**2 + mz**2),
            'my_norm': my / np.sqrt(mx**2 + my**2 + mz**2),
            'mz_norm': mz / np.sqrt(mx**2 + my**2 + mz**2)
        }
        
        coordinates = {'x': X, 'y': Y, 'z': Z}
        
        # Calculate energies
        energy_result = analyzer.calculate_energy_densities(magnetization, coordinates)
        
        # Check that exchange energy is positive (domain wall costs energy)
        exchange_total = energy_result['exchange_energy']['total']
        assert exchange_total > 0, "Exchange energy should be positive for domain wall"
        
        # Check that energy values are finite
        for energy_type in ['exchange_energy', 'demagnetization_energy', 'anisotropy_energy']:
            total_energy = energy_result[energy_type]['total']
            assert np.isfinite(total_energy), f"{energy_type} should be finite"
    
    def test_domain_detection_stability(self, analyzer):
        """Test domain detection stability with noise."""
        # Create base magnetization pattern
        size = 20
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Create stripe domains
        mx = np.sign(np.sin(5 * np.pi * X))
        my = np.zeros_like(mx)
        mz = np.zeros_like(mx)
        
        # Add small amount of noise
        noise_level = 0.1
        mx_noisy = mx + noise_level * np.random.normal(size=mx.shape)
        my_noisy = my + noise_level * np.random.normal(size=my.shape)
        mz_noisy = mz + noise_level * np.random.normal(size=mz.shape)
        
        # Normalize
        mag = np.sqrt(mx_noisy**2 + my_noisy**2 + mz_noisy**2)
        mx_noisy /= mag
        my_noisy /= mag
        mz_noisy /= mag
        
        magnetization = {
            'mx': mx_noisy,
            'my': my_noisy,
            'mz': mz_noisy,
            'magnitude': np.ones_like(mx),
            'theta': np.arccos(np.clip(mz_noisy, -1, 1)),
            'phi': np.arctan2(my_noisy, mx_noisy),
            'mx_norm': mx_noisy,
            'my_norm': my_noisy,
            'mz_norm': mz_noisy
        }
        
        coordinates = {
            'x': X,
            'y': Y,
            'z': np.zeros_like(X)
        }
        
        # Analyze domains
        domain_result = analyzer.analyze_domains(magnetization, coordinates)
        
        # Should detect multiple domains in stripe pattern
        assert domain_result['num_domains'] > 1, "Should detect multiple domains in stripe pattern"
        
        # Domain wall density should be reasonable
        assert 0 < domain_result['domain_wall_density'] < 1, "Domain wall density should be reasonable"


class TestAnalysisIntegration:
    """Integration tests for analysis module."""
    
    def test_full_analysis_workflow(self, tmp_path):
        """Test complete analysis workflow from file to visualization."""
        # Create complex magnetization pattern
        size = 15
        ovf_content = f"""# OOMMF OVF 2.0
# Segment count: 1
# Begin: Segment
# Begin: Header
# Title: Integration test
# meshtype: rectangular
# meshunit: m
# xnodes: {size}
# ynodes: {size}
# znodes: 1
# xstepsize: 1e-9
# ystepsize: 1e-9
# zstepsize: 1e-9
# valuedim: 3
# valueunit: A/m
# End: Header
# Begin: Data Text
"""
        
        # Create vortex pattern
        center = size // 2
        for j in range(size):
            for i in range(size):
                x = i - center
                y = j - center
                r = np.sqrt(x**2 + y**2)
                
                if r > 0:
                    # Vortex field
                    mx = -y / (r**2 + 1)
                    my = x / (r**2 + 1)
                    mz = np.tanh(r - center/2)
                else:
                    mx, my, mz = 0, 0, 1
                
                # Normalize
                mag = np.sqrt(mx**2 + my**2 + mz**2)
                if mag > 0:
                    mx /= mag
                    my /= mag
                    mz /= mag
                
                ovf_content += f"{mx:.6f} {my:.6f} {mz:.6f}\n"
        
        ovf_content += "# End: Data Text\n# End: Segment\n"
        
        ovf_file = tmp_path / "integration_test.ovf"
        ovf_file.write_text(ovf_content)
        
        # Run complete analysis
        analyzer = MagnetizationAnalyzer()
        results = analyzer.analyze_ovf_file(ovf_file)
        
        # Verify comprehensive analysis was performed
        expected_sections = [
            'domain_analysis', 'energy_analysis', 'spatial_analysis',
            'topological_analysis', 'texture_analysis'
        ]
        
        for section in expected_sections:
            assert section in results, f"Missing analysis section: {section}"
        
        # Check that vortex was detected
        topo_analysis = results['topological_analysis']
        assert topo_analysis['num_topological_defects'] >= 1, "Should detect topological defects in vortex"
        
        # Check that pattern recognition works
        texture_analysis = results['texture_analysis']
        patterns = texture_analysis['pattern_analysis']
        assert patterns['vortex_state'] is True, "Should detect vortex pattern"
        assert patterns['uniform_state'] is False, "Should not detect uniform state in vortex"
        
        # Test visualization generation
        fig = analyzer.plot_magnetization_map(results, component='mz')
        assert fig is not None
        
        # Clean up
        import matplotlib.pyplot as plt
        plt.close(fig)
    
    def test_analysis_error_handling(self, tmp_path):
        """Test error handling in analysis methods."""
        analyzer = MagnetizationAnalyzer()
        
        # Test with non-existent file
        with pytest.raises(Exception):  # Should raise some kind of error
            analyzer.analyze_ovf_file("nonexistent.ovf")
        
        # Test with corrupted OVF file
        corrupted_file = tmp_path / "corrupted.ovf"
        corrupted_file.write_text("This is not a valid OVF file")
        
        with pytest.raises(Exception):
            analyzer.analyze_ovf_file(corrupted_file)
    
    def test_analysis_consistency(self, tmp_path):
        """Test that repeated analysis gives consistent results."""
        # Create deterministic test case
        ovf_content = """# OOMMF OVF 2.0
# Segment count: 1
# Begin: Segment
# Begin: Header
# Title: Consistency test
# meshtype: rectangular
# meshunit: m
# xnodes: 5
# ynodes: 5
# znodes: 1
# xstepsize: 1e-9
# ystepsize: 1e-9
# zstepsize: 1e-9
# valuedim: 3
# valueunit: A/m
# End: Header
# Begin: Data Text
1.0 0.0 0.0
0.8 0.6 0.0
0.0 1.0 0.0
-0.8 0.6 0.0
-1.0 0.0 0.0
0.6 -0.8 0.0
0.0 -1.0 0.0
-0.6 -0.8 0.0
1.0 0.0 0.0
0.8 0.6 0.0
0.0 1.0 0.0
-0.8 0.6 0.0
-1.0 0.0 0.0
0.6 -0.8 0.0
0.0 -1.0 0.0
-0.6 -0.8 0.0
1.0 0.0 0.0
0.8 0.6 0.0
0.0 1.0 0.0
-0.8 0.6 0.0
-1.0 0.0 0.0
0.6 -0.8 0.0
0.0 -1.0 0.0
-0.6 -0.8 0.0
1.0 0.0 0.0
# End: Data Text
# End: Segment
"""
        
        ovf_file = tmp_path / "consistency.ovf"
        ovf_file.write_text(ovf_content)
        
        analyzer = MagnetizationAnalyzer()
        
        # Run analysis multiple times
        results1 = analyzer.analyze_ovf_file(ovf_file)
        results2 = analyzer.analyze_ovf_file(ovf_file)
        
        # Check that key metrics are consistent
        topo1 = results1['topological_analysis']['total_topological_charge']
        topo2 = results2['topological_analysis']['total_topological_charge']
        assert abs(topo1 - topo2) < 1e-10, "Topological charge should be consistent"
        
        energy1 = results1['energy_analysis']['total_energy']['total']
        energy2 = results2['energy_analysis']['total_energy']['total']
        assert abs(energy1 - energy2) < 1e-10, "Total energy should be consistent"


@pytest.mark.benchmark
class TestAnalysisPerformance:
    """Performance tests for analysis module."""
    
    def test_large_system_analysis_performance(self, benchmark, tmp_path):
        """Test performance of analyzing large magnetic systems."""
        # Create large system
        size = 50
        total_points = size * size
        
        ovf_content = f"""# OOMMF OVF 2.0
# Segment count: 1
# Begin: Segment
# Begin: Header
# Title: Performance test
# meshtype: rectangular
# meshunit: m
# xnodes: {size}
# ynodes: {size}
# znodes: 1
# xstepsize: 1e-9
# ystepsize: 1e-9
# zstepsize: 1e-9
# valuedim: 3
# valueunit: A/m
# End: Header
# Begin: Data Text
"""
        
        # Create spiral pattern for interesting topology
        center = size // 2
        for j in range(size):
            for i in range(size):
                x = (i - center) / center
                y = (j - center) / center
                r = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x) + r * np.pi
                
                mx = np.cos(theta) * np.exp(-r)
                my = np.sin(theta) * np.exp(-r)
                mz = np.tanh(2 * (r - 0.5))
                
                # Normalize
                mag = np.sqrt(mx**2 + my**2 + mz**2)
                if mag > 0:
                    mx /= mag
                    my /= mag
                    mz /= mag
                
                ovf_content += f"{mx:.6f} {my:.6f} {mz:.6f}\n"
        
        ovf_content += "# End: Data Text\n# End: Segment\n"
        
        ovf_file = tmp_path / "performance_test.ovf"
        ovf_file.write_text(ovf_content)
        
        analyzer = MagnetizationAnalyzer()
        
        def analyze_large_system():
            return analyzer.analyze_ovf_file(ovf_file)
        
        result = benchmark(analyze_large_system)
        
        # Verify that analysis completed successfully
        assert 'domain_analysis' in result
        assert 'energy_analysis' in result
        assert 'topological_analysis' in result
        
        # Check that reasonable number of domains detected
        num_domains = result['domain_analysis']['num_domains']
        assert 0 <= num_domains <= total_points // 10  # Reasonable upper bound


if __name__ == "__main__":
    pytest.main([__file__])
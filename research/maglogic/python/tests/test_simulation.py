"""
Tests for MagLogic simulation module.

This module contains tests for OOMMF simulation runner and parameter management.

Author: Meshal Alawein
Email: meshal@berkeley.edu
License: MIT
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess

from maglogic.simulation.oommf_runner import OOMMFRunner


class TestOOMMFRunner:
    """Test OOMMF simulation runner functionality."""
    
    @pytest.fixture
    def mock_oommf_path(self, tmp_path):
        """Create mock OOMMF installation."""
        oommf_dir = tmp_path / "oommf"
        oommf_dir.mkdir()
        
        # Create mock oommf.tcl
        oommf_tcl = oommf_dir / "oommf.tcl"
        oommf_tcl.write_text("# Mock OOMMF TCL script")
        
        return str(oommf_dir)
    
    @pytest.fixture
    def runner_with_mock_oommf(self, mock_oommf_path, tmp_path):
        """Create OOMMF runner with mock installation."""
        working_dir = tmp_path / "work"
        working_dir.mkdir()
        
        with patch.object(OOMMFRunner, '_verify_oommf_installation'):
            runner = OOMMFRunner(oommf_path=mock_oommf_path, working_dir=str(working_dir))
        
        return runner
    
    @pytest.fixture
    def sample_mif_content(self):
        """Create sample MIF file content."""
        return """# Sample MIF file for testing
# MagLogic Test Simulation

SetOptions {
    basename test_simulation
    scalar_output_format %.17g
    vector_field_output_format {text %#.17g}
}

# Parameters
Parameter Ms {{Ms}}
Parameter A {{A}}
Parameter alpha {{alpha}}

# Mesh
Specify Oxs_BoxAtlas:ORCHEX {
    xrange {0 100e-9}
    yrange {0 100e-9}
    zrange {0 5e-9}
}

Specify Oxs_RectangularMesh:mesh {
    cellsize {2e-9 2e-9 5e-9}
    ORCHEX :ORCHEX
}

# Material
Specify Oxs_UniformExchange {
    A $A
}

Specify Oxs_Demag {}

# Evolver
Specify Oxs_CGEvolve:evolve {
    method conjugate_gradient
}

# Driver
Specify Oxs_MinDriver {
    evolver :evolve
    stopping_mxHxm 0.01
    mesh :mesh
    Ms $Ms
    m0 {1 0 0}
}

Destination table mmArchive
Destination mags mmArchive

Schedule DataTable table Stage 1
Schedule Oxs_MinDriver::Magnetization mags Stage 1
"""
    
    def test_runner_initialization_with_path(self, mock_oommf_path, tmp_path):
        """Test runner initialization with explicit OOMMF path."""
        working_dir = tmp_path / "work"
        
        with patch.object(OOMMFRunner, '_verify_oommf_installation'):
            runner = OOMMFRunner(oommf_path=mock_oommf_path, working_dir=str(working_dir))
        
        assert runner.oommf_path == mock_oommf_path
        assert runner.working_dir == working_dir
        assert working_dir.exists()
    
    def test_runner_initialization_auto_detect(self, tmp_path):
        """Test runner initialization with auto-detection."""
        working_dir = tmp_path / "work"
        
        with patch.object(OOMMFRunner, '_find_oommf_path', return_value="/mock/oommf"):
            with patch.object(OOMMFRunner, '_verify_oommf_installation'):
                runner = OOMMFRunner(working_dir=str(working_dir))
        
        assert runner.oommf_path == "/mock/oommf"
        assert runner.working_dir == working_dir
    
    def test_find_oommf_path_environment(self):
        """Test OOMMF path detection from environment variable."""
        with patch.dict('os.environ', {'OOMMF_ROOT': '/env/oommf'}):
            with patch('pathlib.Path.exists', return_value=True):
                runner = OOMMFRunner.__new__(OOMMFRunner)  # Create without calling __init__
                path = runner._find_oommf_path()
                assert path == '/env/oommf'
    
    def test_find_oommf_path_command(self):
        """Test OOMMF path detection from command in PATH."""
        with patch('shutil.which', return_value='/usr/bin/oommf'):
            runner = OOMMFRunner.__new__(OOMMFRunner)
            path = runner._find_oommf_path()
            assert path == '/usr/bin/oommf'
    
    def test_find_oommf_path_not_found(self):
        """Test OOMMF path detection when not found."""
        with patch('shutil.which', return_value=None):
            with patch('pathlib.Path.exists', return_value=False):
                runner = OOMMFRunner.__new__(OOMMFRunner)
                path = runner._find_oommf_path()
                assert path is None
    
    def test_parameter_substitution(self, runner_with_mock_oommf, sample_mif_content):
        """Test parameter substitution in MIF content."""
        parameters = {
            'Ms': 8.6e5,
            'A': 1.3e-11,
            'alpha': 0.008
        }
        
        result = runner_with_mock_oommf._substitute_parameters(sample_mif_content, parameters)
        
        # Check that parameters were substituted
        assert '{{Ms}}' not in result
        assert '{{A}}' not in result
        assert '{{alpha}}' not in result
        assert str(parameters['Ms']) in result
        assert str(parameters['A']) in result
        assert str(parameters['alpha']) in result
    
    def test_prepare_mif_file_from_content(self, runner_with_mock_oommf, sample_mif_content):
        """Test MIF file preparation from content string."""
        parameters = {'Ms': 8.6e5, 'A': 1.3e-11}
        sim_dir = runner_with_mock_oommf.working_dir / "test_sim"
        sim_dir.mkdir()
        
        mif_path = runner_with_mock_oommf._prepare_mif_file(
            sample_mif_content, parameters, sim_dir
        )
        
        assert mif_path.exists()
        assert mif_path.suffix == '.mif'
        
        # Check that parameters were substituted
        content = mif_path.read_text()
        assert '{{Ms}}' not in content
        assert str(parameters['Ms']) in content
    
    def test_prepare_mif_file_from_file(self, runner_with_mock_oommf, sample_mif_content, tmp_path):
        """Test MIF file preparation from existing file."""
        # Create source MIF file
        source_mif = tmp_path / "source.mif"
        source_mif.write_text(sample_mif_content)
        
        parameters = {'Ms': 8.6e5}
        sim_dir = runner_with_mock_oommf.working_dir / "test_sim"
        sim_dir.mkdir()
        
        mif_path = runner_with_mock_oommf._prepare_mif_file(
            str(source_mif), parameters, sim_dir
        )
        
        assert mif_path.exists()
        assert mif_path.name == "source.mif"
        
        # Check that parameters were substituted
        content = mif_path.read_text()
        assert str(parameters['Ms']) in content
    
    @patch('subprocess.run')
    def test_execute_oommf_simulation_success(self, mock_run, runner_with_mock_oommf, tmp_path):
        """Test successful OOMMF simulation execution."""
        # Mock successful subprocess run
        mock_run.return_value = Mock(returncode=0)
        
        mif_file = tmp_path / "test.mif"
        mif_file.write_text("# Test MIF")
        
        sim_dir = tmp_path / "sim"
        sim_dir.mkdir()
        
        result = runner_with_mock_oommf._execute_oommf_simulation(mif_file, sim_dir)
        
        assert result is True
        mock_run.assert_called_once()
        
        # Check that oommf.tcl was called
        call_args = mock_run.call_args[0][0]
        assert 'oommf.tcl' in ' '.join(call_args)
        assert 'boxsi' in call_args
    
    @patch('subprocess.run')
    def test_execute_oommf_simulation_failure(self, mock_run, runner_with_mock_oommf, tmp_path):
        """Test failed OOMMF simulation execution."""
        # Mock failed subprocess run
        mock_run.return_value = Mock(returncode=1)
        
        mif_file = tmp_path / "test.mif"
        mif_file.write_text("# Test MIF")
        
        sim_dir = tmp_path / "sim"
        sim_dir.mkdir()
        
        result = runner_with_mock_oommf._execute_oommf_simulation(mif_file, sim_dir)
        
        assert result is False
    
    @patch('subprocess.run')
    def test_execute_oommf_simulation_timeout(self, mock_run, runner_with_mock_oommf, tmp_path):
        """Test OOMMF simulation timeout."""
        # Mock timeout
        mock_run.side_effect = subprocess.TimeoutExpired('cmd', 10)
        
        mif_file = tmp_path / "test.mif"
        mif_file.write_text("# Test MIF")
        
        sim_dir = tmp_path / "sim"
        sim_dir.mkdir()
        
        result = runner_with_mock_oommf._execute_oommf_simulation(mif_file, sim_dir)
        
        assert result is False
    
    def test_collect_simulation_results_empty(self, runner_with_mock_oommf, tmp_path):
        """Test collecting results from empty simulation directory."""
        sim_dir = tmp_path / "empty_sim"
        sim_dir.mkdir()
        
        mif_file = tmp_path / "test.mif"
        mif_file.write_text("# Test MIF")
        
        results = runner_with_mock_oommf._collect_simulation_results(sim_dir, mif_file)
        
        assert 'magnetization_files' in results
        assert 'table_data' in results
        assert 'metadata' in results
        assert 'log_info' in results
        
        assert len(results['magnetization_files']) == 0
        assert results['table_data'] is None
        assert results['metadata']['num_ovf_files'] == 0
    
    def test_collect_simulation_results_with_files(self, runner_with_mock_oommf, tmp_path):
        """Test collecting results with simulation files."""
        sim_dir = tmp_path / "sim_with_files"
        sim_dir.mkdir()
        
        # Create mock OVF file
        ovf_content = """# OOMMF OVF 2.0
# Segment count: 1
# Begin: Segment
# Begin: Header
# Title: Test
# meshtype: rectangular
# meshunit: m
# xnodes: 2
# ynodes: 2
# znodes: 1
# xstepsize: 1e-9
# ystepsize: 1e-9
# zstepsize: 1e-9
# valuedim: 3
# valueunit: A/m
# End: Header
# Begin: Data Text
1 0 0
0 1 0
-1 0 0
0 -1 0
# End: Data Text
# End: Segment
"""
        (sim_dir / "test.ovf").write_text(ovf_content)
        
        # Create mock ODT file
        odt_content = """# ODT 1.0
# Columns: Time Energy mx my mz
0.0 -1.0e-12 1.0 0.0 0.0
1.0e-12 -1.1e-12 0.9 0.1 0.0
"""
        (sim_dir / "test.odt").write_text(odt_content)
        
        # Create mock log file
        log_content = "OOMMF simulation log\nEnd of run\n"
        (sim_dir / "oommf_output.log").write_text(log_content)
        
        mif_file = tmp_path / "test.mif"
        mif_file.write_text("# Test MIF")
        
        results = runner_with_mock_oommf._collect_simulation_results(sim_dir, mif_file)
        
        assert len(results['magnetization_files']) == 1
        assert results['table_data'] is not None
        assert results['log_info'] is not None
        assert results['metadata']['num_ovf_files'] == 1
        assert results['metadata']['has_table_data'] is True
        assert results['metadata']['has_log'] is True
    
    def test_extract_success_indicators(self, runner_with_mock_oommf):
        """Test extraction of success indicators from log."""
        log_content = """
OOMMF simulation starting...
Step 100 completed
Final energy: -1.234e-12 J
End of run
Simulation completed successfully
"""
        
        indicators = runner_with_mock_oommf._extract_success_indicators(log_content)
        
        assert 'End of run' in indicators
        assert 'Simulation completed' in indicators
        assert 'Final energy' in indicators
    
    def test_extract_warnings(self, runner_with_mock_oommf):
        """Test extraction of warnings from log."""
        log_content = """
OOMMF simulation starting...
Warning: Large time step detected
Step 100 completed
WARNING: Convergence may be slow
Simulation completed
"""
        
        warnings = runner_with_mock_oommf._extract_warnings(log_content)
        
        assert len(warnings) == 2
        assert "Warning: Large time step detected" in warnings
        assert "WARNING: Convergence may be slow" in warnings
    
    def test_extract_timing_info(self, runner_with_mock_oommf):
        """Test extraction of timing information from log."""
        log_content = """
OOMMF simulation starting...
Step 100 completed
Step 200 completed
Step 500 completed
Total time: 45.67 s
Simulation completed
"""
        
        timing = runner_with_mock_oommf._extract_timing_info(log_content)
        
        assert 'total_time' in timing
        assert timing['total_time'] == 45.67
        assert 'num_steps' in timing
        assert timing['num_steps'] == 500
    
    @patch.object(OOMMFRunner, '_execute_oommf_simulation', return_value=True)
    @patch.object(OOMMFRunner, '_collect_simulation_results')
    def test_run_simulation_success(self, mock_collect, mock_execute, runner_with_mock_oommf, sample_mif_content):
        """Test successful simulation run."""
        # Mock collect results
        mock_results = {
            'magnetization_files': [],
            'table_data': None,
            'metadata': {'test': True},
            'log_info': None
        }
        mock_collect.return_value = mock_results
        
        parameters = {'Ms': 8.6e5, 'A': 1.3e-11}
        
        results = runner_with_mock_oommf.run_simulation(
            sample_mif_content, 
            parameters=parameters,
            cleanup=True
        )
        
        assert 'metadata' in results
        assert 'simulation_time' in results['metadata']
        assert 'start_time' in results['metadata']
        assert 'end_time' in results['metadata']
        
        mock_execute.assert_called_once()
        mock_collect.assert_called_once()
    
    @patch.object(OOMMFRunner, '_execute_oommf_simulation', return_value=False)
    def test_run_simulation_failure(self, mock_execute, runner_with_mock_oommf, sample_mif_content):
        """Test simulation run failure."""
        with pytest.raises(RuntimeError, match="OOMMF simulation failed"):
            runner_with_mock_oommf.run_simulation(sample_mif_content)
        
        mock_execute.assert_called_once()
    
    def test_run_simulation_no_oommf(self, tmp_path):
        """Test simulation run without OOMMF available."""
        runner = OOMMFRunner.__new__(OOMMFRunner)
        runner.oommf_path = None
        runner.working_dir = tmp_path
        runner.parser = Mock()
        
        with pytest.raises(RuntimeError, match="OOMMF not available"):
            runner.run_simulation("# Test MIF")
    
    def test_create_parameter_sweep(self, runner_with_mock_oommf, sample_mif_content, tmp_path):
        """Test parameter sweep creation."""
        param_ranges = {
            'Ms': [8.0e5, 8.6e5, 9.0e5],
            'A': [1.0e-11, 1.3e-11, 1.5e-11]
        }
        
        output_dir = tmp_path / "sweep"
        
        # Mock run_simulation to avoid actual OOMMF calls
        def mock_run_simulation(mif, parameters=None, output_dir=None, cleanup=False):
            return {
                'metadata': {'test': True},
                'magnetization_files': [],
                'table_data': None
            }
        
        with patch.object(runner_with_mock_oommf, 'run_simulation', side_effect=mock_run_simulation):
            results = runner_with_mock_oommf.create_parameter_sweep(
                sample_mif_content, param_ranges, output_dir
            )
        
        # Should have 3 * 3 = 9 combinations
        assert len(results) == 9
        
        # Check that all combinations were tried
        param_combinations = [r['sweep_info']['parameters'] for r in results]
        assert len(set(tuple(p.items()) for p in param_combinations)) == 9
        
        # Check that sweep summary was created
        summary_file = output_dir / "sweep_summary.json"
        assert summary_file.exists()
    
    def test_analyze_convergence_no_data(self, runner_with_mock_oommf):
        """Test convergence analysis with no data."""
        result = runner_with_mock_oommf.analyze_convergence(None)
        
        assert result['converged'] is False
        assert 'reason' in result
    
    def test_analyze_convergence_with_data(self, runner_with_mock_oommf):
        """Test convergence analysis with time series data."""
        # Create mock time series that converges
        time_steps = 1000
        time = np.linspace(0, 1e-9, time_steps)
        
        # Energy that converges
        energy = -1e-12 * (1 + np.exp(-time * 1e10))
        
        # Add small noise to final portion to test stability
        energy[-100:] += 1e-15 * np.random.normal(size=100)
        
        # Magnetization components
        mx = np.ones(time_steps) * 0.9
        my = np.ones(time_steps) * 0.1
        mz = np.zeros(time_steps)
        
        # Add small variations to magnetization
        mx[-100:] += 1e-8 * np.random.normal(size=100)
        my[-100:] += 1e-8 * np.random.normal(size=100)
        
        table_data = {
            'time_series': {
                'Time': time,
                'E_total': energy,
                'mx': mx,
                'my': my,
                'mz': mz
            }
        }
        
        result = runner_with_mock_oommf.analyze_convergence(table_data)
        
        assert 'converged' in result
        assert 'energy_convergence' in result
        assert 'magnetization_convergence' in result
        
        # Check energy convergence analysis
        energy_conv = result['energy_convergence']['E_total']
        assert 'final_value' in energy_conv
        assert 'relative_stability' in energy_conv
        assert 'converged' in energy_conv
    
    def test_get_simulation_info(self, runner_with_mock_oommf):
        """Test getting simulation runner information."""
        info = runner_with_mock_oommf.get_simulation_info()
        
        assert 'oommf_path' in info
        assert 'working_directory' in info
        assert 'oommf_available' in info
        assert 'working_dir_exists' in info
        assert 'working_dir_writable' in info
        
        assert info['oommf_available'] is True
        assert info['working_dir_exists'] is True
        assert info['oommf_path'] == runner_with_mock_oommf.oommf_path


class TestSimulationIntegration:
    """Integration tests for simulation module."""
    
    def test_simulation_workflow_mock(self, tmp_path):
        """Test complete simulation workflow with mocked OOMMF."""
        # Create mock OOMMF installation
        oommf_dir = tmp_path / "oommf"
        oommf_dir.mkdir()
        (oommf_dir / "oommf.tcl").write_text("# Mock OOMMF")
        
        # Create working directory
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        
        # Mock MIF content
        mif_content = """
# Test MIF
Parameter Ms {{Ms}}
Parameter A {{A}}
# ... rest of MIF content ...
"""
        
        with patch.object(OOMMFRunner, '_verify_oommf_installation'):
            runner = OOMMFRunner(oommf_path=str(oommf_dir), working_dir=str(work_dir))
        
        # Mock the actual simulation execution
        def mock_execute(mif_path, sim_dir):
            # Create mock output files
            (sim_dir / "test.ovf").write_text("# Mock OVF file")
            (sim_dir / "test.odt").write_text("# Mock ODT file")
            (sim_dir / "oommf_output.log").write_text("End of run\nTotal time: 10.5 s")
            return True
        
        with patch.object(runner, '_execute_oommf_simulation', side_effect=mock_execute):
            results = runner.run_simulation(
                mif_content,
                parameters={'Ms': 8.6e5, 'A': 1.3e-11},
                cleanup=False
            )
        
        assert 'metadata' in results
        assert 'simulation_time' in results['metadata']
        assert results['metadata']['simulation_time'] > 0
    
    def test_error_handling_integration(self, tmp_path):
        """Test error handling in simulation workflow."""
        # Test with no OOMMF installation
        runner = OOMMFRunner.__new__(OOMMFRunner)
        runner.oommf_path = None
        runner.working_dir = tmp_path
        runner.parser = Mock()
        
        with pytest.raises(RuntimeError):
            runner.run_simulation("# Test MIF")


@pytest.mark.benchmark
class TestSimulationPerformance:
    """Performance tests for simulation module."""
    
    def test_parameter_substitution_performance(self, benchmark, tmp_path):
        """Test performance of parameter substitution."""
        # Create large MIF content with many parameters
        mif_content = """# Large MIF file
SetOptions { basename test }

# Many parameters
"""
        
        for i in range(100):
            mif_content += f"Parameter param{i} {{{{param{i}}}}}\n"
        
        mif_content += """
# More content...
Specify Oxs_BoxAtlas:ORCHEX { xrange {0 100e-9} }
"""
        
        # Create many parameters
        parameters = {f'param{i}': f'value{i}' for i in range(100)}
        
        # Create mock runner
        oommf_dir = tmp_path / "oommf"
        oommf_dir.mkdir()
        (oommf_dir / "oommf.tcl").write_text("# Mock")
        
        with patch.object(OOMMFRunner, '_verify_oommf_installation'):
            runner = OOMMFRunner(oommf_path=str(oommf_dir))
        
        def substitute_params():
            return runner._substitute_parameters(mif_content, parameters)
        
        result = benchmark(substitute_params)
        
        # Verify that substitution worked
        assert '{{param0}}' not in result
        assert 'value0' in result
        assert '{{param99}}' not in result
        assert 'value99' in result


if __name__ == "__main__":
    pytest.main([__file__])
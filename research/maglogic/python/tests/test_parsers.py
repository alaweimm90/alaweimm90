"""
Tests for MagLogic parsers.

This module contains comprehensive tests for OOMMF and MuMax3 file parsers,
ensuring robust handling of various file formats and edge cases.

Author: Meshal Alawein
Email: meshal@berkeley.edu
License: MIT
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from maglogic.parsers import OOMMFParser, MuMax3Parser, BaseParser
from maglogic.parsers.base_parser import ParseError, UnsupportedFormatError, CorruptedFileError


class TestBaseParser:
    """Test BaseParser abstract functionality."""
    
    def test_base_parser_cannot_be_instantiated(self):
        """Test that BaseParser cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseParser()
    
    def test_standardize_coordinates(self):
        """Test coordinate standardization."""
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        z = np.array([7, 8, 9])
        
        coords = BaseParser.standardize_coordinates(x, y, z)
        
        assert 'x' in coords
        assert 'y' in coords
        assert 'z' in coords
        np.testing.assert_array_equal(coords['x'], x)
        np.testing.assert_array_equal(coords['y'], y)
        np.testing.assert_array_equal(coords['z'], z)
    
    def test_standardize_magnetization(self):
        """Test magnetization standardization."""
        mx = np.array([1.0, 0.0, -1.0])
        my = np.array([0.0, 1.0, 0.0])
        mz = np.array([0.0, 0.0, 0.0])
        
        mag = BaseParser.standardize_magnetization(mx, my, mz)
        
        # Check basic components
        np.testing.assert_array_equal(mag['mx'], mx)
        np.testing.assert_array_equal(mag['my'], my)
        np.testing.assert_array_equal(mag['mz'], mz)
        
        # Check derived quantities
        expected_magnitude = np.array([1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(mag['magnitude'], expected_magnitude)
        
        # Check normalized components
        np.testing.assert_array_almost_equal(mag['mx_norm'], [1.0, 0.0, -1.0])
        np.testing.assert_array_almost_equal(mag['my_norm'], [0.0, 1.0, 0.0])
        
        # Check angles
        assert 'theta' in mag
        assert 'phi' in mag
    
    def test_calculate_volume_average(self):
        """Test volume average calculation."""
        data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        
        avg = BaseParser.calculate_volume_average(data)
        expected = np.mean(data)
        
        assert avg == expected
    
    def test_calculate_volume_average_with_weights(self):
        """Test volume average calculation with weights."""
        data = np.array([1, 2, 3, 4])
        weights = np.array([1, 2, 3, 4])
        
        avg = BaseParser.calculate_volume_average(data, weights)
        expected = np.average(data, weights=weights)
        
        assert avg == expected


class TestOOMMFParser:
    """Test OOMMF parser functionality."""
    
    @pytest.fixture
    def parser(self):
        """Create OOMMF parser instance."""
        return OOMMFParser(verbose=False)
    
    @pytest.fixture
    def sample_ovf_content(self):
        """Create sample OVF file content."""
        return """# OOMMF OVF 2.0
# Segment count: 1
# Begin: Segment
# Begin: Header
# Title: Test magnetization
# meshtype: rectangular
# meshunit: m
# xnodes: 2
# ynodes: 2
# znodes: 1
# xstepsize: 1e-9
# ystepsize: 1e-9
# zstepsize: 1e-9
# xmin: 0
# ymin: 0
# zmin: 0
# xmax: 1e-9
# ymax: 1e-9
# zmax: 0
# valuedim: 3
# valueunit: A/m
# valuemultiplier: 8.6e5
# End: Header
# Begin: Data Text
1.0 0.0 0.0
0.0 1.0 0.0
-1.0 0.0 0.0
0.0 -1.0 0.0
# End: Data Text
# End: Segment
"""
    
    @pytest.fixture
    def sample_odt_content(self):
        """Create sample ODT file content."""
        return """# ODT 1.0
# Table Start
# Columns: Time {s} Energy {J} mx {} my {} mz {}
# Units:   s     J       {}   {}   {}
0.0 -1.23e-12 1.0 0.0 0.0
1e-12 -1.24e-12 0.9 0.1 0.0
2e-12 -1.25e-12 0.8 0.2 0.0
"""
    
    def test_parser_initialization(self, parser):
        """Test parser initialization."""
        assert parser.verbose is False
        assert '.ovf' in parser.supported_extensions
        assert '.odt' in parser.supported_extensions
    
    def test_validate_file_exists(self, parser, tmp_path):
        """Test file validation for existing file."""
        test_file = tmp_path / "test.ovf"
        test_file.write_text("test content")
        
        assert parser.validate_file(test_file) is True
    
    def test_validate_file_not_exists(self, parser):
        """Test file validation for non-existing file."""
        assert parser.validate_file("nonexistent.ovf") is False
    
    def test_get_file_info(self, parser, tmp_path):
        """Test getting file information."""
        test_file = tmp_path / "test.ovf"
        test_content = "test content"
        test_file.write_text(test_content)
        
        info = parser.get_file_info(test_file)
        
        assert info['exists'] is True
        assert info['size'] == len(test_content)
        assert info['extension'] == '.ovf'
        assert info['filename'] == 'test.ovf'
    
    def test_parse_ovf_file(self, parser, tmp_path, sample_ovf_content):
        """Test parsing OVF file."""
        ovf_file = tmp_path / "test.ovf"
        ovf_file.write_text(sample_ovf_content)
        
        result = parser.parse_ovf(ovf_file)
        
        # Check structure
        assert 'magnetization' in result
        assert 'coordinates' in result
        assert 'metadata' in result
        assert 'header' in result
        
        # Check metadata
        metadata = result['metadata']
        assert metadata['grid_size'] == [2, 2, 1]
        assert metadata['total_cells'] == 4
        assert metadata['file_format'] == 'OVF'
        
        # Check magnetization data
        mag = result['magnetization']
        assert 'mx' in mag
        assert 'my' in mag
        assert 'mz' in mag
        assert 'magnitude' in mag
        
        # Check shapes
        assert mag['mx'].shape == (1, 2, 2)
        assert mag['my'].shape == (1, 2, 2)
        assert mag['mz'].shape == (1, 2, 2)
    
    def test_parse_ovf_invalid_file(self, parser, tmp_path):
        """Test parsing invalid OVF file."""
        ovf_file = tmp_path / "invalid.ovf"
        ovf_file.write_text("invalid content")
        
        with pytest.raises(ParseError):
            parser.parse_ovf(ovf_file)
    
    def test_parse_odt_file(self, parser, tmp_path, sample_odt_content):
        """Test parsing ODT file."""
        odt_file = tmp_path / "test.odt"
        odt_file.write_text(sample_odt_content)
        
        result = parser.parse_odt(odt_file)
        
        # Check structure
        assert 'time_series' in result
        assert 'metadata' in result
        assert 'header' in result
        
        # Check time series data
        time_series = result['time_series']
        assert 'Time' in time_series
        assert 'Energy' in time_series
        assert 'mx' in time_series
        
        # Check data values
        assert len(time_series['Time']) == 3
        np.testing.assert_array_equal(time_series['Time'], [0.0, 1e-12, 2e-12])
        np.testing.assert_array_equal(time_series['mx'], [1.0, 0.9, 0.8])
    
    def test_parse_file_auto_detection(self, parser, tmp_path, sample_ovf_content):
        """Test automatic file format detection."""
        ovf_file = tmp_path / "test.ovf"
        ovf_file.write_text(sample_ovf_content)
        
        result = parser.parse_file(ovf_file)
        
        # Should detect as OVF and parse correctly
        assert 'magnetization' in result
        assert result['metadata']['file_format'] == 'OVF'
    
    def test_get_ovf_info(self, parser, tmp_path, sample_ovf_content):
        """Test getting OVF file information without full parsing."""
        ovf_file = tmp_path / "test.ovf"
        ovf_file.write_text(sample_ovf_content)
        
        info = parser.get_ovf_info(ovf_file)
        
        assert info['grid_size'] == [2, 2, 1]
        assert info['total_cells'] == 4
        assert info['data_format'] == 'text'
        assert info['ovf_version'] == '2.0'


class TestMuMax3Parser:
    """Test MuMax3 parser functionality."""
    
    @pytest.fixture
    def parser(self):
        """Create MuMax3 parser instance."""
        return MuMax3Parser(verbose=False)
    
    @pytest.fixture
    def sample_table_content(self):
        """Create sample MuMax3 table content."""
        return """# t (s)\tE_total (J)\tmx ()\tmy ()\tmz ()
0\t-1.234e-12\t1\t0\t0
1e-12\t-1.235e-12\t0.9\t0.1\t0
2e-12\t-1.236e-12\t0.8\t0.2\t0
"""
    
    @pytest.fixture
    def sample_json_content(self):
        """Create sample MuMax3 JSON content."""
        return """{
    "Msat": 860000,
    "Aex": 1.3e-11,
    "alpha": 0.008,
    "GridSize": [64, 64, 1],
    "CellSize": [2e-9, 2e-9, 1e-9]
}"""
    
    def test_parser_initialization(self, parser):
        """Test parser initialization."""
        assert parser.verbose is False
        assert '.txt' in parser.supported_extensions
        assert '.ovf' in parser.supported_extensions
        assert '.json' in parser.supported_extensions
    
    def test_parse_table_file(self, parser, tmp_path, sample_table_content):
        """Test parsing MuMax3 table file."""
        table_file = tmp_path / "table.txt"
        table_file.write_text(sample_table_content)
        
        result = parser.parse_table(table_file)
        
        # Check structure
        assert 'time_series' in result
        assert 'metadata' in result
        assert 'header' in result
        
        # Check time series data
        time_series = result['time_series']
        assert 't' in time_series
        assert 'E_total' in time_series
        assert 'mx' in time_series
        
        # Check metadata
        metadata = result['metadata']
        assert metadata['file_format'] == 'MuMax3_table'
        assert metadata['num_rows'] == 3
        assert metadata['num_columns'] == 5
        
        # Check timing statistics
        assert 'time_range' in metadata
        assert 'time_step' in metadata
        assert 'simulation_time' in metadata
    
    def test_parse_json_file(self, parser, tmp_path, sample_json_content):
        """Test parsing MuMax3 JSON file."""
        json_file = tmp_path / "params.json"
        json_file.write_text(sample_json_content)
        
        result = parser.parse_json(json_file)
        
        assert 'parameters' in result
        assert 'metadata' in result
        
        params = result['parameters']
        assert params['Msat'] == 860000
        assert params['Aex'] == 1.3e-11
        assert params['alpha'] == 0.008
    
    def test_parse_output_log(self, parser, tmp_path):
        """Test parsing MuMax3 output log."""
        log_content = """MuMax3 3.10 linux_amd64 go1.17.8
Using CUDA device 0: GeForce RTX 3080
Ms = 8.6e+05 A/m
A = 1.3e-11 J/m
SetGridSize(64, 64, 1)
SetCellSize(2e-09, 2e-09, 1e-09)
//output directory: .
tableAdd(E_total)
Step 1000, time 1e-09 s
total time: 45.2 s
"""
        
        log_file = tmp_path / "simulation.out"
        log_file.write_text(log_content)
        
        result = parser.parse_output_log(log_file)
        
        assert 'log_info' in result
        assert 'metadata' in result
        
        log_info = result['log_info']
        assert 'parameters' in log_info
        assert 'performance' in log_info
        
        # Check extracted parameters
        params = log_info['parameters']
        assert 'Msat' in params
        assert 'Aex' in params
        
        # Check performance info
        perf = log_info['performance']
        assert 'total_time' in perf
        assert 'gpu_name' in perf
    
    def test_parse_simulation_directory(self, parser, tmp_path, sample_table_content):
        """Test parsing entire simulation directory."""
        # Create simulation directory structure
        sim_dir = tmp_path / "simulation"
        sim_dir.mkdir()
        
        # Add table file
        table_file = sim_dir / "table.txt"
        table_file.write_text(sample_table_content)
        
        # Add OVF files
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
        (sim_dir / "m000001.ovf").write_text(ovf_content)
        (sim_dir / "m000002.ovf").write_text(ovf_content)
        
        result = parser.parse_simulation_directory(sim_dir)
        
        # Check structure
        assert 'table_data' in result
        assert 'magnetization_files' in result
        assert 'metadata' in result
        
        # Check that files were found and parsed
        assert result['table_data'] is not None
        assert len(result['magnetization_files']) == 2
        
        # Check metadata
        metadata = result['metadata']
        assert metadata['simulation_name'] == 'simulation'
        assert metadata['num_magnetization_files'] == 2
        assert metadata['has_table_data'] is True
    
    def test_get_table_info(self, parser, tmp_path, sample_table_content):
        """Test getting table file information."""
        table_file = tmp_path / "table.txt"
        table_file.write_text(sample_table_content)
        
        info = parser.get_table_info(table_file)
        
        assert info['estimated_rows'] == 3
        assert info['estimated_columns'] == 5
        assert info['has_header'] is True
        assert len(info['columns']) == 5


class TestParserIntegration:
    """Integration tests for parsers."""
    
    def test_parser_consistency(self, tmp_path):
        """Test that both parsers handle OVF files consistently."""
        ovf_content = """# OOMMF OVF 2.0
# Segment count: 1
# Begin: Segment
# Begin: Header
# Title: Consistency test
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
        
        ovf_file = tmp_path / "consistency.ovf"
        ovf_file.write_text(ovf_content)
        
        # Parse with both parsers
        oommf_parser = OOMMFParser()
        mumax3_parser = MuMax3Parser()
        
        oommf_result = oommf_parser.parse_ovf(ovf_file)
        mumax3_result = mumax3_parser.parse_ovf(ovf_file)
        
        # Check that both parsers extract the same grid information
        oommf_meta = oommf_result['metadata']
        mumax3_meta = mumax3_result['metadata']
        
        assert oommf_meta['grid_size'] == mumax3_meta['grid_size']
        assert oommf_meta['total_cells'] == mumax3_meta['total_cells']
        
        # Check that magnetization data is consistent
        oommf_mag = oommf_result['magnetization']
        mumax3_mag = mumax3_result['magnetization']
        
        np.testing.assert_array_almost_equal(oommf_mag['mx'], mumax3_mag['mx'])
        np.testing.assert_array_almost_equal(oommf_mag['my'], mumax3_mag['my'])
        np.testing.assert_array_almost_equal(oommf_mag['mz'], mumax3_mag['mz'])
    
    def test_error_handling(self, tmp_path):
        """Test error handling across parsers."""
        # Test corrupted file
        corrupted_file = tmp_path / "corrupted.ovf"
        corrupted_file.write_text("This is not a valid OVF file")
        
        parser = OOMMFParser()
        
        with pytest.raises(ParseError):
            parser.parse_ovf(corrupted_file)
        
        # Test missing file
        with pytest.raises(ParseError):
            parser.parse_ovf(tmp_path / "nonexistent.ovf")


@pytest.mark.benchmark
class TestParserPerformance:
    """Performance tests for parsers."""
    
    def test_large_ovf_parsing_performance(self, benchmark, tmp_path):
        """Test performance of parsing large OVF files."""
        # Create large OVF file
        grid_size = 50
        total_points = grid_size * grid_size
        
        ovf_content = f"""# OOMMF OVF 2.0
# Segment count: 1
# Begin: Segment
# Begin: Header
# Title: Performance test
# meshtype: rectangular
# meshunit: m
# xnodes: {grid_size}
# ynodes: {grid_size}
# znodes: 1
# xstepsize: 1e-9
# ystepsize: 1e-9
# zstepsize: 1e-9
# valuedim: 3
# valueunit: A/m
# End: Header
# Begin: Data Text
"""
        
        # Add magnetization data
        for i in range(total_points):
            angle = 2 * np.pi * i / total_points
            mx = np.cos(angle)
            my = np.sin(angle)
            ovf_content += f"{mx:.6f} {my:.6f} 0.0\n"
        
        ovf_content += "# End: Data Text\n# End: Segment\n"
        
        ovf_file = tmp_path / "large.ovf"
        ovf_file.write_text(ovf_content)
        
        parser = OOMMFParser()
        
        def parse_large_ovf():
            return parser.parse_ovf(ovf_file)
        
        result = benchmark(parse_large_ovf)
        
        # Verify result structure
        assert result['metadata']['total_cells'] == total_points
        assert result['magnetization']['mx'].size == total_points


if __name__ == "__main__":
    pytest.main([__file__])
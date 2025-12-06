"""
OOMMF file parser for OVF and ODT formats.

This module provides comprehensive parsing capabilities for OOMMF simulation
output files, including vector field files (OVF) and data table files (ODT).

Author: Meshal Alawein
Email: meshal@berkeley.edu
License: MIT
"""

import struct
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import pandas as pd

from .base_parser import BaseParser, ParseError, UnsupportedFormatError, CorruptedFileError

class OOMMFParser(BaseParser):
    """
    Parser for OOMMF simulation output files.
    
    Supports:
    - OVF 1.0 and 2.0 vector field files (text and binary)
    - ODT data table files
    - Automatic format detection
    - Comprehensive error handling and validation
    
    Example:
        >>> parser = OOMMFParser(verbose=True)
        >>> data = parser.parse_ovf('magnetization.ovf')
        >>> time_series = parser.parse_odt('simulation.odt')
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize OOMMF parser."""
        super().__init__(verbose)
        self.supported_extensions = ['.ovf', '.odt', '.omf', '.ohf']
    
    def parse_file(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse OOMMF file with automatic format detection.
        
        Args:
            filepath: Path to OOMMF file
            
        Returns:
            Parsed data dictionary
        """
        filepath = Path(filepath)
        
        if not self.validate_file(filepath):
            raise ParseError(f"Invalid file: {filepath}")
        
        # Detect file type based on extension and content
        if filepath.suffix.lower() in ['.ovf', '.omf', '.ohf']:
            return self.parse_ovf(filepath)
        elif filepath.suffix.lower() == '.odt':
            return self.parse_odt(filepath)
        else:
            # Try to detect based on content
            try:
                return self.parse_ovf(filepath)
            except:
                try:
                    return self.parse_odt(filepath)
                except:
                    raise UnsupportedFormatError(f"Cannot determine file format: {filepath}")
    
    def parse_ovf(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse OOMMF Vector Field (OVF) file.
        
        Args:
            filepath: Path to OVF file
            
        Returns:
            Dictionary containing:
            - magnetization: Magnetization components and derived quantities
            - coordinates: Spatial coordinate arrays
            - metadata: Grid information and simulation parameters
            - header: Original file header
        """
        filepath = Path(filepath)
        self._log_info(f"Parsing OVF file: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                # Parse header
                header = self._parse_ovf_header(f)
                
                # Validate header
                self._validate_ovf_header(header)
                
                # Read data based on format
                if header.get('data_format', 'text').lower() == 'binary':
                    magnetization_data = self._read_ovf_binary_data(f, header)
                else:
                    magnetization_data = self._read_ovf_text_data(f, header)
                
                # Create coordinate grids
                coordinates = self._create_coordinate_grids(header)
                
                # Standardize magnetization data
                nx, ny, nz = header['xnodes'], header['ynodes'], header['znodes']
                
                # Reshape magnetization data
                mx = magnetization_data[:, 0].reshape((nz, ny, nx))
                my = magnetization_data[:, 1].reshape((nz, ny, nx))
                mz = magnetization_data[:, 2].reshape((nz, ny, nx))
                
                magnetization = self.standardize_magnetization(mx, my, mz)
                
                # Calculate volume averages
                avg_magnetization = {
                    'mx_avg': self.calculate_volume_average(mx),
                    'my_avg': self.calculate_volume_average(my),
                    'mz_avg': self.calculate_volume_average(mz),
                    'mag_avg': self.calculate_volume_average(magnetization['magnitude'])
                }
                
                # Create metadata
                metadata = {
                    'grid_size': [nx, ny, nz],
                    'cell_size': [header['xstepsize'], header['ystepsize'], header['zstepsize']],
                    'total_cells': nx * ny * nz,
                    'file_format': 'OVF',
                    'data_format': header.get('data_format', 'text'),
                    'ovf_version': header.get('ovf_version', '1.0'),
                    'filename': str(filepath),
                    **avg_magnetization
                }
                
                result = {
                    'magnetization': magnetization,
                    'coordinates': coordinates,
                    'metadata': metadata,
                    'header': header
                }
                
                self._log_info(f"Successfully parsed OVF file: {nx}×{ny}×{nz} grid")
                return result
                
        except Exception as e:
            raise ParseError(f"Failed to parse OVF file {filepath}: {e}")
    
    def parse_odt(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse OOMMF Data Table (ODT) file.
        
        Args:
            filepath: Path to ODT file
            
        Returns:
            Dictionary containing time series data and metadata
        """
        filepath = Path(filepath)
        self._log_info(f"Parsing ODT file: {filepath}")
        
        try:
            # Read header information
            header_info = self._parse_odt_header(filepath)
            
            # Read numeric data
            try:
                # Try pandas first (faster and more robust)
                data = pd.read_csv(filepath, comment='#', sep=r'\s+', header=None)
                data_array = data.values
            except:
                # Fallback to numpy
                data_array = np.loadtxt(filepath, comments='#')
            
            # Create time series dictionary
            time_series = {}
            columns = header_info.get('columns', [])
            
            if len(columns) == data_array.shape[1]:
                for i, col_name in enumerate(columns):
                    # Clean column name for dictionary key
                    clean_name = re.sub(r'[^\w\s-]', '', col_name)
                    clean_name = re.sub(r'[-\s]+', '_', clean_name)
                    clean_name = clean_name.strip('_')
                    
                    if not clean_name:
                        clean_name = f'column_{i}'
                    
                    time_series[clean_name] = data_array[:, i]
            else:
                # Generic column names if mismatch
                for i in range(data_array.shape[1]):
                    time_series[f'column_{i}'] = data_array[:, i]
            
            # Create metadata
            metadata = {
                'file_format': 'ODT',
                'num_rows': data_array.shape[0],
                'num_columns': data_array.shape[1],
                'column_names': list(time_series.keys()),
                'filename': str(filepath)
            }
            
            # Add derived quantities if time and magnetization data present
            if 'Time' in time_series or 'time' in [k.lower() for k in time_series.keys()]:
                time_key = next((k for k in time_series.keys() if k.lower() == 'time'), None)
                if time_key:
                    metadata['time_range'] = [time_series[time_key].min(), time_series[time_key].max()]
                    metadata['time_step'] = np.mean(np.diff(time_series[time_key]))
            
            result = {
                'time_series': time_series,
                'metadata': metadata,
                'header': header_info
            }
            
            self._log_info(f"Successfully parsed ODT file: {data_array.shape[0]} rows, {data_array.shape[1]} columns")
            return result
            
        except Exception as e:
            raise ParseError(f"Failed to parse ODT file {filepath}: {e}")
    
    def _parse_ovf_header(self, file_handle) -> Dict[str, Any]:
        """Parse OVF file header."""
        header = {}
        file_handle.seek(0)
        
        # Read header line by line
        while True:
            pos = file_handle.tell()
            line = file_handle.readline()
            
            if not line:
                break
                
            try:
                line = line.decode('utf-8').strip()
            except UnicodeDecodeError:
                line = line.decode('latin-1').strip()
            
            # Check for end of header
            if line.startswith('# Begin: Data'):
                break
            
            # Parse header fields
            if line.startswith('#'):
                self._parse_ovf_header_line(line, header)
            elif line.strip() == '':
                continue
            else:
                # Non-comment line before data section
                file_handle.seek(pos)
                break
        
        return header
    
    def _parse_ovf_header_line(self, line: str, header: Dict[str, Any]):
        """Parse individual OVF header line."""
        line = line[1:].strip()  # Remove '#' and whitespace
        
        # Version information
        if line.startswith('OOMMF OVF'):
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                header['ovf_version'] = match.group(1)
        
        # Grid dimensions
        elif line.startswith('xnodes:'):
            header['xnodes'] = int(line.split(':')[1].strip())
        elif line.startswith('ynodes:'):
            header['ynodes'] = int(line.split(':')[1].strip())
        elif line.startswith('znodes:'):
            header['znodes'] = int(line.split(':')[1].strip())
        
        # Step sizes
        elif line.startswith('xstepsize:'):
            header['xstepsize'] = float(line.split(':')[1].strip())
        elif line.startswith('ystepsize:'):
            header['ystepsize'] = float(line.split(':')[1].strip())
        elif line.startswith('zstepsize:'):
            header['zstepsize'] = float(line.split(':')[1].strip())
        
        # Base point
        elif line.startswith('xbase:'):
            header['xbase'] = float(line.split(':')[1].strip())
        elif line.startswith('ybase:'):
            header['ybase'] = float(line.split(':')[1].strip())
        elif line.startswith('zbase:'):
            header['zbase'] = float(line.split(':')[1].strip())
        
        # Data format
        elif 'binary' in line.lower():
            header['data_format'] = 'binary'
            if '4' in line:
                header['binary_precision'] = 4
            elif '8' in line:
                header['binary_precision'] = 8
        elif 'text' in line.lower():
            header['data_format'] = 'text'
        
        # Value multiplier
        elif line.startswith('valuemultiplier:'):
            header['valuemultiplier'] = float(line.split(':')[1].strip())
        
        # Value range
        elif line.startswith('valuerange:'):
            range_str = line.split(':')[1].strip()
            try:
                header['valuerange'] = [float(x) for x in range_str.split()]
            except:
                pass
        
        # Units
        elif line.startswith('valueunit:'):
            header['valueunit'] = line.split(':')[1].strip()
        elif line.startswith('meshunit:'):
            header['meshunit'] = line.split(':')[1].strip()
        
        # Title and description
        elif line.startswith('Title:'):
            header['title'] = line.split(':', 1)[1].strip()
        elif line.startswith('Desc:'):
            header['description'] = line.split(':', 1)[1].strip()
    
    def _validate_ovf_header(self, header: Dict[str, Any]):
        """Validate OVF header for required fields."""
        required_fields = ['xnodes', 'ynodes', 'znodes', 'xstepsize', 'ystepsize', 'zstepsize']
        
        for field in required_fields:
            if field not in header:
                raise CorruptedFileError(f"Missing required header field: {field}")
        
        # Validate dimensions
        if any(header[f] <= 0 for f in ['xnodes', 'ynodes', 'znodes']):
            raise CorruptedFileError("Invalid grid dimensions in header")
        
        if any(header[f] <= 0 for f in ['xstepsize', 'ystepsize', 'zstepsize']):
            raise CorruptedFileError("Invalid step sizes in header")
    
    def _read_ovf_binary_data(self, file_handle, header: Dict[str, Any]) -> np.ndarray:
        """Read binary OVF data."""
        nx, ny, nz = header['xnodes'], header['ynodes'], header['znodes']
        total_points = nx * ny * nz
        
        # Determine precision
        precision = header.get('binary_precision', 4)
        if precision == 8:
            dtype = np.float64
            fmt = 'd'
        else:
            dtype = np.float32
            fmt = 'f'
        
        # Skip the binary data marker (should be 1234567.0)
        try:
            marker = struct.unpack(f'<{fmt}', file_handle.read(precision))[0]
            expected_marker = 1234567.0
            if abs(marker - expected_marker) > 1e-6:
                # Try big-endian
                file_handle.seek(-precision, 1)
                marker = struct.unpack(f'>{fmt}', file_handle.read(precision))[0]
                if abs(marker - expected_marker) > 1e-6:
                    self._log_warning(f"Unexpected binary marker: {marker}")
                endian = '>'
            else:
                endian = '<'
        except:
            raise CorruptedFileError("Cannot read binary data marker")
        
        # Read magnetization data
        try:
            data_bytes = file_handle.read(total_points * 3 * precision)
            if len(data_bytes) != total_points * 3 * precision:
                raise CorruptedFileError("Insufficient binary data")
            
            # Unpack data
            data = struct.unpack(f'{endian}{total_points * 3}{fmt}', data_bytes)
            data_array = np.array(data, dtype=dtype).reshape((total_points, 3))
            
        except Exception as e:
            raise CorruptedFileError(f"Error reading binary magnetization data: {e}")
        
        return data_array
    
    def _read_ovf_text_data(self, file_handle, header: Dict[str, Any]) -> np.ndarray:
        """Read text OVF data."""
        nx, ny, nz = header['xnodes'], header['ynodes'], header['znodes']
        total_points = nx * ny * nz
        
        data_list = []
        
        try:
            for line in file_handle:
                try:
                    line = line.decode('utf-8').strip()
                except UnicodeDecodeError:
                    line = line.decode('latin-1').strip()
                
                if not line or line.startswith('#'):
                    continue
                
                values = line.split()
                if len(values) >= 3:
                    try:
                        mx, my, mz = float(values[0]), float(values[1]), float(values[2])
                        data_list.append([mx, my, mz])
                    except ValueError:
                        continue
                
                if len(data_list) >= total_points:
                    break
            
            if len(data_list) != total_points:
                raise CorruptedFileError(f"Expected {total_points} data points, got {len(data_list)}")
            
            return np.array(data_list)
            
        except Exception as e:
            raise CorruptedFileError(f"Error reading text magnetization data: {e}")
    
    def _create_coordinate_grids(self, header: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Create coordinate grids from header information."""
        nx, ny, nz = header['xnodes'], header['ynodes'], header['znodes']
        dx, dy, dz = header['xstepsize'], header['ystepsize'], header['zstepsize']
        
        # Base points (default to 0 if not specified)
        x0 = header.get('xbase', 0.0)
        y0 = header.get('ybase', 0.0)
        z0 = header.get('zbase', 0.0)
        
        # Create coordinate vectors
        x_vec = x0 + np.arange(nx) * dx
        y_vec = y0 + np.arange(ny) * dy
        z_vec = z0 + np.arange(nz) * dz
        
        # Create meshgrids
        x_grid, y_grid, z_grid = np.meshgrid(x_vec, y_vec, z_vec, indexing='ij')
        
        # Transpose to match magnetization data ordering (z, y, x)
        x_grid = np.transpose(x_grid, (2, 1, 0))
        y_grid = np.transpose(y_grid, (2, 1, 0))
        z_grid = np.transpose(z_grid, (2, 1, 0))
        
        return self.standardize_coordinates(x_grid, y_grid, z_grid)
    
    def _parse_odt_header(self, filepath: Path) -> Dict[str, Any]:
        """Parse ODT file header to extract column information."""
        header_info = {'columns': [], 'units': [], 'comments': []}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    
                    if not line.startswith('#'):
                        break  # End of header
                    
                    # Remove '#' and strip whitespace
                    content = line[1:].strip()
                    
                    if content.startswith('Columns:'):
                        # Extract column names
                        col_line = content[8:].strip()  # Remove 'Columns:'
                        columns = col_line.split()
                        header_info['columns'] = columns
                    
                    elif content.startswith('Units:'):
                        # Extract units
                        unit_line = content[6:].strip()  # Remove 'Units:'
                        units = unit_line.split()
                        header_info['units'] = units
                    
                    else:
                        # General comment
                        header_info['comments'].append(content)
                    
                    # Limit header reading to avoid large files
                    if line_num > 100:
                        break
        
        except Exception as e:
            self._log_warning(f"Could not parse ODT header: {e}")
            # Provide default column names
            header_info['columns'] = ['Time', 'Energy', 'mx', 'my', 'mz']
        
        return header_info
    
    def get_ovf_info(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Get OVF file information without loading full data.
        
        Args:
            filepath: Path to OVF file
            
        Returns:
            Dictionary with file information
        """
        filepath = Path(filepath)
        
        try:
            with open(filepath, 'rb') as f:
                header = self._parse_ovf_header(f)
                
            info = self.get_file_info(filepath)
            info.update({
                'grid_size': [header.get('xnodes', 0), header.get('ynodes', 0), header.get('znodes', 0)],
                'cell_size': [header.get('xstepsize', 0), header.get('ystepsize', 0), header.get('zstepsize', 0)],
                'data_format': header.get('data_format', 'text'),
                'ovf_version': header.get('ovf_version', 'unknown'),
                'title': header.get('title', ''),
                'total_cells': header.get('xnodes', 0) * header.get('ynodes', 0) * header.get('znodes', 0)
            })
            
            return info
            
        except Exception as e:
            info = self.get_file_info(filepath)
            info['error'] = str(e)
            return info
"""
MuMax3 file parser for output formats.

This module provides parsing capabilities for MuMax3 simulation output files,
including table files and OVF files with MuMax3-specific extensions.

Author: Meshal Alawein
Email: meshal@berkeley.edu
License: MIT
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import numpy as np
import pandas as pd

from .base_parser import BaseParser, ParseError, UnsupportedFormatError, CorruptedFileError
from .oommf_parser import OOMMFParser

class MuMax3Parser(BaseParser):
    """
    Parser for MuMax3 simulation output files.
    
    Supports:
    - MuMax3 table.txt files (time series data)
    - MuMax3 OVF files (inherits from OOMMF parser)
    - MuMax3 metadata files
    - JSON parameter files
    
    Example:
        >>> parser = MuMax3Parser(verbose=True)
        >>> data = parser.parse_table('table.txt')
        >>> magnetization = parser.parse_ovf('m000001.ovf')
    """
    
    def __init__(self, verbose: bool = False):
        """Initialize MuMax3 parser."""
        super().__init__(verbose)
        self.supported_extensions = ['.txt', '.ovf', '.json', '.out']
        self.ovf_parser = OOMMFParser(verbose=verbose)
    
    def parse_file(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse MuMax3 file with automatic format detection.
        
        Args:
            filepath: Path to MuMax3 file
            
        Returns:
            Parsed data dictionary
        """
        filepath = Path(filepath)
        
        if not self.validate_file(filepath):
            raise ParseError(f"Invalid file: {filepath}")
        
        # Detect file type
        if filepath.name.lower() == 'table.txt':
            return self.parse_table(filepath)
        elif filepath.suffix.lower() == '.ovf':
            return self.parse_ovf(filepath)
        elif filepath.suffix.lower() == '.json':
            return self.parse_json(filepath)
        elif filepath.suffix.lower() == '.out':
            return self.parse_output_log(filepath)
        else:
            # Try table format first, then others
            try:
                return self.parse_table(filepath)
            except:
                try:
                    return self.parse_ovf(filepath)
                except:
                    raise UnsupportedFormatError(f"Cannot determine MuMax3 file format: {filepath}")
    
    def parse_table(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse MuMax3 table.txt file.
        
        Args:
            filepath: Path to table.txt file
            
        Returns:
            Dictionary containing time series data and metadata
        """
        filepath = Path(filepath)
        self._log_info(f"Parsing MuMax3 table file: {filepath}")
        
        try:
            # Read file and detect structure
            with open(filepath, 'r', encoding='utf-8') as f:
                # Read first few lines to understand format
                lines = []
                for i, line in enumerate(f):
                    lines.append(line.strip())
                    if i > 10:  # Read enough to detect header
                        break
            
            # Find header line (starts with #)
            header_line = None
            data_start_line = 0
            
            for i, line in enumerate(lines):
                if line.startswith('#'):
                    header_line = line
                    data_start_line = i + 1
                    break
                elif line and not line.startswith('#'):
                    # Data starts immediately
                    data_start_line = i
                    break
            
            # Parse column names from header
            columns = []
            if header_line:
                # Remove '#' and split
                header_content = header_line[1:].strip()
                columns = header_content.split('\t') if '\t' in header_content else header_content.split()
                # Clean column names
                columns = [col.strip() for col in columns if col.strip()]
            
            # Read numeric data
            try:
                # Try pandas for better performance
                data = pd.read_csv(filepath, sep=r'\s+', comment='#', header=None, skiprows=data_start_line)
                data_array = data.values
            except:
                # Fallback to numpy
                data_array = np.loadtxt(filepath, comments='#', skiprows=data_start_line)
            
            # Handle single column case
            if data_array.ndim == 1:
                data_array = data_array.reshape(-1, 1)
            
            # Create time series dictionary
            time_series = {}
            
            if columns and len(columns) == data_array.shape[1]:
                for i, col_name in enumerate(columns):
                    clean_name = self._clean_column_name(col_name)
                    time_series[clean_name] = data_array[:, i]
            else:
                # Generate default column names
                default_columns = self._generate_default_columns(data_array.shape[1])
                for i, col_name in enumerate(default_columns):
                    time_series[col_name] = data_array[:, i]
                columns = default_columns
            
            # Create metadata
            metadata = {
                'file_format': 'MuMax3_table',
                'num_rows': data_array.shape[0],
                'num_columns': data_array.shape[1],
                'column_names': list(time_series.keys()),
                'original_columns': columns,
                'filename': str(filepath)
            }
            
            # Add derived statistics
            self._add_time_series_stats(time_series, metadata)
            
            # Detect and parse regional data
            regional_data = self._detect_regional_data(time_series)
            if regional_data:
                metadata['regions'] = regional_data
            
            result = {
                'time_series': time_series,
                'metadata': metadata,
                'header': {'columns': columns}
            }
            
            self._log_info(f"Successfully parsed MuMax3 table: {data_array.shape[0]} rows, {data_array.shape[1]} columns")
            return result
            
        except Exception as e:
            raise ParseError(f"Failed to parse MuMax3 table file {filepath}: {e}")
    
    def parse_ovf(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse MuMax3 OVF file (uses OOMMF parser with MuMax3 extensions).
        
        Args:
            filepath: Path to OVF file
            
        Returns:
            Dictionary containing magnetization data
        """
        # Use OOMMF parser as base
        result = self.ovf_parser.parse_ovf(filepath)
        
        # Add MuMax3-specific metadata
        result['metadata']['file_format'] = 'MuMax3_OVF'
        result['metadata']['simulator'] = 'MuMax3'
        
        # Try to extract additional info from filename
        filepath = Path(filepath)
        filename_info = self._parse_mumax3_filename(filepath.name)
        if filename_info:
            result['metadata'].update(filename_info)
        
        return result
    
    def parse_json(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse MuMax3 JSON parameter/metadata file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Dictionary containing parameters and metadata
        """
        filepath = Path(filepath)
        self._log_info(f"Parsing MuMax3 JSON file: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            result = {
                'parameters': json_data,
                'metadata': {
                    'file_format': 'MuMax3_JSON',
                    'filename': str(filepath)
                }
            }
            
            return result
            
        except Exception as e:
            raise ParseError(f"Failed to parse MuMax3 JSON file {filepath}: {e}")
    
    def parse_output_log(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse MuMax3 output log file.
        
        Args:
            filepath: Path to .out file
            
        Returns:
            Dictionary containing log information and extracted data
        """
        filepath = Path(filepath)
        self._log_info(f"Parsing MuMax3 output log: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            # Extract key information from log
            info = {
                'log_content': log_content,
                'parameters': self._extract_parameters_from_log(log_content),
                'errors': self._extract_errors_from_log(log_content),
                'warnings': self._extract_warnings_from_log(log_content),
                'performance': self._extract_performance_from_log(log_content)
            }
            
            result = {
                'log_info': info,
                'metadata': {
                    'file_format': 'MuMax3_log',
                    'filename': str(filepath)
                }
            }
            
            return result
            
        except Exception as e:
            raise ParseError(f"Failed to parse MuMax3 log file {filepath}: {e}")
    
    def parse_simulation_directory(self, directory: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse entire MuMax3 simulation directory.
        
        Args:
            directory: Path to simulation output directory
            
        Returns:
            Dictionary containing all parsed files
        """
        directory = Path(directory)
        self._log_info(f"Parsing MuMax3 simulation directory: {directory}")
        
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")
        
        result = {
            'table_data': None,
            'magnetization_files': [],
            'field_files': [],
            'energy_files': [],
            'parameters': None,
            'log_info': None,
            'metadata': {
                'directory': str(directory),
                'simulation_name': directory.name
            }
        }
        
        # Parse table.txt
        table_file = directory / 'table.txt'
        if table_file.exists():
            try:
                result['table_data'] = self.parse_table(table_file)
            except Exception as e:
                self._log_warning(f"Could not parse table.txt: {e}")
        
        # Find and parse OVF files
        ovf_files = sorted(directory.glob('*.ovf'))
        
        for ovf_file in ovf_files:
            try:
                ovf_data = self.parse_ovf(ovf_file)
                
                # Categorize by filename pattern
                filename = ovf_file.name.lower()
                if filename.startswith('m'):
                    result['magnetization_files'].append(ovf_data)
                elif filename.startswith('b_') or filename.startswith('h_'):
                    result['field_files'].append(ovf_data)
                elif filename.startswith('e'):
                    result['energy_files'].append(ovf_data)
                else:
                    result['magnetization_files'].append(ovf_data)  # Default
                    
            except Exception as e:
                self._log_warning(f"Could not parse OVF file {ovf_file}: {e}")
        
        # Parse parameter files
        for param_file in directory.glob('*.json'):
            try:
                param_data = self.parse_json(param_file)
                result['parameters'] = param_data
                break  # Use first JSON file found
            except Exception as e:
                self._log_warning(f"Could not parse parameter file {param_file}: {e}")
        
        # Parse log files
        for log_file in directory.glob('*.out'):
            try:
                log_data = self.parse_output_log(log_file)
                result['log_info'] = log_data
                break  # Use first log file found
            except Exception as e:
                self._log_warning(f"Could not parse log file {log_file}: {e}")
        
        # Update metadata
        result['metadata'].update({
            'num_magnetization_files': len(result['magnetization_files']),
            'num_field_files': len(result['field_files']),
            'num_energy_files': len(result['energy_files']),
            'has_table_data': result['table_data'] is not None,
            'has_parameters': result['parameters'] is not None,
            'has_log': result['log_info'] is not None
        })
        
        self._log_info(f"Parsed simulation directory: {len(ovf_files)} OVF files, table data: {result['table_data'] is not None}")
        return result
    
    def _clean_column_name(self, name: str) -> str:
        """Clean column name for dictionary key."""
        # Remove special characters and replace with underscores
        clean = re.sub(r'[^\w\s-]', '', str(name))
        clean = re.sub(r'[-\s]+', '_', clean)
        clean = clean.strip('_')
        
        # Handle empty names
        if not clean:
            clean = 'unnamed'
        
        return clean
    
    def _generate_default_columns(self, num_cols: int) -> List[str]:
        """Generate default column names."""
        common_names = ['t', 'E_total', 'mx', 'my', 'mz', 'E_exch', 'E_demag', 'E_Zeeman', 'E_anis']
        
        if num_cols <= len(common_names):
            return common_names[:num_cols]
        else:
            # Extend with generic names
            names = common_names.copy()
            for i in range(len(common_names), num_cols):
                names.append(f'col_{i}')
            return names
    
    def _add_time_series_stats(self, time_series: Dict[str, np.ndarray], metadata: Dict[str, Any]):
        """Add statistical information about time series."""
        # Find time column
        time_key = None
        for key in ['t', 'time', 'Time']:
            if key in time_series:
                time_key = key
                break
        
        if time_key and len(time_series[time_key]) > 1:
            time_data = time_series[time_key]
            metadata.update({
                'time_range': [float(time_data.min()), float(time_data.max())],
                'time_step': float(np.mean(np.diff(time_data))),
                'simulation_time': float(time_data.max() - time_data.min()),
                'num_time_steps': len(time_data)
            })
        
        # Add statistics for energy columns
        energy_cols = [k for k in time_series.keys() if k.startswith('E_') or 'energy' in k.lower()]
        if energy_cols:
            energy_stats = {}
            for col in energy_cols:
                data = time_series[col]
                energy_stats[col] = {
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'mean': float(data.mean()),
                    'final': float(data[-1])
                }
            metadata['energy_statistics'] = energy_stats
    
    def _detect_regional_data(self, time_series: Dict[str, np.ndarray]) -> Optional[Dict[str, Any]]:
        """Detect regional magnetization data in time series."""
        regional_data = {}
        
        # Look for regional magnetization columns
        region_pattern = re.compile(r'(m[xyz]?)_?(region|r)_?(\d+)', re.IGNORECASE)
        
        for key in time_series.keys():
            match = region_pattern.search(key)
            if match:
                component = match.group(1) if match.group(1) else 'm'
                region_id = int(match.group(3))
                
                if region_id not in regional_data:
                    regional_data[region_id] = {}
                
                regional_data[region_id][component] = key
        
        return regional_data if regional_data else None
    
    def _parse_mumax3_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Extract information from MuMax3 filename patterns."""
        info = {}
        
        # Extract frame number from magnetization files (m000001.ovf)
        m_match = re.match(r'm(\d+)\.ovf', filename.lower())
        if m_match:
            info['frame_number'] = int(m_match.group(1))
            info['data_type'] = 'magnetization'
            return info
        
        # Extract field information (B_ext000001.ovf)
        field_match = re.match(r'([bhm]_\w+)(\d+)\.ovf', filename.lower())
        if field_match:
            info['field_type'] = field_match.group(1)
            info['frame_number'] = int(field_match.group(2))
            info['data_type'] = 'field'
            return info
        
        return None
    
    def _extract_parameters_from_log(self, log_content: str) -> Dict[str, Any]:
        """Extract simulation parameters from log content."""
        params = {}
        
        # Extract common parameters
        patterns = {
            'Msat': r'Msat\s*=\s*([\d.e+-]+)',
            'Aex': r'Aex\s*=\s*([\d.e+-]+)',
            'alpha': r'alpha\s*=\s*([\d.e+-]+)',
            'dt': r'dt\s*=\s*([\d.e+-]+)',
            'grid_size': r'SetGridSize\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)',
            'cell_size': r'SetCellSize\s*\(\s*([\d.e+-]+)\s*,\s*([\d.e+-]+)\s*,\s*([\d.e+-]+)\s*\)'
        }
        
        for param_name, pattern in patterns.items():
            matches = re.findall(pattern, log_content, re.IGNORECASE)
            if matches:
                if param_name in ['grid_size', 'cell_size']:
                    params[param_name] = [float(x) for x in matches[-1]]
                else:
                    params[param_name] = float(matches[-1])
        
        return params
    
    def _extract_errors_from_log(self, log_content: str) -> List[str]:
        """Extract error messages from log content."""
        error_pattern = r'(?:ERROR|Error|error):\s*(.+)$'
        errors = re.findall(error_pattern, log_content, re.MULTILINE)
        return errors
    
    def _extract_warnings_from_log(self, log_content: str) -> List[str]:
        """Extract warning messages from log content."""
        warning_pattern = r'(?:WARNING|Warning|warning):\s*(.+)$'
        warnings = re.findall(warning_pattern, log_content, re.MULTILINE)
        return warnings
    
    def _extract_performance_from_log(self, log_content: str) -> Dict[str, Any]:
        """Extract performance information from log content."""
        perf = {}
        
        # Extract runtime
        runtime_pattern = r'total time:\s*([\d.]+)\s*s'
        runtime_match = re.search(runtime_pattern, log_content, re.IGNORECASE)
        if runtime_match:
            perf['total_time'] = float(runtime_match.group(1))
        
        # Extract GPU information
        gpu_pattern = r'Using CUDA device\s*(\d+):\s*(.+)'
        gpu_match = re.search(gpu_pattern, log_content)
        if gpu_match:
            perf['gpu_device'] = int(gpu_match.group(1))
            perf['gpu_name'] = gpu_match.group(2).strip()
        
        return perf
    
    def get_table_info(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Get table file information without loading full data.
        
        Args:
            filepath: Path to table file
            
        Returns:
            Dictionary with file information
        """
        filepath = Path(filepath)
        
        try:
            # Read just the first few lines
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = [f.readline().strip() for _ in range(10)]
            
            # Find header
            header_line = None
            for line in lines:
                if line.startswith('#'):
                    header_line = line
                    break
            
            # Count approximate number of lines
            with open(filepath, 'r', encoding='utf-8') as f:
                num_lines = sum(1 for _ in f)
            
            info = self.get_file_info(filepath)
            
            columns = []
            if header_line:
                header_content = header_line[1:].strip()
                columns = header_content.split('\t') if '\t' in header_content else header_content.split()
            
            info.update({
                'estimated_rows': num_lines - (1 if header_line else 0),
                'estimated_columns': len(columns) if columns else 0,
                'columns': columns,
                'has_header': header_line is not None
            })
            
            return info
            
        except Exception as e:
            info = self.get_file_info(filepath)
            info['error'] = str(e)
            return info
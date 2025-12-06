#!/usr/bin/env python3
"""
SpinCirc Data Processing Module

Advanced data analysis and post-processing for SpinCirc MATLAB simulations.

Features:
- MATLAB .mat file loading and processing
- Statistical analysis of simulation results
- Parameter extraction and curve fitting
- Data validation and quality assessment
- Export to various formats (CSV, HDF5, JSON)

Author: Meshal Alawein <meshal@berkeley.edu>
Copyright © 2025 Meshal Alawein — All rights reserved.
License: MIT
"""

import numpy as np
import scipy.io
import scipy.optimize
import scipy.stats
import pandas as pd
import h5py
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for data processing operations"""
    remove_outliers: bool = True
    outlier_threshold: float = 3.0  # Standard deviations
    interpolate_missing: bool = True
    smooth_data: bool = False
    smoothing_window: int = 5
    normalize_data: bool = False
    verbose: bool = True

class SpinCircDataProcessor:
    """Main data processing class for SpinCirc simulation results"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the data processor
        
        Args:
            config: Processing configuration options
        """
        self.config = config or ProcessingConfig()
        self.data = {}
        self.metadata = {}
        self.processed_data = {}
        
    def load_matlab_data(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load MATLAB .mat file with comprehensive error handling
        
        Args:
            filepath: Path to .mat file
            
        Returns:
            Dictionary containing loaded data
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
            
        try:
            # Try scipy.io first (for -v7.3 and older formats)
            data = scipy.io.loadmat(str(filepath), squeeze_me=True, struct_as_record=False)
            
            # Remove MATLAB metadata keys
            matlab_keys = ['__header__', '__version__', '__globals__']
            for key in matlab_keys:
                data.pop(key, None)
                
            logger.info(f"Loaded MATLAB data from {filepath}")
            logger.info(f"Variables found: {list(data.keys())}")
            
        except NotImplementedError:
            # Try h5py for -v7.3 files
            try:
                data = {}
                with h5py.File(filepath, 'r') as f:
                    for key in f.keys():
                        if key.startswith('#'):
                            continue  # Skip HDF5 metadata
                        data[key] = np.array(f[key])
                        
                logger.info(f"Loaded HDF5/MATLAB v7.3 data from {filepath}")
                
            except Exception as e:
                raise IOError(f"Failed to load file {filepath}: {e}")
                
        self.data = data
        self._extract_metadata()
        
        return data
    
    def _extract_metadata(self):
        """Extract metadata from loaded data"""
        metadata = {
            'variables': list(self.data.keys()),
            'load_time': pd.Timestamp.now(),
            'data_types': {k: str(type(v)) for k, v in self.data.items()},
            'array_shapes': {k: v.shape if hasattr(v, 'shape') else 'scalar' 
                           for k, v in self.data.items()}
        }
        
        # Try to extract SpinCirc-specific metadata
        if 'solution_info' in self.data:
            info = self.data['solution_info']
            if hasattr(info, '_fieldnames'):
                metadata['spincirc_info'] = {field: getattr(info, field) 
                                           for field in info._fieldnames}
        
        self.metadata = metadata
        
    def extract_transport_data(self, data_key: str = 'I_s') -> Dict[str, np.ndarray]:
        """
        Extract and organize spin transport data
        
        Args:
            data_key: Key for transport data in loaded data
            
        Returns:
            Dictionary with organized transport data
        """
        if data_key not in self.data:
            raise KeyError(f"Transport data key '{data_key}' not found")
            
        transport_data = self.data[data_key]
        
        # Handle different data formats
        if hasattr(transport_data, '_fieldnames'):
            # MATLAB struct format
            result = {}
            for field in transport_data._fieldnames:
                result[field] = np.array(getattr(transport_data, field))
        else:
            # Assume array format [charge, spin_x, spin_y, spin_z]
            if transport_data.ndim == 2 and transport_data.shape[0] == 4:
                result = {
                    'charge': transport_data[0, :],
                    'spin_x': transport_data[1, :],
                    'spin_y': transport_data[2, :],
                    'spin_z': transport_data[3, :]
                }
            else:
                result = {'data': transport_data}
                
        return result
    
    def extract_magnetization_data(self, data_key: str = 'm') -> Dict[str, np.ndarray]:
        """
        Extract and validate magnetization trajectory data
        
        Args:
            data_key: Key for magnetization data
            
        Returns:
            Dictionary with magnetization components and derived quantities
        """
        if data_key not in self.data:
            raise KeyError(f"Magnetization data key '{data_key}' not found")
            
        m_data = np.array(self.data[data_key])
        
        # Ensure correct shape (3 x N or 3 x N x M)
        if m_data.shape[0] != 3:
            raise ValueError(f"Magnetization data must have 3 components, got {m_data.shape[0]}")
            
        result = {
            'mx': m_data[0, ...],
            'my': m_data[1, ...],
            'mz': m_data[2, ...]
        }
        
        # Calculate derived quantities
        result['magnitude'] = np.sqrt(m_data[0, ...]**2 + m_data[1, ...]**2 + m_data[2, ...]**2)
        result['theta'] = np.arccos(np.clip(m_data[2, ...] / result['magnitude'], -1, 1))
        result['phi'] = np.arctan2(m_data[1, ...], m_data[0, ...])
        
        # Validate magnetization magnitude
        mag_error = np.abs(result['magnitude'] - 1.0)
        if np.max(mag_error) > 0.01:
            warnings.warn(f"Magnetization magnitude error: max = {np.max(mag_error):.4f}")
            
        return result
    
    def calculate_tmr(self, resistance_data: np.ndarray, 
                     configuration: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate tunnel magnetoresistance from resistance data
        
        Args:
            resistance_data: Array of resistance values
            configuration: Array indicating P (1) or AP (-1) states
            
        Returns:
            Dictionary with TMR statistics
        """
        if configuration is not None:
            # Separate P and AP states
            p_indices = configuration > 0
            ap_indices = configuration < 0
            
            if not np.any(p_indices) or not np.any(ap_indices):
                raise ValueError("Both parallel and antiparallel states required")
                
            r_p = np.mean(resistance_data[p_indices])
            r_ap = np.mean(resistance_data[ap_indices])
            
        else:
            # Assume first half is P, second half is AP
            mid_point = len(resistance_data) // 2
            r_p = np.mean(resistance_data[:mid_point])
            r_ap = np.mean(resistance_data[mid_point:])
            
        tmr = (r_ap - r_p) / r_p * 100  # Percentage
        
        return {
            'tmr_percent': tmr,
            'resistance_parallel': r_p,
            'resistance_antiparallel': r_ap,
            'resistance_ratio': r_ap / r_p
        }
    
    def fit_llg_precession(self, time: np.ndarray, magnetization: np.ndarray) -> Dict[str, float]:
        """
        Fit LLG precession data to extract physical parameters
        
        Args:
            time: Time array
            magnetization: Magnetization trajectory (3 x N)
            
        Returns:
            Dictionary with fitted parameters
        """
        # Extract in-plane components
        mx = magnetization[0, :]
        my = magnetization[1, :]
        mz = magnetization[2, :]
        
        # Fit precession frequency
        def precession_model(t, freq, phase, amplitude, offset, damping):
            return amplitude * np.exp(-damping * t) * np.cos(2 * np.pi * freq * t + phase) + offset
        
        try:
            # Fit mx component
            popt_x, _ = scipy.optimize.curve_fit(
                precession_model, time, mx,
                p0=[1e9, 0, 0.5, 0, 1e9],  # Initial guess
                bounds=([1e6, -np.pi, 0, -1, 0], [1e12, np.pi, 2, 1, 1e12])
            )
            
            # Fit my component  
            popt_y, _ = scipy.optimize.curve_fit(
                precession_model, time, my,
                p0=[1e9, np.pi/2, 0.5, 0, 1e9],
                bounds=([1e6, -np.pi, 0, -1, 0], [1e12, np.pi, 2, 1, 1e12])
            )
            
            # Extract parameters
            freq_x, phase_x, amp_x, offset_x, damp_x = popt_x
            freq_y, phase_y, amp_y, offset_y, damp_y = popt_y
            
            # Calculate derived quantities
            gamma = freq_x * 2 * np.pi / 0.1  # Assuming 0.1 T field
            alpha = damp_x / (2 * np.pi * freq_x)
            
            return {
                'precession_frequency': (freq_x + freq_y) / 2,
                'damping_rate': (damp_x + damp_y) / 2,
                'gyromagnetic_ratio': gamma,
                'gilbert_damping': alpha,
                'fit_quality_x': np.corrcoef(mx, precession_model(time, *popt_x))[0, 1]**2,
                'fit_quality_y': np.corrcoef(my, precession_model(time, *popt_y))[0, 1]**2
            }
            
        except Exception as e:
            logger.warning(f"Precession fitting failed: {e}")
            return {'error': str(e)}
    
    def analyze_switching_dynamics(self, time: np.ndarray, 
                                 magnetization: np.ndarray) -> Dict[str, float]:
        """
        Analyze magnetization switching dynamics
        
        Args:
            time: Time array
            magnetization: Magnetization trajectory
            
        Returns:
            Dictionary with switching analysis results
        """
        mz = magnetization[2, :] if magnetization.ndim > 1 else magnetization
        
        # Find switching events (sign changes in mz)
        switching_points = []
        for i in range(1, len(mz)):
            if mz[i] * mz[i-1] < 0:  # Sign change
                switching_points.append(i)
                
        if not switching_points:
            return {'switching_events': 0, 'switching_times': []}
        
        switching_times = time[switching_points]
        
        # Calculate switching rates
        switching_rates = []
        for i, switch_idx in enumerate(switching_points):
            # Find 10% to 90% switching time
            start_val = mz[0] if i == 0 else mz[switching_points[i-1]]
            end_val = -start_val
            
            val_10 = start_val + 0.1 * (end_val - start_val)
            val_90 = start_val + 0.9 * (end_val - start_val)
            
            # Find indices
            if end_val > start_val:
                idx_10 = np.where(mz[:switch_idx] >= val_10)[0]
                idx_90 = np.where(mz[:switch_idx] >= val_90)[0]
            else:
                idx_10 = np.where(mz[:switch_idx] <= val_10)[0]
                idx_90 = np.where(mz[:switch_idx] <= val_90)[0]
                
            if len(idx_10) > 0 and len(idx_90) > 0:
                t_10_90 = time[idx_90[-1]] - time[idx_10[0]]
                switching_rates.append(1 / t_10_90 if t_10_90 > 0 else np.inf)
        
        return {
            'switching_events': len(switching_points),
            'switching_times': switching_times.tolist(),
            'average_switching_rate': np.mean(switching_rates) if switching_rates else 0,
            'switching_rate_std': np.std(switching_rates) if switching_rates else 0
        }
    
    def statistical_analysis(self, data: np.ndarray, 
                           variable_name: str = 'data') -> Dict[str, float]:
        """
        Comprehensive statistical analysis of data
        
        Args:
            data: Data array to analyze
            variable_name: Name for reporting
            
        Returns:
            Dictionary with statistical measures
        """
        # Remove NaN values
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) == 0:
            return {'error': 'No valid data points'}
            
        # Basic statistics
        stats = {
            'mean': np.mean(clean_data),
            'median': np.median(clean_data),
            'std': np.std(clean_data),
            'var': np.var(clean_data),
            'min': np.min(clean_data),
            'max': np.max(clean_data),
            'range': np.ptp(clean_data),
            'count': len(clean_data),
            'valid_fraction': len(clean_data) / len(data)
        }
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 75, 90, 95, 99]
        for p in percentiles:
            stats[f'p{p}'] = np.percentile(clean_data, p)
            
        # Distribution tests
        try:
            # Normality test
            _, p_normal = scipy.stats.normaltest(clean_data)
            stats['normality_p_value'] = p_normal
            stats['is_normal'] = p_normal > 0.05
            
            # Skewness and kurtosis
            stats['skewness'] = scipy.stats.skew(clean_data)
            stats['kurtosis'] = scipy.stats.kurtosis(clean_data)
            
        except Exception as e:
            logger.warning(f"Distribution analysis failed for {variable_name}: {e}")
            
        return stats
    
    def export_processed_data(self, output_path: Union[str, Path], 
                            format: str = 'hdf5') -> None:
        """
        Export processed data to various formats
        
        Args:
            output_path: Output file path
            format: Export format ('hdf5', 'csv', 'json', 'matlab')
        """
        output_path = Path(output_path)
        
        if format.lower() == 'hdf5':
            with h5py.File(output_path, 'w') as f:
                # Raw data
                raw_group = f.create_group('raw_data')
                for key, value in self.data.items():
                    if isinstance(value, np.ndarray):
                        raw_group.create_dataset(key, data=value)
                        
                # Processed data
                if self.processed_data:
                    proc_group = f.create_group('processed_data')
                    for key, value in self.processed_data.items():
                        if isinstance(value, (np.ndarray, list)):
                            proc_group.create_dataset(key, data=value)
                            
                # Metadata
                meta_group = f.create_group('metadata')
                meta_group.attrs['export_time'] = str(pd.Timestamp.now())
                
        elif format.lower() == 'csv':
            # Convert data to DataFrame and export
            df_data = {}
            for key, value in self.data.items():
                if isinstance(value, np.ndarray) and value.ndim == 1:
                    df_data[key] = value
            
            if df_data:
                df = pd.DataFrame(df_data)
                df.to_csv(output_path, index=False)
                
        elif format.lower() == 'json':
            # Convert arrays to lists for JSON serialization
            json_data = {}
            for key, value in self.data.items():
                if isinstance(value, np.ndarray):
                    json_data[key] = value.tolist()
                else:
                    json_data[key] = value
                    
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
                
        elif format.lower() == 'matlab':
            scipy.io.savemat(output_path, self.data)
            
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
        logger.info(f"Data exported to {output_path} in {format} format")
    
    def generate_report(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Generate comprehensive analysis report
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Report string
        """
        report_lines = [
            "SpinCirc Data Analysis Report",
            "=" * 35,
            f"Generated: {pd.Timestamp.now()}",
            "",
            "Data Overview:",
            "-" * 15
        ]
        
        # Metadata summary
        if self.metadata:
            report_lines.extend([
                f"Variables loaded: {len(self.metadata['variables'])}",
                f"Variable names: {', '.join(self.metadata['variables'])}",
                ""
            ])
            
        # Statistical summaries for numeric arrays
        for key, value in self.data.items():
            if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number):
                stats = self.statistical_analysis(value.flatten(), key)
                if 'error' not in stats:
                    report_lines.extend([
                        f"Analysis of '{key}':",
                        f"  Shape: {value.shape}",
                        f"  Mean: {stats['mean']:.6g}",
                        f"  Std: {stats['std']:.6g}",
                        f"  Range: [{stats['min']:.6g}, {stats['max']:.6g}]",
                        f"  Valid data: {stats['valid_fraction']:.1%}",
                        ""
                    ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
            
        return report

# Convenience functions
def load_spincirc_data(filepath: Union[str, Path], 
                      config: Optional[ProcessingConfig] = None) -> SpinCircDataProcessor:
    """
    Convenience function to load SpinCirc data
    
    Args:
        filepath: Path to data file
        config: Processing configuration
        
    Returns:
        Initialized data processor
    """
    processor = SpinCircDataProcessor(config)
    processor.load_matlab_data(filepath)
    return processor

def analyze_transport_sweep(filepath: Union[str, Path], 
                           current_key: str = 'I_s',
                           voltage_key: str = 'V') -> Dict[str, Any]:
    """
    Analyze transport sweep data
    
    Args:
        filepath: Path to data file
        current_key: Key for current data
        voltage_key: Key for voltage data
        
    Returns:
        Analysis results
    """
    processor = load_spincirc_data(filepath)
    
    # Extract transport data
    transport_data = processor.extract_transport_data(current_key)
    
    # Calculate conductance
    if voltage_key in processor.data:
        voltage = np.array(processor.data[voltage_key])
        conductance = {}
        for component, current in transport_data.items():
            if len(current) == len(voltage):
                # Avoid division by zero
                nonzero_mask = np.abs(voltage) > 1e-12
                if np.any(nonzero_mask):
                    conductance[f'{component}_conductance'] = current[nonzero_mask] / voltage[nonzero_mask]
    
    return {
        'transport_data': transport_data,
        'conductance': conductance if 'conductance' in locals() else {},
        'statistics': {k: processor.statistical_analysis(v, k) for k, v in transport_data.items()}
    }

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Process SpinCirc simulation data")
    parser.add_argument("input_file", help="Input .mat file")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--format", "-f", choices=['hdf5', 'csv', 'json', 'matlab'], 
                        default='hdf5', help="Output format")
    parser.add_argument("--report", action="store_true", help="Generate analysis report")
    
    args = parser.parse_args()
    
    # Process data
    processor = load_spincirc_data(args.input_file)
    
    # Generate report if requested
    if args.report:
        report_path = Path(args.input_file).with_suffix('.txt')
        processor.generate_report(report_path)
        print(f"Analysis report saved to {report_path}")
    
    # Export data if output specified
    if args.output:
        processor.export_processed_data(args.output, args.format)
        print(f"Data exported to {args.output} in {args.format} format")
    
    print("Data processing completed successfully!")

#!/usr/bin/env python3
"""
File I/O Module
Provides utilities for saving and loading scientific data in various formats,
including HDF5, NumPy, and CSV. Handles wavefunctions, results, and metadata.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import os
import h5py
import numpy as np
import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
from datetime import datetime
import warnings
def create_output_directory(base_path: str,
                          experiment_name: Optional[str] = None,
                          timestamp: bool = True) -> Path:
    """
    Create organized output directory structure for scientific results.
    Parameters
    ----------
    base_path : str
        Base directory path
    experiment_name : str, optional
        Name of the experiment
    timestamp : bool, default True
        Whether to include timestamp in directory name
    Returns
    -------
    Path
        Created directory path
    """
    base_dir = Path(base_path)
    if experiment_name is None:
        experiment_name = "experiment"
    if timestamp:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{experiment_name}_{timestamp_str}"
    else:
        dir_name = experiment_name
    output_dir = base_dir / dir_name
    # Create subdirectories
    subdirs = ['data', 'figures', 'logs', 'configs', 'checkpoints']
    for subdir in subdirs:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    return output_dir
def save_data(data: Union[np.ndarray, Dict[str, Any]],
              filepath: Union[str, Path],
              format: str = 'auto',
              metadata: Optional[Dict[str, Any]] = None,
              compression: bool = True) -> None:
    """
    Save scientific data in specified format with metadata.
    Parameters
    ----------
    data : array or dict
        Data to save
    filepath : str or Path
        Output file path
    format : str, default 'auto'
        Output format ('hdf5', 'npy', 'csv', 'json', 'pickle', 'auto')
    metadata : dict, optional
        Additional metadata to save
    compression : bool, default True
        Whether to use compression for applicable formats
    """
    filepath = Path(filepath)
    if format == 'auto':
        format = _infer_format(filepath)
    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # Add metadata
    if metadata is None:
        metadata = {}
    metadata.update({
        'timestamp': datetime.now().isoformat(),
        'format': format,
        'scicomp_version': '1.0.0'
    })
    if format == 'hdf5':
        _save_hdf5(data, filepath, metadata, compression)
    elif format == 'npy':
        _save_npy(data, filepath, metadata)
    elif format == 'csv':
        _save_csv(data, filepath, metadata)
    elif format == 'json':
        _save_json(data, filepath, metadata)
    elif format == 'pickle':
        _save_pickle(data, filepath, metadata)
    else:
        raise ValueError(f"Unsupported format: {format}")
def load_data(filepath: Union[str, Path],
              format: str = 'auto') -> Tuple[Union[np.ndarray, Dict], Dict]:
    """
    Load scientific data from file with metadata.
    Parameters
    ----------
    filepath : str or Path
        Input file path
    format : str, default 'auto'
        Input format ('hdf5', 'npy', 'csv', 'json', 'pickle', 'auto')
    Returns
    -------
    data : array or dict
        Loaded data
    metadata : dict
        Associated metadata
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if format == 'auto':
        format = _infer_format(filepath)
    if format == 'hdf5':
        return _load_hdf5(filepath)
    elif format == 'npy':
        return _load_npy(filepath)
    elif format == 'csv':
        return _load_csv(filepath)
    elif format == 'json':
        return _load_json(filepath)
    elif format == 'pickle':
        return _load_pickle(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")
def save_wavefunction(psi: np.ndarray,
                     x: np.ndarray,
                     filepath: Union[str, Path],
                     params: Optional[Dict[str, Any]] = None,
                     time: Optional[float] = None) -> None:
    """
    Save quantum wavefunction with coordinate grid and parameters.
    Parameters
    ----------
    psi : ndarray
        Wavefunction values
    x : ndarray
        Coordinate grid
    filepath : str or Path
        Output file path
    params : dict, optional
        Physical parameters
    time : float, optional
        Time value for time-dependent wavefunctions
    """
    data = {
        'wavefunction': psi,
        'coordinates': x,
        'probability_density': np.abs(psi)**2,
        'norm': np.trapz(np.abs(psi)**2, x)
    }
    metadata = {
        'type': 'quantum_wavefunction',
        'dimensions': psi.ndim,
        'grid_points': len(x),
        'is_normalized': np.abs(np.trapz(np.abs(psi)**2, x) - 1.0) < 1e-10
    }
    if params is not None:
        metadata['parameters'] = params
    if time is not None:
        metadata['time'] = time
        data['time'] = time
    save_data(data, filepath, metadata=metadata)
def load_wavefunction(filepath: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load quantum wavefunction with coordinate grid and parameters.
    Parameters
    ----------
    filepath : str or Path
        Input file path
    Returns
    -------
    psi : ndarray
        Wavefunction values
    x : ndarray
        Coordinate grid
    metadata : dict
        Parameters and metadata
    """
    data, metadata = load_data(filepath)
    if metadata.get('type') != 'quantum_wavefunction':
        warnings.warn("File may not contain a quantum wavefunction")
    return data['wavefunction'], data['coordinates'], metadata
def export_results(results: Dict[str, Any],
                  output_dir: Union[str, Path],
                  experiment_name: str = "results",
                  include_plots: bool = True) -> None:
    """
    Export comprehensive results package with multiple formats.
    Parameters
    ----------
    results : dict
        Results dictionary
    output_dir : str or Path
        Output directory
    experiment_name : str
        Experiment name for file naming
    include_plots : bool
        Whether to save plots if present
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    # Save main results in HDF5 format
    main_file = output_path / f"{experiment_name}_results.h5"
    save_data(results, main_file, format='hdf5')
    # Save summary in JSON format
    summary = _create_results_summary(results)
    summary_file = output_path / f"{experiment_name}_summary.json"
    save_data(summary, summary_file, format='json')
    # Save numerical data in CSV format if applicable
    if 'data_arrays' in results:
        for key, array in results['data_arrays'].items():
            if array.ndim <= 2:  # Can be saved as CSV
                csv_file = output_path / f"{experiment_name}_{key}.csv"
                save_data(array, csv_file, format='csv')
    # Save plots if present
    if include_plots and 'figures' in results:
        figures_dir = output_path / 'figures'
        figures_dir.mkdir(exist_ok=True)
        for fig_name, fig_obj in results['figures'].items():
            fig_path = figures_dir / f"{fig_name}.png"
            if hasattr(fig_obj, 'savefig'):
                fig_obj.savefig(fig_path, dpi=300, bbox_inches='tight')
def _infer_format(filepath: Path) -> str:
    """Infer file format from extension."""
    suffix = filepath.suffix.lower()
    format_map = {
        '.h5': 'hdf5',
        '.hdf5': 'hdf5',
        '.npy': 'npy',
        '.npz': 'npy',
        '.csv': 'csv',
        '.json': 'json',
        '.pkl': 'pickle',
        '.pickle': 'pickle'
    }
    if suffix in format_map:
        return format_map[suffix]
    else:
        raise ValueError(f"Cannot infer format from extension: {suffix}")
def _save_hdf5(data: Union[np.ndarray, Dict], filepath: Path,
               metadata: Dict, compression: bool) -> None:
    """Save data in HDF5 format."""
    with h5py.File(filepath, 'w') as f:
        # Save metadata as attributes
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                f.attrs[key] = value
            else:
                f.attrs[key] = json.dumps(value)
        # Save data
        if isinstance(data, np.ndarray):
            if compression:
                f.create_dataset('data', data=data, compression='gzip', compression_opts=9)
            else:
                f.create_dataset('data', data=data)
        elif isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    if compression:
                        f.create_dataset(key, data=value, compression='gzip', compression_opts=9)
                    else:
                        f.create_dataset(key, data=value)
                else:
                    f.attrs[f"data_{key}"] = json.dumps(value)
def _load_hdf5(filepath: Path) -> Tuple[Union[np.ndarray, Dict], Dict]:
    """Load data from HDF5 format."""
    with h5py.File(filepath, 'r') as f:
        # Load metadata
        metadata = {}
        for key, value in f.attrs.items():
            if key.startswith('data_'):
                continue
            try:
                metadata[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                metadata[key] = value
        # Load data
        if 'data' in f:
            # Single array
            data = f['data'][:]
        else:
            # Multiple datasets
            data = {}
            for key in f.keys():
                data[key] = f[key][:]
            # Add scalar data from attributes
            for key, value in f.attrs.items():
                if key.startswith('data_'):
                    data_key = key[5:]  # Remove 'data_' prefix
                    try:
                        data[data_key] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        data[data_key] = value
    return data, metadata
def _save_npy(data: Union[np.ndarray, Dict], filepath: Path, metadata: Dict) -> None:
    """Save data in NumPy format."""
    if isinstance(data, np.ndarray):
        np.save(filepath, data)
    else:
        np.savez_compressed(filepath.with_suffix('.npz'), **data)
    # Save metadata separately
    metadata_file = filepath.with_suffix('.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
def _load_npy(filepath: Path) -> Tuple[Union[np.ndarray, Dict], Dict]:
    """Load data from NumPy format."""
    if filepath.suffix == '.npy':
        data = np.load(filepath)
    else:
        npz_file = np.load(filepath)
        data = {key: npz_file[key] for key in npz_file.files}
    # Load metadata
    metadata_file = filepath.with_suffix('.json')
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    return data, metadata
def _save_csv(data: Union[np.ndarray, Dict], filepath: Path, metadata: Dict) -> None:
    """Save data in CSV format."""
    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            df = pd.DataFrame({'values': data})
        elif data.ndim == 2:
            df = pd.DataFrame(data)
        else:
            raise ValueError("Cannot save >2D arrays as CSV")
    else:
        df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    # Save metadata separately
    metadata_file = filepath.with_suffix('.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
def _load_csv(filepath: Path) -> Tuple[pd.DataFrame, Dict]:
    """Load data from CSV format."""
    data = pd.read_csv(filepath)
    # Load metadata
    metadata_file = filepath.with_suffix('.json')
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    return data, metadata
def _save_json(data: Union[np.ndarray, Dict], filepath: Path, metadata: Dict) -> None:
    """Save data in JSON format."""
    output_data = {'data': data, 'metadata': metadata}
    # Convert numpy arrays to lists
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    output_data = convert_numpy(output_data)
    with open(filepath, 'w') as f:
        json.dump(output_data, f, indent=2)
def _load_json(filepath: Path) -> Tuple[Union[np.ndarray, Dict], Dict]:
    """Load data from JSON format."""
    with open(filepath, 'r') as f:
        content = json.load(f)
    data = content.get('data', content)
    metadata = content.get('metadata', {})
    return data, metadata
def _save_pickle(data: Union[np.ndarray, Dict], filepath: Path, metadata: Dict) -> None:
    """Save data in pickle format."""
    output_data = {'data': data, 'metadata': metadata}
    with open(filepath, 'wb') as f:
        pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)
def _load_pickle(filepath: Path) -> Tuple[Union[np.ndarray, Dict], Dict]:
    """Load data from pickle format."""
    with open(filepath, 'rb') as f:
        content = pickle.load(f)
    if isinstance(content, dict) and 'data' in content:
        data = content['data']
        metadata = content.get('metadata', {})
    else:
        data = content
        metadata = {}
    return data, metadata
def _create_results_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Create a summary of results for JSON export."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'experiment_type': results.get('experiment_type', 'unknown'),
        'parameters': results.get('parameters', {}),
        'data_shapes': {},
        'statistics': {}
    }
    # Add array shapes and basic statistics
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            summary['data_shapes'][key] = list(value.shape)
            if value.size > 0:
                summary['statistics'][key] = {
                    'mean': float(np.mean(value)),
                    'std': float(np.std(value)),
                    'min': float(np.min(value)),
                    'max': float(np.max(value))
                }
    return summary
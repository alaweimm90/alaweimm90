# file_io
**Module:** `Python/utils/file_io.py`
## Overview
File I/O Module
Provides utilities for saving and loading scientific data in various formats,
including HDF5, NumPy, and CSV. Handles wavefunctions, results, and metadata.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
## Functions
### `create_output_directory(base_path, experiment_name, timestamp)`
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
**Source:** [Line 25](Python/utils/file_io.py#L25)
### `save_data(data, filepath, format, metadata, compression)`
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
**Source:** [Line 65](Python/utils/file_io.py#L65)
### `load_data(filepath, format)`
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
**Source:** [Line 117](Python/utils/file_io.py#L117)
### `save_wavefunction(psi, x, filepath, params, time)`
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
**Source:** [Line 157](Python/utils/file_io.py#L157)
### `load_wavefunction(filepath)`
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
**Source:** [Line 201](Python/utils/file_io.py#L201)
### `export_results(results, output_dir, experiment_name, include_plots)`
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
**Source:** [Line 226](Python/utils/file_io.py#L226)
### `_infer_format(filepath)`
Infer file format from extension.
**Source:** [Line 273](Python/utils/file_io.py#L273)
### `_save_hdf5(data, filepath, metadata, compression)`
Save data in HDF5 format.
**Source:** [Line 292](Python/utils/file_io.py#L292)
### `_load_hdf5(filepath)`
Load data from HDF5 format.
**Source:** [Line 319](Python/utils/file_io.py#L319)
### `_save_npy(data, filepath, metadata)`
Save data in NumPy format.
**Source:** [Line 353](Python/utils/file_io.py#L353)
### `_load_npy(filepath)`
Load data from NumPy format.
**Source:** [Line 365](Python/utils/file_io.py#L365)
### `_save_csv(data, filepath, metadata)`
Save data in CSV format.
**Source:** [Line 383](Python/utils/file_io.py#L383)
### `_load_csv(filepath)`
Load data from CSV format.
**Source:** [Line 402](Python/utils/file_io.py#L402)
### `_save_json(data, filepath, metadata)`
Save data in JSON format.
**Source:** [Line 416](Python/utils/file_io.py#L416)
### `_load_json(filepath)`
Load data from JSON format.
**Source:** [Line 440](Python/utils/file_io.py#L440)
### `_save_pickle(data, filepath, metadata)`
Save data in pickle format.
**Source:** [Line 450](Python/utils/file_io.py#L450)
### `_load_pickle(filepath)`
Load data from pickle format.
**Source:** [Line 457](Python/utils/file_io.py#L457)
### `_create_results_summary(results)`
Create a summary of results for JSON export.
**Source:** [Line 471](Python/utils/file_io.py#L471)
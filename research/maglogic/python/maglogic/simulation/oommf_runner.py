"""
OOMMF simulation runner for MagLogic.

This module provides a Python interface for running OOMMF simulations,
managing simulation parameters, and collecting results.

Author: Meshal Alawein
Email: meshal@berkeley.edu
License: MIT
"""

import os
import subprocess
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import yaml
import json
import logging

from ..core.constants import PHYSICAL_CONSTANTS
from ..parsers import OOMMFParser

logger = logging.getLogger(__name__)


class OOMMFRunner:
    """
    OOMMF simulation runner and manager.
    
    Provides a Python interface for setting up, running, and analyzing
    OOMMF micromagnetic simulations with automatic parameter management
    and result collection.
    
    Example:
        >>> runner = OOMMFRunner()
        >>> params = {'Ms': 8.6e5, 'A': 1.3e-11, 'alpha': 0.008}
        >>> results = runner.run_simulation('triangle.mif', params)
    """
    
    def __init__(self, oommf_path: Optional[str] = None, working_dir: Optional[str] = None):
        """
        Initialize OOMMF runner.
        
        Args:
            oommf_path: Path to OOMMF installation (auto-detected if None)
            working_dir: Working directory for simulations (temp dir if None)
        """
        self.oommf_path = oommf_path or self._find_oommf_path()
        self.working_dir = Path(working_dir) if working_dir else Path(tempfile.mkdtemp())
        self.parser = OOMMFParser(verbose=False)
        
        # Ensure working directory exists
        self.working_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify OOMMF installation
        self._verify_oommf_installation()
        
        logger.info(f"OOMMF Runner initialized with OOMMF at: {self.oommf_path}")
        logger.info(f"Working directory: {self.working_dir}")
    
    def _find_oommf_path(self) -> Optional[str]:
        """Auto-detect OOMMF installation path."""
        # Common OOMMF installation paths
        possible_paths = [
            '/usr/local/oommf',
            '/opt/oommf',
            '~/oommf',
            'C:\\oommf',
            'C:\\Program Files\\oommf'
        ]
        
        # Check environment variable
        if 'OOMMF_ROOT' in os.environ:
            possible_paths.insert(0, os.environ['OOMMF_ROOT'])
        
        # Check if oommf command is in PATH
        if shutil.which('oommf'):
            return shutil.which('oommf')
        
        # Check common installation directories
        for path in possible_paths:
            path = Path(path).expanduser()
            if path.exists() and (path / 'oommf.tcl').exists():
                return str(path)
        
        logger.warning("OOMMF installation not found. Please set OOMMF_ROOT environment variable.")
        return None
    
    def _verify_oommf_installation(self):
        """Verify OOMMF installation and get version info."""
        if not self.oommf_path:
            raise RuntimeError("OOMMF not found. Please install OOMMF or set OOMMF_ROOT.")
        
        try:
            # Try to run OOMMF version command
            result = subprocess.run([
                'tclsh', os.path.join(self.oommf_path, 'oommf.tcl'), '+version'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info(f"OOMMF version: {result.stdout.strip()}")
            else:
                logger.warning("Could not verify OOMMF version")
                
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning(f"OOMMF verification failed: {e}")
    
    def run_simulation(self, mif_file: Union[str, Path], 
                      parameters: Optional[Dict[str, Any]] = None,
                      output_dir: Optional[Union[str, Path]] = None,
                      cleanup: bool = True) -> Dict[str, Any]:
        """
        Run OOMMF simulation with given MIF file and parameters.
        
        Args:
            mif_file: Path to MIF file or MIF content string
            parameters: Dictionary of simulation parameters to override
            output_dir: Directory to save outputs (working_dir if None)
            cleanup: Whether to clean up temporary files
            
        Returns:
            Dictionary containing simulation results and metadata
        """
        if not self.oommf_path:
            raise RuntimeError("OOMMF not available")
        
        # Setup simulation directory
        sim_id = f"sim_{int(time.time())}"
        sim_dir = self.working_dir / sim_id
        sim_dir.mkdir(exist_ok=True)
        
        try:
            # Prepare MIF file
            mif_path = self._prepare_mif_file(mif_file, parameters, sim_dir)
            
            # Run simulation
            start_time = time.time()
            success = self._execute_oommf_simulation(mif_path, sim_dir)
            end_time = time.time()
            
            if not success:
                raise RuntimeError("OOMMF simulation failed")
            
            # Collect results
            results = self._collect_simulation_results(sim_dir, mif_path)
            
            # Add timing information
            results['metadata']['simulation_time'] = end_time - start_time
            results['metadata']['start_time'] = start_time
            results['metadata']['end_time'] = end_time
            
            # Copy results to output directory if specified
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                self._copy_results(sim_dir, output_path)
                results['metadata']['output_directory'] = str(output_path)
            
            logger.info(f"Simulation completed successfully in {end_time - start_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            raise
        finally:
            # Cleanup temporary files if requested
            if cleanup and sim_dir.exists():
                shutil.rmtree(sim_dir, ignore_errors=True)
    
    def _prepare_mif_file(self, mif_input: Union[str, Path], 
                         parameters: Optional[Dict[str, Any]], 
                         sim_dir: Path) -> Path:
        """Prepare MIF file with parameter substitution."""
        # Determine if input is file path or content
        if isinstance(mif_input, (str, Path)) and Path(mif_input).exists():
            # Read from file
            mif_path = Path(mif_input)
            with open(mif_path, 'r') as f:
                mif_content = f.read()
            output_name = mif_path.stem
        else:
            # Treat as content string
            mif_content = str(mif_input)
            output_name = "simulation"
        
        # Apply parameter substitutions
        if parameters:
            mif_content = self._substitute_parameters(mif_content, parameters)
        
        # Write to simulation directory
        output_mif = sim_dir / f"{output_name}.mif"
        with open(output_mif, 'w') as f:
            f.write(mif_content)
        
        return output_mif
    
    def _substitute_parameters(self, mif_content: str, parameters: Dict[str, Any]) -> str:
        """Substitute parameters in MIF file content."""
        # Simple parameter substitution using string replacement
        # Format: {{parameter_name}} in MIF file
        
        for param_name, param_value in parameters.items():
            placeholder = f"{{{{{param_name}}}}}"
            mif_content = mif_content.replace(placeholder, str(param_value))
        
        # Also handle some common parameter formats
        common_substitutions = {
            'Ms': 'Ms',
            'A': 'A',
            'Aex': 'A',  # Alternative name
            'alpha': 'alpha',
            'K1': 'K1',
            'Ku': 'K1',  # Alternative name
            'dt': 'dt',
            'mesh_size': 'mesh_size',
            'cell_size': 'cell_size'
        }
        
        for param_name, param_value in parameters.items():
            if param_name in common_substitutions:
                # Try multiple common formats
                for format_str in [f"${param_name}", f"$({param_name})", f"{{{param_name}}}"]:
                    mif_content = mif_content.replace(format_str, str(param_value))
        
        return mif_content
    
    def _execute_oommf_simulation(self, mif_path: Path, sim_dir: Path) -> bool:
        """Execute OOMMF simulation."""
        try:
            # Change to simulation directory
            original_cwd = os.getcwd()
            os.chdir(sim_dir)
            
            # Prepare OOMMF command
            oommf_cmd = [
                'tclsh',
                os.path.join(self.oommf_path, 'oommf.tcl'),
                'boxsi',
                str(mif_path)
            ]
            
            # Run simulation
            logger.info(f"Running OOMMF simulation: {' '.join(oommf_cmd)}")
            
            with open('oommf_output.log', 'w') as log_file:
                process = subprocess.run(
                    oommf_cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=3600  # 1 hour timeout
                )
            
            success = process.returncode == 0
            
            if not success:
                logger.error(f"OOMMF simulation failed with return code: {process.returncode}")
                # Log error details
                if Path('oommf_output.log').exists():
                    with open('oommf_output.log', 'r') as f:
                        error_log = f.read()
                        logger.error(f"OOMMF error log:\n{error_log}")
            
            return success
            
        except subprocess.TimeoutExpired:
            logger.error("OOMMF simulation timed out")
            return False
        except Exception as e:
            logger.error(f"Error executing OOMMF simulation: {e}")
            return False
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
    
    def _collect_simulation_results(self, sim_dir: Path, mif_path: Path) -> Dict[str, Any]:
        """Collect and parse simulation results."""
        results = {
            'magnetization_files': [],
            'table_data': None,
            'metadata': {
                'simulation_directory': str(sim_dir),
                'mif_file': str(mif_path),
                'files_found': []
            },
            'log_info': None
        }
        
        # Find all OVF files
        ovf_files = sorted(sim_dir.glob('*.ovf'))
        for ovf_file in ovf_files:
            try:
                ovf_data = self.parser.parse_ovf(ovf_file)
                results['magnetization_files'].append(ovf_data)
                results['metadata']['files_found'].append(str(ovf_file.name))
            except Exception as e:
                logger.warning(f"Could not parse OVF file {ovf_file}: {e}")
        
        # Find ODT files
        odt_files = sorted(sim_dir.glob('*.odt'))
        if odt_files:
            try:
                odt_data = self.parser.parse_odt(odt_files[0])  # Use first ODT file
                results['table_data'] = odt_data
                results['metadata']['files_found'].append(str(odt_files[0].name))
            except Exception as e:
                logger.warning(f"Could not parse ODT file {odt_files[0]}: {e}")
        
        # Read log file
        log_file = sim_dir / 'oommf_output.log'
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    log_content = f.read()
                
                results['log_info'] = {
                    'log_content': log_content,
                    'success_indicators': self._extract_success_indicators(log_content),
                    'warnings': self._extract_warnings(log_content),
                    'timing_info': self._extract_timing_info(log_content)
                }
            except Exception as e:
                logger.warning(f"Could not read log file: {e}")
        
        # Summary statistics
        results['metadata']['num_ovf_files'] = len(results['magnetization_files'])
        results['metadata']['has_table_data'] = results['table_data'] is not None
        results['metadata']['has_log'] = results['log_info'] is not None
        
        return results
    
    def _extract_success_indicators(self, log_content: str) -> List[str]:
        """Extract success indicators from OOMMF log."""
        indicators = []
        
        success_patterns = [
            'End of run',
            'Simulation completed',
            'Final energy',
            'Run completed'
        ]
        
        for pattern in success_patterns:
            if pattern.lower() in log_content.lower():
                indicators.append(pattern)
        
        return indicators
    
    def _extract_warnings(self, log_content: str) -> List[str]:
        """Extract warnings from OOMMF log."""
        warnings = []
        
        for line in log_content.split('\n'):
            if 'warning' in line.lower() or 'warn' in line.lower():
                warnings.append(line.strip())
        
        return warnings
    
    def _extract_timing_info(self, log_content: str) -> Dict[str, Any]:
        """Extract timing information from OOMMF log."""
        timing_info = {}
        
        # Look for common timing patterns
        import re
        
        # Total time pattern
        time_pattern = r'Total time:\s*([0-9.]+)\s*s'
        match = re.search(time_pattern, log_content, re.IGNORECASE)
        if match:
            timing_info['total_time'] = float(match.group(1))
        
        # Step count
        step_pattern = r'Step\s*(\d+)'
        steps = re.findall(step_pattern, log_content, re.IGNORECASE)
        if steps:
            timing_info['num_steps'] = int(steps[-1])
        
        return timing_info
    
    def _copy_results(self, sim_dir: Path, output_dir: Path):
        """Copy simulation results to output directory."""
        # Copy all relevant files
        extensions = ['.ovf', '.odt', '.log', '.mif']
        
        for ext in extensions:
            for file_path in sim_dir.glob(f'*{ext}'):
                shutil.copy2(file_path, output_dir / file_path.name)
    
    def create_parameter_sweep(self, base_mif: Union[str, Path],
                             param_ranges: Dict[str, List],
                             output_dir: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Create parameter sweep simulations.
        
        Args:
            base_mif: Base MIF file or content
            param_ranges: Dictionary of parameter names to value lists
            output_dir: Directory to save all sweep results
            
        Returns:
            List of simulation results for each parameter combination
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate all parameter combinations
        import itertools
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        combinations = list(itertools.product(*param_values))
        
        results = []
        
        for i, combination in enumerate(combinations):
            # Create parameter dictionary for this combination
            params = dict(zip(param_names, combination))
            
            # Create unique output directory for this combination
            sweep_dir = output_path / f"sweep_{i:04d}"
            
            # Create parameter description
            param_str = "_".join([f"{k}={v}" for k, v in params.items()])
            logger.info(f"Running parameter sweep {i+1}/{len(combinations)}: {param_str}")
            
            try:
                # Run simulation
                result = self.run_simulation(
                    base_mif, 
                    parameters=params,
                    output_dir=sweep_dir,
                    cleanup=False
                )
                
                # Add sweep metadata
                result['sweep_info'] = {
                    'sweep_index': i,
                    'parameters': params,
                    'parameter_string': param_str
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Parameter sweep {i} failed: {e}")
                # Add failed result
                results.append({
                    'sweep_info': {
                        'sweep_index': i,
                        'parameters': params,
                        'parameter_string': param_str,
                        'failed': True,
                        'error': str(e)
                    }
                })
        
        # Save sweep summary
        sweep_summary = {
            'param_ranges': param_ranges,
            'num_combinations': len(combinations),
            'successful_runs': len([r for r in results if not r.get('sweep_info', {}).get('failed', False)]),
            'results_summary': [r.get('sweep_info', {}) for r in results]
        }
        
        with open(output_path / 'sweep_summary.json', 'w') as f:
            json.dump(sweep_summary, f, indent=2)
        
        logger.info(f"Parameter sweep completed: {sweep_summary['successful_runs']}/{len(combinations)} successful")
        
        return results
    
    def analyze_convergence(self, table_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze simulation convergence from table data.
        
        Args:
            table_data: Parsed ODT table data
            
        Returns:
            Convergence analysis results
        """
        if not table_data or 'time_series' not in table_data:
            return {'converged': False, 'reason': 'No table data available'}
        
        time_series = table_data['time_series']
        
        # Look for energy convergence
        energy_keys = [k for k in time_series.keys() if 'energy' in k.lower() or k.startswith('E_')]
        
        convergence_results = {
            'converged': False,
            'energy_convergence': {},
            'magnetization_convergence': {},
            'final_state_analysis': {}
        }
        
        # Analyze energy convergence
        if energy_keys:
            for energy_key in energy_keys:
                energy_data = time_series[energy_key]
                if len(energy_data) > 10:  # Need sufficient data points
                    
                    # Check if energy has stabilized (last 10% of simulation)
                    final_portion = energy_data[int(0.9 * len(energy_data)):]
                    energy_std = np.std(final_portion)
                    energy_mean = np.mean(final_portion)
                    
                    # Relative stability criterion
                    relative_stability = energy_std / abs(energy_mean) if abs(energy_mean) > 1e-15 else float('inf')
                    
                    convergence_results['energy_convergence'][energy_key] = {
                        'final_value': float(energy_data[-1]),
                        'final_std': float(energy_std),
                        'relative_stability': float(relative_stability),
                        'converged': relative_stability < 1e-6
                    }
        
        # Analyze magnetization convergence
        mag_keys = [k for k in time_series.keys() if k.startswith('m') and k[1:] in ['x', 'y', 'z']]
        
        for mag_key in mag_keys:
            mag_data = time_series[mag_key]
            if len(mag_data) > 10:
                final_portion = mag_data[int(0.9 * len(mag_data)):]
                mag_std = np.std(final_portion)
                
                convergence_results['magnetization_convergence'][mag_key] = {
                    'final_value': float(mag_data[-1]),
                    'final_std': float(mag_std),
                    'converged': mag_std < 1e-6
                }
        
        # Overall convergence assessment
        energy_converged = all(
            ec.get('converged', False) 
            for ec in convergence_results['energy_convergence'].values()
        )
        
        mag_converged = all(
            mc.get('converged', False)
            for mc in convergence_results['magnetization_convergence'].values()
        )
        
        convergence_results['converged'] = energy_converged and mag_converged
        
        if not convergence_results['converged']:
            reasons = []
            if not energy_converged:
                reasons.append('Energy not converged')
            if not mag_converged:
                reasons.append('Magnetization not converged')
            convergence_results['reason'] = '; '.join(reasons)
        
        return convergence_results
    
    def get_simulation_info(self) -> Dict[str, Any]:
        """Get information about OOMMF installation and runner status."""
        return {
            'oommf_path': self.oommf_path,
            'working_directory': str(self.working_dir),
            'oommf_available': self.oommf_path is not None,
            'working_dir_exists': self.working_dir.exists(),
            'working_dir_writable': os.access(self.working_dir, os.W_OK) if self.working_dir.exists() else False
        }
#!/usr/bin/env python3
"""
Multi-objective optimization for spintronic device design.

Uses NSGA-II/III, Bayesian optimization, and genetic algorithms to optimize
device performance metrics like TMR, switching energy, and speed.

Author: Meshal Alawein <meshal@berkeley.edu>
Copyright © 2025 Meshal Alawein — All rights reserved.
License: MIT
"""

import numpy as np
import scipy.optimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import optuna
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import logging
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
import json
import pickle

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Optimization algorithm settings"""
    algorithm: str = 'nsga2'
    n_generations: int = 100
    population_size: int = 50
    n_trials: int = 1000
    objectives: List[str] = None
    constraints: List[str] = None
    random_state: int = 42
    parallel: bool = True
    save_history: bool = True
    verbose: bool = True

class SpintronicDeviceProblem(Problem):
    """Pymoo problem wrapper for device optimization"""
    
    def __init__(self, device_simulator: Callable, design_space: Dict[str, Tuple],
                 objectives: List[str], constraints: List[str] = None):
        self.device_simulator = device_simulator
        self.design_space = design_space
        self.param_names = list(design_space.keys())
        
        xl = np.array([bounds[0] for bounds in design_space.values()])
        xu = np.array([bounds[1] for bounds in design_space.values()])
        
        n_var = len(design_space)
        n_obj = len(objectives)
        n_constr = len(constraints) if constraints else 0
        
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)
        
        self.objectives = objectives
        self.constraints = constraints or []
        self.evaluation_history = []
        
    def _evaluate(self, X, out, *args, **kwargs):
        """Run device simulations and extract objective/constraint values"""
        n_samples = X.shape[0]
        objectives = np.zeros((n_samples, self.n_obj))
        constraints = np.zeros((n_samples, self.n_constr)) if self.n_constr > 0 else None
        
        for i in range(n_samples):
            params = dict(zip(self.param_names, X[i, :]))
            
            try:
                results = self.device_simulator(params)
                
                for j, obj_name in enumerate(self.objectives):
                    if obj_name in results:
                        objectives[i, j] = results[obj_name]
                    else:
                        objectives[i, j] = np.inf
                
                if self.n_constr > 0:
                    for j, constr_name in enumerate(self.constraints):
                        if constr_name in results:
                            constraints[i, j] = results[constr_name]
                        else:
                            constraints[i, j] = 1e6
                
                self.evaluation_history.append({
                    'parameters': params.copy(),
                    'objectives': dict(zip(self.objectives, objectives[i, :])),
                    'constraints': dict(zip(self.constraints, constraints[i, :] if self.n_constr > 0 else []))
                })
                
            except Exception as e:
                logger.warning(f"Simulation failed for {params}: {e}")
                objectives[i, :] = np.inf
                if self.n_constr > 0:
                    constraints[i, :] = 1e6
        
        out["F"] = objectives
        if self.n_constr > 0:
            out["G"] = constraints

class DeviceOptimizer:
    """Multi-objective optimizer for MTJs, spin valves, and ASL devices"""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.optimization_history = []
        self.best_solutions = []
        self.pareto_front = None
        self._setup_logging()
        
    def optimize_mtj_device(self, design_space: Dict[str, Tuple]) -> Dict[str, Any]:
        """
        Optimize MTJ device parameters
        
        Args:
            design_space: Parameter bounds dictionary
            
        Returns:
            Optimization results
        """
        
        def mtj_simulator(params):
            """Simulate MTJ device with given parameters"""
            # Extract parameters
            area = params['area']
            tox = params['tox'] 
            tmr0 = params['tmr0']
            ra_p = params['ra_p']
            
            # Simple MTJ model (would be replaced with full simulation)
            try:
                # Calculate basic properties
                resistance_p = ra_p * 1e-12 / area  # Convert to Ohms
                resistance_ap = resistance_p * (1 + tmr0/100)
                
                # Switching energy (simplified)
                volume = area * tox
                switching_energy = 1e-19 * volume * 1e27  # Approximate in J
                
                # Switching speed (simplified)
                switching_speed = 1e-9 / (tox * 1e9)  # Approximate in s
                
                # Reliability metric (higher is better)
                reliability = tmr0 / (tox * 1e9)**2
                
                return {
                    # Objectives (minimize energy, maximize TMR, minimize switching time)
                    'energy': switching_energy,
                    'tmr': -tmr0,  # Negative for minimization
                    'switching_time': switching_speed,
                    # Additional metrics
                    'resistance_p': resistance_p,
                    'resistance_ap': resistance_ap,
                    'reliability': reliability,
                    # Constraints
                    'area_constraint': area - 1e-15,  # Minimum area
                    'tox_constraint': 0.5e-9 - tox   # Minimum thickness
                }
                
            except Exception as e:
                logger.error(f"MTJ simulation failed: {e}")
                return {
                    'energy': np.inf,
                    'tmr': np.inf,
                    'switching_time': np.inf,
                    'area_constraint': 1e6,
                    'tox_constraint': 1e6
                }
        
        # Define objectives and constraints
        objectives = ['energy', 'tmr', 'switching_time']
        constraints = ['area_constraint', 'tox_constraint']
        
        return self._run_optimization(mtj_simulator, design_space, objectives, constraints)
    
    def optimize_spin_valve(self, design_space: Dict[str, Tuple]) -> Dict[str, Any]:
        """
        Optimize spin valve device parameters
        
        Args:
            design_space: Parameter bounds dictionary
            
        Returns:
            Optimization results
        """
        
        def spin_valve_simulator(params):
            """Simulate spin valve device"""
            length = params['length']
            width = params['width'] 
            thickness = params['thickness']
            lambda_sf = params['lambda_sf']
            
            try:
                # Geometric factors
                area = width * thickness
                volume = length * width * thickness
                aspect_ratio = length / width
                
                # Transport properties (simplified)
                spin_signal = np.exp(-length / lambda_sf)
                resistance = length / (area * 1e6)  # Simplified
                
                # Power consumption
                power = resistance * (0.1**2)  # At 0.1V bias
                
                # Speed (inversely related to RC delay)
                capacitance = 1e-15 * area * 1e12  # Simplified
                speed = 1 / (resistance * capacitance)
                
                return {
                    # Objectives
                    'power': power,
                    'signal': -spin_signal,  # Maximize signal
                    'speed': -speed,  # Maximize speed
                    # Constraints
                    'aspect_ratio_constraint': aspect_ratio - 10,  # Max aspect ratio
                    'volume_constraint': volume - 1e-21  # Minimum volume
                }
                
            except Exception as e:
                logger.error(f"Spin valve simulation failed: {e}")
                return {
                    'power': np.inf,
                    'signal': np.inf, 
                    'speed': np.inf,
                    'aspect_ratio_constraint': 1e6,
                    'volume_constraint': 1e6
                }
        
        objectives = ['power', 'signal', 'speed']
        constraints = ['aspect_ratio_constraint', 'volume_constraint']
        
        return self._run_optimization(spin_valve_simulator, design_space, objectives, constraints)
    
    def optimize_asl_device(self, design_space: Dict[str, Tuple]) -> Dict[str, Any]:
        """
        Optimize All-Spin Logic device
        
        Args:
            design_space: Parameter bounds dictionary
            
        Returns:
            Optimization results
        """
        
        def asl_simulator(params):
            """Simulate ASL device"""
            channel_length = params['channel_length']
            channel_width = params['channel_width']
            magnet_length = params['magnet_length']
            lambda_sf = params['lambda_sf']
            
            try:
                # Channel properties
                channel_resistance = channel_length / (channel_width * 2e-9 * 1e6)
                
                # Spin transport
                spin_decay = np.exp(-channel_length / lambda_sf)
                spin_signal = 0.5 * spin_decay  # Simplified
                
                # Logic metrics
                on_off_ratio = 1 / (1 - spin_signal) if spin_signal < 1 else 1
                
                # Speed estimation
                switching_time = 1e-10 * channel_length * 1e9
                
                # Power estimation
                power = channel_resistance * (0.05**2)  # Low voltage operation
                
                # Noise margin
                noise_margin = spin_signal * 0.5
                
                return {
                    # Objectives
                    'power': power,
                    'delay': switching_time,
                    'on_off_ratio': -on_off_ratio,  # Maximize
                    'noise_margin': -noise_margin,  # Maximize
                    # Constraints
                    'signal_constraint': 0.1 - spin_signal,  # Minimum signal
                    'aspect_constraint': channel_length/channel_width - 20  # Max aspect
                }
                
            except Exception as e:
                logger.error(f"ASL simulation failed: {e}")
                return {
                    'power': np.inf,
                    'delay': np.inf,
                    'on_off_ratio': np.inf,
                    'noise_margin': np.inf,
                    'signal_constraint': 1e6,
                    'aspect_constraint': 1e6
                }
        
        objectives = ['power', 'delay', 'on_off_ratio', 'noise_margin']
        constraints = ['signal_constraint', 'aspect_constraint']
        
        return self._run_optimization(asl_simulator, design_space, objectives, constraints)
    
    def _run_optimization(self, simulator: Callable, design_space: Dict[str, Tuple],
                         objectives: List[str], constraints: List[str] = None) -> Dict[str, Any]:
        """Run multi-objective optimization"""
        
        # Create optimization problem
        problem = SpintronicDeviceProblem(simulator, design_space, objectives, constraints)
        
        # Select algorithm
        if self.config.algorithm == 'nsga2':
            algorithm = NSGA2(pop_size=self.config.population_size)
        elif self.config.algorithm == 'nsga3':
            # Create reference directions for NSGA-III
            from pymoo.util.ref_dirs import get_reference_directions
            ref_dirs = get_reference_directions("das-dennis", len(objectives), n_partitions=12)
            algorithm = NSGA3(pop_size=self.config.population_size, ref_dirs=ref_dirs)
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")
        
        # Run optimization
        logger.info(f"Starting {self.config.algorithm} optimization with {self.config.n_generations} generations")
        
        result = minimize(
            problem,
            algorithm,
            ('n_gen', self.config.n_generations),
            seed=self.config.random_state,
            verbose=self.config.verbose
        )
        
        # Extract results
        pareto_front = result.F
        pareto_set = result.X
        
        # Convert to parameter dictionaries
        pareto_solutions = []
        for i in range(len(pareto_set)):
            params = dict(zip(problem.param_names, pareto_set[i, :]))
            obj_values = dict(zip(objectives, pareto_front[i, :]))
            
            pareto_solutions.append({
                'parameters': params,
                'objectives': obj_values
            })
        
        # Store results
        self.pareto_front = pareto_front
        self.best_solutions = pareto_solutions
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(pareto_front, objectives)
        
        return {
            'pareto_solutions': pareto_solutions,
            'pareto_front': pareto_front,
            'n_evaluations': len(problem.evaluation_history),
            'metrics': metrics,
            'evaluation_history': problem.evaluation_history,
            'convergence_data': result.history if hasattr(result, 'history') else None
        }
    
    def _calculate_performance_metrics(self, pareto_front: np.ndarray, 
                                     objectives: List[str]) -> Dict[str, float]:
        """Calculate optimization performance metrics"""
        metrics = {}
        
        try:
            # Hypervolume (requires reference point)
            # Use nadir point as reference
            ref_point = np.max(pareto_front, axis=0) * 1.1
            
            from pymoo.indicators.hv import HV
            hv = HV(ref_point=ref_point)
            metrics['hypervolume'] = float(hv(pareto_front))
            
            # Spread/diversity metrics
            if len(pareto_front) > 1:
                # Calculate distances between consecutive solutions
                distances = []
                for i in range(len(pareto_front) - 1):
                    dist = np.linalg.norm(pareto_front[i] - pareto_front[i+1])
                    distances.append(dist)
                
                metrics['average_distance'] = float(np.mean(distances))
                metrics['distance_std'] = float(np.std(distances))
            
            # Range metrics for each objective
            for i, obj_name in enumerate(objectives):
                obj_values = pareto_front[:, i]
                metrics[f'{obj_name}_range'] = float(np.max(obj_values) - np.min(obj_values))
                metrics[f'{obj_name}_mean'] = float(np.mean(obj_values))
                metrics[f'{obj_name}_std'] = float(np.std(obj_values))
                
        except Exception as e:
            logger.warning(f"Failed to calculate some performance metrics: {e}")
        
        return metrics
    
    def bayesian_optimization(self, simulator: Callable, design_space: Dict[str, Tuple],
                            objective: str, n_trials: int = None) -> Dict[str, Any]:
        """Single-objective Bayesian optimization"""
        
        n_trials = n_trials or self.config.n_trials
        
        def objective_wrapper(trial):
            """Optuna objective function"""
            params = {}
            for param_name, (min_val, max_val) in design_space.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = trial.suggest_int(param_name, min_val, max_val)
                else:
                    params[param_name] = trial.suggest_float(param_name, min_val, max_val)
            
            try:
                results = simulator(params)
                return results[objective]
            except Exception as e:
                logger.warning(f"Simulation failed: {e}")
                return np.inf
        
        # Create study
        study = optuna.create_study(direction='minimize')
        
        # Optimize
        study.optimize(objective_wrapper, n_trials=n_trials, 
                      show_progress_bar=self.config.verbose)
        
        return {
            'best_parameters': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'study': study
        }
    
    def plot_pareto_front(self, results: Dict[str, Any], objectives: List[str] = None,
                         save_path: str = None):
        """Plot Pareto front visualization"""
        
        pareto_front = results['pareto_front']
        objectives = objectives or list(results['pareto_solutions'][0]['objectives'].keys())
        
        if len(objectives) == 2:
            # 2D plot
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 8))
            plt.scatter(pareto_front[:, 0], pareto_front[:, 1], 
                       c='blue', s=50, alpha=0.7)
            plt.xlabel(objectives[0])
            plt.ylabel(objectives[1])
            plt.title('Pareto Front')
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        elif len(objectives) == 3:
            # 3D plot
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection='3d')
            
            scatter = ax.scatter(pareto_front[:, 0], pareto_front[:, 1], pareto_front[:, 2],
                               c='blue', s=50, alpha=0.7)
            
            ax.set_xlabel(objectives[0])
            ax.set_ylabel(objectives[1])
            ax.set_zlabel(objectives[2])
            ax.set_title('3D Pareto Front')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
        else:
            # Use pymoo's parallel coordinates plot for higher dimensions
            try:
                from pymoo.visualization.pcp import PCP
                
                plot = PCP()
                plot.add(pareto_front, color="red", alpha=0.8)
                plot.show()
                
            except ImportError:
                logger.warning("Cannot plot high-dimensional Pareto front without pymoo visualization")
    
    def analyze_trade_offs(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trade-offs in Pareto solutions"""
        
        pareto_solutions = results['pareto_solutions']
        
        if len(pareto_solutions) < 2:
            return {'error': 'Need at least 2 Pareto solutions for trade-off analysis'}
        
        # Extract objective values
        obj_names = list(pareto_solutions[0]['objectives'].keys())
        obj_matrix = np.array([[sol['objectives'][name] for name in obj_names] 
                              for sol in pareto_solutions])
        
        # Calculate correlation matrix
        correlations = np.corrcoef(obj_matrix.T)
        
        # Find extreme solutions
        extreme_solutions = {}
        for i, obj_name in enumerate(obj_names):
            # Best and worst for each objective
            best_idx = np.argmin(obj_matrix[:, i])
            worst_idx = np.argmax(obj_matrix[:, i])
            
            extreme_solutions[f'{obj_name}_best'] = {
                'solution_index': int(best_idx),
                'parameters': pareto_solutions[best_idx]['parameters'],
                'objectives': pareto_solutions[best_idx]['objectives']
            }
            
            extreme_solutions[f'{obj_name}_worst'] = {
                'solution_index': int(worst_idx),
                'parameters': pareto_solutions[worst_idx]['parameters'],
                'objectives': pareto_solutions[worst_idx]['objectives']
            }
        
        # Calculate compromise solutions (closest to utopia point)
        utopia_point = np.min(obj_matrix, axis=0)
        distances = np.linalg.norm(obj_matrix - utopia_point, axis=1)
        compromise_idx = np.argmin(distances)
        
        compromise_solution = {
            'solution_index': int(compromise_idx),
            'parameters': pareto_solutions[compromise_idx]['parameters'],
            'objectives': pareto_solutions[compromise_idx]['objectives'],
            'distance_to_utopia': float(distances[compromise_idx])
        }
        
        return {
            'objective_correlations': correlations.tolist(),
            'objective_names': obj_names,
            'extreme_solutions': extreme_solutions,
            'compromise_solution': compromise_solution,
            'utopia_point': utopia_point.tolist(),
            'nadir_point': np.max(obj_matrix, axis=0).tolist()
        }
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """Save optimization results to file"""
        
        # Prepare data for serialization
        save_data = {
            'pareto_solutions': results['pareto_solutions'],
            'metrics': results['metrics'],
            'n_evaluations': results['n_evaluations'],
            'config': self.config.__dict__
        }
        
        # Convert numpy arrays to lists for JSON serialization
        if 'pareto_front' in results:
            save_data['pareto_front'] = results['pareto_front'].tolist()
        
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=2)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
        
        logger.info(f"Optimization results saved to {filepath}")
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """Load optimization results from file"""
        
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                results = json.load(f)
        else:
            with open(filepath, 'rb') as f:
                results = pickle.load(f)
        
        # Convert lists back to numpy arrays if needed
        if 'pareto_front' in results:
            results['pareto_front'] = np.array(results['pareto_front'])
        
        logger.info(f"Optimization results loaded from {filepath}")
        return results
    
    def _setup_logging(self):
        """Setup logging configuration"""
        if self.config.verbose:
            logging.basicConfig(level=logging.INFO, 
                              format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        else:
            logging.basicConfig(level=logging.WARNING)
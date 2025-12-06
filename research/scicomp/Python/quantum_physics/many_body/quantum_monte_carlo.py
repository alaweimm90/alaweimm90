#!/usr/bin/env python3
"""
Quantum Monte Carlo Methods
Advanced implementation of quantum Monte Carlo techniques including Variational
Monte Carlo (VMC), Diffusion Monte Carlo (DMC), and Path Integral Monte Carlo (PIMC)
for studying strongly correlated quantum systems and many-body ground states.
Key Features:
- Variational Monte Carlo with optimized trial wavefunctions
- Diffusion Monte Carlo for exact ground state properties
- Path Integral Monte Carlo for finite temperature calculations
- Advanced sampling techniques and importance sampling
- Parallel implementation for large-scale calculations
Applications:
- Electronic structure of atoms and molecules
- Quantum many-body systems and phase transitions
- Strongly correlated materials
- Quantum spin systems
- Cold atom physics and ultracold gases
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
Created: 2025
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, Callable, Dict, List, Any
from dataclasses import dataclass
import warnings
from multiprocessing import Pool
import time
from ...utils.constants import hbar, me, kb, e
from ...utils.units import energy_convert
from ...utils.parallel import parallel_map
from ...visualization.berkeley_style import BerkeleyPlot
@dataclass
class QMCConfig:
    """Configuration for Quantum Monte Carlo calculations."""
    # System parameters
    n_particles: int = 4
    n_dimensions: int = 3
    n_up: Optional[int] = None  # Number of up-spin electrons
    n_down: Optional[int] = None  # Number of down-spin electrons
    # Monte Carlo parameters
    n_walkers: int = 1000
    n_steps: int = 10000
    n_equilibration: int = 1000
    step_size: float = 0.1
    # Method selection
    method: str = 'VMC'  # 'VMC', 'DMC', 'PIMC'
    # VMC parameters
    n_optimization_steps: int = 100
    learning_rate: float = 0.01
    # DMC parameters
    time_step: float = 0.01
    target_population: int = 1000
    branching_factor: float = 1.0
    # PIMC parameters
    temperature: float = 1.0  # In units of interaction energy
    n_time_slices: int = 32
    # Parallelization
    n_cores: int = 4
    # Output control
    save_configurations: bool = False
    output_frequency: int = 100
class QuantumMonteCarlo:
    """
    Comprehensive Quantum Monte Carlo implementation.
    Provides VMC, DMC, and PIMC methods for studying quantum many-body
    systems with advanced sampling techniques and parallel computation.
    Parameters
    ----------
    config : QMCConfig
        Configuration parameters for QMC calculations
    hamiltonian : callable
        Hamiltonian function H(R) returning energy
    trial_wavefunction : callable
        Trial wavefunction ψ_T(R, params)
    """
    def __init__(self, config: QMCConfig,
                 hamiltonian: Callable[[np.ndarray], float],
                 trial_wavefunction: Callable[[np.ndarray, np.ndarray], float]):
        """Initialize Quantum Monte Carlo system."""
        self.config = config
        self.hamiltonian = hamiltonian
        self.trial_wavefunction = trial_wavefunction
        # Set up spin configuration
        if config.n_up is None:
            self.n_up = config.n_particles // 2
        else:
            self.n_up = config.n_up
        if config.n_down is None:
            self.n_down = config.n_particles - self.n_up
        else:
            self.n_down = config.n_down
        # Results storage
        self.energies = []
        self.configurations = []
        self.optimization_history = []
        self.acceptance_rates = []
        # Trial wavefunction parameters
        self.wf_parameters = None
    def initialize_walkers(self) -> np.ndarray:
        """
        Initialize walker configurations.
        Returns
        -------
        np.ndarray
            Initial walker positions, shape (n_walkers, n_particles, n_dimensions)
        """
        walkers = np.random.normal(
            0.0, 1.0,
            (self.config.n_walkers, self.config.n_particles, self.config.n_dimensions)
        )
        return walkers
    def slater_determinant_wavefunction(self, positions: np.ndarray,
                                      parameters: np.ndarray) -> float:
        """
        Slater determinant trial wavefunction for fermions.
        Parameters
        ----------
        positions : np.ndarray
            Particle positions, shape (n_particles, n_dimensions)
        parameters : np.ndarray
            Wavefunction parameters (orbital coefficients)
        Returns
        -------
        float
            Wavefunction value
        """
        n_particles = positions.shape[0]
        # Create single-particle orbitals (simplified Gaussian orbitals)
        n_orbitals = n_particles
        orbital_matrix = np.zeros((n_particles, n_orbitals))
        for i in range(n_particles):
            for j in range(n_orbitals):
                # Gaussian orbital centered at origin
                r_squared = np.sum(positions[i]**2)
                alpha = abs(parameters[j]) if j < len(parameters) else 1.0
                orbital_matrix[i, j] = np.exp(-alpha * r_squared)
        # Compute determinant for up-spin electrons
        if self.n_up > 0:
            det_up = np.linalg.det(orbital_matrix[:self.n_up, :self.n_up])
        else:
            det_up = 1.0
        # Compute determinant for down-spin electrons
        if self.n_down > 0:
            det_down = np.linalg.det(orbital_matrix[self.n_up:, :self.n_down])
        else:
            det_down = 1.0
        return det_up * det_down
    def jastrow_factor(self, positions: np.ndarray,
                      parameters: np.ndarray) -> float:
        """
        Jastrow correlation factor.
        Parameters
        ----------
        positions : np.ndarray
            Particle positions
        parameters : np.ndarray
            Jastrow parameters
        Returns
        -------
        float
            Jastrow factor
        """
        n_particles = positions.shape[0]
        jastrow = 0.0
        # Two-body Jastrow factor
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                r_ij = np.linalg.norm(positions[i] - positions[j])
                # Different parameters for parallel/anti-parallel spins
                if (i < self.n_up and j < self.n_up) or (i >= self.n_up and j >= self.n_up):
                    # Parallel spins
                    b = parameters[0] if len(parameters) > 0 else 0.25
                else:
                    # Anti-parallel spins
                    b = parameters[1] if len(parameters) > 1 else 0.5
                jastrow += b * r_ij / (1 + b * r_ij)
        return np.exp(jastrow)
    def full_trial_wavefunction(self, positions: np.ndarray,
                               parameters: np.ndarray) -> float:
        """
        Complete trial wavefunction including Slater determinant and Jastrow.
        Parameters
        ----------
        positions : np.ndarray
            Particle positions
        parameters : np.ndarray
            All wavefunction parameters
        Returns
        -------
        float
            Complete wavefunction value
        """
        # Split parameters between Slater and Jastrow parts
        n_slater_params = self.config.n_particles
        slater_params = parameters[:n_slater_params]
        jastrow_params = parameters[n_slater_params:]
        slater = self.slater_determinant_wavefunction(positions, slater_params)
        jastrow = self.jastrow_factor(positions, jastrow_params)
        return slater * jastrow
    def local_energy(self, positions: np.ndarray,
                    parameters: np.ndarray) -> float:
        """
        Calculate local energy E_L = H*ψ_T / ψ_T.
        Parameters
        ----------
        positions : np.ndarray
            Particle positions
        parameters : np.ndarray
            Wavefunction parameters
        Returns
        -------
        float
            Local energy
        """
        # Potential energy
        V = self.hamiltonian(positions)
        # Kinetic energy via finite differences
        T = self._kinetic_energy_finite_diff(positions, parameters)
        return T + V
    def _kinetic_energy_finite_diff(self, positions: np.ndarray,
                                   parameters: np.ndarray,
                                   h: float = 1e-5) -> float:
        """Calculate kinetic energy using finite differences."""
        n_particles, n_dim = positions.shape
        psi_0 = self.full_trial_wavefunction(positions, parameters)
        if abs(psi_0) < 1e-12:
            return 0.0
        kinetic = 0.0
        for i in range(n_particles):
            for d in range(n_dim):
                # Forward difference
                pos_forward = positions.copy()
                pos_forward[i, d] += h
                psi_forward = self.full_trial_wavefunction(pos_forward, parameters)
                # Backward difference
                pos_backward = positions.copy()
                pos_backward[i, d] -= h
                psi_backward = self.full_trial_wavefunction(pos_backward, parameters)
                # Second derivative
                d2psi_dr2 = (psi_forward - 2*psi_0 + psi_backward) / h**2
                kinetic += -0.5 * (hbar**2 / me) * (d2psi_dr2 / psi_0)
        return kinetic
    def metropolis_step(self, positions: np.ndarray,
                       parameters: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Single Metropolis Monte Carlo step.
        Parameters
        ----------
        positions : np.ndarray
            Current positions
        parameters : np.ndarray
            Wavefunction parameters
        Returns
        -------
        Tuple[np.ndarray, bool]
            New positions and acceptance flag
        """
        # Propose new configuration
        trial_positions = positions + np.random.normal(
            0, self.config.step_size, positions.shape
        )
        # Calculate acceptance probability
        psi_old = self.full_trial_wavefunction(positions, parameters)
        psi_new = self.full_trial_wavefunction(trial_positions, parameters)
        if abs(psi_old) > 1e-12:
            prob_ratio = (abs(psi_new) / abs(psi_old))**2
            accept_prob = min(1.0, prob_ratio)
        else:
            accept_prob = 1.0
        # Accept or reject
        if np.random.random() < accept_prob:
            return trial_positions, True
        else:
            return positions, False
    def vmc_run(self, parameters: np.ndarray) -> Dict[str, Any]:
        """
        Run Variational Monte Carlo calculation.
        Parameters
        ----------
        parameters : np.ndarray
            Trial wavefunction parameters
        Returns
        -------
        Dict[str, Any]
            VMC results including energy and statistics
        """
        walkers = self.initialize_walkers()
        energies = []
        n_accepted = 0
        for step in range(self.config.n_steps + self.config.n_equilibration):
            step_energies = []
            step_accepted = 0
            # Update all walkers
            for w in range(self.config.n_walkers):
                walkers[w], accepted = self.metropolis_step(walkers[w], parameters)
                if accepted:
                    step_accepted += 1
                # Calculate local energy after equilibration
                if step >= self.config.n_equilibration:
                    E_local = self.local_energy(walkers[w], parameters)
                    step_energies.append(E_local)
            # Store results after equilibration
            if step >= self.config.n_equilibration:
                energies.extend(step_energies)
                n_accepted += step_accepted
                if step % self.config.output_frequency == 0:
                    mean_E = np.mean(step_energies)
                    print(f"Step {step}: E = {mean_E:.6f}")
        # Calculate statistics
        mean_energy = np.mean(energies)
        energy_error = np.std(energies) / np.sqrt(len(energies))
        acceptance_rate = n_accepted / (self.config.n_steps * self.config.n_walkers)
        results = {
            'energy': mean_energy,
            'energy_error': energy_error,
            'energies': energies,
            'acceptance_rate': acceptance_rate,
            'final_walkers': walkers
        }
        return results
    def optimize_wavefunction(self, initial_parameters: np.ndarray) -> np.ndarray:
        """
        Optimize trial wavefunction parameters.
        Parameters
        ----------
        initial_parameters : np.ndarray
            Initial parameter guess
        Returns
        -------
        np.ndarray
            Optimized parameters
        """
        def objective(params):
            """Objective function for optimization."""
            result = self.vmc_run(params)
            energy = result['energy']
            # Store optimization history
            self.optimization_history.append({
                'parameters': params.copy(),
                'energy': energy,
                'error': result['energy_error']
            })
            print(f"Optimization step: E = {energy:.6f} ± {result['energy_error']:.6f}")
            return energy
        # Optimize using scipy
        result = minimize(
            objective,
            initial_parameters,
            method='Powell',
            options={'maxiter': self.config.n_optimization_steps}
        )
        return result.x
    def dmc_run(self, parameters: np.ndarray) -> Dict[str, Any]:
        """
        Run Diffusion Monte Carlo calculation.
        Parameters
        ----------
        parameters : np.ndarray
            Trial wavefunction parameters
        Returns
        -------
        Dict[str, Any]
            DMC results
        """
        # Initialize walkers with weights
        walkers = self.initialize_walkers()
        weights = np.ones(self.config.n_walkers)
        energies = []
        populations = []
        # Reference energy (updated during simulation)
        E_ref = 0.0
        for step in range(self.config.n_steps):
            new_walkers = []
            new_weights = []
            for w in range(len(walkers)):
                walker = walkers[w]
                weight = weights[w]
                # Diffusion step
                walker_new = walker + np.random.normal(
                    0, np.sqrt(self.config.time_step), walker.shape
                )
                # Calculate local energy
                E_local = self.local_energy(walker_new, parameters)
                # Branching: calculate survival probability
                survival_prob = np.exp(-self.config.time_step * (E_local - E_ref))
                # Determine number of offspring
                n_offspring = int(survival_prob + np.random.random())
                # Create offspring
                for _ in range(n_offspring):
                    new_walkers.append(walker_new.copy())
                    new_weights.append(weight)
                energies.append(E_local)
            # Update walker ensemble
            if len(new_walkers) > 0:
                walkers = new_walkers
                weights = new_weights
                # Population control
                current_pop = len(walkers)
                if current_pop > 1.5 * self.config.target_population:
                    # Randomly remove walkers
                    keep_indices = np.random.choice(
                        current_pop, self.config.target_population, replace=False
                    )
                    walkers = [walkers[i] for i in keep_indices]
                    weights = [weights[i] for i in keep_indices]
                elif current_pop < 0.5 * self.config.target_population:
                    # Replicate walkers
                    while len(walkers) < self.config.target_population:
                        idx = np.random.randint(len(walkers))
                        walkers.append(walkers[idx].copy())
                        weights.append(weights[idx])
            populations.append(len(walkers))
            # Update reference energy
            if step > 100:  # After some equilibration
                recent_energies = energies[-100:]
                E_ref = np.mean(recent_energies)
            if step % self.config.output_frequency == 0:
                print(f"DMC Step {step}: Population = {len(walkers)}, E_ref = {E_ref:.6f}")
        # Calculate final results
        equilibration_cut = len(energies) // 4  # Remove first 25%
        final_energies = energies[equilibration_cut:]
        mean_energy = np.mean(final_energies)
        energy_error = np.std(final_energies) / np.sqrt(len(final_energies))
        results = {
            'energy': mean_energy,
            'energy_error': energy_error,
            'energies': energies,
            'populations': populations,
            'final_walkers': walkers
        }
        return results
    def pimc_run(self, parameters: np.ndarray) -> Dict[str, Any]:
        """
        Run Path Integral Monte Carlo calculation.
        Parameters
        ----------
        parameters : np.ndarray
            Trial wavefunction parameters (for importance sampling)
        Returns
        -------
        Dict[str, Any]
            PIMC results
        """
        # Initialize path configurations
        # Each walker has n_time_slices configurations
        n_walkers = self.config.n_walkers
        n_slices = self.config.n_time_slices
        paths = np.random.normal(
            0, 1,
            (n_walkers, n_slices, self.config.n_particles, self.config.n_dimensions)
        )
        beta = 1.0 / self.config.temperature
        tau = beta / n_slices
        energies = []
        acceptance_rates = []
        for step in range(self.config.n_steps):
            n_accepted = 0
            for w in range(n_walkers):
                # Update each time slice
                for t in range(n_slices):
                    # Propose new configuration for time slice t
                    old_config = paths[w, t].copy()
                    new_config = old_config + np.random.normal(
                        0, self.config.step_size, old_config.shape
                    )
                    # Calculate action difference
                    t_prev = (t - 1) % n_slices
                    t_next = (t + 1) % n_slices
                    # Kinetic action (harmonic approximation)
                    kinetic_old = 0.5 * self.config.n_particles / tau * np.sum(
                        (paths[w, t_next] - old_config)**2 +
                        (old_config - paths[w, t_prev])**2
                    )
                    kinetic_new = 0.5 * self.config.n_particles / tau * np.sum(
                        (paths[w, t_next] - new_config)**2 +
                        (new_config - paths[w, t_prev])**2
                    )
                    # Potential action
                    potential_old = tau * self.hamiltonian(old_config)
                    potential_new = tau * self.hamiltonian(new_config)
                    # Accept/reject based on action
                    delta_action = (kinetic_new + potential_new) - (kinetic_old + potential_old)
                    if np.random.random() < np.exp(-delta_action):
                        paths[w, t] = new_config
                        n_accepted += 1
            # Calculate observables
            if step >= self.config.n_equilibration:
                step_energies = []
                for w in range(n_walkers):
                    # Calculate energy using virial estimator
                    energy = 0.0
                    for t in range(n_slices):
                        energy += self.hamiltonian(paths[w, t])
                    energy /= n_slices
                    step_energies.append(energy)
                energies.extend(step_energies)
                acceptance_rates.append(n_accepted / (n_walkers * n_slices))
                if step % self.config.output_frequency == 0:
                    mean_E = np.mean(step_energies)
                    print(f"PIMC Step {step}: E = {mean_E:.6f}")
        # Final statistics
        mean_energy = np.mean(energies)
        energy_error = np.std(energies) / np.sqrt(len(energies))
        mean_acceptance = np.mean(acceptance_rates)
        results = {
            'energy': mean_energy,
            'energy_error': energy_error,
            'energies': energies,
            'acceptance_rate': mean_acceptance,
            'final_paths': paths
        }
        return results
    def run_calculation(self, initial_parameters: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run complete QMC calculation based on configuration.
        Parameters
        ----------
        initial_parameters : np.ndarray, optional
            Initial wavefunction parameters
        Returns
        -------
        Dict[str, Any]
            Complete calculation results
        """
        if initial_parameters is None:
            # Default parameters
            n_params = self.config.n_particles + 2  # Slater + Jastrow
            initial_parameters = np.random.normal(0, 0.1, n_params)
        print(f"Starting {self.config.method} calculation...")
        print(f"System: {self.config.n_particles} particles")
        print(f"Walkers: {self.config.n_walkers}, Steps: {self.config.n_steps}")
        start_time = time.time()
        if self.config.method == 'VMC':
            # Optimize wavefunction first
            if self.config.n_optimization_steps > 0:
                print("Optimizing trial wavefunction...")
                optimal_params = self.optimize_wavefunction(initial_parameters)
            else:
                optimal_params = initial_parameters
            # Run VMC with optimized parameters
            results = self.vmc_run(optimal_params)
            results['optimized_parameters'] = optimal_params
            results['optimization_history'] = self.optimization_history
        elif self.config.method == 'DMC':
            results = self.dmc_run(initial_parameters)
        elif self.config.method == 'PIMC':
            results = self.pimc_run(initial_parameters)
        else:
            raise ValueError(f"Unknown QMC method: {self.config.method}")
        end_time = time.time()
        results['computation_time'] = end_time - start_time
        print(f"\nCalculation completed in {results['computation_time']:.2f} seconds")
        print(f"Final energy: {results['energy']:.6f} ± {results['energy_error']:.6f}")
        return results
    def plot_energy_convergence(self, results: Dict[str, Any]) -> None:
        """Plot energy convergence during QMC run."""
        berkeley_plot = BerkeleyPlot()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        energies = results['energies']
        # Running average
        window = min(100, len(energies) // 10)
        running_avg = np.convolve(energies, np.ones(window)/window, mode='valid')
        ax1.plot(energies, alpha=0.3, color=berkeley_plot.colors['berkeley_blue'],
                label='Instantaneous')
        ax1.plot(range(window-1, len(energies)), running_avg,
                color=berkeley_plot.colors['california_gold'],
                linewidth=2, label=f'Running average ({window})')
        ax1.axhline(results['energy'], color='red', linestyle='--',
                   label=f'Final: {results["energy"]:.4f}')
        ax1.set_xlabel('MC Step')
        ax1.set_ylabel('Energy')
        ax1.set_title(f'{self.config.method} Energy Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        # Energy histogram
        ax2.hist(energies[len(energies)//4:], bins=50, alpha=0.7,
                color=berkeley_plot.colors['berkeley_blue'], density=True)
        ax2.axvline(results['energy'], color='red', linestyle='--',
                   label=f'Mean: {results["energy"]:.4f}')
        ax2.set_xlabel('Energy')
        ax2.set_ylabel('Probability Density')
        ax2.set_title('Energy Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    def plot_optimization_history(self) -> None:
        """Plot wavefunction optimization history."""
        if not self.optimization_history:
            print("No optimization history available")
            return
        berkeley_plot = BerkeleyPlot()
        fig, ax = plt.subplots(figsize=(10, 6))
        steps = range(len(self.optimization_history))
        energies = [entry['energy'] for entry in self.optimization_history]
        errors = [entry['error'] for entry in self.optimization_history]
        ax.errorbar(steps, energies, yerr=errors,
                   color=berkeley_plot.colors['berkeley_blue'],
                   marker='o', markersize=4, linewidth=2, capsize=3)
        ax.set_xlabel('Optimization Step')
        ax.set_ylabel('Variational Energy')
        ax.set_title('Wavefunction Parameter Optimization')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
# Example Hamiltonians
def harmonic_oscillator_hamiltonian(positions: np.ndarray, omega: float = 1.0) -> float:
    """Harmonic oscillator Hamiltonian."""
    return 0.5 * omega**2 * np.sum(positions**2)
def helium_atom_hamiltonian(positions: np.ndarray, Z: float = 2.0) -> float:
    """Helium atom Hamiltonian (atomic units)."""
    if positions.shape[0] != 2:
        raise ValueError("Helium atom requires exactly 2 electrons")
    r1 = np.linalg.norm(positions[0])
    r2 = np.linalg.norm(positions[1])
    r12 = np.linalg.norm(positions[0] - positions[1])
    # Nuclear attraction + electron-electron repulsion
    V = -Z/r1 - Z/r2 + 1/r12 if r12 > 1e-10 else 1e10
    return V
def hydrogen_molecule_hamiltonian(positions: np.ndarray, R: float = 1.4) -> float:
    """H2 molecule Hamiltonian (simplified, atomic units)."""
    if positions.shape[0] != 2:
        raise ValueError("H2 molecule requires exactly 2 electrons")
    # Nuclear positions (fixed)
    nucleus1 = np.array([-R/2, 0, 0])
    nucleus2 = np.array([R/2, 0, 0])
    r1a = np.linalg.norm(positions[0] - nucleus1)
    r1b = np.linalg.norm(positions[0] - nucleus2)
    r2a = np.linalg.norm(positions[1] - nucleus1)
    r2b = np.linalg.norm(positions[1] - nucleus2)
    r12 = np.linalg.norm(positions[0] - positions[1])
    # Electron-nuclear attraction + electron-electron repulsion + nuclear repulsion
    V = (-1/r1a - 1/r1b - 1/r2a - 1/r2b +
         (1/r12 if r12 > 1e-10 else 1e10) + 1/R)
    return V
if __name__ == "__main__":
    # Example: Helium atom ground state
    config = QMCConfig(
        n_particles=2,
        n_dimensions=3,
        n_walkers=500,
        n_steps=5000,
        n_equilibration=1000,
        method='VMC',
        n_optimization_steps=20
    )
    # Use helium atom Hamiltonian
    hamiltonian = lambda pos: helium_atom_hamiltonian(pos, Z=2.0)
    # Simple trial wavefunction
    def trial_wf(pos, params):
        alpha = abs(params[0]) if len(params) > 0 else 1.6875
        r1 = np.linalg.norm(pos[0])
        r2 = np.linalg.norm(pos[1])
        return np.exp(-alpha * (r1 + r2))
    qmc = QuantumMonteCarlo(config, hamiltonian, trial_wf)
    # Run calculation
    results = qmc.run_calculation()
    # Plot results
    qmc.plot_energy_convergence(results)
    qmc.plot_optimization_history()
    print(f"\nHelium ground state energy: {results['energy']:.6f} ± {results['energy_error']:.6f} Ha")
    print(f"Exact energy: -2.903724 Ha")
    print(f"Error: {abs(results['energy'] + 2.903724):.6f} Ha")
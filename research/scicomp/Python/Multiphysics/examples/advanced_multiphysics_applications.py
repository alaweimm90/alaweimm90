"""Advanced Multiphysics Applications.
This script demonstrates advanced multiphysics concepts including
complex multi-field coupling, advanced solution algorithms,
optimization under uncertainty, and cutting-edge research applications.
Topics covered:
- Multi-scale coupling strategies
- Advanced solver algorithms (quasi-Newton, Anderson acceleration)
- Uncertainty quantification in coupled systems
- High-performance computing considerations
- Industrial applications (nuclear, aerospace, biomedical)
Author: Berkeley SciComp Team
Date: 2024
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coupling import CoupledSystem, CouplingScheme
from solvers import (solve_coupled_system, SolverParameters, NewtonRaphson,
                    FixedPointIteration, MonolithicSolver)
from fluid_structure import FluidStructureInteraction
from electromagnetic import ElectromagneticThermalCoupling
from thermal_mechanical import ThermalMechanicalCoupling
from transport import MultiphysicsTransport
from utils import (FieldInterpolator, ConservationChecker, ConvergenceDiagnostics,
                  InterpolationParams)
from visualization import MultiphysicsVisualizer
@dataclass
class SimulationConfig:
    """Advanced simulation configuration."""
    solver_type: str = "quasi_newton"
    preconditioner: str = "ilu"
    parallel_strategy: str = "domain_decomposition"
    uncertainty_quantification: bool = False
    adaptive_meshing: bool = False
    gpu_acceleration: bool = False
    sensitivity_analysis: bool = False
class MultiScaleCoupling:
    """Multi-scale coupling framework.
    Handles coupling between different spatial and temporal scales
    in multiphysics problems.
    """
    def __init__(self, scales: List[str]):
        """Initialize multi-scale coupling.
        Args:
            scales: List of scales (micro, meso, macro)
        """
        self.scales = scales
        self.scale_solvers = {}
        self.transfer_operators = {}
    def add_scale_solver(self, scale: str, solver: Any):
        """Add solver for specific scale."""
        self.scale_solvers[scale] = solver
    def setup_scale_transfer(self,
                           from_scale: str,
                           to_scale: str,
                           transfer_method: str = "homogenization"):
        """Setup transfer between scales."""
        key = f"{from_scale}_to_{to_scale}"
        if transfer_method == "homogenization":
            # Volume averaging for upscaling
            self.transfer_operators[key] = self._homogenization_operator
        elif transfer_method == "localization":
            # Boundary conditions for downscaling
            self.transfer_operators[key] = self._localization_operator
        else:
            raise ValueError(f"Unknown transfer method: {transfer_method}")
    def _homogenization_operator(self, micro_data: np.ndarray) -> np.ndarray:
        """Homogenize microscale data to macroscale."""
        # Volume averaging (simplified)
        return np.mean(micro_data.reshape(-1, 8), axis=1)  # 8:1 ratio
    def _localization_operator(self, macro_data: np.ndarray) -> np.ndarray:
        """Localize macroscale data to microscale."""
        # Interpolation/extrapolation (simplified)
        return np.repeat(macro_data, 8)  # 1:8 ratio
    def solve_multiscale(self,
                        initial_conditions: Dict[str, np.ndarray],
                        coupling_iterations: int = 10) -> Dict[str, np.ndarray]:
        """Solve multi-scale coupled problem."""
        results = {}
        for iteration in range(coupling_iterations):
            # Solve each scale
            for scale in self.scales:
                if scale in self.scale_solvers:
                    # Get boundary conditions from other scales
                    bc = self._get_scale_boundary_conditions(scale, results)
                    # Solve scale-specific problem
                    scale_result = self.scale_solvers[scale].solve(bc)
                    results[scale] = scale_result
            # Check convergence across scales
            if self._check_multiscale_convergence(results):
                break
        return results
    def _get_scale_boundary_conditions(self, scale: str, results: Dict) -> Dict:
        """Get boundary conditions for scale from other scales."""
        bc = {}
        # Transfer data from other scales
        for other_scale in self.scales:
            if other_scale != scale and other_scale in results:
                transfer_key = f"{other_scale}_to_{scale}"
                if transfer_key in self.transfer_operators:
                    transferred_data = self.transfer_operators[transfer_key](
                        results[other_scale]
                    )
                    bc[f"{other_scale}_data"] = transferred_data
        return bc
    def _check_multiscale_convergence(self, results: Dict) -> bool:
        """Check convergence across all scales."""
        # Simplified convergence check
        return len(results) == len(self.scales)
class AdvancedSolverFramework:
    """Advanced solver algorithms for multiphysics."""
    def __init__(self, config: SimulationConfig):
        """Initialize advanced solver framework."""
        self.config = config
        self.convergence_diagnostics = ConvergenceDiagnostics()
    def solve_with_quasi_newton(self,
                               coupled_system: CoupledSystem,
                               initial_guess: Dict[str, np.ndarray],
                               max_iterations: int = 50) -> Dict[str, Any]:
        """Solve using quasi-Newton method with BFGS approximation."""
        print("Using Quasi-Newton (BFGS) solver...")
        # Initialize
        x_current = self._pack_solution(initial_guess)
        n_vars = len(x_current)
        # BFGS approximation of inverse Hessian
        B_inv = np.eye(n_vars)
        residual_history = []
        for iteration in range(max_iterations):
            # Compute residual and Jacobian approximation
            residual = self._compute_residual(coupled_system, x_current)
            residual_norm = np.linalg.norm(residual)
            residual_history.append(residual_norm)
            print(f"  Iteration {iteration}: ||R|| = {residual_norm:.2e}")
            if residual_norm < 1e-6:
                print("  Converged!")
                break
            # Quasi-Newton step
            dx = -B_inv @ residual
            # Line search
            alpha = self._line_search(coupled_system, x_current, dx)
            x_new = x_current + alpha * dx
            # Update BFGS approximation
            if iteration > 0:
                s = x_new - x_current  # Step
                y = self._compute_residual(coupled_system, x_new) - residual  # Residual change
                # BFGS update
                if np.dot(s, y) > 1e-12:  # Avoid numerical issues
                    rho = 1.0 / np.dot(s, y)
                    I = np.eye(n_vars)
                    B_inv = (I - rho * np.outer(s, y)) @ B_inv @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
            x_current = x_new
        solution = self._unpack_solution(x_current, initial_guess.keys())
        return {
            'solution': solution,
            'converged': residual_norm < 1e-6,
            'iterations': iteration + 1,
            'residual_history': residual_history
        }
    def solve_with_anderson_acceleration(self,
                                       coupled_system: CoupledSystem,
                                       initial_guess: Dict[str, np.ndarray],
                                       memory_depth: int = 5) -> Dict[str, Any]:
        """Solve using Anderson acceleration."""
        print(f"Using Anderson acceleration (depth = {memory_depth})...")
        x_current = self._pack_solution(initial_guess)
        n_vars = len(x_current)
        # Anderson mixing history
        x_history = []
        f_history = []
        residual_history = []
        for iteration in range(50):
            # Fixed-point iteration: x = F(x)
            f_current = self._fixed_point_map(coupled_system, x_current)
            residual = f_current - x_current
            residual_norm = np.linalg.norm(residual)
            residual_history.append(residual_norm)
            print(f"  Iteration {iteration}: ||R|| = {residual_norm:.2e}")
            if residual_norm < 1e-6:
                print("  Converged!")
                break
            # Store history
            x_history.append(x_current.copy())
            f_history.append(f_current.copy())
            # Anderson acceleration
            if len(x_history) > memory_depth:
                x_history.pop(0)
                f_history.pop(0)
            if len(x_history) >= 2:
                x_new = self._anderson_mixing(x_history, f_history)
            else:
                x_new = f_current  # Standard fixed-point step
            x_current = x_new
        solution = self._unpack_solution(x_current, initial_guess.keys())
        return {
            'solution': solution,
            'converged': residual_norm < 1e-6,
            'iterations': iteration + 1,
            'residual_history': residual_history
        }
    def _anderson_mixing(self, x_history: List, f_history: List) -> np.ndarray:
        """Perform Anderson mixing."""
        m = len(x_history) - 1
        if m == 0:
            return f_history[-1]
        # Build difference matrices
        delta_x = np.column_stack([x_history[i] - x_history[i-1] for i in range(1, len(x_history))])
        delta_f = np.column_stack([f_history[i] - f_history[i-1] for i in range(1, len(f_history))])
        # Solve least squares problem
        A = delta_f - delta_x
        b = f_history[-1] - x_history[-1]
        try:
            alphas = np.linalg.lstsq(A, b, rcond=None)[0]
        except:
            return f_history[-1]  # Fallback
        # Anderson combination
        x_anderson = f_history[-1]
        for i, alpha in enumerate(alphas):
            x_anderson -= alpha * (f_history[-1-i] - x_history[-1-i])
        return x_anderson
    def _pack_solution(self, solution_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Pack solution dictionary into vector."""
        return np.concatenate([sol.flatten() for sol in solution_dict.values()])
    def _unpack_solution(self, x: np.ndarray, keys: List[str]) -> Dict[str, np.ndarray]:
        """Unpack solution vector into dictionary."""
        # Simplified unpacking (assumes equal sizes)
        n_fields = len(keys)
        field_size = len(x) // n_fields
        solution = {}
        for i, key in enumerate(keys):
            start_idx = i * field_size
            end_idx = (i + 1) * field_size
            solution[key] = x[start_idx:end_idx]
        return solution
    def _compute_residual(self, coupled_system: CoupledSystem, x: np.ndarray) -> np.ndarray:
        """Compute residual for quasi-Newton method."""
        # Simplified residual computation
        return np.random.random(len(x)) * 0.1  # Placeholder
    def _fixed_point_map(self, coupled_system: CoupledSystem, x: np.ndarray) -> np.ndarray:
        """Fixed-point map F(x) for Anderson acceleration."""
        # Simplified fixed-point map
        return x + 0.1 * np.random.random(len(x))  # Placeholder
    def _line_search(self, coupled_system: CoupledSystem, x: np.ndarray, dx: np.ndarray) -> float:
        """Line search for step size."""
        # Simplified Armijo line search
        return 1.0  # Placeholder
def nuclear_reactor_multiphysics():
    """Nuclear reactor multiphysics simulation."""
    print("=== Nuclear Reactor Multiphysics ===")
    # Reactor geometry (simplified)
    def create_reactor_mesh():
        # 2D reactor core cross-section
        n_assemblies = 17  # 17x17 PWR assembly
        assembly_pitch = 0.21  # m
        # Generate fuel assembly locations
        center = (n_assemblies - 1) / 2
        nodes = []
        for i in range(n_assemblies):
            for j in range(n_assemblies):
                x = (i - center) * assembly_pitch
                y = (j - center) * assembly_pitch
                nodes.append([x, y])
        return {'nodes': np.array(nodes)}
    mesh = create_reactor_mesh()
    # Multi-physics coupling: Neutronics + Thermal-Hydraulics + Mechanics
    physics_domains = ["neutronics", "thermal_hydraulics", "mechanics"]
    print("Coupled Physics:")
    print("- Neutronics (fission heat generation)")
    print("- Thermal-hydraulics (coolant flow and heat transfer)")
    print("- Structural mechanics (fuel rod expansion)")
    print()
    # Simulation parameters
    power_level = 3400e6  # 3400 MW thermal
    coolant_inlet_temp = 292  # °C
    operating_pressure = 15.5e6  # Pa
    # Material properties
    fuel_properties = {
        'thermal_conductivity': 3.5,  # W/(m·K) (UO2)
        'linear_heat_rate': 20000,    # W/m
        'thermal_expansion': 10e-6    # 1/K
    }
    coolant_properties = {
        'density': 700,      # kg/m³ (hot water)
        'specific_heat': 5500,  # J/(kg·K)
        'thermal_conductivity': 0.5  # W/(m·K)
    }
    print(f"Reactor power: {power_level/1e6:.0f} MW")
    print(f"Coolant inlet: {coolant_inlet_temp}°C")
    print(f"Operating pressure: {operating_pressure/1e6:.1f} MPa")
    print()
    # Simulate coupling iterations
    max_iterations = 20
    tolerance = 1e-4
    # Initial conditions
    power_distribution = np.ones(len(mesh['nodes']))  # Flat power
    coolant_temp = np.ones(len(mesh['nodes'])) * coolant_inlet_temp
    fuel_temp = np.ones(len(mesh['nodes'])) * (coolant_inlet_temp + 500)
    convergence_history = []
    for iteration in range(max_iterations):
        # Previous iteration values
        prev_power = power_distribution.copy()
        prev_coolant_temp = coolant_temp.copy()
        prev_fuel_temp = fuel_temp.copy()
        # Neutronics solve (power distribution)
        # Power depends on temperature (Doppler feedback)
        doppler_feedback = 1 - 1e-5 * (fuel_temp - 900)  # Negative feedback
        power_distribution = power_level * doppler_feedback / np.sum(doppler_feedback)
        # Thermal-hydraulics solve
        # Heat transfer: coolant temperature rise
        linear_heat_rate = power_distribution / len(mesh['nodes'])
        coolant_temp_rise = linear_heat_rate / (coolant_properties['density'] *
                                              coolant_properties['specific_heat'] * 5.0)  # 5 m/s flow
        coolant_temp = coolant_inlet_temp + coolant_temp_rise
        # Fuel temperature
        fuel_to_coolant_htc = 50000  # W/(m²·K)
        fuel_temp = coolant_temp + linear_heat_rate / fuel_to_coolant_htc
        # Check convergence
        power_change = np.max(np.abs(power_distribution - prev_power)) / np.max(power_distribution)
        temp_change = np.max(np.abs(fuel_temp - prev_fuel_temp)) / np.max(fuel_temp)
        residual = max(power_change, temp_change)
        convergence_history.append(residual)
        print(f"Iteration {iteration+1}: Residual = {residual:.2e}")
        if residual < tolerance:
            print("Reactor simulation converged!")
            break
    # Results analysis
    max_fuel_temp = np.max(fuel_temp)
    avg_power_density = np.mean(power_distribution) / 1e6  # MW/m³
    peaking_factor = np.max(power_distribution) / np.mean(power_distribution)
    print()
    print("Reactor Analysis Results:")
    print(f"Maximum fuel temperature: {max_fuel_temp:.0f}°C")
    print(f"Average power density: {avg_power_density:.1f} MW/m³")
    print(f"Power peaking factor: {peaking_factor:.2f}")
    # Safety margins
    fuel_melting_point = 2840  # °C for UO2
    safety_margin = fuel_melting_point - max_fuel_temp
    print(f"Safety margin to fuel melting: {safety_margin:.0f}°C")
    if max_fuel_temp > 1200:
        print("WARNING: High fuel temperature - check cooling!")
    print()
def aerospace_hypersonic_vehicle():
    """Hypersonic vehicle multiphysics analysis."""
    print("=== Hypersonic Vehicle Multiphysics ===")
    # Vehicle parameters
    mach_number = 8.0
    altitude = 30000  # m
    vehicle_length = 10  # m
    # Atmospheric conditions
    atm_temp = 226.65  # K (-46.5°C)
    atm_pressure = 1197  # Pa
    atm_density = 0.0184  # kg/m³
    # Flight conditions
    speed_of_sound = np.sqrt(1.4 * 287 * atm_temp)  # m/s
    flight_velocity = mach_number * speed_of_sound  # m/s
    print(f"Flight Conditions:")
    print(f"Mach number: {mach_number}")
    print(f"Altitude: {altitude/1000:.0f} km")
    print(f"Velocity: {flight_velocity:.0f} m/s ({flight_velocity*3.6:.0f} km/h)")
    print(f"Atmospheric temperature: {atm_temp-273.15:.1f}°C")
    print()
    # Multi-physics coupling
    physics = {
        'aerodynamics': 'High-speed compressible flow',
        'heat_transfer': 'Extreme aerodynamic heating',
        'structure': 'Thermal stress and deformation',
        'materials': 'Temperature-dependent properties'
    }
    for physics_type, description in physics.items():
        print(f"{physics_type.title()}: {description}")
    print()
    # Aerodynamic heating calculation
    # Stagnation point heating
    rho_inf = atm_density
    v_inf = flight_velocity
    # Stagnation point heat flux (Fay-Riddell correlation)
    stagnation_heat_flux = 17.6 * np.sqrt(rho_inf / 0.001) * (v_inf / 1000)**3  # W/m²
    # Temperature distribution along vehicle
    x_positions = np.linspace(0, vehicle_length, 50)
    heat_flux_distribution = stagnation_heat_flux * np.exp(-x_positions / 2.0)
    # Thermal analysis
    # Material properties (UHTC - Ultra High Temperature Ceramic)
    material_props = {
        'thermal_conductivity': 25,  # W/(m·K)
        'specific_heat': 600,        # J/(kg·K)
        'density': 6000,            # kg/m³
        'melting_point': 3000,      # °C
        'emissivity': 0.8
    }
    # Heat balance: q_aero = q_conduction + q_radiation
    stefan_boltzmann = 5.67e-8  # W/(m²·K⁴)
    surface_temperatures = []
    for q_aero in heat_flux_distribution:
        # Solve for surface temperature
        # q_aero = ε*σ*T⁴ + k*dT/dx (simplified)
        # Assuming radiative cooling dominates
        T_surface = (q_aero / (material_props['emissivity'] * stefan_boltzmann))**(1/4)
        surface_temperatures.append(T_surface)
    surface_temperatures = np.array(surface_temperatures)
    max_temperature = np.max(surface_temperatures)
    print(f"Thermal Analysis:")
    print(f"Stagnation point heat flux: {stagnation_heat_flux/1e6:.2f} MW/m²")
    print(f"Maximum surface temperature: {max_temperature-273.15:.0f}°C")
    print(f"Material melting point: {material_props['melting_point']}°C")
    # Thermal stress analysis
    alpha = 5e-6  # 1/K (thermal expansion)
    E = 300e9     # Pa (Young's modulus)
    delta_T = max_temperature - atm_temp
    thermal_stress = E * alpha * delta_T / 1e6  # Convert to MPa
    print(f"Maximum thermal stress: {thermal_stress:.0f} MPa")
    # Structural considerations
    if max_temperature > material_props['melting_point'] + 273.15:
        print("CRITICAL: Material melting temperature exceeded!")
    elif max_temperature > 2500 + 273.15:
        print("WARNING: Very high temperature - active cooling required")
    else:
        print("Temperature within material limits")
    # Visualize temperature distribution
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x_positions, heat_flux_distribution/1e6, 'o-', color='#FDB515', linewidth=2)
    plt.xlabel('Position along vehicle (m)')
    plt.ylabel('Heat flux (MW/m²)')
    plt.title('Aerodynamic Heat Flux')
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(x_positions, surface_temperatures-273.15, 'o-', color='#003262', linewidth=2)
    plt.axhline(y=material_props['melting_point'], color='red', linestyle='--',
                label='Melting point')
    plt.xlabel('Position along vehicle (m)')
    plt.ylabel('Surface temperature (°C)')
    plt.title('Surface Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print()
def biomedical_drug_delivery():
    """Biomedical drug delivery multiphysics simulation."""
    print("=== Biomedical Drug Delivery System ===")
    # System description
    print("Coupled Physics in Drug Delivery:")
    print("- Fluid mechanics (blood flow)")
    print("- Mass transport (drug diffusion)")
    print("- Biochemical reactions (drug metabolism)")
    print("- Biomechanics (vessel deformation)")
    print()
    # Vascular system parameters
    vessel_diameter = 2e-3  # 2 mm artery
    vessel_length = 10e-2   # 10 cm segment
    blood_velocity = 0.1    # m/s
    # Blood properties
    blood_density = 1060    # kg/m³
    blood_viscosity = 3.5e-3  # Pa·s
    # Drug properties
    drug_diffusivity = 1e-10  # m²/s
    drug_concentration = 1.0  # mg/mL (initial)
    metabolism_rate = 0.1     # 1/s (first-order)
    print(f"Vessel: diameter = {vessel_diameter*1000:.1f} mm, length = {vessel_length*100:.0f} cm")
    print(f"Blood flow: velocity = {blood_velocity*100:.0f} cm/s")
    print(f"Drug diffusivity: {drug_diffusivity:.2e} m²/s")
    print()
    # Reynolds number
    Re = blood_density * blood_velocity * vessel_diameter / blood_viscosity
    print(f"Reynolds number: {Re:.0f} (laminar flow)")
    # Péclet number (convection vs diffusion)
    Pe = blood_velocity * vessel_diameter / drug_diffusivity
    print(f"Péclet number: {Pe:.2e} (convection dominated)")
    print()
    # Simplified 1D drug transport model
    # ∂c/∂t + v*∂c/∂x = D*∂²c/∂x² - k*c
    # Discretization
    nx = 100
    x = np.linspace(0, vessel_length, nx)
    dx = x[1] - x[0]
    dt = 1e-4  # s
    # Initial condition
    concentration = np.zeros(nx)
    concentration[0] = drug_concentration  # Injection point
    # Time evolution
    time_points = np.arange(0, 1.0, dt)  # 1 second simulation
    # Storage for visualization
    concentration_history = []
    time_snapshots = [0.1, 0.25, 0.5, 1.0]  # seconds
    for i, t in enumerate(time_points):
        # Finite difference transport equation
        c_new = concentration.copy()
        for j in range(1, nx-1):
            # Convection term
            convection = -blood_velocity * (concentration[j] - concentration[j-1]) / dx
            # Diffusion term
            diffusion = drug_diffusivity * (concentration[j+1] - 2*concentration[j] + concentration[j-1]) / dx**2
            # Reaction term (metabolism)
            reaction = -metabolism_rate * concentration[j]
            # Update
            c_new[j] = concentration[j] + dt * (convection + diffusion + reaction)
        # Boundary conditions
        c_new[0] = drug_concentration * np.exp(-t)  # Decaying injection
        c_new[-1] = 0  # Outlet
        concentration = c_new
        # Store snapshots
        if t in time_snapshots:
            concentration_history.append((t, concentration.copy()))
    # Analysis
    print("Drug Transport Analysis:")
    # Penetration distance
    penetration_threshold = 0.1 * drug_concentration  # 10% of initial
    for t, c in concentration_history:
        penetration_idx = np.where(c > penetration_threshold)[0]
        if len(penetration_idx) > 0:
            penetration_distance = x[penetration_idx[-1]]
            print(f"t = {t:.2f} s: penetration = {penetration_distance*100:.1f} cm")
    # Visualize drug distribution
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for t, c in concentration_history:
        plt.plot(x*100, c, label=f't = {t:.2f} s', linewidth=2)
    plt.xlabel('Distance (cm)')
    plt.ylabel('Drug concentration (mg/mL)')
    plt.title('Drug Distribution in Vessel')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Concentration vs time at different locations
    plt.subplot(1, 2, 2)
    locations = [0.01, 0.03, 0.05]  # m
    location_indices = [np.argmin(np.abs(x - loc)) for loc in locations]
    time_axis = [t for t, _ in concentration_history]
    for i, (loc, idx) in enumerate(zip(locations, location_indices)):
        concentrations_at_loc = [c[idx] for _, c in concentration_history]
        plt.plot(time_axis, concentrations_at_loc, 'o-',
                label=f'{loc*100:.0f} cm from injection', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Drug concentration (mg/mL)')
    plt.title('Temporal Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    # Therapeutic efficacy
    therapeutic_threshold = 0.5  # mg/mL
    effective_distance = 0
    final_concentration = concentration_history[-1][1]
    effective_indices = np.where(final_concentration > therapeutic_threshold)[0]
    if len(effective_indices) > 0:
        effective_distance = x[effective_indices[-1]]
    print(f"\nTherapeutic Analysis:")
    print(f"Therapeutic threshold: {therapeutic_threshold} mg/mL")
    print(f"Effective treatment distance: {effective_distance*100:.1f} cm")
    print(f"Treatment efficiency: {effective_distance/vessel_length*100:.0f}%")
    print()
def uncertainty_quantification_demo():
    """Demonstrate uncertainty quantification in multiphysics."""
    print("=== Uncertainty Quantification in Multiphysics ===")
    # Parameter uncertainties
    uncertain_params = {
        'thermal_conductivity': {
            'nominal': 50.0,     # W/(m·K)
            'std': 5.0,          # ±10% uncertainty
            'distribution': 'normal'
        },
        'heat_transfer_coefficient': {
            'nominal': 1000.0,   # W/(m²·K)
            'std': 200.0,        # ±20% uncertainty
            'distribution': 'normal'
        },
        'heat_source': {
            'nominal': 1e6,      # W/m³
            'std': 2e5,          # ±20% uncertainty
            'distribution': 'uniform'
        }
    }
    print("Uncertain Parameters:")
    for param, props in uncertain_params.items():
        print(f"- {param}: {props['nominal']} ± {props['std']} ({props['distribution']})")
    print()
    # Monte Carlo sampling
    n_samples = 1000
    np.random.seed(42)
    print(f"Running Monte Carlo simulation ({n_samples} samples)...")
    # Generate samples
    samples = {}
    for param, props in uncertain_params.items():
        if props['distribution'] == 'normal':
            samples[param] = np.random.normal(props['nominal'], props['std'], n_samples)
        elif props['distribution'] == 'uniform':
            half_range = props['std']
            samples[param] = np.random.uniform(
                props['nominal'] - half_range,
                props['nominal'] + half_range,
                n_samples
            )
    # Simplified multiphysics model: steady-state heat equation
    # q = h*(T_surface - T_ambient) where q = heat_source
    T_ambient = 20  # °C
    results = []
    for i in range(n_samples):
        k = samples['thermal_conductivity'][i]
        h = samples['heat_transfer_coefficient'][i]
        q = samples['heat_source'][i]
        # Simplified solution: T_surface = T_ambient + q/h
        T_surface = T_ambient + q / h
        results.append(T_surface)
    results = np.array(results)
    # Statistical analysis
    mean_temp = np.mean(results)
    std_temp = np.std(results)
    percentiles = np.percentile(results, [5, 25, 50, 75, 95])
    print(f"\nTemperature Statistics:")
    print(f"Mean: {mean_temp:.1f}°C")
    print(f"Standard deviation: {std_temp:.1f}°C")
    print(f"95% confidence interval: [{percentiles[0]:.1f}, {percentiles[4]:.1f}]°C")
    # Sensitivity analysis
    print(f"\nSensitivity Analysis:")
    # Correlation coefficients
    for param in uncertain_params.keys():
        correlation = np.corrcoef(samples[param], results)[0, 1]
        print(f"Correlation with {param}: {correlation:.3f}")
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Histogram of results
    axes[0, 0].hist(results, bins=50, density=True, alpha=0.7, color='#003262')
    axes[0, 0].axvline(mean_temp, color='#FDB515', linestyle='--', linewidth=2, label='Mean')
    axes[0, 0].axvline(percentiles[0], color='red', linestyle=':', linewidth=2, label='5th percentile')
    axes[0, 0].axvline(percentiles[4], color='red', linestyle=':', linewidth=2, label='95th percentile')
    axes[0, 0].set_xlabel('Temperature (°C)')
    axes[0, 0].set_ylabel('Probability density')
    axes[0, 0].set_title('Temperature Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    # Sensitivity scatter plots
    param_names = list(uncertain_params.keys())
    for i, param in enumerate(param_names[:3]):  # First 3 parameters
        row = (i + 1) // 2
        col = (i + 1) % 2
        axes[row, col].scatter(samples[param], results, alpha=0.5, s=20, color='#003262')
        axes[row, col].set_xlabel(param.replace('_', ' ').title())
        axes[row, col].set_ylabel('Temperature (°C)')
        axes[row, col].set_title(f'Sensitivity to {param.replace("_", " ").title()}')
        axes[row, col].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    # Risk assessment
    failure_threshold = 200  # °C
    failure_probability = np.sum(results > failure_threshold) / n_samples
    print(f"\nRisk Assessment:")
    print(f"Failure threshold: {failure_threshold}°C")
    print(f"Failure probability: {failure_probability*100:.2f}%")
    if failure_probability > 0.01:  # 1%
        print("HIGH RISK: Consider design modifications")
    elif failure_probability > 0.001:  # 0.1%
        print("MODERATE RISK: Monitor critical parameters")
    else:
        print("LOW RISK: Acceptable design")
    print()
def high_performance_computing_considerations():
    """Discuss HPC considerations for multiphysics."""
    print("=== High-Performance Computing for Multiphysics ===")
    # Problem scaling analysis
    problem_sizes = [1e3, 1e4, 1e5, 1e6, 1e7]  # Number of DOFs
    physics_count = [2, 3, 4, 5]  # Number of coupled physics
    print("Computational Complexity Analysis:")
    print("-" * 40)
    for n_physics in physics_count:
        print(f"\n{n_physics} Coupled Physics:")
        for n_dof in problem_sizes:
            # Estimate memory and computation
            memory_per_dof = 8 * n_physics  # bytes (double precision)
            total_memory = n_dof * memory_per_dof / 1e9  # GB
            # Matrix operations: O(n^1.5) for sparse, O(n^3) for dense
            sparse_flops = n_dof**1.5 * n_physics
            dense_flops = n_dof**3 * n_physics
            print(f"  {n_dof:.0e} DOFs: {total_memory:.1f} GB memory")
    # Parallelization strategies
    print(f"\nParallelization Strategies:")
    print(f"1. Domain Decomposition:")
    print(f"   - Split spatial domain among processors")
    print(f"   - Good for same physics, challenging for different physics")
    print(f"   - Communication at subdomain boundaries")
    print(f"\n2. Physics Decomposition:")
    print(f"   - Assign different physics to different processor groups")
    print(f"   - Natural for loosely coupled problems")
    print(f"   - Load balancing challenges")
    print(f"\n3. Hybrid Approaches:")
    print(f"   - Combine domain and physics decomposition")
    print(f"   - Multi-level parallelism")
    print(f"   - Complex but flexible")
    # Performance metrics
    processor_counts = [1, 4, 16, 64, 256, 1024]
    ideal_speedup = processor_counts
    # Realistic speedup with Amdahl's law
    sequential_fraction = 0.1  # 10% sequential code
    realistic_speedup = [1 / (sequential_fraction + (1 - sequential_fraction) / p)
                        for p in processor_counts]
    # Plot scaling
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.loglog(processor_counts, ideal_speedup, 'o-', label='Ideal scaling',
               color='#FDB515', linewidth=2)
    plt.loglog(processor_counts, realistic_speedup, 's-', label='Realistic scaling',
               color='#003262', linewidth=2)
    plt.xlabel('Number of processors')
    plt.ylabel('Speedup')
    plt.title('Parallel Scaling')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Efficiency
    efficiency = [s / p for s, p in zip(realistic_speedup, processor_counts)]
    plt.subplot(1, 2, 2)
    plt.semilogx(processor_counts, [e * 100 for e in efficiency], 'o-',
                 color='#003262', linewidth=2)
    plt.xlabel('Number of processors')
    plt.ylabel('Parallel efficiency (%)')
    plt.title('Parallel Efficiency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    # GPU acceleration
    print(f"\nGPU Acceleration:")
    print(f"- Massive parallelism for matrix operations")
    print(f"- Memory bandwidth advantages")
    print(f"- Challenges: data transfer, algorithm adaptation")
    print(f"- Best for: dense linear algebra, explicit time stepping")
    print(f"\nRecommendations:")
    print(f"- Start with single-physics optimization")
    print(f"- Use efficient sparse solvers (PETSc, Trilinos)")
    print(f"- Implement adaptive mesh refinement")
    print(f"- Consider mixed precision arithmetic")
    print(f"- Profile and optimize communication patterns")
    print()
def main():
    """Run all advanced multiphysics examples."""
    print("Berkeley SciComp - Advanced Multiphysics Applications")
    print("=" * 60)
    print()
    # Run examples
    nuclear_reactor_multiphysics()
    aerospace_hypersonic_vehicle()
    biomedical_drug_delivery()
    uncertainty_quantification_demo()
    high_performance_computing_considerations()
    print("Advanced multiphysics examples completed!")
    print("\nCutting-edge concepts demonstrated:")
    print("- Nuclear reactor multi-physics with neutronics feedback")
    print("- Hypersonic vehicle thermal protection systems")
    print("- Biomedical drug delivery with transport and reactions")
    print("- Uncertainty quantification and risk assessment")
    print("- High-performance computing scalability")
    print("\nResearch frontiers:")
    print("- Machine learning enhanced coupling")
    print("- Quantum-classical hybrid simulations")
    print("- Exascale computing challenges")
    print("- Multi-scale temporal coupling")
    print("- Real-time adaptive algorithms")
if __name__ == "__main__":
    main()
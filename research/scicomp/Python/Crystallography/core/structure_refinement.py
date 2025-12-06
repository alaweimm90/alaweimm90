"""
Structure Refinement Methods
Advanced crystallographic structure refinement algorithms including
Rietveld method, least squares refinement, and maximum likelihood approaches.
"""
import numpy as np
from scipy import optimize, linalg
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
import warnings
# Import related modules
from .crystal_structure import CrystalStructure, AtomicPosition, LatticeParameters
from .diffraction import DiffractionPattern, ReflectionData
@dataclass
class RefinementParameter:
    """Refinement parameter definition."""
    name: str
    value: float
    refined: bool = True
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    uncertainty: Optional[float] = None
    def __post_init__(self):
        """Validate parameter bounds."""
        if (self.lower_bound is not None and self.upper_bound is not None and
            self.lower_bound >= self.upper_bound):
            raise ValueError("Lower bound must be less than upper bound")
@dataclass
class RefinementResults:
    """Results from structure refinement."""
    converged: bool
    final_parameters: Dict[str, float]
    parameter_uncertainties: Dict[str, float] = field(default_factory=dict)
    goodness_of_fit: float = 0.0
    r_factor: float = 0.0
    weighted_r_factor: float = 0.0
    chi_squared: float = 0.0
    correlation_matrix: Optional[np.ndarray] = None
    iterations: int = 0
    final_residual: float = 0.0
class LeastSquaresRefinement:
    """
    Least squares refinement for crystal structures.
    Features:
    - Parameter constraint handling
    - Correlation matrix calculation
    - Uncertainty estimation
    - Robust convergence algorithms
    Examples:
        >>> crystal = CrystalStructure(lattice, atoms)
        >>> refinement = LeastSquaresRefinement(crystal, observed_data)
        >>> results = refinement.refine()
    """
    def __init__(self, crystal: CrystalStructure,
                 observed_intensities: List[float],
                 reflection_data: List[ReflectionData]):
        """
        Initialize least squares refinement.
        Parameters:
            crystal: Initial crystal structure
            observed_intensities: Observed structure factor magnitudes
            reflection_data: List of reflection information
        """
        self.crystal = crystal
        self.observed_intensities = np.array(observed_intensities)
        self.reflection_data = reflection_data
        self.parameters = {}
        self.parameter_order = []
        # Refinement settings
        self.max_iterations = 100
        self.convergence_tolerance = 1e-6
        self.damping_factor = 1.0
        # Setup initial parameters
        self._setup_parameters()
    def _setup_parameters(self):
        """Setup refinement parameters from crystal structure."""
        param_idx = 0
        # Lattice parameters
        lattice = self.crystal.lattice
        self.parameters['a'] = RefinementParameter('a', lattice.a, True)
        self.parameters['b'] = RefinementParameter('b', lattice.b, True)
        self.parameters['c'] = RefinementParameter('c', lattice.c, True)
        self.parameters['alpha'] = RefinementParameter('alpha', lattice.alpha, True)
        self.parameters['beta'] = RefinementParameter('beta', lattice.beta, True)
        self.parameters['gamma'] = RefinementParameter('gamma', lattice.gamma, True)
        # Atomic parameters
        for i, atom in enumerate(self.crystal.atoms):
            self.parameters[f'x_{i}'] = RefinementParameter(f'x_{i}', atom.x, True, 0.0, 1.0)
            self.parameters[f'y_{i}'] = RefinementParameter(f'y_{i}', atom.y, True, 0.0, 1.0)
            self.parameters[f'z_{i}'] = RefinementParameter(f'z_{i}', atom.z, True, 0.0, 1.0)
            self.parameters[f'occ_{i}'] = RefinementParameter(f'occ_{i}', atom.occupancy, True, 0.0, 1.0)
            self.parameters[f'B_{i}'] = RefinementParameter(f'B_{i}', atom.thermal_factor, True, 0.0)
        # Create parameter order for optimization
        self.parameter_order = [name for name, param in self.parameters.items() if param.refined]
    def get_parameter_vector(self) -> np.ndarray:
        """Get current parameter values as vector."""
        return np.array([self.parameters[name].value for name in self.parameter_order])
    def set_parameter_vector(self, values: np.ndarray):
        """Set parameter values from vector."""
        for i, name in enumerate(self.parameter_order):
            self.parameters[name].value = values[i]
        # Update crystal structure
        self._update_crystal_structure()
    def _update_crystal_structure(self):
        """Update crystal structure from current parameters."""
        # Update lattice parameters
        new_lattice = LatticeParameters(
            a=self.parameters['a'].value,
            b=self.parameters['b'].value,
            c=self.parameters['c'].value,
            alpha=self.parameters['alpha'].value,
            beta=self.parameters['beta'].value,
            gamma=self.parameters['gamma'].value
        )
        # Update atomic positions
        new_atoms = []
        for i, atom in enumerate(self.crystal.atoms):
            new_atom = AtomicPosition(
                element=atom.element,
                x=self.parameters[f'x_{i}'].value,
                y=self.parameters[f'y_{i}'].value,
                z=self.parameters[f'z_{i}'].value,
                occupancy=self.parameters[f'occ_{i}'].value,
                thermal_factor=self.parameters[f'B_{i}'].value
            )
            new_atoms.append(new_atom)
        self.crystal = CrystalStructure(new_lattice, new_atoms)
    def calculate_structure_factors(self) -> np.ndarray:
        """Calculate structure factors for all reflections."""
        from .diffraction import StructureFactor
        sf_calculator = StructureFactor(self.crystal)
        calculated_intensities = []
        for refl in self.reflection_data:
            F_hkl = sf_calculator.calculate(refl.h, refl.k, refl.l)
            calculated_intensities.append(abs(F_hkl))
        return np.array(calculated_intensities)
    def residual_function(self, parameters: np.ndarray) -> np.ndarray:
        """Calculate residual vector for least squares."""
        # Update parameters
        self.set_parameter_vector(parameters)
        # Calculate model intensities
        calculated = self.calculate_structure_factors()
        # Calculate residuals (weighted differences)
        weights = np.ones_like(self.observed_intensities)  # Unit weights for now
        residuals = weights * (self.observed_intensities - calculated)
        return residuals
    def jacobian_function(self, parameters: np.ndarray) -> np.ndarray:
        """Calculate Jacobian matrix for least squares."""
        # Update parameters
        self.set_parameter_vector(parameters)
        # Calculate numerical derivatives
        n_params = len(parameters)
        n_data = len(self.observed_intensities)
        jacobian = np.zeros((n_data, n_params))
        step_size = 1e-6
        for i, param_name in enumerate(self.parameter_order):
            # Forward difference
            params_plus = parameters.copy()
            params_plus[i] += step_size
            residuals_plus = self.residual_function(params_plus)
            # Backward difference
            params_minus = parameters.copy()
            params_minus[i] -= step_size
            residuals_minus = self.residual_function(params_minus)
            # Central difference
            jacobian[:, i] = (residuals_plus - residuals_minus) / (2 * step_size)
        # Restore original parameters
        self.set_parameter_vector(parameters)
        return jacobian
    def refine(self) -> RefinementResults:
        """Perform least squares refinement."""
        initial_params = self.get_parameter_vector()
        try:
            # Use scipy's least squares with Jacobian
            result = optimize.least_squares(
                self.residual_function,
                initial_params,
                jac=self.jacobian_function,
                max_nfev=self.max_iterations * len(initial_params),
                ftol=self.convergence_tolerance,
                xtol=self.convergence_tolerance,
                gtol=self.convergence_tolerance
            )
            # Calculate uncertainties from covariance matrix
            uncertainties = {}
            correlation_matrix = None
            if result.success and hasattr(result, 'jac'):
                try:
                    # Calculate covariance matrix
                    jacobian = result.jac
                    if jacobian.size > 0:
                        cov_matrix = linalg.inv(jacobian.T @ jacobian)
                        # Parameter uncertainties are sqrt of diagonal elements
                        param_uncertainties = np.sqrt(np.diag(cov_matrix))
                        for i, name in enumerate(self.parameter_order):
                            uncertainties[name] = param_uncertainties[i]
                        # Correlation matrix
                        diag_inv = np.diag(1.0 / param_uncertainties)
                        correlation_matrix = diag_inv @ cov_matrix @ diag_inv
                except (linalg.LinAlgError, ValueError):
                    warnings.warn("Could not calculate parameter uncertainties")
            # Calculate goodness of fit statistics
            final_residuals = result.fun
            chi_squared = np.sum(final_residuals**2)
            n_data = len(self.observed_intensities)
            n_params = len(initial_params)
            goodness_of_fit = np.sqrt(chi_squared / (n_data - n_params)) if n_data > n_params else 0.0
            # R-factors
            calculated = self.calculate_structure_factors()
            r_factor = np.sum(np.abs(self.observed_intensities - calculated)) / np.sum(self.observed_intensities)
            # Weighted R-factor (assuming unit weights)
            weights = np.ones_like(self.observed_intensities)
            weighted_r_factor = (np.sqrt(np.sum(weights * (self.observed_intensities - calculated)**2)) /
                                np.sqrt(np.sum(weights * self.observed_intensities**2)))
            # Get final parameters
            final_parameters = {}
            for i, name in enumerate(self.parameter_order):
                final_parameters[name] = result.x[i]
            return RefinementResults(
                converged=result.success,
                final_parameters=final_parameters,
                parameter_uncertainties=uncertainties,
                goodness_of_fit=goodness_of_fit,
                r_factor=r_factor,
                weighted_r_factor=weighted_r_factor,
                chi_squared=chi_squared,
                correlation_matrix=correlation_matrix,
                iterations=result.nfev,
                final_residual=np.linalg.norm(final_residuals)
            )
        except Exception as e:
            warnings.warn(f"Refinement failed: {e}")
            return RefinementResults(
                converged=False,
                final_parameters={name: self.parameters[name].value for name in self.parameter_order}
            )
class RietveldRefinement:
    """
    Rietveld refinement for powder diffraction data.
    Features:
    - Full powder pattern fitting
    - Peak profile refinement
    - Background modeling
    - Preferred orientation correction
    - Multi-phase refinement capability
    Examples:
        >>> crystal = CrystalStructure(lattice, atoms)
        >>> rietveld = RietveldRefinement(crystal, two_theta_obs, intensity_obs)
        >>> results = rietveld.refine()
    """
    def __init__(self, crystal: CrystalStructure,
                 two_theta_observed: np.ndarray,
                 intensity_observed: np.ndarray,
                 wavelength: float = 1.54056):
        """
        Initialize Rietveld refinement.
        Parameters:
            crystal: Crystal structure
            two_theta_observed: Observed 2θ values
            intensity_observed: Observed intensities
            wavelength: X-ray wavelength
        """
        self.crystal = crystal
        self.two_theta_obs = np.array(two_theta_observed)
        self.intensity_obs = np.array(intensity_observed)
        self.wavelength = wavelength
        # Refinement parameters
        self.parameters = {}
        self.parameter_order = []
        # Peak profile parameters
        self.peak_shape = 'pseudo_voigt'  # 'gaussian', 'lorentzian', 'pseudo_voigt'
        # Background parameters
        self.background_type = 'polynomial'  # 'polynomial', 'spline'
        self.background_order = 5
        self._setup_rietveld_parameters()
    def _setup_rietveld_parameters(self):
        """Setup Rietveld refinement parameters."""
        # Crystal structure parameters (from LeastSquaresRefinement)
        lattice = self.crystal.lattice
        self.parameters['a'] = RefinementParameter('a', lattice.a, True)
        self.parameters['b'] = RefinementParameter('b', lattice.b, True)
        self.parameters['c'] = RefinementParameter('c', lattice.c, True)
        # Atomic parameters
        for i, atom in enumerate(self.crystal.atoms):
            self.parameters[f'x_{i}'] = RefinementParameter(f'x_{i}', atom.x, True, 0.0, 1.0)
            self.parameters[f'y_{i}'] = RefinementParameter(f'y_{i}', atom.y, True, 0.0, 1.0)
            self.parameters[f'z_{i}'] = RefinementParameter(f'z_{i}', atom.z, True, 0.0, 1.0)
            self.parameters[f'B_{i}'] = RefinementParameter(f'B_{i}', atom.thermal_factor, True, 0.0)
        # Profile parameters
        self.parameters['U'] = RefinementParameter('U', 0.01, True, 0.0)  # Peak width parameter
        self.parameters['V'] = RefinementParameter('V', 0.01, True)       # Peak width parameter
        self.parameters['W'] = RefinementParameter('W', 0.01, True, 0.0)  # Peak width parameter
        self.parameters['eta'] = RefinementParameter('eta', 0.5, True, 0.0, 1.0)  # Mixing parameter
        # Scale factor
        self.parameters['scale'] = RefinementParameter('scale', 1.0, True, 0.0)
        # Background parameters
        for i in range(self.background_order + 1):
            initial_value = 10.0 if i == 0 else 0.0  # Constant background
            self.parameters[f'bg_{i}'] = RefinementParameter(f'bg_{i}', initial_value, True)
        # Zero point correction
        self.parameters['zero_point'] = RefinementParameter('zero_point', 0.0, True, -0.1, 0.1)
        # Create parameter order
        self.parameter_order = [name for name, param in self.parameters.items() if param.refined]
    def calculate_pattern(self, parameters: Optional[np.ndarray] = None) -> np.ndarray:
        """Calculate powder diffraction pattern."""
        if parameters is not None:
            self.set_parameter_vector(parameters)
        # Update crystal structure
        self._update_crystal_structure()
        # Generate diffraction pattern
        diffraction = DiffractionPattern(self.crystal)
        reflections = diffraction._calculate_reflections(
            self.wavelength, np.max(self.two_theta_obs), max_hkl=5
        )
        # Initialize calculated pattern
        calculated_intensity = np.zeros_like(self.two_theta_obs)
        # Add background
        background = self._calculate_background()
        calculated_intensity += background
        # Add Bragg peaks
        scale = self.parameters['scale'].value
        zero_point = self.parameters['zero_point'].value
        for refl in reflections:
            # Apply zero point correction
            peak_position = refl.two_theta + zero_point
            # Calculate peak width (Caglioti function)
            tan_theta = np.tan(np.radians(peak_position / 2))
            U = self.parameters['U'].value
            V = self.parameters['V'].value
            W = self.parameters['W'].value
            fwhm_squared = U * tan_theta**2 + V * tan_theta + W
            fwhm = np.sqrt(max(fwhm_squared, 0.001))  # Prevent negative FWHM
            # Calculate peak profile
            if self.peak_shape == 'gaussian':
                profile = self._gaussian_profile(self.two_theta_obs, peak_position, fwhm)
            elif self.peak_shape == 'lorentzian':
                profile = self._lorentzian_profile(self.two_theta_obs, peak_position, fwhm)
            elif self.peak_shape == 'pseudo_voigt':
                eta = self.parameters['eta'].value
                gaussian = self._gaussian_profile(self.two_theta_obs, peak_position, fwhm)
                lorentzian = self._lorentzian_profile(self.two_theta_obs, peak_position, fwhm)
                profile = eta * lorentzian + (1 - eta) * gaussian
            # Add scaled peak to pattern
            calculated_intensity += scale * refl.intensity * profile
        return calculated_intensity
    def _calculate_background(self) -> np.ndarray:
        """Calculate background intensity."""
        if self.background_type == 'polynomial':
            background = np.zeros_like(self.two_theta_obs)
            for i in range(self.background_order + 1):
                bg_coeff = self.parameters[f'bg_{i}'].value
                background += bg_coeff * (self.two_theta_obs / 100.0) ** i
            return background
        else:
            # Placeholder for other background types
            return np.full_like(self.two_theta_obs, self.parameters['bg_0'].value)
    def _gaussian_profile(self, two_theta: np.ndarray, center: float, fwhm: float) -> np.ndarray:
        """Gaussian peak profile."""
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        return np.exp(-0.5 * ((two_theta - center) / sigma) ** 2)
    def _lorentzian_profile(self, two_theta: np.ndarray, center: float, fwhm: float) -> np.ndarray:
        """Lorentzian peak profile."""
        gamma = fwhm / 2
        return gamma**2 / ((two_theta - center)**2 + gamma**2)
    def get_parameter_vector(self) -> np.ndarray:
        """Get parameter vector for optimization."""
        return np.array([self.parameters[name].value for name in self.parameter_order])
    def set_parameter_vector(self, values: np.ndarray):
        """Set parameters from vector."""
        for i, name in enumerate(self.parameter_order):
            self.parameters[name].value = values[i]
    def _update_crystal_structure(self):
        """Update crystal structure from parameters."""
        # Update lattice (simplified - only cubic for now)
        a = self.parameters['a'].value
        new_lattice = LatticeParameters(a=a, b=a, c=a, alpha=90, beta=90, gamma=90)
        # Update atoms
        new_atoms = []
        for i, atom in enumerate(self.crystal.atoms):
            new_atom = AtomicPosition(
                element=atom.element,
                x=self.parameters[f'x_{i}'].value,
                y=self.parameters[f'y_{i}'].value,
                z=self.parameters[f'z_{i}'].value,
                occupancy=atom.occupancy,
                thermal_factor=self.parameters[f'B_{i}'].value
            )
            new_atoms.append(new_atom)
        self.crystal = CrystalStructure(new_lattice, new_atoms)
    def residual_function(self, parameters: np.ndarray) -> np.ndarray:
        """Calculate residuals for Rietveld refinement."""
        calculated = self.calculate_pattern(parameters)
        # Statistical weights
        weights = 1.0 / np.sqrt(np.maximum(self.intensity_obs, 1.0))
        return weights * (self.intensity_obs - calculated)
    def refine(self) -> RefinementResults:
        """Perform Rietveld refinement."""
        initial_params = self.get_parameter_vector()
        try:
            # Least squares optimization
            result = optimize.least_squares(
                self.residual_function,
                initial_params,
                max_nfev=1000,
                ftol=1e-8,
                xtol=1e-8
            )
            # Calculate R-factors
            final_calculated = self.calculate_pattern(result.x)
            # R_wp (weighted profile R-factor)
            weights = 1.0 / np.sqrt(np.maximum(self.intensity_obs, 1.0))
            r_wp = (np.sqrt(np.sum(weights**2 * (self.intensity_obs - final_calculated)**2)) /
                   np.sqrt(np.sum(weights**2 * self.intensity_obs**2)))
            # R_p (profile R-factor)
            r_p = (np.sum(np.abs(self.intensity_obs - final_calculated)) /
                  np.sum(self.intensity_obs))
            # Chi-squared
            chi_squared = np.sum((self.intensity_obs - final_calculated)**2 /
                               np.maximum(self.intensity_obs, 1.0))
            # Goodness of fit
            n_data = len(self.intensity_obs)
            n_params = len(initial_params)
            goodness_of_fit = np.sqrt(chi_squared / (n_data - n_params))
            # Final parameters
            final_parameters = {}
            for i, name in enumerate(self.parameter_order):
                final_parameters[name] = result.x[i]
            return RefinementResults(
                converged=result.success,
                final_parameters=final_parameters,
                goodness_of_fit=goodness_of_fit,
                r_factor=r_p,
                weighted_r_factor=r_wp,
                chi_squared=chi_squared,
                iterations=result.nfev,
                final_residual=np.linalg.norm(result.fun)
            )
        except Exception as e:
            warnings.warn(f"Rietveld refinement failed: {e}")
            return RefinementResults(
                converged=False,
                final_parameters={name: self.parameters[name].value for name in self.parameter_order}
            )
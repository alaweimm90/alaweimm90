"""
Continuum Mechanics for Elastic Deformation
Advanced continuum mechanics formulations including finite strain analysis,
nonlinear elasticity, and hyperelastic material models.
"""
import numpy as np
from typing import Tuple, Optional, Callable, Union, List
from dataclasses import dataclass
from scipy import optimize, linalg
import warnings
from .stress_strain import StressTensor, StrainTensor, ElasticConstants
@dataclass
class DeformationGradient:
    """Deformation gradient tensor and related measures."""
    F: np.ndarray  # Deformation gradient tensor
    def __post_init__(self):
        """Validate deformation gradient."""
        if self.F.shape != (3, 3):
            raise ValueError("Deformation gradient must be 3x3 tensor")
        # Check for positive determinant (no material inversion)
        if np.linalg.det(self.F) <= 0:
            warnings.warn("Deformation gradient has non-positive determinant")
    @property
    def jacobian(self) -> float:
        """Jacobian of deformation (volume change ratio)."""
        return np.linalg.det(self.F)
    @property
    def right_cauchy_green(self) -> np.ndarray:
        """Right Cauchy-Green tensor C = F^T F."""
        return self.F.T @ self.F
    @property
    def left_cauchy_green(self) -> np.ndarray:
        """Left Cauchy-Green tensor B = F F^T."""
        return self.F @ self.F.T
    @property
    def green_lagrange_strain(self) -> np.ndarray:
        """Green-Lagrange strain tensor E = 1/2(C - I)."""
        C = self.right_cauchy_green
        I = np.eye(3)
        return 0.5 * (C - I)
    @property
    def almansi_strain(self) -> np.ndarray:
        """Almansi strain tensor e = 1/2(I - B^-1)."""
        B = self.left_cauchy_green
        B_inv = np.linalg.inv(B)
        I = np.eye(3)
        return 0.5 * (I - B_inv)
    def polar_decomposition(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Polar decomposition F = RU = VR.
        Returns:
            Tuple of (rotation_tensor_R, right_stretch_tensor_U)
        """
        # Compute right stretch tensor U
        C = self.right_cauchy_green
        eigenvals, eigenvecs = np.linalg.eigh(C)
        # Ensure positive eigenvalues
        eigenvals = np.maximum(eigenvals, 1e-12)
        sqrt_eigenvals = np.sqrt(eigenvals)
        U = eigenvecs @ np.diag(sqrt_eigenvals) @ eigenvecs.T
        # Compute rotation tensor R
        R = self.F @ np.linalg.inv(U)
        return R, U
    def principal_stretches(self) -> np.ndarray:
        """Calculate principal stretches (eigenvalues of U)."""
        C = self.right_cauchy_green
        eigenvals = np.linalg.eigvals(C)
        return np.sqrt(np.maximum(eigenvals, 1e-12))
    def strain_invariants(self) -> Tuple[float, float, float]:
        """
        Calculate strain invariants I1, I2, I3.
        Returns:
            Tuple of strain invariants based on C
        """
        C = self.right_cauchy_green
        I1 = np.trace(C)
        I2 = 0.5 * (np.trace(C)**2 - np.trace(C @ C))
        I3 = np.linalg.det(C)
        return I1, I2, I3
class FiniteStrainAnalysis:
    """
    Finite strain analysis for large deformations.
    Features:
    - Multiple strain measures
    - Stress-strain relationships for finite deformations
    - Material frame indifference
    - Objectivity analysis
    Examples:
        >>> analysis = FiniteStrainAnalysis()
        >>> F = create_simple_shear_gradient(gamma=0.5)
        >>> stress = analysis.second_piola_kirchhoff_stress(F, material_model)
    """
    def __init__(self):
        """Initialize finite strain analysis."""
        pass
    def create_deformation_gradient(self, displacement_gradient: np.ndarray) -> DeformationGradient:
        """
        Create deformation gradient from displacement gradient.
        Parameters:
            displacement_gradient: ∇u tensor
        Returns:
            DeformationGradient object
        """
        F = np.eye(3) + displacement_gradient
        return DeformationGradient(F)
    def logarithmic_strain(self, F: np.ndarray) -> np.ndarray:
        """
        Calculate logarithmic (true) strain tensor.
        Parameters:
            F: Deformation gradient
        Returns:
            Logarithmic strain tensor
        """
        deform_grad = DeformationGradient(F)
        C = deform_grad.right_cauchy_green
        # Compute log(C)/2 = log(U)
        eigenvals, eigenvecs = np.linalg.eigh(C)
        log_eigenvals = 0.5 * np.log(np.maximum(eigenvals, 1e-12))
        log_strain = eigenvecs @ np.diag(log_eigenvals) @ eigenvecs.T
        return log_strain
    def cauchy_stress_from_kirchhoff(self, kirchhoff_stress: np.ndarray,
                                   jacobian: float) -> np.ndarray:
        """Convert Kirchhoff stress to Cauchy stress."""
        return kirchhoff_stress / jacobian
    def second_piola_kirchhoff_from_cauchy(self, cauchy_stress: np.ndarray,
                                         F: np.ndarray) -> np.ndarray:
        """Convert Cauchy stress to 2nd Piola-Kirchhoff stress."""
        J = np.linalg.det(F)
        F_inv = np.linalg.inv(F)
        return J * F_inv @ cauchy_stress @ F_inv.T
    def push_forward_operation(self, material_tensor: np.ndarray,
                             F: np.ndarray) -> np.ndarray:
        """Push-forward operation for tensor quantities."""
        J = np.linalg.det(F)
        F_inv = np.linalg.inv(F)
        # For 2nd order tensors: σ = (1/J) F S F^T
        return (1/J) * F @ material_tensor @ F.T
    def pull_back_operation(self, spatial_tensor: np.ndarray,
                          F: np.ndarray) -> np.ndarray:
        """Pull-back operation for tensor quantities."""
        J = np.linalg.det(F)
        F_inv = np.linalg.inv(F)
        # For 2nd order tensors: S = J F^-1 σ F^-T
        return J * F_inv @ spatial_tensor @ F_inv.T
    def rate_of_deformation_tensor(self, velocity_gradient: np.ndarray) -> np.ndarray:
        """Calculate rate of deformation tensor D = sym(grad v)."""
        return 0.5 * (velocity_gradient + velocity_gradient.T)
    def spin_tensor(self, velocity_gradient: np.ndarray) -> np.ndarray:
        """Calculate spin tensor W = skew(grad v)."""
        return 0.5 * (velocity_gradient - velocity_gradient.T)
    def jaumann_stress_rate(self, stress_rate: np.ndarray, stress: np.ndarray,
                          spin: np.ndarray) -> np.ndarray:
        """Calculate Jaumann (corotational) stress rate."""
        return stress_rate - spin @ stress + stress @ spin
class HyperelasticMaterial:
    """
    Hyperelastic material models for finite strain analysis.
    Features:
    - Neo-Hookean model
    - Mooney-Rivlin model
    - Ogden model
    - Nearly incompressible formulations
    Examples:
        >>> material = HyperelasticMaterial.neo_hookean(mu=80e3, bulk_modulus=200e6)
        >>> stress = material.second_piola_kirchhoff_stress(F)
        >>> tangent = material.material_tangent(F)
    """
    def __init__(self, strain_energy_function: Callable,
                 stress_function: Callable, tangent_function: Callable):
        """
        Initialize hyperelastic material.
        Parameters:
            strain_energy_function: W(I1, I2, J) function
            stress_function: Function to compute 2nd P-K stress
            tangent_function: Function to compute material tangent
        """
        self.strain_energy = strain_energy_function
        self.stress_function = stress_function
        self.tangent_function = tangent_function
    @classmethod
    def neo_hookean(cls, mu: float, bulk_modulus: float) -> 'HyperelasticMaterial':
        """
        Create Neo-Hookean hyperelastic material.
        Parameters:
            mu: Shear modulus
            bulk_modulus: Bulk modulus
        Returns:
            HyperelasticMaterial object
        """
        def strain_energy(I1: float, I2: float, J: float) -> float:
            """Neo-Hookean strain energy: W = μ/2(I1 - 3) + K/2(J-1)²"""
            return mu/2 * (I1 - 3) + bulk_modulus/2 * (J - 1)**2
        def stress_function(F: np.ndarray) -> np.ndarray:
            """2nd Piola-Kirchhoff stress for Neo-Hookean material."""
            deform_grad = DeformationGradient(F)
            C = deform_grad.right_cauchy_green
            C_inv = np.linalg.inv(C)
            J = deform_grad.jacobian
            I = np.eye(3)
            # S = μ(I - C^-1) + K(J-1)J C^-1
            S = mu * (I - C_inv) + bulk_modulus * (J - 1) * J * C_inv
            return S
        def tangent_function(F: np.ndarray) -> np.ndarray:
            """Material tangent moduli."""
            deform_grad = DeformationGradient(F)
            C = deform_grad.right_cauchy_green
            C_inv = np.linalg.inv(C)
            J = deform_grad.jacobian
            # Simplified tangent - full implementation would be more complex
            # This is a placeholder for the 4th order elasticity tensor
            tangent = np.zeros((6, 6))
            # Diagonal terms (simplified)
            for i in range(3):
                tangent[i, i] = 2 * mu + bulk_modulus * J * (2*J - 1)
            for i in range(3, 6):
                tangent[i, i] = mu
            return tangent
        return cls(strain_energy, stress_function, tangent_function)
    @classmethod
    def mooney_rivlin(cls, c10: float, c01: float, bulk_modulus: float) -> 'HyperelasticMaterial':
        """
        Create Mooney-Rivlin hyperelastic material.
        Parameters:
            c10, c01: Mooney-Rivlin parameters
            bulk_modulus: Bulk modulus
        """
        def strain_energy(I1: float, I2: float, J: float) -> float:
            """Mooney-Rivlin strain energy."""
            return c10 * (I1 - 3) + c01 * (I2 - 3) + bulk_modulus/2 * (J - 1)**2
        def stress_function(F: np.ndarray) -> np.ndarray:
            """2nd Piola-Kirchhoff stress for Mooney-Rivlin material."""
            deform_grad = DeformationGradient(F)
            C = deform_grad.right_cauchy_green
            C_inv = np.linalg.inv(C)
            J = deform_grad.jacobian
            I = np.eye(3)
            I1, I2, I3 = deform_grad.strain_invariants()
            # S = 2c10(I - C^-1) + 2c01(I1 I - C - C^-1) + K(J-1)J C^-1
            S = (2 * c10 * (I - C_inv) +
                 2 * c01 * (I1 * I - C - C_inv) +
                 bulk_modulus * (J - 1) * J * C_inv)
            return S
        def tangent_function(F: np.ndarray) -> np.ndarray:
            """Material tangent moduli."""
            # Simplified implementation
            tangent = np.zeros((6, 6))
            mu_eff = 2 * (c10 + c01)
            for i in range(3):
                tangent[i, i] = 2 * mu_eff + bulk_modulus
            for i in range(3, 6):
                tangent[i, i] = mu_eff
            return tangent
        return cls(strain_energy, stress_function, tangent_function)
    @classmethod
    def ogden(cls, mu_params: List[float], alpha_params: List[float],
             bulk_modulus: float) -> 'HyperelasticMaterial':
        """
        Create Ogden hyperelastic material.
        Parameters:
            mu_params: List of μᵢ parameters
            alpha_params: List of αᵢ parameters
            bulk_modulus: Bulk modulus
        """
        if len(mu_params) != len(alpha_params):
            raise ValueError("mu_params and alpha_params must have same length")
        def strain_energy(stretches: np.ndarray, J: float) -> float:
            """Ogden strain energy."""
            W = 0
            for mu_i, alpha_i in zip(mu_params, alpha_params):
                W += mu_i/alpha_i * (stretches[0]**alpha_i + stretches[1]**alpha_i +
                                   stretches[2]**alpha_i - 3)
            W += bulk_modulus/2 * (J - 1)**2
            return W
        def stress_function(F: np.ndarray) -> np.ndarray:
            """2nd Piola-Kirchhoff stress for Ogden material."""
            deform_grad = DeformationGradient(F)
            stretches = deform_grad.principal_stretches()
            # Complex implementation - would need principal directions
            # Simplified version using Neo-Hookean approximation
            mu_eff = sum(mu_params)
            neo_hookean = HyperelasticMaterial.neo_hookean(mu_eff, bulk_modulus)
            return neo_hookean.stress_function(F)
        def tangent_function(F: np.ndarray) -> np.ndarray:
            """Material tangent moduli."""
            mu_eff = sum(mu_params)
            neo_hookean = HyperelasticMaterial.neo_hookean(mu_eff, bulk_modulus)
            return neo_hookean.tangent_function(F)
        return cls(strain_energy, stress_function, tangent_function)
    def second_piola_kirchhoff_stress(self, F: np.ndarray) -> np.ndarray:
        """Calculate 2nd Piola-Kirchhoff stress."""
        return self.stress_function(F)
    def cauchy_stress(self, F: np.ndarray) -> np.ndarray:
        """Calculate Cauchy stress."""
        S = self.second_piola_kirchhoff_stress(F)
        J = np.linalg.det(F)
        return (1/J) * F @ S @ F.T
    def material_tangent(self, F: np.ndarray) -> np.ndarray:
        """Calculate material tangent moduli."""
        return self.tangent_function(F)
class PlasticityModel:
    """
    Elastoplasticity models for permanent deformation.
    Features:
    - Von Mises plasticity
    - Kinematic hardening
    - Isotropic hardening
    - Rate-dependent plasticity
    Examples:
        >>> plasticity = PlasticityModel.von_mises(yield_stress=250e6, hardening_modulus=1e9)
        >>> stress, plastic_strain = plasticity.integrate_stress(strain_increment, state)
    """
    def __init__(self, yield_function: Callable, flow_rule: Callable,
                 hardening_rule: Callable):
        """
        Initialize plasticity model.
        Parameters:
            yield_function: f(stress, internal_variables)
            flow_rule: g(stress, internal_variables)
            hardening_rule: h(plastic_strain, internal_variables)
        """
        self.yield_function = yield_function
        self.flow_rule = flow_rule
        self.hardening_rule = hardening_rule
    @classmethod
    def von_mises(cls, initial_yield_stress: float, hardening_modulus: float = 0) -> 'PlasticityModel':
        """
        Create von Mises plasticity model.
        Parameters:
            initial_yield_stress: Initial yield stress
            hardening_modulus: Linear hardening modulus
        """
        def yield_function(stress: StressTensor, equivalent_plastic_strain: float) -> float:
            """Von Mises yield function."""
            von_mises_stress = stress.von_mises_stress()
            current_yield_stress = initial_yield_stress + hardening_modulus * equivalent_plastic_strain
            return von_mises_stress - current_yield_stress
        def flow_rule(stress: StressTensor, equivalent_plastic_strain: float) -> np.ndarray:
            """Associated flow rule (normal to yield surface)."""
            deviatoric = stress.deviatoric()
            von_mises = stress.von_mises_stress()
            if von_mises > 1e-12:
                return 1.5 * deviatoric.tensor / von_mises
            else:
                return np.zeros((3, 3))
        def hardening_rule(equivalent_plastic_strain: float) -> float:
            """Linear isotropic hardening."""
            return initial_yield_stress + hardening_modulus * equivalent_plastic_strain
        return cls(yield_function, flow_rule, hardening_rule)
    def check_yielding(self, stress: StressTensor, equivalent_plastic_strain: float) -> bool:
        """Check if material is yielding."""
        f = self.yield_function(stress, equivalent_plastic_strain)
        return f > 1e-12
    def plastic_strain_increment(self, stress: StressTensor,
                               equivalent_plastic_strain: float,
                               plastic_multiplier: float) -> np.ndarray:
        """Calculate plastic strain increment."""
        flow_direction = self.flow_rule(stress, equivalent_plastic_strain)
        return plastic_multiplier * flow_direction
    def return_mapping(self, trial_stress: StressTensor,
                      equivalent_plastic_strain: float,
                      elastic_modulus: float) -> Tuple[StressTensor, float, float]:
        """
        Radial return mapping algorithm.
        Parameters:
            trial_stress: Trial elastic stress
            equivalent_plastic_strain: Current equivalent plastic strain
            elastic_modulus: Elastic shear modulus
        Returns:
            Tuple of (corrected_stress, plastic_multiplier, new_equivalent_plastic_strain)
        """
        # Check yield condition
        f_trial = self.yield_function(trial_stress, equivalent_plastic_strain)
        if f_trial <= 1e-12:
            # Elastic step
            return trial_stress, 0.0, equivalent_plastic_strain
        # Plastic step - radial return
        von_mises_trial = trial_stress.von_mises_stress()
        current_yield_stress = self.hardening_rule(equivalent_plastic_strain)
        # Plastic multiplier
        plastic_multiplier = (von_mises_trial - current_yield_stress) / (3 * elastic_modulus)
        # Corrected stress
        stress_correction = 3 * elastic_modulus * plastic_multiplier
        corrected_von_mises = von_mises_trial - stress_correction
        if von_mises_trial > 1e-12:
            stress_factor = corrected_von_mises / von_mises_trial
            corrected_stress_tensor = trial_stress * stress_factor
        else:
            corrected_stress_tensor = trial_stress
        # Update equivalent plastic strain
        new_equivalent_plastic_strain = equivalent_plastic_strain + plastic_multiplier
        return corrected_stress_tensor, plastic_multiplier, new_equivalent_plastic_strain
class ViscoelasticModel:
    """
    Viscoelastic material models for time-dependent behavior.
    Features:
    - Maxwell model
    - Kelvin-Voigt model
    - Standard linear solid
    - Generalized models with multiple relaxation times
    """
    def __init__(self, model_type: str, parameters: dict):
        """
        Initialize viscoelastic model.
        Parameters:
            model_type: 'maxwell', 'kelvin_voigt', or 'standard_linear_solid'
            parameters: Model parameters
        """
        self.model_type = model_type
        self.parameters = parameters
        if model_type == 'maxwell':
            self.E = parameters['elastic_modulus']
            self.eta = parameters['viscosity']
            self.relaxation_time = self.eta / self.E
        elif model_type == 'kelvin_voigt':
            self.E = parameters['elastic_modulus']
            self.eta = parameters['viscosity']
        elif model_type == 'standard_linear_solid':
            self.E1 = parameters['E1']  # Instantaneous modulus
            self.E2 = parameters['E2']  # Delayed modulus
            self.eta = parameters['eta']  # Viscosity
            self.relaxation_time = self.eta / self.E2
    def relaxation_modulus(self, t: np.ndarray) -> np.ndarray:
        """Calculate relaxation modulus E(t)."""
        if self.model_type == 'maxwell':
            return self.E * np.exp(-t / self.relaxation_time)
        elif self.model_type == 'standard_linear_solid':
            E_inf = self.E1 * self.E2 / (self.E1 + self.E2)
            E_0 = self.E1
            return E_inf + (E_0 - E_inf) * np.exp(-t / self.relaxation_time)
        else:
            raise NotImplementedError(f"Relaxation modulus for {self.model_type} not implemented")
    def creep_compliance(self, t: np.ndarray) -> np.ndarray:
        """Calculate creep compliance J(t) = 1/E(t)."""
        if self.model_type == 'maxwell':
            return 1/self.E + t/self.eta
        elif self.model_type == 'kelvin_voigt':
            return (1/self.E) * (1 - np.exp(-self.E * t / self.eta))
        elif self.model_type == 'standard_linear_solid':
            J_0 = 1 / self.E1
            J_inf = (self.E1 + self.E2) / (self.E1 * self.E2)
            return J_inf - (J_inf - J_0) * np.exp(-t / self.relaxation_time)
        else:
            raise NotImplementedError(f"Creep compliance for {self.model_type} not implemented")
    def stress_response(self, strain_history: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Calculate stress response to given strain history using convolution.
        Parameters:
            strain_history: Strain as function of time
            t: Time array
        Returns:
            Stress response
        """
        dt = t[1] - t[0]  # Assume uniform time step
        relaxation_mod = self.relaxation_modulus(t)
        # Convolution integral
        stress = np.zeros_like(t)
        for i in range(len(t)):
            for j in range(i + 1):
                if j == 0:
                    strain_rate = strain_history[0] / dt
                else:
                    strain_rate = (strain_history[j] - strain_history[j-1]) / dt
                stress[i] += relaxation_mod[i-j] * strain_rate * dt
        return stress
# continuum_mechanics
**Module:** `Python/Elasticity/core/continuum_mechanics.py`
## Overview
Continuum Mechanics for Elastic Deformation
Advanced continuum mechanics formulations including finite strain analysis,
nonlinear elasticity, and hyperelastic material models.
## Classes
### `DeformationGradient`
Deformation gradient tensor and related measures.
#### Methods
##### `__post_init__(self)`
Validate deformation gradient.
**Source:** [Line 22](Python/Elasticity/core/continuum_mechanics.py#L22)
##### `jacobian(self)`
Jacobian of deformation (volume change ratio).
**Source:** [Line 32](Python/Elasticity/core/continuum_mechanics.py#L32)
##### `right_cauchy_green(self)`
Right Cauchy-Green tensor C = F^T F.
**Source:** [Line 37](Python/Elasticity/core/continuum_mechanics.py#L37)
##### `left_cauchy_green(self)`
Left Cauchy-Green tensor B = F F^T.
**Source:** [Line 42](Python/Elasticity/core/continuum_mechanics.py#L42)
##### `green_lagrange_strain(self)`
Green-Lagrange strain tensor E = 1/2(C - I).
**Source:** [Line 47](Python/Elasticity/core/continuum_mechanics.py#L47)
##### `almansi_strain(self)`
Almansi strain tensor e = 1/2(I - B^-1).
**Source:** [Line 54](Python/Elasticity/core/continuum_mechanics.py#L54)
##### `polar_decomposition(self)`
Polar decomposition F = RU = VR.
Returns:
Tuple of (rotation_tensor_R, right_stretch_tensor_U)
**Source:** [Line 61](Python/Elasticity/core/continuum_mechanics.py#L61)
##### `principal_stretches(self)`
Calculate principal stretches (eigenvalues of U).
**Source:** [Line 83](Python/Elasticity/core/continuum_mechanics.py#L83)
##### `strain_invariants(self)`
Calculate strain invariants I1, I2, I3.
Returns:
Tuple of strain invariants based on C
**Source:** [Line 89](Python/Elasticity/core/continuum_mechanics.py#L89)
**Class Source:** [Line 18](Python/Elasticity/core/continuum_mechanics.py#L18)
### `FiniteStrainAnalysis`
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
#### Methods
##### `__init__(self)`
Initialize finite strain analysis.
**Source:** [Line 121](Python/Elasticity/core/continuum_mechanics.py#L121)
##### `create_deformation_gradient(self, displacement_gradient)`
Create deformation gradient from displacement gradient.
Parameters:
displacement_gradient: ∇u tensor
Returns:
DeformationGradient object
**Source:** [Line 125](Python/Elasticity/core/continuum_mechanics.py#L125)
##### `logarithmic_strain(self, F)`
Calculate logarithmic (true) strain tensor.
Parameters:
F: Deformation gradient
Returns:
Logarithmic strain tensor
**Source:** [Line 138](Python/Elasticity/core/continuum_mechanics.py#L138)
##### `cauchy_stress_from_kirchhoff(self, kirchhoff_stress, jacobian)`
Convert Kirchhoff stress to Cauchy stress.
**Source:** [Line 158](Python/Elasticity/core/continuum_mechanics.py#L158)
##### `second_piola_kirchhoff_from_cauchy(self, cauchy_stress, F)`
Convert Cauchy stress to 2nd Piola-Kirchhoff stress.
**Source:** [Line 163](Python/Elasticity/core/continuum_mechanics.py#L163)
##### `push_forward_operation(self, material_tensor, F)`
Push-forward operation for tensor quantities.
**Source:** [Line 170](Python/Elasticity/core/continuum_mechanics.py#L170)
##### `pull_back_operation(self, spatial_tensor, F)`
Pull-back operation for tensor quantities.
**Source:** [Line 179](Python/Elasticity/core/continuum_mechanics.py#L179)
##### `rate_of_deformation_tensor(self, velocity_gradient)`
Calculate rate of deformation tensor D = sym(grad v).
**Source:** [Line 188](Python/Elasticity/core/continuum_mechanics.py#L188)
##### `spin_tensor(self, velocity_gradient)`
Calculate spin tensor W = skew(grad v).
**Source:** [Line 192](Python/Elasticity/core/continuum_mechanics.py#L192)
##### `jaumann_stress_rate(self, stress_rate, stress, spin)`
Calculate Jaumann (corotational) stress rate.
**Source:** [Line 196](Python/Elasticity/core/continuum_mechanics.py#L196)
**Class Source:** [Line 105](Python/Elasticity/core/continuum_mechanics.py#L105)
### `HyperelasticMaterial`
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
#### Methods
##### `__init__(self, strain_energy_function, stress_function, tangent_function)`
Initialize hyperelastic material.
Parameters:
strain_energy_function: W(I1, I2, J) function
stress_function: Function to compute 2nd P-K stress
tangent_function: Function to compute material tangent
**Source:** [Line 218](Python/Elasticity/core/continuum_mechanics.py#L218)
##### `neo_hookean(cls, mu, bulk_modulus)`
Create Neo-Hookean hyperelastic material.
Parameters:
mu: Shear modulus
bulk_modulus: Bulk modulus
Returns:
HyperelasticMaterial object
**Source:** [Line 233](Python/Elasticity/core/continuum_mechanics.py#L233)
##### `mooney_rivlin(cls, c10, c01, bulk_modulus)`
Create Mooney-Rivlin hyperelastic material.
Parameters:
c10, c01: Mooney-Rivlin parameters
bulk_modulus: Bulk modulus
**Source:** [Line 283](Python/Elasticity/core/continuum_mechanics.py#L283)
##### `ogden(cls, mu_params, alpha_params, bulk_modulus)`
Create Ogden hyperelastic material.
Parameters:
mu_params: List of μᵢ parameters
alpha_params: List of αᵢ parameters
bulk_modulus: Bulk modulus
**Source:** [Line 326](Python/Elasticity/core/continuum_mechanics.py#L326)
##### `second_piola_kirchhoff_stress(self, F)`
Calculate 2nd Piola-Kirchhoff stress.
**Source:** [Line 367](Python/Elasticity/core/continuum_mechanics.py#L367)
##### `cauchy_stress(self, F)`
Calculate Cauchy stress.
**Source:** [Line 371](Python/Elasticity/core/continuum_mechanics.py#L371)
##### `material_tangent(self, F)`
Calculate material tangent moduli.
**Source:** [Line 377](Python/Elasticity/core/continuum_mechanics.py#L377)
**Class Source:** [Line 202](Python/Elasticity/core/continuum_mechanics.py#L202)
### `PlasticityModel`
Elastoplasticity models for permanent deformation.
Features:
- Von Mises plasticity
- Kinematic hardening
- Isotropic hardening
- Rate-dependent plasticity
Examples:
>>> plasticity = PlasticityModel.von_mises(yield_stress=250e6, hardening_modulus=1e9)
>>> stress, plastic_strain = plasticity.integrate_stress(strain_increment, state)
#### Methods
##### `__init__(self, yield_function, flow_rule, hardening_rule)`
Initialize plasticity model.
Parameters:
yield_function: f(stress, internal_variables)
flow_rule: g(stress, internal_variables)
hardening_rule: h(plastic_strain, internal_variables)
**Source:** [Line 397](Python/Elasticity/core/continuum_mechanics.py#L397)
##### `von_mises(cls, initial_yield_stress, hardening_modulus)`
Create von Mises plasticity model.
Parameters:
initial_yield_stress: Initial yield stress
hardening_modulus: Linear hardening modulus
**Source:** [Line 412](Python/Elasticity/core/continuum_mechanics.py#L412)
##### `check_yielding(self, stress, equivalent_plastic_strain)`
Check if material is yielding.
**Source:** [Line 442](Python/Elasticity/core/continuum_mechanics.py#L442)
##### `plastic_strain_increment(self, stress, equivalent_plastic_strain, plastic_multiplier)`
Calculate plastic strain increment.
**Source:** [Line 447](Python/Elasticity/core/continuum_mechanics.py#L447)
##### `return_mapping(self, trial_stress, equivalent_plastic_strain, elastic_modulus)`
Radial return mapping algorithm.
Parameters:
trial_stress: Trial elastic stress
equivalent_plastic_strain: Current equivalent plastic strain
elastic_modulus: Elastic shear modulus
Returns:
Tuple of (corrected_stress, plastic_multiplier, new_equivalent_plastic_strain)
**Source:** [Line 454](Python/Elasticity/core/continuum_mechanics.py#L454)
**Class Source:** [Line 382](Python/Elasticity/core/continuum_mechanics.py#L382)
### `ViscoelasticModel`
Viscoelastic material models for time-dependent behavior.
Features:
- Maxwell model
- Kelvin-Voigt model
- Standard linear solid
- Generalized models with multiple relaxation times
#### Methods
##### `__init__(self, model_type, parameters)`
Initialize viscoelastic model.
Parameters:
model_type: 'maxwell', 'kelvin_voigt', or 'standard_linear_solid'
parameters: Model parameters
**Source:** [Line 509](Python/Elasticity/core/continuum_mechanics.py#L509)
##### `relaxation_modulus(self, t)`
Calculate relaxation modulus E(t).
**Source:** [Line 535](Python/Elasticity/core/continuum_mechanics.py#L535)
##### `creep_compliance(self, t)`
Calculate creep compliance J(t) = 1/E(t).
**Source:** [Line 548](Python/Elasticity/core/continuum_mechanics.py#L548)
##### `stress_response(self, strain_history, t)`
Calculate stress response to given strain history using convolution.
Parameters:
strain_history: Strain as function of time
t: Time array
Returns:
Stress response
**Source:** [Line 564](Python/Elasticity/core/continuum_mechanics.py#L564)
**Class Source:** [Line 498](Python/Elasticity/core/continuum_mechanics.py#L498)
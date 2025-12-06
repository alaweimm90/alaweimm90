# stress_strain
**Module:** `Python/Elasticity/core/stress_strain.py`
## Overview
Stress-Strain Analysis for Elastic Materials
Comprehensive stress and strain tensor operations, elastic moduli calculations,
and constitutive relationship modeling for isotropic and anisotropic materials.
## Functions
### `create_material_database()`
Create database of common engineering materials.
**Source:** [Line 635](Python/Elasticity/core/stress_strain.py#L635)
## Classes
### `ElasticConstants`
Elastic constants for materials.
#### Methods
##### `__post_init__(self)`
Calculate derived elastic constants.
**Source:** [Line 26](Python/Elasticity/core/stress_strain.py#L26)
**Class Source:** [Line 17](Python/Elasticity/core/stress_strain.py#L17)
### `StressTensor`
3D stress tensor representation and operations.
Features:
- Stress tensor creation and manipulation
- Principal stress calculation
- Invariant computation
- Coordinate transformations
- Von Mises stress calculation
Examples:
>>> stress = StressTensor(sigma_xx=100e6, sigma_yy=50e6, sigma_zz=25e6)
>>> principal_stresses = stress.principal_stresses()
>>> von_mises = stress.von_mises_stress()
#### Methods
##### `__init__(self, sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz)`
Initialize stress tensor.
Parameters:
sigma_xx, sigma_yy, sigma_zz: Normal stress components (Pa)
sigma_xy, sigma_xz, sigma_yz: Shear stress components (Pa)
**Source:** [Line 67](Python/Elasticity/core/stress_strain.py#L67)
##### `from_array(cls, tensor)`
Create stress tensor from 3x3 array.
**Source:** [Line 83](Python/Elasticity/core/stress_strain.py#L83)
##### `from_voigt(cls, voigt)`
Create stress tensor from Voigt notation [σxx, σyy, σzz, σyz, σxz, σxy].
**Source:** [Line 93](Python/Elasticity/core/stress_strain.py#L93)
##### `to_voigt(self)`
Convert to Voigt notation.
**Source:** [Line 103](Python/Elasticity/core/stress_strain.py#L103)
##### `principal_stresses(self)`
Calculate principal stresses and directions.
Returns:
Tuple of (principal_values, principal_directions)
**Source:** [Line 114](Python/Elasticity/core/stress_strain.py#L114)
##### `invariants(self)`
Calculate stress tensor invariants.
Returns:
Tuple of (I1, I2, I3) stress invariants
**Source:** [Line 130](Python/Elasticity/core/stress_strain.py#L130)
##### `deviatoric(self)`
Calculate deviatoric stress tensor.
**Source:** [Line 145](Python/Elasticity/core/stress_strain.py#L145)
##### `von_mises_stress(self)`
Calculate von Mises equivalent stress.
**Source:** [Line 151](Python/Elasticity/core/stress_strain.py#L151)
##### `maximum_shear_stress(self)`
Calculate maximum shear stress.
**Source:** [Line 156](Python/Elasticity/core/stress_strain.py#L156)
##### `octahedral_shear_stress(self)`
Calculate octahedral shear stress.
**Source:** [Line 161](Python/Elasticity/core/stress_strain.py#L161)
##### `transform(self, rotation_matrix)`
Transform stress tensor to new coordinate system.
**Source:** [Line 167](Python/Elasticity/core/stress_strain.py#L167)
##### `__add__(self, other)`
Add stress tensors.
**Source:** [Line 176](Python/Elasticity/core/stress_strain.py#L176)
##### `__sub__(self, other)`
Subtract stress tensors.
**Source:** [Line 180](Python/Elasticity/core/stress_strain.py#L180)
##### `__mul__(self, scalar)`
Multiply stress tensor by scalar.
**Source:** [Line 184](Python/Elasticity/core/stress_strain.py#L184)
##### `__str__(self)`
String representation.
**Source:** [Line 188](Python/Elasticity/core/stress_strain.py#L188)
**Class Source:** [Line 50](Python/Elasticity/core/stress_strain.py#L50)
### `StrainTensor`
3D strain tensor representation and operations.
Features:
- Strain tensor creation and manipulation
- Principal strain calculation
- Strain invariants
- Compatibility checks
- Finite vs infinitesimal strain
Examples:
>>> strain = StrainTensor(epsilon_xx=0.001, epsilon_yy=0.0005)
>>> principal_strains = strain.principal_strains()
>>> volumetric_strain = strain.volumetric_strain()
#### Methods
##### `__init__(self, epsilon_xx, epsilon_yy, epsilon_zz, gamma_xy, gamma_xz, gamma_yz, engineering_strain)`
Initialize strain tensor.
Parameters:
epsilon_xx, epsilon_yy, epsilon_zz: Normal strain components
gamma_xy, gamma_xz, gamma_yz: Shear strain components
engineering_strain: If True, use engineering shear strains (γ = 2ε)
**Source:** [Line 210](Python/Elasticity/core/stress_strain.py#L210)
##### `from_array(cls, tensor)`
Create strain tensor from 3x3 array.
**Source:** [Line 231](Python/Elasticity/core/stress_strain.py#L231)
##### `from_voigt(cls, voigt, engineering_strain)`
Create strain tensor from Voigt notation.
**Source:** [Line 241](Python/Elasticity/core/stress_strain.py#L241)
##### `to_voigt(self, engineering_strain)`
Convert to Voigt notation.
**Source:** [Line 252](Python/Elasticity/core/stress_strain.py#L252)
##### `principal_strains(self)`
Calculate principal strains and directions.
**Source:** [Line 265](Python/Elasticity/core/stress_strain.py#L265)
##### `volumetric_strain(self)`
Calculate volumetric strain (trace of strain tensor).
**Source:** [Line 276](Python/Elasticity/core/stress_strain.py#L276)
##### `deviatoric(self)`
Calculate deviatoric strain tensor.
**Source:** [Line 280](Python/Elasticity/core/stress_strain.py#L280)
##### `equivalent_strain(self)`
Calculate equivalent strain (von Mises equivalent).
**Source:** [Line 286](Python/Elasticity/core/stress_strain.py#L286)
##### `maximum_shear_strain(self)`
Calculate maximum shear strain.
**Source:** [Line 291](Python/Elasticity/core/stress_strain.py#L291)
##### `transform(self, rotation_matrix)`
Transform strain tensor to new coordinate system.
**Source:** [Line 296](Python/Elasticity/core/stress_strain.py#L296)
##### `__add__(self, other)`
Add strain tensors.
**Source:** [Line 305](Python/Elasticity/core/stress_strain.py#L305)
##### `__sub__(self, other)`
Subtract strain tensors.
**Source:** [Line 309](Python/Elasticity/core/stress_strain.py#L309)
##### `__mul__(self, scalar)`
Multiply strain tensor by scalar.
**Source:** [Line 313](Python/Elasticity/core/stress_strain.py#L313)
**Class Source:** [Line 193](Python/Elasticity/core/stress_strain.py#L193)
### `IsotropicElasticity`
Isotropic linear elasticity relationships.
Features:
- Stress-strain constitutive relationships
- Compliance and stiffness matrices
- Elastic wave velocities
- Energy calculations
Examples:
>>> elasticity = IsotropicElasticity(youngs_modulus=200e9, poissons_ratio=0.3)
>>> stress = elasticity.stress_from_strain(strain_tensor)
>>> strain = elasticity.strain_from_stress(stress_tensor)
#### Methods
##### `__init__(self, youngs_modulus, poissons_ratio, density)`
Initialize isotropic elastic material.
Parameters:
youngs_modulus: Young's modulus (Pa)
poissons_ratio: Poisson's ratio
density: Material density (kg/m³)
**Source:** [Line 334](Python/Elasticity/core/stress_strain.py#L334)
##### `_compute_stiffness_matrix(self)`
Compute 6x6 stiffness matrix in Voigt notation.
**Source:** [Line 350](Python/Elasticity/core/stress_strain.py#L350)
##### `_compute_compliance_matrix(self)`
Compute 6x6 compliance matrix in Voigt notation.
**Source:** [Line 369](Python/Elasticity/core/stress_strain.py#L369)
##### `stiffness_matrix(self)`
Get stiffness matrix.
**Source:** [Line 387](Python/Elasticity/core/stress_strain.py#L387)
##### `compliance_matrix(self)`
Get compliance matrix.
**Source:** [Line 392](Python/Elasticity/core/stress_strain.py#L392)
##### `stress_from_strain(self, strain)`
Calculate stress from strain using Hooke's law.
**Source:** [Line 396](Python/Elasticity/core/stress_strain.py#L396)
##### `strain_from_stress(self, stress)`
Calculate strain from stress using compliance.
**Source:** [Line 402](Python/Elasticity/core/stress_strain.py#L402)
##### `elastic_wave_velocities(self)`
Calculate elastic wave velocities.
Returns:
Tuple of (longitudinal_velocity, transverse_velocity) in m/s
**Source:** [Line 408](Python/Elasticity/core/stress_strain.py#L408)
##### `elastic_energy_density(self, stress, strain)`
Calculate elastic energy density.
**Source:** [Line 427](Python/Elasticity/core/stress_strain.py#L427)
##### `bulk_modulus_from_stress(self, hydrostatic_stress)`
Calculate bulk modulus from hydrostatic stress state.
**Source:** [Line 433](Python/Elasticity/core/stress_strain.py#L433)
##### `shear_modulus_from_stress(self, shear_stress, shear_strain)`
Calculate shear modulus from pure shear state.
**Source:** [Line 437](Python/Elasticity/core/stress_strain.py#L437)
**Class Source:** [Line 318](Python/Elasticity/core/stress_strain.py#L318)
### `AnisotropicElasticity`
Anisotropic linear elasticity for general materials.
Features:
- General 6x6 stiffness matrix handling
- Orthotropic and transversely isotropic materials
- Coordinate transformations
- Engineering constants calculation
Examples:
>>> # Orthotropic material
>>> stiffness = create_orthotropic_stiffness(Ex, Ey, Ez, Gxy, Gxz, Gyz, nuxy, nuxz, nuyz)
>>> elasticity = AnisotropicElasticity(stiffness)
>>> stress = elasticity.stress_from_strain(strain)
#### Methods
##### `__init__(self, stiffness_matrix, density)`
Initialize anisotropic elastic material.
Parameters:
stiffness_matrix: 6x6 stiffness matrix in Voigt notation
density: Material density (kg/m³)
**Source:** [Line 463](Python/Elasticity/core/stress_strain.py#L463)
##### `orthotropic(cls, Ex, Ey, Ez, Gxy, Gxz, Gyz, nuxy, nuxz, nuyz, density)`
Create orthotropic material from engineering constants.
Parameters:
Ex, Ey, Ez: Young's moduli in x, y, z directions
Gxy, Gxz, Gyz: Shear moduli in xy, xz, yz planes
nuxy, nuxz, nuyz: Poisson's ratios
density: Material density
**Source:** [Line 488](Python/Elasticity/core/stress_strain.py#L488)
##### `transversely_isotropic(cls, E1, E2, G12, G23, nu12, nu23, density)`
Create transversely isotropic material.
Parameters:
E1: Young's modulus in fiber direction
E2: Young's modulus transverse to fiber
G12: In-plane shear modulus
G23: Out-of-plane shear modulus
nu12: In-plane Poisson's ratio
nu23: Out-of-plane Poisson's ratio
density: Material density
**Source:** [Line 526](Python/Elasticity/core/stress_strain.py#L526)
##### `stiffness_matrix(self)`
Get stiffness matrix.
**Source:** [Line 548](Python/Elasticity/core/stress_strain.py#L548)
##### `compliance_matrix(self)`
Get compliance matrix.
**Source:** [Line 553](Python/Elasticity/core/stress_strain.py#L553)
##### `stress_from_strain(self, strain)`
Calculate stress from strain.
**Source:** [Line 557](Python/Elasticity/core/stress_strain.py#L557)
##### `strain_from_stress(self, stress)`
Calculate strain from stress.
**Source:** [Line 563](Python/Elasticity/core/stress_strain.py#L563)
##### `transform_stiffness(self, rotation_matrix)`
Transform stiffness matrix to new coordinate system.
**Source:** [Line 569](Python/Elasticity/core/stress_strain.py#L569)
##### `_create_voigt_transformation_matrix(self, R)`
Create 6x6 transformation matrix for Voigt notation.
**Source:** [Line 582](Python/Elasticity/core/stress_strain.py#L582)
##### `elastic_energy_density(self, stress, strain)`
Calculate elastic energy density.
**Source:** [Line 628](Python/Elasticity/core/stress_strain.py#L628)
**Class Source:** [Line 446](Python/Elasticity/core/stress_strain.py#L446)
# structure_refinement
**Module:** `Python/Crystallography/core/structure_refinement.py`
## Overview
Structure Refinement Methods
Advanced crystallographic structure refinement algorithms including
Rietveld method, least squares refinement, and maximum likelihood approaches.
## Classes
### `RefinementParameter`
Refinement parameter definition.
#### Methods
##### `__post_init__(self)`
Validate parameter bounds.
**Source:** [Line 29](Python/Crystallography/core/structure_refinement.py#L29)
**Class Source:** [Line 20](Python/Crystallography/core/structure_refinement.py#L20)
### `RefinementResults`
Results from structure refinement.
**Class Source:** [Line 37](Python/Crystallography/core/structure_refinement.py#L37)
### `LeastSquaresRefinement`
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
#### Methods
##### `__init__(self, crystal, observed_intensities, reflection_data)`
Initialize least squares refinement.
Parameters:
crystal: Initial crystal structure
observed_intensities: Observed structure factor magnitudes
reflection_data: List of reflection information
**Source:** [Line 67](Python/Crystallography/core/structure_refinement.py#L67)
##### `_setup_parameters(self)`
Setup refinement parameters from crystal structure.
**Source:** [Line 92](Python/Crystallography/core/structure_refinement.py#L92)
##### `get_parameter_vector(self)`
Get current parameter values as vector.
**Source:** [Line 116](Python/Crystallography/core/structure_refinement.py#L116)
##### `set_parameter_vector(self, values)`
Set parameter values from vector.
**Source:** [Line 120](Python/Crystallography/core/structure_refinement.py#L120)
##### `_update_crystal_structure(self)`
Update crystal structure from current parameters.
**Source:** [Line 128](Python/Crystallography/core/structure_refinement.py#L128)
##### `calculate_structure_factors(self)`
Calculate structure factors for all reflections.
**Source:** [Line 155](Python/Crystallography/core/structure_refinement.py#L155)
##### `residual_function(self, parameters)`
Calculate residual vector for least squares.
**Source:** [Line 168](Python/Crystallography/core/structure_refinement.py#L168)
##### `jacobian_function(self, parameters)`
Calculate Jacobian matrix for least squares.
**Source:** [Line 182](Python/Crystallography/core/structure_refinement.py#L182)
##### `refine(self)`
Perform least squares refinement.
**Source:** [Line 213](Python/Crystallography/core/structure_refinement.py#L213)
**Class Source:** [Line 51](Python/Crystallography/core/structure_refinement.py#L51)
### `RietveldRefinement`
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
#### Methods
##### `__init__(self, crystal, two_theta_observed, intensity_observed, wavelength)`
Initialize Rietveld refinement.
Parameters:
crystal: Crystal structure
two_theta_observed: Observed 2θ values
intensity_observed: Observed intensities
wavelength: X-ray wavelength
**Source:** [Line 313](Python/Crystallography/core/structure_refinement.py#L313)
##### `_setup_rietveld_parameters(self)`
Setup Rietveld refinement parameters.
**Source:** [Line 344](Python/Crystallography/core/structure_refinement.py#L344)
##### `calculate_pattern(self, parameters)`
Calculate powder diffraction pattern.
**Source:** [Line 379](Python/Crystallography/core/structure_refinement.py#L379)
##### `_calculate_background(self)`
Calculate background intensity.
**Source:** [Line 433](Python/Crystallography/core/structure_refinement.py#L433)
##### `_gaussian_profile(self, two_theta, center, fwhm)`
Gaussian peak profile.
**Source:** [Line 445](Python/Crystallography/core/structure_refinement.py#L445)
##### `_lorentzian_profile(self, two_theta, center, fwhm)`
Lorentzian peak profile.
**Source:** [Line 450](Python/Crystallography/core/structure_refinement.py#L450)
##### `get_parameter_vector(self)`
Get parameter vector for optimization.
**Source:** [Line 455](Python/Crystallography/core/structure_refinement.py#L455)
##### `set_parameter_vector(self, values)`
Set parameters from vector.
**Source:** [Line 459](Python/Crystallography/core/structure_refinement.py#L459)
##### `_update_crystal_structure(self)`
Update crystal structure from parameters.
**Source:** [Line 464](Python/Crystallography/core/structure_refinement.py#L464)
##### `residual_function(self, parameters)`
Calculate residuals for Rietveld refinement.
**Source:** [Line 485](Python/Crystallography/core/structure_refinement.py#L485)
##### `refine(self)`
Perform Rietveld refinement.
**Source:** [Line 494](Python/Crystallography/core/structure_refinement.py#L494)
**Class Source:** [Line 296](Python/Crystallography/core/structure_refinement.py#L296)
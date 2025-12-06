# crystal_structure
**Module:** `Python/Crystallography/core/crystal_structure.py`
## Overview
Crystal Structure Analysis
Comprehensive crystallographic structure representation and analysis tools.
Includes lattice parameter calculations, unit cell operations, and
crystallographic coordinate transformations.
## Classes
### `LatticeParameters`
Crystal lattice parameters container.
#### Methods
##### `__post_init__(self)`
Validate lattice parameters.
**Source:** [Line 25](Python/Crystallography/core/crystal_structure.py#L25)
##### `alpha_rad(self)`
Alpha angle in radians.
**Source:** [Line 34](Python/Crystallography/core/crystal_structure.py#L34)
##### `beta_rad(self)`
Beta angle in radians.
**Source:** [Line 39](Python/Crystallography/core/crystal_structure.py#L39)
##### `gamma_rad(self)`
Gamma angle in radians.
**Source:** [Line 44](Python/Crystallography/core/crystal_structure.py#L44)
**Class Source:** [Line 16](Python/Crystallography/core/crystal_structure.py#L16)
### `AtomicPosition`
Atomic position in crystal structure.
#### Methods
##### `__post_init__(self)`
Validate atomic position parameters.
**Source:** [Line 59](Python/Crystallography/core/crystal_structure.py#L59)
**Class Source:** [Line 50](Python/Crystallography/core/crystal_structure.py#L50)
### `CrystalStructure`
Crystal structure representation and analysis.
Features:
- Lattice parameter calculations
- Unit cell volume and density
- Coordinate transformations (fractional ↔ Cartesian)
- Interatomic distance calculations
- Miller indices operations
- Structure factor calculations
Examples:
>>> lattice = LatticeParameters(a=5.0, b=5.0, c=5.0, alpha=90, beta=90, gamma=90)
>>> atoms = [AtomicPosition('Si', 0.0, 0.0, 0.0), AtomicPosition('Si', 0.5, 0.5, 0.5)]
>>> crystal = CrystalStructure(lattice, atoms)
>>> print(f"Unit cell volume: {crystal.unit_cell_volume():.2f} Å³")
#### Methods
##### `__init__(self, lattice_parameters, atoms)`
Initialize crystal structure.
Parameters:
lattice_parameters: Crystal lattice parameters
atoms: List of atomic positions
**Source:** [Line 87](Python/Crystallography/core/crystal_structure.py#L87)
##### `_compute_matrices(self)`
Compute fundamental crystallographic matrices.
**Source:** [Line 105](Python/Crystallography/core/crystal_structure.py#L105)
##### `direct_matrix(self)`
Direct lattice matrix (3x3).
**Source:** [Line 136](Python/Crystallography/core/crystal_structure.py#L136)
##### `reciprocal_matrix(self)`
Reciprocal lattice matrix (3x3).
**Source:** [Line 141](Python/Crystallography/core/crystal_structure.py#L141)
##### `metric_tensor(self)`
Metric tensor (3x3).
**Source:** [Line 146](Python/Crystallography/core/crystal_structure.py#L146)
##### `unit_cell_volume(self)`
Calculate unit cell volume in Å³.
**Source:** [Line 150](Python/Crystallography/core/crystal_structure.py#L150)
##### `density(self, molecular_weight, z)`
Calculate crystal density.
Parameters:
molecular_weight: Molecular weight (g/mol)
z: Number of formula units per unit cell
Returns:
Density in g/cm³
**Source:** [Line 154](Python/Crystallography/core/crystal_structure.py#L154)
##### `fractional_to_cartesian(self, fractional_coords)`
Convert fractional coordinates to Cartesian coordinates.
Parameters:
fractional_coords: Fractional coordinates (Nx3 or 3,)
Returns:
Cartesian coordinates in Å
**Source:** [Line 170](Python/Crystallography/core/crystal_structure.py#L170)
##### `cartesian_to_fractional(self, cartesian_coords)`
Convert Cartesian coordinates to fractional coordinates.
Parameters:
cartesian_coords: Cartesian coordinates in Å (Nx3 or 3,)
Returns:
Fractional coordinates
**Source:** [Line 183](Python/Crystallography/core/crystal_structure.py#L183)
##### `interatomic_distance(self, atom1_idx, atom2_idx, include_symmetry)`
Calculate distance between two atoms.
Parameters:
atom1_idx: Index of first atom
atom2_idx: Index of second atom
include_symmetry: Whether to consider periodic boundary conditions
Returns:
Distance in Å
**Source:** [Line 196](Python/Crystallography/core/crystal_structure.py#L196)
##### `d_spacing(self, h, k, l)`
Calculate d-spacing for Miller indices (hkl).
Parameters:
h, k, l: Miller indices
Returns:
d-spacing in Å
**Source:** [Line 226](Python/Crystallography/core/crystal_structure.py#L226)
##### `bragg_angle(self, h, k, l, wavelength)`
Calculate Bragg angle for reflection (hkl).
Parameters:
h, k, l: Miller indices
wavelength: X-ray wavelength in Å
Returns:
Bragg angle in degrees
**Source:** [Line 248](Python/Crystallography/core/crystal_structure.py#L248)
##### `systematic_absences(self, space_group)`
Determine systematic absences for given space group.
Parameters:
space_group: Space group symbol (e.g., 'Fm3m', 'P21/c')
Returns:
List of forbidden reflections
**Source:** [Line 267](Python/Crystallography/core/crystal_structure.py#L267)
##### `coordination_number(self, atom_idx, cutoff_radius)`
Calculate coordination number for an atom.
Parameters:
atom_idx: Index of central atom
cutoff_radius: Maximum distance for coordination (Å)
Returns:
Coordination number
**Source:** [Line 302](Python/Crystallography/core/crystal_structure.py#L302)
##### `powder_pattern_positions(self, wavelength, max_2theta, min_d_spacing)`
Calculate powder diffraction peak positions.
Parameters:
wavelength: X-ray wavelength in Å
max_2theta: Maximum 2θ angle in degrees
min_d_spacing: Minimum d-spacing to consider (Å)
Returns:
List of reflection information dictionaries
**Source:** [Line 352](Python/Crystallography/core/crystal_structure.py#L352)
##### `_calculate_multiplicity(self, h, k, l)`
Calculate multiplicity factor for reflection (simplified).
**Source:** [Line 404](Python/Crystallography/core/crystal_structure.py#L404)
##### `get_fractional_coordinates(self)`
Get all atomic fractional coordinates as array.
**Source:** [Line 418](Python/Crystallography/core/crystal_structure.py#L418)
##### `get_cartesian_coordinates(self)`
Get all atomic Cartesian coordinates as array.
**Source:** [Line 425](Python/Crystallography/core/crystal_structure.py#L425)
##### `supercell(self, nx, ny, nz)`
Create supercell structure.
Parameters:
nx, ny, nz: Supercell dimensions
Returns:
New CrystalStructure representing supercell
**Source:** [Line 430](Python/Crystallography/core/crystal_structure.py#L430)
**Class Source:** [Line 68](Python/Crystallography/core/crystal_structure.py#L68)
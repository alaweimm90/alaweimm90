# space_groups
**Module:** `Python/Crystallography/core/space_groups.py`
## Overview
Space Group Operations
Crystallographic space group analysis and symmetry operations.
Includes space group identification, symmetry operation generation,
and equivalent position calculations.
## Functions
### `identify_space_group(lattice_parameters, systematic_absences)`
Identify possible space groups from lattice parameters and systematic absences.
Parameters:
lattice_parameters: Crystal lattice parameters
systematic_absences: List of systematically absent reflections
Returns:
List of possible space group symbols
**Source:** [Line 453](Python/Crystallography/core/space_groups.py#L453)
## Classes
### `CrystalSystem`
Crystal system enumeration.
**Class Source:** [Line 16](Python/Crystallography/core/space_groups.py#L16)
### `LatticeType`
Bravais lattice type enumeration.
**Class Source:** [Line 27](Python/Crystallography/core/space_groups.py#L27)
### `SymmetryOperation`
Crystallographic symmetry operation.
#### Methods
##### `__post_init__(self)`
Validate symmetry operation.
**Source:** [Line 45](Python/Crystallography/core/space_groups.py#L45)
##### `apply(self, position)`
Apply symmetry operation to position.
Parameters:
position: Fractional coordinates (3,)
Returns:
Transformed fractional coordinates
**Source:** [Line 58](Python/Crystallography/core/space_groups.py#L58)
##### `inverse(self)`
Get inverse symmetry operation.
**Source:** [Line 70](Python/Crystallography/core/space_groups.py#L70)
##### `__mul__(self, other)`
Compose two symmetry operations.
**Source:** [Line 81](Python/Crystallography/core/space_groups.py#L81)
##### `order(self)`
Calculate order of rotation operation.
**Source:** [Line 92](Python/Crystallography/core/space_groups.py#L92)
**Class Source:** [Line 39](Python/Crystallography/core/space_groups.py#L39)
### `SpaceGroup`
Crystallographic space group representation.
Features:
- Space group symbol parsing
- Symmetry operation generation
- Equivalent position calculation
- Systematic absence determination
- Crystal system identification
Examples:
>>> sg = SpaceGroup("P21/c")
>>> print(f"Crystal system: {sg.crystal_system}")
>>> ops = sg.generate_symmetry_operations()
>>> print(f"Number of operations: {len(ops)}")
#### Methods
##### `__init__(self, symbol)`
Initialize space group.
Parameters:
symbol: Hermann-Mauguin space group symbol
**Source:** [Line 136](Python/Crystallography/core/space_groups.py#L136)
##### `_parse_symbol(self)`
Parse Hermann-Mauguin symbol.
**Source:** [Line 153](Python/Crystallography/core/space_groups.py#L153)
##### `_initialize_database(self)`
Initialize space group database (simplified).
**Source:** [Line 166](Python/Crystallography/core/space_groups.py#L166)
##### `generate_symmetry_operations(self)`
Generate all symmetry operations for this space group.
**Source:** [Line 201](Python/Crystallography/core/space_groups.py#L201)
##### `_parse_operation_string(self, op_string)`
Parse symmetry operation string like 'x,y,z' or '-x,y+1/2,-z+1/2'.
**Source:** [Line 239](Python/Crystallography/core/space_groups.py#L239)
##### `equivalent_positions(self, position)`
Generate all equivalent positions for given fractional coordinates.
Parameters:
position: Fractional coordinates (3,)
Returns:
List of equivalent positions
**Source:** [Line 284](Python/Crystallography/core/space_groups.py#L284)
##### `multiplicity(self, position)`
Calculate multiplicity of a position.
**Source:** [Line 328](Python/Crystallography/core/space_groups.py#L328)
##### `systematic_absences(self)`
Determine systematic absences for this space group.
Returns:
Dictionary of absence conditions and forbidden reflections
**Source:** [Line 332](Python/Crystallography/core/space_groups.py#L332)
##### `is_centrosymmetric(self)`
Check if space group is centrosymmetric.
**Source:** [Line 376](Python/Crystallography/core/space_groups.py#L376)
##### `point_group_operations(self)`
Extract point group operations (rotations only, no translations).
**Source:** [Line 389](Python/Crystallography/core/space_groups.py#L389)
##### `wyckoff_positions(self)`
Get Wyckoff positions for this space group (simplified).
Returns:
Dictionary of Wyckoff positions with multiplicities and site symmetries
**Source:** [Line 410](Python/Crystallography/core/space_groups.py#L410)
##### `__str__(self)`
String representation.
**Source:** [Line 442](Python/Crystallography/core/space_groups.py#L442)
##### `__repr__(self)`
Detailed representation.
**Source:** [Line 446](Python/Crystallography/core/space_groups.py#L446)
**Class Source:** [Line 118](Python/Crystallography/core/space_groups.py#L118)
# units
**Module:** `Python/utils/units.py`
## Overview
Unit Conversion Module
Provides comprehensive unit conversion capabilities for scientific computing,
including energy, length, time, mass, and other physical quantities.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
## Functions
### `energy_convert(value, from_unit, to_unit)`
Convert energy units.
**Source:** [Line 283](Python/utils/units.py#L283)
### `length_convert(value, from_unit, to_unit)`
Convert length units.
**Source:** [Line 289](Python/utils/units.py#L289)
### `time_convert(value, from_unit, to_unit)`
Convert time units.
**Source:** [Line 295](Python/utils/units.py#L295)
### `mass_convert(value, from_unit, to_unit)`
Convert mass units.
**Source:** [Line 301](Python/utils/units.py#L301)
### `temperature_convert(value, from_unit, to_unit)`
Convert temperature units.
**Source:** [Line 307](Python/utils/units.py#L307)
### `pressure_convert(value, from_unit, to_unit)`
Convert pressure units.
**Source:** [Line 313](Python/utils/units.py#L313)
### `magnetic_field_convert(value, from_unit, to_unit)`
Convert magnetic field units.
**Source:** [Line 319](Python/utils/units.py#L319)
## Classes
### `UnitConverter`
Comprehensive unit conversion utility for scientific computing.
Supports conversion between various units commonly used in physics,
chemistry, and materials science.
#### Methods
##### `__init__(self)`
Initialize the unit converter with predefined conversion factors.
**Source:** [Line 26](Python/utils/units.py#L26)
##### `_init_energy_conversions(self)`
Initialize energy conversion factors to Joules.
**Source:** [Line 36](Python/utils/units.py#L36)
##### `_init_length_conversions(self)`
Initialize length conversion factors to meters.
**Source:** [Line 60](Python/utils/units.py#L60)
##### `_init_time_conversions(self)`
Initialize time conversion factors to seconds.
**Source:** [Line 81](Python/utils/units.py#L81)
##### `_init_mass_conversions(self)`
Initialize mass conversion factors to kilograms.
**Source:** [Line 98](Python/utils/units.py#L98)
##### `_init_temperature_conversions(self)`
Initialize temperature conversion functions.
**Source:** [Line 116](Python/utils/units.py#L116)
##### `_init_pressure_conversions(self)`
Initialize pressure conversion factors to Pascals.
**Source:** [Line 125](Python/utils/units.py#L125)
##### `_init_magnetic_field_conversions(self)`
Initialize magnetic field conversion factors to Tesla.
**Source:** [Line 141](Python/utils/units.py#L141)
##### `convert(self, value, from_unit, to_unit, quantity_type)`
Convert between units of the same physical quantity.
Parameters
----------
value : float or array
Value(s) to convert
from_unit : str
Source unit
to_unit : str
Target unit
quantity_type : str
Type of physical quantity ('energy', 'length', 'time', 'mass',
'temperature', 'pressure', 'magnetic_field')
Returns
-------
float or array
Converted value(s)
**Source:** [Line 154](Python/utils/units.py#L154)
##### `_convert_energy(self, value, from_unit, to_unit)`
Convert energy units.
**Source:** [Line 194](Python/utils/units.py#L194)
##### `_convert_length(self, value, from_unit, to_unit)`
Convert length units.
**Source:** [Line 206](Python/utils/units.py#L206)
##### `_convert_time(self, value, from_unit, to_unit)`
Convert time units.
**Source:** [Line 218](Python/utils/units.py#L218)
##### `_convert_mass(self, value, from_unit, to_unit)`
Convert mass units.
**Source:** [Line 230](Python/utils/units.py#L230)
##### `_convert_temperature(self, value, from_unit, to_unit)`
Convert temperature units.
**Source:** [Line 242](Python/utils/units.py#L242)
##### `_convert_pressure(self, value, from_unit, to_unit)`
Convert pressure units.
**Source:** [Line 258](Python/utils/units.py#L258)
##### `_convert_magnetic_field(self, value, from_unit, to_unit)`
Convert magnetic field units.
**Source:** [Line 270](Python/utils/units.py#L270)
**Class Source:** [Line 18](Python/utils/units.py#L18)
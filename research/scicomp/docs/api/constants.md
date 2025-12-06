# constants
**Module:** `Python/utils/constants.py`
## Overview
Physical Constants Module
Provides fundamental physical constants and unit conversion factors
used throughout scientific computing applications.
All constants are given in SI units unless otherwise specified.
Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT
Copyright © 2025 Meshal Alawein — All rights reserved.
## Constants
- **`NA`**
## Functions
### `celsius_to_kelvin(T_celsius)`
Convert Celsius to Kelvin.
**Source:** [Line 152](Python/utils/constants.py#L152)
### `kelvin_to_celsius(T_kelvin)`
Convert Kelvin to Celsius.
**Source:** [Line 156](Python/utils/constants.py#L156)
### `fahrenheit_to_kelvin(T_fahrenheit)`
Convert Fahrenheit to Kelvin.
**Source:** [Line 160](Python/utils/constants.py#L160)
### `kelvin_to_fahrenheit(T_kelvin)`
Convert Kelvin to Fahrenheit.
**Source:** [Line 164](Python/utils/constants.py#L164)
### `eV_to_wavenumber(energy_eV)`
Convert energy in eV to wavenumber in cm⁻¹.
**Source:** [Line 169](Python/utils/constants.py#L169)
### `wavenumber_to_eV(wavenumber_cm)`
Convert wavenumber in cm⁻¹ to energy in eV.
**Source:** [Line 173](Python/utils/constants.py#L173)
### `eV_to_frequency(energy_eV)`
Convert energy in eV to frequency in Hz.
**Source:** [Line 177](Python/utils/constants.py#L177)
### `frequency_to_eV(frequency_Hz)`
Convert frequency in Hz to energy in eV.
**Source:** [Line 181](Python/utils/constants.py#L181)
### `eV_to_wavelength(energy_eV)`
Convert energy in eV to wavelength in meters.
**Source:** [Line 185](Python/utils/constants.py#L185)
### `wavelength_to_eV(wavelength_m)`
Convert wavelength in meters to energy in eV.
**Source:** [Line 189](Python/utils/constants.py#L189)
### `compton_wavelength(mass_kg)`
Calculate Compton wavelength for a particle of given mass.
**Source:** [Line 194](Python/utils/constants.py#L194)
### `classical_electron_radius()`
Calculate classical electron radius.
**Source:** [Line 198](Python/utils/constants.py#L198)
### `cyclotron_frequency(B_tesla, mass_kg, charge_C)`
Calculate cyclotron frequency for charged particle in magnetic field.
**Source:** [Line 202](Python/utils/constants.py#L202)
### `plasma_frequency(n_density, mass_kg, charge_C)`
Calculate plasma frequency for given particle density.
**Source:** [Line 206](Python/utils/constants.py#L206)
### `debye_length(T_kelvin, n_density)`
Calculate Debye screening length in plasma.
**Source:** [Line 210](Python/utils/constants.py#L210)
### `thermal_de_broglie_wavelength(T_kelvin, mass_kg)`
Calculate thermal de Broglie wavelength.
**Source:** [Line 214](Python/utils/constants.py#L214)
### `convert_to_atomic_units(value, unit_type)`
Convert physical quantities to atomic units.
Parameters
----------
value : float
Value to convert
unit_type : str
Type of unit: 'energy', 'length', 'time', 'mass', 'charge'
Returns
-------
float
Value in atomic units
**Source:** [Line 219](Python/utils/constants.py#L219)
### `convert_from_atomic_units(value, unit_type)`
Convert from atomic units to SI units.
Parameters
----------
value : float
Value in atomic units
unit_type : str
Type of unit: 'energy', 'length', 'time', 'mass', 'charge'
Returns
-------
float
Value in SI units
**Source:** [Line 253](Python/utils/constants.py#L253)
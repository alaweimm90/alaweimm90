"""
Physical and Material Constants for Elasticity
Collection of physical constants, material properties, and unit conversions
commonly used in elasticity and structural mechanics.
"""
import numpy as np
from typing import Dict, Any
# Physical constants
ELASTIC_CONSTANTS = {
    # Fundamental constants
    'avogadro_number': 6.02214076e23,  # mol⁻¹
    'boltzmann_constant': 1.380649e-23,  # J/K
    'gas_constant': 8.314462618,  # J/(mol·K)
    'planck_constant': 6.62607015e-34,  # J·s
    # Gravitational acceleration
    'gravity_earth': 9.80665,  # m/s²
    # Standard conditions
    'standard_temperature': 293.15,  # K (20°C)
    'standard_pressure': 101325,  # Pa
}
# Material properties database
MATERIAL_PROPERTIES = {
    # Metals
    'steel_mild': {
        'density': 7850,  # kg/m³
        'youngs_modulus': 200e9,  # Pa
        'poissons_ratio': 0.30,
        'yield_strength': 250e6,  # Pa
        'ultimate_strength': 400e6,  # Pa
        'thermal_expansion': 12e-6,  # 1/K
        'thermal_conductivity': 50,  # W/(m·K)
        'specific_heat': 490,  # J/(kg·K)
    },
    'steel_stainless_304': {
        'density': 8000,
        'youngs_modulus': 193e9,
        'poissons_ratio': 0.29,
        'yield_strength': 205e6,
        'ultimate_strength': 515e6,
        'thermal_expansion': 17.3e-6,
        'thermal_conductivity': 16.2,
        'specific_heat': 500,
    },
    'aluminum_6061': {
        'density': 2700,
        'youngs_modulus': 68.9e9,
        'poissons_ratio': 0.33,
        'yield_strength': 276e6,
        'ultimate_strength': 310e6,
        'thermal_expansion': 23.1e-6,
        'thermal_conductivity': 167,
        'specific_heat': 896,
    },
    'copper': {
        'density': 8960,
        'youngs_modulus': 110e9,
        'poissons_ratio': 0.35,
        'yield_strength': 70e6,
        'ultimate_strength': 220e6,
        'thermal_expansion': 16.5e-6,
        'thermal_conductivity': 401,
        'specific_heat': 385,
    },
    'titanium_ti6al4v': {
        'density': 4430,
        'youngs_modulus': 114e9,
        'poissons_ratio': 0.32,
        'yield_strength': 880e6,
        'ultimate_strength': 950e6,
        'thermal_expansion': 8.6e-6,
        'thermal_conductivity': 6.7,
        'specific_heat': 526,
    },
    # Non-metals
    'concrete': {
        'density': 2400,
        'youngs_modulus': 30e9,
        'poissons_ratio': 0.20,
        'compressive_strength': 40e6,
        'tensile_strength': 4e6,
        'thermal_expansion': 10e-6,
        'thermal_conductivity': 1.7,
        'specific_heat': 880,
    },
    'glass_soda_lime': {
        'density': 2500,
        'youngs_modulus': 70e9,
        'poissons_ratio': 0.22,
        'compressive_strength': 1000e6,
        'tensile_strength': 50e6,
        'thermal_expansion': 9e-6,
        'thermal_conductivity': 1.05,
        'specific_heat': 840,
    },
    'wood_pine': {
        'density': 500,  # Parallel to grain
        'youngs_modulus_parallel': 12e9,
        'youngs_modulus_perpendicular': 0.4e9,
        'poissons_ratio_major': 0.37,
        'poissons_ratio_minor': 0.02,
        'compressive_strength_parallel': 40e6,
        'tensile_strength_parallel': 100e6,
        'thermal_expansion_parallel': 3e-6,
        'thermal_expansion_perpendicular': 30e-6,
        'thermal_conductivity': 0.13,
        'specific_heat': 1380,
    },
    # Polymers
    'abs_plastic': {
        'density': 1050,
        'youngs_modulus': 2.3e9,
        'poissons_ratio': 0.35,
        'yield_strength': 40e6,
        'ultimate_strength': 45e6,
        'thermal_expansion': 100e-6,
        'thermal_conductivity': 0.25,
        'specific_heat': 1400,
    },
    'polypropylene': {
        'density': 900,
        'youngs_modulus': 1.5e9,
        'poissons_ratio': 0.42,
        'yield_strength': 25e6,
        'ultimate_strength': 35e6,
        'thermal_expansion': 150e-6,
        'thermal_conductivity': 0.12,
        'specific_heat': 1920,
    },
    'peek': {
        'density': 1300,
        'youngs_modulus': 3.6e9,
        'poissons_ratio': 0.38,
        'yield_strength': 90e6,
        'ultimate_strength': 100e6,
        'thermal_expansion': 47e-6,
        'thermal_conductivity': 0.25,
        'specific_heat': 1340,
    },
    # Composites
    'carbon_fiber_epoxy': {
        'density': 1600,
        'youngs_modulus_fiber': 150e9,  # Fiber direction
        'youngs_modulus_matrix': 10e9,  # Matrix direction
        'poissons_ratio_major': 0.30,
        'poissons_ratio_minor': 0.02,
        'shear_modulus': 5e9,
        'tensile_strength_fiber': 1500e6,
        'tensile_strength_matrix': 50e6,
        'thermal_expansion_fiber': -0.5e-6,
        'thermal_expansion_matrix': 30e-6,
        'thermal_conductivity': 1.0,
        'specific_heat': 1050,
    },
    'glass_fiber_epoxy': {
        'density': 2000,
        'youngs_modulus_fiber': 40e9,
        'youngs_modulus_matrix': 8e9,
        'poissons_ratio_major': 0.28,
        'poissons_ratio_minor': 0.06,
        'shear_modulus': 4e9,
        'tensile_strength_fiber': 1000e6,
        'tensile_strength_matrix': 40e6,
        'thermal_expansion_fiber': 6e-6,
        'thermal_expansion_matrix': 25e-6,
        'thermal_conductivity': 0.3,
        'specific_heat': 1000,
    },
    # Ceramics
    'alumina': {
        'density': 3900,
        'youngs_modulus': 370e9,
        'poissons_ratio': 0.22,
        'compressive_strength': 2000e6,
        'tensile_strength': 300e6,
        'thermal_expansion': 8.5e-6,
        'thermal_conductivity': 25,
        'specific_heat': 775,
    },
    'silicon_carbide': {
        'density': 3200,
        'youngs_modulus': 410e9,
        'poissons_ratio': 0.21,
        'compressive_strength': 3500e6,
        'tensile_strength': 400e6,
        'thermal_expansion': 4.3e-6,
        'thermal_conductivity': 85,
        'specific_heat': 675,
    },
}
# Typical failure criteria parameters
FAILURE_CRITERIA = {
    'von_mises': {
        'description': 'Von Mises equivalent stress criterion',
        'formula': 'σ_eq = sqrt(3/2 * S_ij * S_ij)',
        'applicable_to': ['ductile_metals'],
    },
    'maximum_principal_stress': {
        'description': 'Maximum principal stress criterion',
        'formula': 'σ_1 ≤ σ_allowable',
        'applicable_to': ['brittle_materials'],
    },
    'mohr_coulomb': {
        'description': 'Mohr-Coulomb failure criterion',
        'formula': 'τ = c + σ * tan(φ)',
        'applicable_to': ['soils', 'concrete', 'rock'],
    },
    'tsai_wu': {
        'description': 'Tsai-Wu failure criterion for composites',
        'formula': 'F_i * σ_i + F_ij * σ_i * σ_j = 1',
        'applicable_to': ['fiber_composites'],
    },
    'maximum_strain': {
        'description': 'Maximum strain criterion',
        'formula': 'ε_1 ≤ ε_allowable',
        'applicable_to': ['brittle_materials', 'ceramics'],
    },
}
# Unit conversion factors
UNIT_CONVERSIONS = {
    # Length
    'length': {
        'mm_to_m': 1e-3,
        'cm_to_m': 1e-2,
        'inch_to_m': 0.0254,
        'ft_to_m': 0.3048,
        'mil_to_m': 25.4e-6,  # 1 mil = 0.001 inch
    },
    # Area
    'area': {
        'mm2_to_m2': 1e-6,
        'cm2_to_m2': 1e-4,
        'inch2_to_m2': 6.4516e-4,
        'ft2_to_m2': 0.092903,
    },
    # Volume
    'volume': {
        'mm3_to_m3': 1e-9,
        'cm3_to_m3': 1e-6,
        'liter_to_m3': 1e-3,
        'inch3_to_m3': 1.6387e-5,
        'ft3_to_m3': 0.028317,
    },
    # Force
    'force': {
        'N_to_N': 1.0,
        'kN_to_N': 1e3,
        'MN_to_N': 1e6,
        'lbf_to_N': 4.4482,
        'kip_to_N': 4448.2,
        'dyne_to_N': 1e-5,
    },
    # Pressure/Stress
    'stress': {
        'Pa_to_Pa': 1.0,
        'kPa_to_Pa': 1e3,
        'MPa_to_Pa': 1e6,
        'GPa_to_Pa': 1e9,
        'psi_to_Pa': 6894.76,
        'ksi_to_Pa': 6.89476e6,
        'bar_to_Pa': 1e5,
        'atm_to_Pa': 101325,
        'torr_to_Pa': 133.322,
    },
    # Energy
    'energy': {
        'J_to_J': 1.0,
        'kJ_to_J': 1e3,
        'MJ_to_J': 1e6,
        'cal_to_J': 4.184,
        'kcal_to_J': 4184,
        'Btu_to_J': 1055.06,
        'kWh_to_J': 3.6e6,
        'eV_to_J': 1.602176634e-19,
    },
    # Temperature
    'temperature': {
        'celsius_to_kelvin': lambda C: C + 273.15,
        'fahrenheit_to_kelvin': lambda F: (F - 32) * 5/9 + 273.15,
        'rankine_to_kelvin': lambda R: R * 5/9,
    },
    # Mass
    'mass': {
        'kg_to_kg': 1.0,
        'g_to_kg': 1e-3,
        'mg_to_kg': 1e-6,
        'lb_to_kg': 0.453592,
        'oz_to_kg': 0.0283495,
        'slug_to_kg': 14.5939,
    },
    # Density
    'density': {
        'kg_m3_to_kg_m3': 1.0,
        'g_cm3_to_kg_m3': 1e3,
        'lb_ft3_to_kg_m3': 16.0185,
        'lb_inch3_to_kg_m3': 2.768e4,
    },
    # Moment of inertia
    'moment_of_inertia': {
        'm4_to_m4': 1.0,
        'mm4_to_m4': 1e-12,
        'cm4_to_m4': 1e-8,
        'inch4_to_m4': 4.16231e-7,
    },
}
# Common engineering formulas and relationships
ENGINEERING_FORMULAS = {
    'elastic_relationships': {
        'shear_modulus_from_E_nu': lambda E, nu: E / (2 * (1 + nu)),
        'bulk_modulus_from_E_nu': lambda E, nu: E / (3 * (1 - 2 * nu)),
        'lame_first_parameter': lambda E, nu: (E * nu) / ((1 + nu) * (1 - 2 * nu)),
        'lame_second_parameter': lambda E, nu: E / (2 * (1 + nu)),
        'poissons_ratio_from_wave_velocities': lambda vp, vs: (vp**2 - 2*vs**2) / (2 * (vp**2 - vs**2)),
    },
    'wave_velocities': {
        'longitudinal_wave_velocity': lambda E, rho, nu: np.sqrt(E * (1 - nu) / (rho * (1 + nu) * (1 - 2 * nu))),
        'shear_wave_velocity': lambda G, rho: np.sqrt(G / rho),
        'rayleigh_wave_velocity': lambda vs, nu: vs * (0.87 + 1.12 * nu) / (1 + nu),
        'plate_wave_velocity': lambda E, rho, h, nu: np.sqrt(E * h**2 / (12 * rho * (1 - nu**2))),
    },
    'stress_concentration_factors': {
        'circular_hole_tension': 3.0,
        'elliptical_hole_tension': lambda a, b: 1 + 2 * a / b,  # a = major axis, b = minor axis
        'notched_bar_tension': lambda r, d: 1 + 2 * np.sqrt(d / r),  # r = notch radius, d = bar width
        'shoulder_fillet': lambda r, D, d: 1.0,  # Simplified - actual formula is complex
    },
    'fatigue_relationships': {
        'basquin_law': lambda sigma_a, sigma_f_prime, b, N: sigma_f_prime * N**b,
        'coffin_manson': lambda epsilon_a, epsilon_f_prime, c, N: epsilon_f_prime * N**c,
        'miner_damage_rule': lambda n_i, N_i: np.sum(n_i / N_i),  # Cumulative damage
    },
    'fracture_mechanics': {
        'stress_intensity_factor_mode_I': lambda sigma, a, Y: Y * sigma * np.sqrt(np.pi * a),
        'critical_crack_length': lambda K_IC, sigma, Y: (K_IC / (Y * sigma))**2 / np.pi,
        'paris_law_crack_growth': lambda da_dN, C, m, delta_K: C * delta_K**m,
    },
}
# Safety factors and design factors
DESIGN_FACTORS = {
    'safety_factors': {
        'steel_structures': 1.5,
        'aluminum_structures': 1.65,
        'concrete_structures': 2.0,
        'wood_structures': 2.5,
        'pressure_vessels': 4.0,
        'aerospace_structures': 1.5,
        'automotive_structures': 2.0,
        'marine_structures': 2.5,
    },
    'load_factors': {
        'dead_load': 1.2,
        'live_load': 1.6,
        'wind_load': 1.0,
        'earthquake_load': 1.0,
        'snow_load': 1.2,
        'impact_load': 2.0,
    },
    'service_life_factors': {
        'temporary_structures': 1.0,
        'ordinary_structures': 1.0,
        'important_structures': 1.1,
        'critical_structures': 1.2,
    },
}
# Environmental conditions
ENVIRONMENTAL_CONDITIONS = {
    'temperature_ranges': {
        'room_temperature': (293.15, 298.15),  # K
        'high_temperature': (373.15, 773.15),  # K
        'cryogenic': (4.2, 77.0),  # K
        'aerospace': (173.15, 398.15),  # K
        'automotive': (233.15, 398.15),  # K
    },
    'humidity_effects': {
        'dry_conditions': 0.0,  # 0% RH
        'normal_conditions': 0.5,  # 50% RH
        'humid_conditions': 0.9,  # 90% RH
        'saturated_conditions': 1.0,  # 100% RH
    },
    'corrosion_rates': {
        'mild_steel_atmosphere': 0.1e-3,  # mm/year
        'mild_steel_marine': 0.5e-3,  # mm/year
        'stainless_steel_atmosphere': 0.001e-3,  # mm/year
        'aluminum_atmosphere': 0.01e-3,  # mm/year
    },
}
# Standard test conditions
TEST_CONDITIONS = {
    'tensile_test': {
        'strain_rate': 8.33e-5,  # s⁻¹ (ASTM E8)
        'temperature': 296.15,  # K (23°C)
        'humidity': 0.5,  # 50% RH
    },
    'fatigue_test': {
        'frequency': 10.0,  # Hz (typical)
        'stress_ratio': 0.1,  # R = σ_min/σ_max
        'temperature': 296.15,  # K
    },
    'creep_test': {
        'temperature_range': (773.15, 1273.15),  # K
        'stress_levels': (0.3, 0.8),  # fraction of yield strength
        'duration': 1000 * 3600,  # hours to seconds
    },
    'impact_test': {
        'charpy_temperature': 296.15,  # K
        'izod_temperature': 296.15,  # K
        'impact_velocity': 5.5,  # m/s
    },
}
def get_material_property(material: str, property_name: str) -> float:
    """
    Get material property from database.
    Parameters:
        material: Material name
        property_name: Property name
    Returns:
        Property value
    Raises:
        KeyError: If material or property not found
    """
    if material not in MATERIAL_PROPERTIES:
        raise KeyError(f"Material '{material}' not found in database")
    if property_name not in MATERIAL_PROPERTIES[material]:
        raise KeyError(f"Property '{property_name}' not found for material '{material}'")
    return MATERIAL_PROPERTIES[material][property_name]
def convert_units(value: float, from_unit: str, to_unit: str,
                 unit_type: str) -> float:
    """
    Convert between units.
    Parameters:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit
        unit_type: Type of unit (e.g., 'length', 'stress', etc.)
    Returns:
        Converted value
    """
    if unit_type not in UNIT_CONVERSIONS:
        raise KeyError(f"Unit type '{unit_type}' not supported")
    conversions = UNIT_CONVERSIONS[unit_type]
    # Special handling for temperature
    if unit_type == 'temperature':
        if from_unit in conversions:
            return conversions[from_unit](value)
        else:
            raise KeyError(f"Temperature conversion from '{from_unit}' not supported")
    # Standard multiplicative conversions
    from_key = f"{from_unit}_to_{to_unit.split('_')[-1]}"
    to_key = f"{to_unit}_to_{from_unit.split('_')[-1]}"
    if from_key in conversions:
        return value * conversions[from_key]
    elif to_key in conversions:
        return value / conversions[to_key]
    else:
        # Try to find base unit conversion
        base_unit = list(conversions.keys())[0].split('_to_')[1]
        from_to_base = f"{from_unit}_to_{base_unit}"
        to_to_base = f"{to_unit}_to_{base_unit}"
        if from_to_base in conversions and to_to_base in conversions:
            base_value = value * conversions[from_to_base]
            return base_value / conversions[to_to_base]
        raise KeyError(f"Conversion from '{from_unit}' to '{to_unit}' not supported")
def estimate_material_properties(known_properties: Dict[str, float],
                                material_class: str = 'metal') -> Dict[str, float]:
    """
    Estimate unknown material properties from known ones.
    Parameters:
        known_properties: Dictionary of known properties
        material_class: Material class ('metal', 'ceramic', 'polymer', etc.)
    Returns:
        Dictionary with estimated properties
    """
    estimated = known_properties.copy()
    # Elastic property relationships
    if 'youngs_modulus' in known_properties and 'poissons_ratio' in known_properties:
        E = known_properties['youngs_modulus']
        nu = known_properties['poissons_ratio']
        if 'shear_modulus' not in estimated:
            estimated['shear_modulus'] = ENGINEERING_FORMULAS['elastic_relationships']['shear_modulus_from_E_nu'](E, nu)
        if 'bulk_modulus' not in estimated:
            estimated['bulk_modulus'] = ENGINEERING_FORMULAS['elastic_relationships']['bulk_modulus_from_E_nu'](E, nu)
    # Typical strength ratios for metals
    if material_class == 'metal':
        if 'youngs_modulus' in known_properties and 'yield_strength' not in estimated:
            E = known_properties['youngs_modulus']
            estimated['yield_strength'] = E / 500  # Typical E/σy ratio
        if 'yield_strength' in known_properties and 'ultimate_strength' not in estimated:
            sy = known_properties['yield_strength']
            estimated['ultimate_strength'] = sy * 1.5  # Typical ratio
    # Wave velocities
    if all(prop in known_properties for prop in ['youngs_modulus', 'density', 'poissons_ratio']):
        E = known_properties['youngs_modulus']
        rho = known_properties['density']
        nu = known_properties['poissons_ratio']
        if 'longitudinal_wave_velocity' not in estimated:
            estimated['longitudinal_wave_velocity'] = ENGINEERING_FORMULAS['wave_velocities']['longitudinal_wave_velocity'](E, rho, nu)
        if 'shear_wave_velocity' not in estimated and 'shear_modulus' in estimated:
            G = estimated['shear_modulus']
            estimated['shear_wave_velocity'] = ENGINEERING_FORMULAS['wave_velocities']['shear_wave_velocity'](G, rho)
    return estimated
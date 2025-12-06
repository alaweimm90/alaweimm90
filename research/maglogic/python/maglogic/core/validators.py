"""
Input validation utilities for micromagnetic simulations.

This module provides comprehensive validation functions for simulation
parameters, ensuring physical consistency and numerical stability.

Author: Meshal Alawein
Email: meshal@berkeley.edu
License: MIT
"""

import numpy as np
from typing import Union, Dict, Any, List, Optional, Tuple
import warnings
from .constants import PARAMETER_RANGES, MATERIAL_CONSTANTS, PHYSICAL_CONSTANTS

class ValidationError(ValueError):
    """Custom exception for validation errors."""
    pass

class ValidationWarning(UserWarning):
    """Custom warning for validation issues that are not errors."""
    pass

def validate_input(value: Union[float, np.ndarray], 
                  name: str,
                  min_val: Optional[float] = None,
                  max_val: Optional[float] = None,
                  positive: bool = False,
                  integer: bool = False,
                  finite: bool = True) -> Union[float, np.ndarray]:
    """
    Validate a numerical input parameter.
    
    Args:
        value: Value to validate
        name: Parameter name for error messages
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        positive: Whether value must be positive
        integer: Whether value must be integer
        finite: Whether value must be finite
        
    Returns:
        Validated value
        
    Raises:
        ValidationError: If validation fails
        
    Example:
        >>> validate_input(100e-9, "cell_size", min_val=1e-10, positive=True)
        1e-07
        >>> validate_input(-1, "damping", positive=True)  # Raises ValidationError
    """
    # Convert to numpy array for consistent handling
    val_array = np.asarray(value)
    original_shape = val_array.shape
    original_type = type(value)
    
    # Check for finite values
    if finite and not np.all(np.isfinite(val_array)):
        raise ValidationError(f"{name} must be finite (no NaN or inf values)")
    
    # Check for integer requirement
    if integer and not np.all(val_array == np.asarray(val_array, dtype=int)):
        raise ValidationError(f"{name} must be integer")
    
    # Check for positive requirement
    if positive and not np.all(val_array > 0):
        raise ValidationError(f"{name} must be positive")
    
    # Check minimum value
    if min_val is not None and not np.all(val_array >= min_val):
        raise ValidationError(f"{name} must be >= {min_val}")
    
    # Check maximum value
    if max_val is not None and not np.all(val_array <= max_val):
        raise ValidationError(f"{name} must be <= {max_val}")
    
    # Return in original format
    if np.isscalar(value):
        return float(val_array) if not integer else int(val_array)
    else:
        return val_array

def validate_material_parameter(material: str, parameter: str, value: float) -> float:
    """
    Validate a material parameter against known physical ranges.
    
    Args:
        material: Material name
        parameter: Parameter name (e.g., 'Ms', 'A_ex')
        value: Parameter value
        
    Returns:
        Validated value
        
    Raises:
        ValidationError: If validation fails
    """
    # Check if parameter has known ranges
    if parameter in PARAMETER_RANGES:
        ranges = PARAMETER_RANGES[parameter]
        min_val = ranges.get("min")
        max_val = ranges.get("max")
        units = ranges.get("units", "")
        
        try:
            validated = validate_input(value, f"{parameter} ({units})", 
                                     min_val=min_val, max_val=max_val, 
                                     finite=True)
        except ValidationError as e:
            raise ValidationError(f"Invalid {parameter} for {material}: {e}")
        
        # Check against typical values for the material if available
        if material in MATERIAL_CONSTANTS:
            typical_value = MATERIAL_CONSTANTS[material].get(parameter)
            if typical_value is not None:
                ratio = abs(value / typical_value)
                if ratio > 10 or ratio < 0.1:
                    warnings.warn(
                        f"{parameter}={value} for {material} differs significantly "
                        f"from typical value {typical_value} (ratio: {ratio:.2f})",
                        ValidationWarning
                    )
        
        return validated
    else:
        # Generic validation for unknown parameters
        return validate_input(value, parameter, finite=True)

def validate_simulation_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a complete set of simulation parameters.
    
    Args:
        params: Dictionary of simulation parameters
        
    Returns:
        Dictionary of validated parameters
        
    Raises:
        ValidationError: If any parameter is invalid
    """
    validated_params = {}
    
    # Required parameters
    required_params = ["Ms", "A_ex", "alpha"]
    for param in required_params:
        if param not in params:
            raise ValidationError(f"Required parameter '{param}' is missing")
    
    # Validate each parameter
    for key, value in params.items():
        try:
            if key == "Ms":
                validated_params[key] = validate_input(
                    value, "saturation magnetization", 
                    min_val=1e3, max_val=3e6, positive=True
                )
            
            elif key == "A_ex":
                validated_params[key] = validate_input(
                    value, "exchange constant",
                    min_val=1e-15, max_val=100e-12, positive=True
                )
            
            elif key == "alpha":
                validated_params[key] = validate_input(
                    value, "Gilbert damping",
                    min_val=1e-5, max_val=1.0, positive=True
                )
            
            elif key == "gamma":
                validated_params[key] = validate_input(
                    value, "gyromagnetic ratio",
                    min_val=1e4, max_val=3e5, positive=True
                )
            
            elif key == "K1":
                validated_params[key] = validate_input(
                    value, "anisotropy constant",
                    min_val=-1e6, max_val=1e6, finite=True
                )
            
            elif key == "temperature":
                validated_params[key] = validate_input(
                    value, "temperature",
                    min_val=0.0, max_val=1000.0
                )
            
            elif key == "cell_size":
                validated_params[key] = validate_input(
                    value, "cell size",
                    min_val=0.1e-9, max_val=1e-6, positive=True
                )
            
            elif key == "time_step":
                validated_params[key] = validate_input(
                    value, "time step",
                    min_val=1e-18, max_val=1e-9, positive=True
                )
            
            elif key == "final_time":
                validated_params[key] = validate_input(
                    value, "final time",
                    min_val=1e-15, max_val=1e-3, positive=True
                )
            
            else:
                # Generic validation for other parameters
                validated_params[key] = validate_input(value, key, finite=True)
                
        except ValidationError as e:
            raise ValidationError(f"Parameter '{key}': {e}")
    
    # Cross-parameter validation
    _validate_parameter_consistency(validated_params)
    
    return validated_params

def _validate_parameter_consistency(params: Dict[str, Any]) -> None:
    """
    Validate consistency between related parameters.
    
    Args:
        params: Dictionary of validated parameters
        
    Raises:
        ValidationError: If parameters are inconsistent
    """
    # Check exchange length vs cell size
    if "Ms" in params and "A_ex" in params and "cell_size" in params:
        Ms = params["Ms"]
        A_ex = params["A_ex"]
        cell_size = params["cell_size"]
        mu_0 = PHYSICAL_CONSTANTS["mu_0"]
        
        # Calculate exchange length
        l_ex = np.sqrt(2 * A_ex / (mu_0 * Ms**2))
        
        # Cell size should be much smaller than exchange length
        if cell_size > l_ex / 2:
            warnings.warn(
                f"Cell size ({cell_size:.2e} m) is larger than l_ex/2 "
                f"({l_ex/2:.2e} m). Consider reducing cell size.",
                ValidationWarning
            )
        
        # Cell size should not be too small (computational efficiency)
        if cell_size < l_ex / 20:
            warnings.warn(
                f"Cell size ({cell_size:.2e} m) is very small compared to "
                f"exchange length ({l_ex:.2e} m). Consider increasing cell size.",
                ValidationWarning
            )
    
    # Check time step vs precession period
    if "gamma" in params and "Ms" in params and "time_step" in params:
        gamma = params["gamma"]
        Ms = params["Ms"]
        time_step = params["time_step"]
        mu_0 = PHYSICAL_CONSTANTS["mu_0"]
        
        # Estimate precession frequency (assuming 1T field)
        H_typical = 1.0 / mu_0  # 1T in A/m
        f_prec = gamma * H_typical / (2 * np.pi)
        T_prec = 1 / f_prec
        
        # Time step should be much smaller than precession period
        if time_step > T_prec / 10:
            warnings.warn(
                f"Time step ({time_step:.2e} s) may be too large for accurate "
                f"precession dynamics (T_prec ~ {T_prec:.2e} s).",
                ValidationWarning
            )
    
    # Check damping vs time step
    if "alpha" in params and "time_step" in params and "gamma" in params:
        alpha = params["alpha"]
        time_step = params["time_step"]
        gamma = params["gamma"]
        
        # Estimate relaxation time
        tau_relax = 1 / (alpha * gamma * Ms) if "Ms" in params else None
        
        if tau_relax and time_step > tau_relax / 10:
            warnings.warn(
                f"Time step ({time_step:.2e} s) may be too large for "
                f"relaxation dynamics (τ_relax ~ {tau_relax:.2e} s).",
                ValidationWarning
            )

def validate_geometry(geometry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate geometry parameters.
    
    Args:
        geometry: Dictionary of geometry parameters
        
    Returns:
        Dictionary of validated geometry parameters
    """
    validated_geometry = {}
    
    # Required geometry parameters
    if "shape" not in geometry:
        raise ValidationError("Geometry must specify 'shape'")
    
    shape = geometry["shape"].lower()
    validated_geometry["shape"] = shape
    
    if shape in ["triangle", "equilateral_triangle"]:
        if "edge_length" not in geometry:
            raise ValidationError("Triangle geometry requires 'edge_length'")
        
        validated_geometry["edge_length"] = validate_input(
            geometry["edge_length"], "edge_length",
            min_val=1e-9, max_val=1e-3, positive=True
        )
    
    elif shape in ["circle", "disk"]:
        if "radius" not in geometry:
            raise ValidationError("Circle geometry requires 'radius'")
        
        validated_geometry["radius"] = validate_input(
            geometry["radius"], "radius",
            min_val=1e-9, max_val=1e-3, positive=True
        )
    
    elif shape in ["rectangle", "square"]:
        if "width" not in geometry or "height" not in geometry:
            raise ValidationError("Rectangle geometry requires 'width' and 'height'")
        
        validated_geometry["width"] = validate_input(
            geometry["width"], "width",
            min_val=1e-9, max_val=1e-3, positive=True
        )
        validated_geometry["height"] = validate_input(
            geometry["height"], "height", 
            min_val=1e-9, max_val=1e-3, positive=True
        )
    
    elif shape == "ellipse":
        if "semi_major" not in geometry or "semi_minor" not in geometry:
            raise ValidationError("Ellipse geometry requires 'semi_major' and 'semi_minor'")
        
        validated_geometry["semi_major"] = validate_input(
            geometry["semi_major"], "semi_major",
            min_val=1e-9, max_val=1e-3, positive=True
        )
        validated_geometry["semi_minor"] = validate_input(
            geometry["semi_minor"], "semi_minor",
            min_val=1e-9, max_val=1e-3, positive=True
        )
    
    # Common parameters
    if "thickness" in geometry:
        validated_geometry["thickness"] = validate_input(
            geometry["thickness"], "thickness",
            min_val=0.1e-9, max_val=1e-6, positive=True
        )
    
    if "position" in geometry:
        position = np.asarray(geometry["position"])
        if position.shape != (3,):
            raise ValidationError("Position must be a 3D vector [x, y, z]")
        validated_geometry["position"] = position
    
    return validated_geometry

def validate_field_profile(field_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate field profile parameters.
    
    Args:
        field_params: Dictionary of field parameters
        
    Returns:
        Dictionary of validated field parameters
    """
    validated_field = {}
    
    # Field type
    if "type" not in field_params:
        raise ValidationError("Field profile must specify 'type'")
    
    field_type = field_params["type"].lower()
    validated_field["type"] = field_type
    
    if field_type == "uniform":
        if "direction" not in field_params or "amplitude" not in field_params:
            raise ValidationError("Uniform field requires 'direction' and 'amplitude'")
        
        # Validate direction vector
        direction = np.asarray(field_params["direction"])
        if direction.shape != (3,):
            raise ValidationError("Field direction must be a 3D vector")
        
        # Normalize direction
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0:
            raise ValidationError("Field direction cannot be zero vector")
        
        validated_field["direction"] = direction / direction_norm
        
        # Validate amplitude
        validated_field["amplitude"] = validate_input(
            field_params["amplitude"], "field amplitude",
            min_val=0, max_val=1e8
        )
    
    elif field_type == "time_varying":
        required = ["amplitude", "frequency", "phase"]
        for param in required:
            if param not in field_params:
                raise ValidationError(f"Time-varying field requires '{param}'")
        
        validated_field["amplitude"] = validate_input(
            field_params["amplitude"], "field amplitude",
            min_val=0, max_val=1e8
        )
        
        validated_field["frequency"] = validate_input(
            field_params["frequency"], "frequency",
            min_val=0, max_val=1e12, positive=True
        )
        
        validated_field["phase"] = validate_input(
            field_params["phase"], "phase",
            min_val=-2*np.pi, max_val=2*np.pi
        )
    
    return validated_field

def check_numerical_stability(params: Dict[str, Any]) -> List[str]:
    """
    Check for potential numerical stability issues.
    
    Args:
        params: Dictionary of simulation parameters
        
    Returns:
        List of stability warnings
    """
    warnings_list = []
    
    # Check CFL condition for time stepping
    if all(key in params for key in ["time_step", "cell_size", "gamma", "Ms"]):
        dt = params["time_step"]
        dx = params["cell_size"]
        gamma = params["gamma"]
        Ms = params["Ms"]
        
        # Estimate maximum possible field
        H_max = Ms  # Order of magnitude estimate
        
        # CFL condition: dt < dx / (gamma * H_max)
        dt_max = dx / (gamma * H_max)
        
        if dt > dt_max:
            warnings_list.append(
                f"Time step ({dt:.2e} s) may violate CFL condition. "
                f"Consider dt < {dt_max:.2e} s"
            )
    
    # Check for very high damping
    if "alpha" in params:
        alpha = params["alpha"]
        if alpha > 0.1:
            warnings_list.append(
                f"Very high damping (α = {alpha}) may cause numerical issues"
            )
    
    # Check for very low damping
    if "alpha" in params:
        alpha = params["alpha"]
        if alpha < 1e-4:
            warnings_list.append(
                f"Very low damping (α = {alpha}) may require smaller time steps"
            )
    
    return warnings_list

def validate_simulation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a complete simulation configuration.
    
    Args:
        config: Complete simulation configuration
        
    Returns:
        Validated configuration
        
    Raises:
        ValidationError: If configuration is invalid
    """
    validated_config = {}
    
    # Validate material parameters
    if "material" in config:
        validated_config["material"] = validate_simulation_parameters(config["material"])
    
    # Validate geometry
    if "geometry" in config:
        validated_config["geometry"] = validate_geometry(config["geometry"])
    
    # Validate fields
    if "fields" in config:
        validated_fields = {}
        for field_name, field_params in config["fields"].items():
            validated_fields[field_name] = validate_field_profile(field_params)
        validated_config["fields"] = validated_fields
    
    # Validate simulation settings
    simulation_keys = ["time_step", "final_time", "cell_size", "output_interval"]
    if any(key in config for key in simulation_keys):
        sim_params = {key: config[key] for key in simulation_keys if key in config}
        validated_config.update(validate_simulation_parameters(sim_params))
    
    # Check numerical stability
    stability_warnings = check_numerical_stability(validated_config)
    if stability_warnings:
        for warning in stability_warnings:
            warnings.warn(warning, ValidationWarning)
    
    return validated_config
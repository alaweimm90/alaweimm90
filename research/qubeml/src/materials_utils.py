"""
Crystal structure descriptors and materials property calculations.

Author: Meshal Alawein (meshal@berkeley.edu)
Institution: University of California, Berkeley
License: MIT License © 2025
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import warnings


def create_crystal_descriptors(
    lattice_params: List[float],
    atomic_numbers: List[int],
    positions: np.ndarray
) -> Dict[str, Union[float, np.ndarray]]:
    """Extract descriptors from crystal structure.
    
    Volume uses triclinic formula. Density approximates mass with atomic number.
    Packing fraction is atom count per unit volume (crude approximation).
    """
    if len(lattice_params) != 6:
        raise ValueError("lattice_params must contain exactly 6 values: a, b, c, alpha, beta, gamma")
    
    if not atomic_numbers:
        raise ValueError("atomic_numbers cannot be empty")
    
    descriptors = {}
    
    a, b, c, alpha, beta, gamma = lattice_params
    descriptors["volume"] = a * b * c * np.sqrt(
        1 + 2*np.cos(np.radians(alpha))*np.cos(np.radians(beta))*np.cos(np.radians(gamma))
        - np.cos(np.radians(alpha))**2 - np.cos(np.radians(beta))**2 - np.cos(np.radians(gamma))**2
    )
    descriptors["density"] = sum(atomic_numbers) / descriptors["volume"]
    descriptors["packing_fraction"] = len(atomic_numbers) / descriptors["volume"]
    
    descriptors["mean_atomic_number"] = np.mean(atomic_numbers)
    descriptors["std_atomic_number"] = np.std(atomic_numbers)
    descriptors["max_atomic_number"] = max(atomic_numbers)
    descriptors["min_atomic_number"] = min(atomic_numbers)
    
    # Structural descriptors
    descriptors["n_atoms"] = len(atomic_numbers)
    descriptors["composition_entropy"] = calculate_composition_entropy(atomic_numbers)
    
    # Geometric descriptors
    positions = np.array(positions)
    descriptors["mean_position"] = np.mean(positions, axis=0)
    descriptors["position_variance"] = np.var(positions, axis=0)
    
    return descriptors


def calculate_composition_entropy(atomic_numbers: List[int]) -> float:
    """
    Calculate the compositional entropy of a structure.
    
    Args:
        atomic_numbers: List of atomic numbers
    
    Returns:
        Compositional entropy
    """
    unique, counts = np.unique(atomic_numbers, return_counts=True)
    probabilities = counts / len(atomic_numbers)
    
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
    return float(entropy)


def calculate_radial_distribution(
    positions: np.ndarray,
    lattice_vectors: np.ndarray,
    r_max: float = 10.0,
    n_bins: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the radial distribution function.
    
    Args:
        positions: Atomic positions (N x 3)
        lattice_vectors: Lattice vectors (3 x 3)
        r_max: Maximum radius
        n_bins: Number of bins
    
    Returns:
        Tuple of (distances, g(r))
    """
    n_atoms = len(positions)
    
    # Calculate pairwise distances
    distances = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            # Account for periodic boundary conditions
            diff = positions[j] - positions[i]
            diff = diff - np.round(diff)  # Wrap to [-0.5, 0.5]
            cart_diff = np.dot(diff, lattice_vectors)
            dist = np.linalg.norm(cart_diff)
            if dist < r_max:
                distances.append(dist)
    
    # Create histogram
    hist, bin_edges = np.histogram(distances, bins=n_bins, range=(0, r_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Normalize by shell volume
    dr = r_max / n_bins
    shell_volumes = 4 * np.pi * bin_centers**2 * dr
    g_r = hist / (shell_volumes * n_atoms * (n_atoms - 1) / 2)
    
    return bin_centers, g_r


def generate_coulomb_matrix(
    atomic_numbers: List[int],
    positions: np.ndarray,
    size: Optional[int] = None
) -> np.ndarray:
    """
    Generate the Coulomb matrix representation of a molecule/crystal.
    
    Args:
        atomic_numbers: List of atomic numbers
        positions: Cartesian coordinates (N x 3) in Angstroms
        size: Pad matrix to this size (for consistent dimensions)
    
    Returns:
        Coulomb matrix
    """
    n_atoms = len(atomic_numbers)
    positions = np.array(positions)
    
    # Initialize Coulomb matrix
    coulomb_matrix = np.zeros((n_atoms, n_atoms))
    
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                # Diagonal elements
                coulomb_matrix[i, j] = 0.5 * atomic_numbers[i] ** 2.4
            else:
                # Off-diagonal elements
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance > 1e-8:
                    coulomb_matrix[i, j] = atomic_numbers[i] * atomic_numbers[j] / distance
    
    # Pad matrix if size specified
    if size and size > n_atoms:
        padded_matrix = np.zeros((size, size))
        padded_matrix[:n_atoms, :n_atoms] = coulomb_matrix
        coulomb_matrix = padded_matrix
    
    return coulomb_matrix


def calculate_band_gap_features(
    valence_electrons: List[int],
    atomic_radii: List[float],
    electronegativities: List[float]
) -> Dict[str, float]:
    """
    Calculate features relevant for band gap prediction.
    
    Args:
        valence_electrons: Number of valence electrons per atom
        atomic_radii: Atomic radii in Angstroms
        electronegativities: Pauling electronegativities
    
    Returns:
        Dictionary of band gap-relevant features
    """
    features = {}
    
    # Electronic features
    features["mean_valence"] = np.mean(valence_electrons)
    features["std_valence"] = np.std(valence_electrons)
    features["valence_range"] = max(valence_electrons) - min(valence_electrons)
    
    # Size features
    features["mean_radius"] = np.mean(atomic_radii)
    features["std_radius"] = np.std(atomic_radii)
    features["radius_ratio"] = min(atomic_radii) / max(atomic_radii) if max(atomic_radii) > 0 else 1.0
    
    # Electronegativity features
    features["mean_electronegativity"] = np.mean(electronegativities)
    features["std_electronegativity"] = np.std(electronegativities)
    features["electronegativity_diff"] = max(electronegativities) - min(electronegativities)
    
    # Combined features
    features["ionic_character"] = features["electronegativity_diff"] / 4.0  # Normalized to [0, 1]
    features["size_mismatch"] = features["std_radius"] / features["mean_radius"] if features["mean_radius"] > 0 else 0
    
    return features


def create_graph_representation(
    positions: np.ndarray,
    atomic_numbers: List[int],
    cutoff_radius: float = 5.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create graph representation of crystal structure for GNNs.
    
    Args:
        positions: Atomic positions (N x 3) in Angstroms
        atomic_numbers: List of atomic numbers
        cutoff_radius: Maximum distance for edges
    
    Returns:
        Tuple of (node_features, edge_index, edge_features)
    """
    n_atoms = len(atomic_numbers)
    positions = np.array(positions)
    
    # Node features (one-hot encoding of atomic numbers for simplicity)
    unique_atoms = np.unique(atomic_numbers)
    node_features = np.zeros((n_atoms, len(unique_atoms)))
    for i, z in enumerate(atomic_numbers):
        idx = np.where(unique_atoms == z)[0][0]
        node_features[i, idx] = 1
    
    # Edge construction based on distance
    edges = []
    edge_features = []
    
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i != j:
                distance = np.linalg.norm(positions[i] - positions[j])
                if distance < cutoff_radius:
                    edges.append([i, j])
                    # Edge features: distance and inverse distance
                    edge_features.append([distance, 1.0 / distance])
    
    edge_index = np.array(edges).T if edges else np.array([[], []], dtype=int)
    edge_features = np.array(edge_features) if edge_features else np.array([])
    
    return node_features, edge_index, edge_features


def calculate_formation_energy_features(
    atomic_numbers: List[int],
    reference_energies: Dict[int, float],
    total_energy: float
) -> float:
    """
    Calculate formation energy from total energy and reference energies.
    
    Args:
        atomic_numbers: List of atomic numbers
        reference_energies: Dictionary mapping atomic number to reference energy
        total_energy: Total energy of the compound
    
    Returns:
        Formation energy per atom
    """
    # Sum of reference energies
    ref_sum = sum(reference_energies.get(z, 0) for z in atomic_numbers)
    
    # Formation energy
    formation_energy = (total_energy - ref_sum) / len(atomic_numbers)
    
    return formation_energy


def generate_2d_material_params(
    material_type: str = "MoS2",
    strain: float = 0.0
) -> Dict[str, Union[float, List]]:
    """
    Generate parameters for 2D material simulations.
    
    Args:
        material_type: Type of 2D material
        strain: Applied strain (percentage)
    
    Returns:
        Dictionary of material parameters
    """
    # Default parameters for common 2D materials
    materials_db = {
        "MoS2": {
            "lattice_constant": 3.16,  # Angstroms
            "bandgap": 1.8,  # eV
            "thickness": 6.5,  # Angstroms
            "spin_orbit": 0.15,  # eV
            "effective_mass": 0.5,  # m_e
        },
        "WSe2": {
            "lattice_constant": 3.28,
            "bandgap": 1.6,
            "thickness": 7.0,
            "spin_orbit": 0.46,
            "effective_mass": 0.4,
        },
        "graphene": {
            "lattice_constant": 2.46,
            "bandgap": 0.0,
            "thickness": 3.35,
            "spin_orbit": 0.001,
            "effective_mass": 0.0,
        },
        "hBN": {
            "lattice_constant": 2.50,
            "bandgap": 5.9,
            "thickness": 3.33,
            "spin_orbit": 0.01,
            "effective_mass": 0.5,
        },
    }
    
    if material_type not in materials_db:
        warnings.warn(f"Unknown material {material_type}, using MoS2 parameters")
        material_type = "MoS2"
    
    params = materials_db[material_type].copy()
    
    # Apply strain
    strain_factor = 1 + strain / 100
    params["lattice_constant"] *= strain_factor
    
    # Strain effects on electronic properties (simplified)
    params["bandgap"] *= (1 - abs(strain) * 0.01)  # Bandgap reduction with strain
    params["effective_mass"] *= (1 + abs(strain) * 0.005)  # Mass increase with strain
    
    return params


def estimate_bulk_modulus(
    volume: float,
    formation_energy: float,
    atomic_numbers: List[int]
) -> float:
    """
    Estimate bulk modulus using empirical correlations.
    
    Args:
        volume: Unit cell volume in cubic Angstroms
        formation_energy: Formation energy per atom in eV
        atomic_numbers: List of atomic numbers
    
    Returns:
        Estimated bulk modulus in GPa
    """
    # Average atomic number as descriptor
    avg_z = np.mean(atomic_numbers)
    
    # Empirical relationship (simplified)
    # Generally: harder materials have more negative formation energies
    stability_factor = abs(formation_energy) / len(atomic_numbers)
    
    # Bulk modulus estimation (empirical formula)
    bulk_modulus = (avg_z * stability_factor * 20) / (volume / len(atomic_numbers)) ** (1/3)
    
    # Reasonable bounds (10-400 GPa for most materials)
    bulk_modulus = np.clip(bulk_modulus, 10, 400)
    
    return float(bulk_modulus)


# TODO: Add more property prediction functions
# Ideas:
# - Thermal conductivity estimation
# - Dielectric constant prediction  
# - Magnetic moment calculation
# 
# RESEARCH: Check Materials Project API for reference data
# Need to validate against experimental values

def calculate_thermal_conductivity():
    """
    Estimate thermal conductivity from structure.
    
    TODO: Implement Slack model or machine learning approach
    References to check:
    - Slack (1973) - Thermal conductivity of pure metals
    - Toberer et al. (2008) - Electronic thermal conductivity
    """
    # FIXME: Just a placeholder for now
    # Need to research proper models
    pass
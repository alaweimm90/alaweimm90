"""Tests for QubeML materials informatics utility functions"""

import numpy as np
import pytest
from src.materials_utils import (
    create_crystal_descriptors,
    calculate_composition_entropy,
    calculate_radial_distribution,
    generate_coulomb_matrix,
    calculate_band_gap_features,
    create_graph_representation,
    calculate_formation_energy_features,
    generate_2d_material_params
)


class TestMaterialsUtils:
    """Test suite for materials utility functions."""
    
    def test_crystal_descriptors(self):
        """Test crystal descriptor generation."""
        # Simple cubic lattice
        lattice_params = [5.0, 5.0, 5.0, 90.0, 90.0, 90.0]
        atomic_numbers = [14, 14, 8, 8]  # Si2O2
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.75]
        ])
        
        descriptors = create_crystal_descriptors(lattice_params, atomic_numbers, positions)
        
        # Check that all expected keys are present
        expected_keys = [
            "volume", "density", "packing_fraction",
            "mean_atomic_number", "std_atomic_number",
            "max_atomic_number", "min_atomic_number",
            "n_atoms", "composition_entropy",
            "mean_position", "position_variance"
        ]
        
        for key in expected_keys:
            assert key in descriptors
        
        # Check specific values
        assert descriptors["n_atoms"] == 4
        assert descriptors["volume"] == 125.0  # 5^3 for cubic
        assert descriptors["max_atomic_number"] == 14
        assert descriptors["min_atomic_number"] == 8
    
    def test_composition_entropy(self):
        """Test composition entropy calculation."""
        # Single element (zero entropy)
        atomic_numbers = [6, 6, 6, 6]
        entropy = calculate_composition_entropy(atomic_numbers)
        assert np.allclose(entropy, 0.0)
        
        # Equal mixture (maximum entropy)
        atomic_numbers = [6, 7, 8, 9]
        entropy = calculate_composition_entropy(atomic_numbers)
        assert entropy > 1.0  # Should be close to ln(4)
        
        # Binary compound
        atomic_numbers = [14, 14, 8, 8]
        entropy = calculate_composition_entropy(atomic_numbers)
        assert 0 < entropy < 1.0
    
    def test_radial_distribution(self):
        """Test radial distribution function."""
        # Simple cubic lattice
        positions = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5]
        ])
        lattice_vectors = np.eye(3) * 10.0  # 10 Angstrom cubic cell
        
        r, g_r = calculate_radial_distribution(positions, lattice_vectors, r_max=10.0, n_bins=50)
        
        # Check output dimensions
        assert len(r) == 50
        assert len(g_r) == 50
        
        # Check that distances are positive
        assert np.all(r > 0)
        
        # Check that g(r) is non-negative
        assert np.all(g_r >= 0)
    
    def test_coulomb_matrix(self):
        """Test Coulomb matrix generation."""
        # H2 molecule
        atomic_numbers = [1, 1]
        positions = np.array([[0, 0, 0], [0.74, 0, 0]])  # H-H bond length ~0.74 Å
        
        cm = generate_coulomb_matrix(atomic_numbers, positions)
        
        # Check shape
        assert cm.shape == (2, 2)
        
        # Check diagonal elements (0.5 * Z^2.4)
        assert np.allclose(cm[0, 0], 0.5 * 1**2.4)
        assert np.allclose(cm[1, 1], 0.5 * 1**2.4)
        
        # Check off-diagonal elements (Z1*Z2/r)
        expected_off_diag = 1 * 1 / 0.74
        assert np.allclose(cm[0, 1], expected_off_diag)
        assert np.allclose(cm[1, 0], expected_off_diag)
        
        # Test padding
        cm_padded = generate_coulomb_matrix(atomic_numbers, positions, size=5)
        assert cm_padded.shape == (5, 5)
        assert np.allclose(cm_padded[:2, :2], cm)
        assert np.allclose(cm_padded[2:, :], 0)
    
    def test_band_gap_features(self):
        """Test band gap feature calculation."""
        # Silicon-like parameters
        valence_electrons = [4, 4, 4, 4]
        atomic_radii = [1.11, 1.11, 1.11, 1.11]
        electronegativities = [1.90, 1.90, 1.90, 1.90]
        
        features = calculate_band_gap_features(
            valence_electrons, atomic_radii, electronegativities
        )
        
        # Check all features are present
        expected_features = [
            "mean_valence", "std_valence", "valence_range",
            "mean_radius", "std_radius", "radius_ratio",
            "mean_electronegativity", "std_electronegativity", "electronegativity_diff",
            "ionic_character", "size_mismatch"
        ]
        
        for feature in expected_features:
            assert feature in features
        
        # For identical atoms, many features should be zero or one
        assert features["std_valence"] == 0
        assert features["valence_range"] == 0
        assert features["radius_ratio"] == 1.0
        assert features["electronegativity_diff"] == 0
        assert features["ionic_character"] == 0
    
    def test_graph_representation(self):
        """Test graph representation for GNNs."""
        # Simple molecule
        positions = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        atomic_numbers = [6, 1, 1, 1]  # CH3
        
        node_features, edge_index, edge_features = create_graph_representation(
            positions, atomic_numbers, cutoff_radius=2.0
        )
        
        # Check node features
        assert node_features.shape[0] == 4  # 4 atoms
        assert node_features.shape[1] == 2  # 2 unique elements
        
        # Check that edges exist
        assert edge_index.shape[0] == 2  # Source and target
        assert edge_index.shape[1] > 0  # At least some edges
        
        # Check edge features
        assert len(edge_features) == edge_index.shape[1]
        if len(edge_features) > 0:
            assert edge_features.shape[1] == 2  # Distance and 1/distance
    
    def test_formation_energy_features(self):
        """Test formation energy calculation."""
        atomic_numbers = [14, 14, 8, 8, 8, 8]  # SiO2
        reference_energies = {14: -5.0, 8: -2.0}  # Example reference energies
        total_energy = -30.0
        
        formation_energy = calculate_formation_energy_features(
            atomic_numbers, reference_energies, total_energy
        )
        
        # Expected: (-30 - (2*-5 + 4*-2)) / 6 = (-30 - (-18)) / 6 = -2.0
        assert np.allclose(formation_energy, -2.0)
    
    def test_2d_material_params(self):
        """Test 2D material parameter generation."""
        # Test MoS2 without strain
        params = generate_2d_material_params("MoS2", strain=0.0)
        
        assert "lattice_constant" in params
        assert "bandgap" in params
        assert "thickness" in params
        assert "spin_orbit" in params
        assert "effective_mass" in params
        
        # Check expected values for MoS2
        assert np.allclose(params["lattice_constant"], 3.16)
        assert np.allclose(params["bandgap"], 1.8)
        
        # Test with strain
        params_strained = generate_2d_material_params("MoS2", strain=2.0)
        
        # Lattice constant should increase with tensile strain
        assert params_strained["lattice_constant"] > params["lattice_constant"]
        
        # Bandgap should decrease with strain
        assert params_strained["bandgap"] < params["bandgap"]
        
        # Test other materials
        for material in ["WSe2", "graphene", "hBN"]:
            params = generate_2d_material_params(material)
            assert "lattice_constant" in params
            assert "bandgap" in params


if __name__ == "__main__":
    pytest.main([__file__])
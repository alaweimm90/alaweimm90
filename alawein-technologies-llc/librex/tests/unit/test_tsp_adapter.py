"""
Unit tests for TSP adapter with enhanced validation

Tests verify distance matrix validation enhancements.
"""

import numpy as np
import pytest

from Librex.adapters.tsp import TSPAdapter


def test_tsp_distance_matrix_from_coordinates():
    """Test distance matrix computation from coordinates"""
    # Simple square: (0,0), (1,0), (1,1), (0,1)
    coordinates = np.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]
    ])

    adapter = TSPAdapter()
    distances = adapter._compute_distance_matrix(coordinates)

    # Check diagonal is zero
    assert np.allclose(np.diag(distances), 0)

    # Check specific distances
    assert abs(distances[0, 1] - 1.0) < 1e-10  # Distance (0,0) to (1,0)
    assert abs(distances[0, 2] - np.sqrt(2)) < 1e-10  # Distance (0,0) to (1,1)

    # Check symmetry
    assert np.allclose(distances, distances.T)


def test_tsp_distance_matrix_symmetry_validation():
    """Test that symmetric distance matrices pass validation"""
    symmetric_distances = np.array([
        [0, 1, 2],
        [1, 0, 3],
        [2, 3, 0]
    ])

    adapter = TSPAdapter()
    # Should not raise any errors
    adapter._validate_distance_matrix(symmetric_distances)


def test_tsp_distance_matrix_negative_values():
    """Test that negative distances are rejected"""
    negative_distances = np.array([
        [0, 1, -2],  # Negative distance
        [1, 0, 3],
        [2, 3, 0]
    ])

    adapter = TSPAdapter()

    with pytest.raises(ValueError, match="negative values"):
        adapter._validate_distance_matrix(negative_distances)


def test_tsp_distance_matrix_non_square():
    """Test that non-square matrices are rejected"""
    non_square = np.array([
        [0, 1, 2],
        [1, 0, 3]
    ])

    adapter = TSPAdapter()

    with pytest.raises(ValueError, match="must be square"):
        adapter._validate_distance_matrix(non_square)


def test_tsp_distance_matrix_nan_values():
    """Test that NaN values are rejected"""
    nan_distances = np.array([
        [0, 1, np.nan],
        [1, 0, 3],
        [2, 3, 0]
    ])

    adapter = TSPAdapter()

    with pytest.raises(ValueError, match="NaN or infinite"):
        adapter._validate_distance_matrix(nan_distances)


def test_tsp_distance_matrix_inf_values():
    """Test that infinite values are rejected"""
    inf_distances = np.array([
        [0, 1, np.inf],
        [1, 0, 3],
        [2, 3, 0]
    ])

    adapter = TSPAdapter()

    with pytest.raises(ValueError, match="NaN or infinite"):
        adapter._validate_distance_matrix(inf_distances)


def test_tsp_compute_objective():
    """Test TSP tour length calculation"""
    # Simple 3-city instance
    distances = np.array([
        [0, 1, 2],
        [1, 0, 3],
        [2, 3, 0]
    ])

    adapter = TSPAdapter()
    adapter.distance_matrix = distances
    adapter.n = 3

    # Tour: 0 -> 1 -> 2 -> 0
    tour = np.array([0, 1, 2])
    length = adapter.compute_objective(tour)

    # Expected: d[0,1] + d[1,2] + d[2,0] = 1 + 3 + 2 = 6
    expected = 1 + 3 + 2
    assert abs(length - expected) < 1e-10


def test_tsp_validate_tour():
    """Test tour validation"""
    adapter = TSPAdapter()
    adapter.n = 3

    # Valid tour
    valid_tour = np.array([0, 1, 2])
    result = adapter.validate_solution(valid_tour)
    assert result.is_valid

    # Invalid tour (duplicate)
    invalid_tour = np.array([0, 0, 1])
    result = adapter.validate_solution(invalid_tour)
    assert not result.is_valid


def test_tsp_encode_with_coordinates():
    """Test encoding TSP problem from coordinates"""
    coordinates = np.array([
        [0, 0],
        [1, 0],
        [1, 1]
    ])

    adapter = TSPAdapter()
    problem = adapter.encode_problem({'coordinates': coordinates})

    assert problem.dimension == 3
    assert adapter.distance_matrix is not None
    assert problem.problem_metadata['has_coordinates']


def test_tsp_encode_with_distance_matrix():
    """Test encoding TSP problem from distance matrix"""
    distances = np.array([
        [0, 1, 2],
        [1, 0, 3],
        [2, 3, 0]
    ])

    adapter = TSPAdapter()
    problem = adapter.encode_problem({'distance_matrix': distances})

    assert problem.dimension == 3
    assert not problem.problem_metadata['has_coordinates']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

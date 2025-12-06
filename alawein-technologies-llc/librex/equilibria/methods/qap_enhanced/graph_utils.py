"""
Graph Structure Detection and Exploitation Utilities

This module provides utilities for detecting and exploiting graph structures
in QAP instances, enabling specialized optimization strategies.

Key Capabilities:
1. Graph pattern detection (grid, tree, planar, etc.)
2. Graph decomposition (connected components, bipartite, etc.)
3. Structure-specific optimizations
4. Graph metric computation
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import (connected_components, minimum_spanning_tree,
                                   shortest_path)


class GraphUtilities:
    """
    Utilities for graph structure analysis and exploitation in QAP.

    This class provides methods to detect special graph structures
    and apply structure-specific optimization techniques.
    """

    def __init__(self):
        """Initialize graph utilities."""
        self.tolerance = 1e-10

    def analyze_graph_structure(self, adjacency_matrix: np.ndarray) -> Dict:
        """
        Comprehensive analysis of graph structure.

        Args:
            adjacency_matrix: Adjacency matrix of the graph

        Returns:
            Dictionary containing structural properties
        """
        n = len(adjacency_matrix)

        # Basic properties
        is_symmetric = np.allclose(adjacency_matrix, adjacency_matrix.T)
        density = np.count_nonzero(adjacency_matrix) / (n * n)

        # Degree statistics
        degrees = np.sum(adjacency_matrix > 0, axis=1)
        avg_degree = np.mean(degrees)
        max_degree = np.max(degrees)
        min_degree = np.min(degrees)

        # Check special structures
        is_bipartite = self._is_bipartite(adjacency_matrix)
        is_planar = self._is_likely_planar(adjacency_matrix)
        is_tree = self._is_tree(adjacency_matrix)

        # Connected components
        n_components = self._count_connected_components(adjacency_matrix)

        # Diameter and radius (for connected graphs)
        diameter, radius = self._compute_diameter_radius(adjacency_matrix)

        # Clustering coefficient
        clustering_coeff = self._compute_clustering_coefficient(adjacency_matrix)

        return {
            "n_vertices": n,
            "n_edges": np.count_nonzero(adjacency_matrix) // (2 if is_symmetric else 1),
            "density": density,
            "is_symmetric": is_symmetric,
            "avg_degree": avg_degree,
            "max_degree": max_degree,
            "min_degree": min_degree,
            "is_bipartite": is_bipartite,
            "is_planar": is_planar,
            "is_tree": is_tree,
            "n_components": n_components,
            "diameter": diameter,
            "radius": radius,
            "clustering_coefficient": clustering_coeff
        }

    def detect_grid_dimensions(self, adjacency_matrix: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect if graph is a grid and return dimensions.

        Args:
            adjacency_matrix: Adjacency matrix

        Returns:
            Grid dimensions (m, n) if graph is a grid, None otherwise
        """
        n = len(adjacency_matrix)

        # Check degree sequence for grid pattern
        degrees = np.sum(adjacency_matrix > 0, axis=1)

        # Count vertices by degree
        # Interior: degree 4, Edge: degree 3, Corner: degree 2
        deg_2_count = np.sum(degrees == 2)
        deg_3_count = np.sum(degrees == 3)
        deg_4_count = np.sum(degrees == 4)

        # For m×n grid:
        # 4 corners (degree 2)
        # 2(m-2) + 2(n-2) edges (degree 3)
        # (m-2)(n-2) interior (degree 4)

        if deg_2_count != 4:
            return None

        # Try to find grid dimensions
        for m in range(2, int(np.sqrt(n)) + 2):
            if n % m == 0:
                n_grid = n // m
                expected_deg_3 = 2 * (m - 2) + 2 * (n_grid - 2) if m > 2 and n_grid > 2 else 0
                expected_deg_4 = (m - 2) * (n_grid - 2) if m > 2 and n_grid > 2 else 0

                if deg_3_count == expected_deg_3 and deg_4_count == expected_deg_4:
                    # Verify grid connectivity pattern
                    if self._verify_grid_pattern(adjacency_matrix, m, n_grid):
                        return (m, n_grid)

        return None

    def _verify_grid_pattern(self, adj_matrix: np.ndarray, m: int, n: int) -> bool:
        """Verify that adjacency matrix represents an m×n grid."""
        size = m * n
        if len(adj_matrix) != size:
            return False

        # Build expected grid adjacency
        expected = np.zeros((size, size))

        for i in range(m):
            for j in range(n):
                idx = i * n + j

                # Connect to neighbors
                if i > 0:  # Up
                    neighbor = (i - 1) * n + j
                    expected[idx, neighbor] = 1
                    expected[neighbor, idx] = 1

                if j > 0:  # Left
                    neighbor = i * n + (j - 1)
                    expected[idx, neighbor] = 1
                    expected[neighbor, idx] = 1

        # Check if patterns match (allowing for permutation)
        # This is a simplified check - full isomorphism test would be more complex
        return np.allclose(np.sort(np.sum(expected, axis=1)),
                          np.sort(np.sum(adj_matrix > 0, axis=1)))

    def _is_bipartite(self, adj_matrix: np.ndarray) -> bool:
        """Check if graph is bipartite using 2-coloring."""
        n = len(adj_matrix)
        color = np.full(n, -1)

        for start in range(n):
            if color[start] != -1:
                continue

            # BFS coloring
            queue = [start]
            color[start] = 0

            while queue:
                u = queue.pop(0)
                current_color = color[u]

                for v in range(n):
                    if adj_matrix[u, v] > 0:
                        if color[v] == -1:
                            color[v] = 1 - current_color
                            queue.append(v)
                        elif color[v] == current_color:
                            return False

        return True

    def _is_likely_planar(self, adj_matrix: np.ndarray) -> bool:
        """
        Heuristic check for planarity using Euler's formula.

        A planar graph with n vertices and m edges satisfies: m ≤ 3n - 6
        """
        n = len(adj_matrix)
        m = np.count_nonzero(adj_matrix) // 2  # Number of edges (undirected)

        # Euler's formula for planar graphs
        return m <= 3 * n - 6

    def _is_tree(self, adj_matrix: np.ndarray) -> bool:
        """Check if graph is a tree (connected and acyclic)."""
        n = len(adj_matrix)
        m = np.count_nonzero(adj_matrix) // 2

        # Tree has exactly n-1 edges
        if m != n - 1:
            return False

        # Check connectivity
        n_components = self._count_connected_components(adj_matrix)

        return n_components == 1

    def _count_connected_components(self, adj_matrix: np.ndarray) -> int:
        """Count number of connected components."""
        sparse_adj = csr_matrix(adj_matrix)
        n_components, _ = connected_components(sparse_adj, directed=False)
        return n_components

    def _compute_diameter_radius(self, adj_matrix: np.ndarray) -> Tuple[float, float]:
        """Compute graph diameter and radius."""
        try:
            # Compute all-pairs shortest paths
            dist_matrix = shortest_path(csr_matrix(adj_matrix), directed=False)

            # Handle disconnected graphs
            if np.any(np.isinf(dist_matrix)):
                return np.inf, np.inf

            # Diameter: maximum distance
            diameter = np.max(dist_matrix)

            # Radius: minimum eccentricity
            eccentricities = np.max(dist_matrix, axis=1)
            radius = np.min(eccentricities)

            return diameter, radius
        except:
            return np.inf, np.inf

    def _compute_clustering_coefficient(self, adj_matrix: np.ndarray) -> float:
        """
        Compute average clustering coefficient.

        For each vertex, count triangles / possible triangles.
        """
        n = len(adj_matrix)
        clustering_coeffs = []

        for i in range(n):
            # Find neighbors
            neighbors = np.where(adj_matrix[i] > 0)[0]
            k = len(neighbors)

            if k < 2:
                continue

            # Count edges between neighbors
            edges_between_neighbors = 0
            for j in range(len(neighbors)):
                for l in range(j + 1, len(neighbors)):
                    if adj_matrix[neighbors[j], neighbors[l]] > 0:
                        edges_between_neighbors += 1

            # Clustering coefficient
            possible_edges = k * (k - 1) / 2
            if possible_edges > 0:
                clustering_coeffs.append(edges_between_neighbors / possible_edges)

        return np.mean(clustering_coeffs) if clustering_coeffs else 0.0

    def decompose_into_blocks(self, matrix: np.ndarray) -> List[Tuple[List[int], np.ndarray]]:
        """
        Decompose matrix into block structure.

        Returns list of (indices, block_matrix) tuples.
        """
        n = len(matrix)

        # Find connected components
        adj_matrix = np.abs(matrix) > self.tolerance
        sparse_adj = csr_matrix(adj_matrix)
        n_components, labels = connected_components(sparse_adj, directed=False)

        blocks = []
        for component_id in range(n_components):
            indices = np.where(labels == component_id)[0]
            if len(indices) > 0:
                # Extract block
                block = matrix[np.ix_(indices, indices)]
                blocks.append((indices.tolist(), block))

        return blocks

    def find_minimum_vertex_separator(self, adj_matrix: np.ndarray) -> List[int]:
        """
        Find approximate minimum vertex separator for divide-and-conquer.

        This uses a simple heuristic based on vertex degrees.
        """
        n = len(adj_matrix)
        degrees = np.sum(adj_matrix > 0, axis=1)

        # Sort vertices by degree (highest first)
        sorted_vertices = np.argsort(degrees)[::-1]

        # Greedily select high-degree vertices as separator
        separator_size = min(int(np.sqrt(n)), n // 4)
        separator = sorted_vertices[:separator_size].tolist()

        return separator

    def compute_graph_bandwidth(self, adj_matrix: np.ndarray) -> int:
        """
        Compute bandwidth of adjacency matrix.

        Bandwidth = max |i - j| for all edges (i, j).
        Low bandwidth indicates good locality.
        """
        n = len(adj_matrix)
        bandwidth = 0

        for i in range(n):
            for j in range(i + 1, n):
                if adj_matrix[i, j] > 0:
                    bandwidth = max(bandwidth, j - i)

        return bandwidth

    def suggest_preprocessing(self, flow_matrix: np.ndarray,
                             distance_matrix: np.ndarray) -> Dict:
        """
        Suggest preprocessing strategies based on matrix structure.

        Returns dictionary of recommended techniques.
        """
        recommendations = {
            "use_fft": False,
            "use_spectral": False,
            "use_decomposition": False,
            "use_multilevel": False,
            "specific_techniques": []
        }

        # Analyze both matrices
        flow_analysis = self.analyze_graph_structure(np.abs(flow_matrix))
        dist_analysis = self.analyze_graph_structure(np.abs(distance_matrix))

        # Check for grid structure
        flow_grid = self.detect_grid_dimensions(np.abs(flow_matrix))
        dist_grid = self.detect_grid_dimensions(np.abs(distance_matrix))

        if flow_grid or dist_grid:
            recommendations["use_fft"] = True
            recommendations["specific_techniques"].append("FFT-based Laplacian operations")

        # Check for low bandwidth
        flow_bandwidth = self.compute_graph_bandwidth(np.abs(flow_matrix))
        dist_bandwidth = self.compute_graph_bandwidth(np.abs(distance_matrix))
        n = len(flow_matrix)

        if flow_bandwidth < n / 4 or dist_bandwidth < n / 4:
            recommendations["use_spectral"] = True
            recommendations["specific_techniques"].append("Bandwidth reduction ordering")

        # Check for decomposability
        if flow_analysis["n_components"] > 1 or dist_analysis["n_components"] > 1:
            recommendations["use_decomposition"] = True
            recommendations["specific_techniques"].append("Block decomposition")

        # Check for hierarchical structure
        if flow_analysis["is_tree"] or dist_analysis["is_tree"]:
            recommendations["use_multilevel"] = True
            recommendations["specific_techniques"].append("Tree decomposition")

        # Bipartite structure
        if flow_analysis["is_bipartite"] or dist_analysis["is_bipartite"]:
            recommendations["specific_techniques"].append("Bipartite matching initialization")

        # Planar structure
        if flow_analysis["is_planar"] or dist_analysis["is_planar"]:
            recommendations["specific_techniques"].append("Planar graph algorithms")

        # High clustering coefficient suggests community structure
        if (flow_analysis["clustering_coefficient"] > 0.3 or
            dist_analysis["clustering_coefficient"] > 0.3):
            recommendations["use_multilevel"] = True
            recommendations["specific_techniques"].append("Community detection")

        return recommendations

    def apply_bandwidth_reduction(self, matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply Cuthill-McKee ordering to reduce bandwidth.

        Returns:
            Tuple of (permutation, reordered_matrix)
        """
        n = len(matrix)
        adj_matrix = np.abs(matrix) > self.tolerance

        # Find peripheral vertex (vertex with maximum eccentricity)
        degrees = np.sum(adj_matrix, axis=1)
        start_vertex = np.argmin(degrees)  # Start with minimum degree vertex

        # Cuthill-McKee algorithm
        visited = np.zeros(n, dtype=bool)
        ordering = []
        queue = [start_vertex]

        while queue:
            v = queue.pop(0)
            if visited[v]:
                continue

            visited[v] = True
            ordering.append(v)

            # Add unvisited neighbors sorted by degree
            neighbors = np.where(adj_matrix[v] > 0)[0]
            unvisited_neighbors = neighbors[~visited[neighbors]]
            neighbor_degrees = [(u, degrees[u]) for u in unvisited_neighbors]
            neighbor_degrees.sort(key=lambda x: x[1])

            for u, _ in neighbor_degrees:
                if not visited[u]:
                    queue.append(u)

        # Add any unvisited vertices
        for v in range(n):
            if not visited[v]:
                ordering.append(v)

        # Create permutation array
        permutation = np.array(ordering)

        # Apply permutation to matrix
        reordered = matrix[np.ix_(permutation, permutation)]

        return permutation, reordered
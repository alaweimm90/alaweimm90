"""
Feature extractor module.

Defines FeatureExtractor and provides a stable import path
`libria_meta.feature_extractor.FeatureExtractor`.
"""

import numpy as np
from typing import Any, List
from sklearn.preprocessing import StandardScaler


class FeatureExtractor:
    """Extract problem instance features for clustering and prediction"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def extract(self, instance: Any) -> np.ndarray:
        """
        Extract feature vector from problem instance

        Features depend on problem type:
        - Assignment: n_agents, n_tasks, cost_matrix_statistics
        - Routing: graph_properties, degree_distribution
        - Resource allocation: n_agents, budget, constraint_tightness

        Returns:
            features: 1D numpy array
        """
        features = []

        # Generic features
        if hasattr(instance, 'n_agents'):
            features.append(instance.n_agents)
        if hasattr(instance, 'n_tasks'):
            features.append(instance.n_tasks)

        # Cost matrix statistics
        if hasattr(instance, 'cost_matrix'):
            C = instance.cost_matrix
            features.extend([
                np.mean(C),
                np.std(C),
                np.min(C),
                np.max(C),
                np.median(C),
                np.percentile(C, 25),
                np.percentile(C, 75),
                np.linalg.norm(C, 'fro'),  # Frobenius norm
            ])

        # Graph properties
        if hasattr(instance, 'adjacency_matrix'):
            A = instance.adjacency_matrix
            n = A.shape[0]
            features.extend([
                n,  # Number of nodes
                np.sum(A) / 2,  # Number of edges (undirected)
                np.sum(A) / (n * (n - 1)) if n > 1 else 0,  # Density
                np.mean(np.sum(A, axis=1)),  # Average degree
                np.std(np.sum(A, axis=1)),  # Degree std
            ])

        # If no features extracted, return default
        if len(features) == 0:
            features = [0.5] * 10  # Default 10-dimensional feature vector

        return np.array(features, dtype=float)

    def fit(self, instances: List[Any]):
        """Fit feature scaler on training instances"""
        features_list = [self.extract(inst) for inst in instances]
        self.scaler.fit(features_list)
        self.fitted = True

    def transform(self, instance: Any) -> np.ndarray:
        """Extract and scale features"""
        features = self.extract(instance)
        if self.fitted:
            return self.scaler.transform([features])[0]
        return features

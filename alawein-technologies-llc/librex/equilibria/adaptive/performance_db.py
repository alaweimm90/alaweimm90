"""
Performance Database for Adaptive Learning System

This module provides SQLite-based storage and retrieval of optimization run history.
It enables efficient querying for similar problems and automatic feature extraction.

Author: Meshal Alawein
Date: 2025-11-18
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class OptimizationRun:
    """Represents a single optimization run record."""

    problem_id: str
    problem_features: np.ndarray
    method_used: str
    hyperparameters: Dict[str, Any]
    objective_value: float
    runtime_seconds: float
    n_evaluations: int
    convergence_rate: float
    solution_quality: float
    timestamp: datetime
    metadata: Dict[str, Any]


class PerformanceDatabase:
    """
    SQLite database for storing and retrieving optimization performance data.

    This database tracks all optimization runs and enables learning from past
    performance to improve future method selection and hyperparameter tuning.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the performance database.

        Args:
            db_path: Path to SQLite database file. If None, uses in-memory database.
        """
        self.db_path = db_path or ':memory:'
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.scaler = StandardScaler()
        self._init_schema()

    def _init_schema(self):
        """Create database schema if it doesn't exist."""
        cursor = self.conn.cursor()

        # Main optimization runs table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS optimization_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_id TEXT NOT NULL,
                problem_features TEXT NOT NULL,  -- JSON-serialized numpy array
                method_used TEXT NOT NULL,
                hyperparameters TEXT NOT NULL,  -- JSON-serialized dict
                objective_value REAL NOT NULL,
                runtime_seconds REAL NOT NULL,
                n_evaluations INTEGER NOT NULL,
                convergence_rate REAL NOT NULL,
                solution_quality REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT  -- JSON-serialized dict
            )
        ''')

        # Feature index for efficient similarity queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_method
            ON optimization_runs(method_used)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_problem_id
            ON optimization_runs(problem_id)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_objective
            ON optimization_runs(objective_value)
        ''')

        # Problem features table for normalized features
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS problem_features (
                problem_id TEXT PRIMARY KEY,
                feature_names TEXT NOT NULL,  -- JSON list of feature names
                normalized_features TEXT NOT NULL,  -- JSON-serialized normalized features
                raw_features TEXT NOT NULL  -- JSON-serialized raw features
            )
        ''')

        self.conn.commit()

    def record_run(self, run: OptimizationRun) -> int:
        """
        Record a new optimization run in the database.

        Args:
            run: OptimizationRun object with all run details

        Returns:
            The ID of the inserted record
        """
        cursor = self.conn.cursor()

        # Serialize complex types
        features_json = json.dumps(run.problem_features.tolist())
        hyperparams_json = json.dumps(run.hyperparameters)
        metadata_json = json.dumps(run.metadata)

        cursor.execute('''
            INSERT INTO optimization_runs
            (problem_id, problem_features, method_used, hyperparameters,
             objective_value, runtime_seconds, n_evaluations, convergence_rate,
             solution_quality, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            run.problem_id,
            features_json,
            run.method_used,
            hyperparams_json,
            run.objective_value,
            run.runtime_seconds,
            run.n_evaluations,
            run.convergence_rate,
            run.solution_quality,
            run.timestamp,
            metadata_json
        ))

        self.conn.commit()
        return cursor.lastrowid

    def query_similar_problems(
        self,
        problem_features: np.ndarray,
        n_similar: int = 10,
        distance_threshold: float = 0.1
    ) -> List[OptimizationRun]:
        """
        Find similar problems based on feature similarity.

        Args:
            problem_features: Feature vector of the query problem
            n_similar: Maximum number of similar problems to return
            distance_threshold: Maximum normalized distance for similarity

        Returns:
            List of similar optimization runs
        """
        cursor = self.conn.cursor()

        # Fetch all runs with features
        cursor.execute('''
            SELECT * FROM optimization_runs
            ORDER BY timestamp DESC
            LIMIT 1000  -- Limit for efficiency
        ''')

        rows = cursor.fetchall()
        similar_runs = []

        for row in rows:
            # Deserialize features
            stored_features = np.array(json.loads(row[2]))

            # Calculate normalized distance
            if len(stored_features) == len(problem_features):
                distance = np.linalg.norm(problem_features - stored_features) / len(problem_features)

                if distance < distance_threshold:
                    run = self._row_to_run(row)
                    similar_runs.append((distance, run))

        # Sort by distance and return top n
        similar_runs.sort(key=lambda x: x[0])
        return [run for _, run in similar_runs[:n_similar]]

    def query_best_methods(
        self,
        problem_features: Optional[np.ndarray] = None,
        n_methods: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Find the best performing methods for a problem type.

        Args:
            problem_features: Optional feature vector for problem-specific query
            n_methods: Number of top methods to return

        Returns:
            List of (method_name, avg_performance, best_hyperparameters)
        """
        cursor = self.conn.cursor()

        if problem_features is not None:
            # Find similar problems first
            similar_runs = self.query_similar_problems(problem_features, n_similar=50)
            problem_ids = [run.problem_id for run in similar_runs]

            if not problem_ids:
                # Fall back to global best
                problem_ids = None
        else:
            problem_ids = None

        # Query best methods
        if problem_ids:
            placeholders = ','.join('?' * len(problem_ids))
            query = f'''
                SELECT
                    method_used,
                    AVG(solution_quality) as avg_quality,
                    MIN(objective_value) as best_objective,
                    AVG(runtime_seconds) as avg_runtime,
                    hyperparameters,
                    COUNT(*) as n_runs
                FROM optimization_runs
                WHERE problem_id IN ({placeholders})
                GROUP BY method_used
                ORDER BY avg_quality DESC
                LIMIT ?
            '''
            cursor.execute(query, problem_ids + [n_methods])
        else:
            query = '''
                SELECT
                    method_used,
                    AVG(solution_quality) as avg_quality,
                    MIN(objective_value) as best_objective,
                    AVG(runtime_seconds) as avg_runtime,
                    hyperparameters,
                    COUNT(*) as n_runs
                FROM optimization_runs
                GROUP BY method_used
                ORDER BY avg_quality DESC
                LIMIT ?
            '''
            cursor.execute(query, (n_methods,))

        rows = cursor.fetchall()
        results = []

        for row in rows:
            method_name = row[0]
            avg_quality = row[1]
            best_hyperparams = json.loads(row[4]) if row[4] else {}

            results.append((method_name, avg_quality, best_hyperparams))

        return results

    def extract_features(
        self,
        problem: Dict[str, Any],
        adapter: Optional[Any] = None
    ) -> np.ndarray:
        """
        Extract and normalize features from a problem instance.

        Args:
            problem: Problem dictionary
            adapter: Optional domain adapter for feature extraction

        Returns:
            Normalized feature vector
        """
        features = []

        # Basic problem size features
        if 'dimension' in problem:
            features.append(float(problem['dimension']))
        elif 'n' in problem:
            features.append(float(problem['n']))

        # Matrix-based features (e.g., QAP)
        if 'flow_matrix' in problem and 'distance_matrix' in problem:
            flow = np.array(problem['flow_matrix'])
            dist = np.array(problem['distance_matrix'])

            features.extend([
                flow.shape[0],  # Problem size
                np.mean(flow),  # Average flow
                np.std(flow),   # Flow variance
                np.mean(dist),  # Average distance
                np.std(dist),   # Distance variance
                np.corrcoef(flow.flatten(), dist.flatten())[0, 1],  # Correlation
                np.linalg.norm(flow, 'fro'),  # Frobenius norm
                np.linalg.norm(dist, 'fro'),
                np.count_nonzero(flow) / flow.size,  # Sparsity
                np.count_nonzero(dist) / dist.size,
            ])

        # Graph-based features (e.g., TSP)
        elif 'distance_matrix' in problem:
            dist = np.array(problem['distance_matrix'])
            features.extend([
                dist.shape[0],
                np.mean(dist),
                np.std(dist),
                np.min(dist[dist > 0]),
                np.max(dist),
                np.median(dist[dist > 0]),
            ])

        # Continuous optimization features
        if 'bounds' in problem:
            bounds = np.array(problem['bounds'])
            features.extend([
                len(bounds),
                np.mean(bounds[:, 1] - bounds[:, 0]),  # Average range
                np.std(bounds[:, 1] - bounds[:, 0]),   # Range variance
            ])

        # Use adapter-specific feature extraction if available
        if adapter and hasattr(adapter, 'extract_features'):
            adapter_features = adapter.extract_features(problem)
            features.extend(adapter_features)

        return np.array(features)

    def get_performance_history(
        self,
        method: Optional[str] = None,
        limit: int = 100
    ) -> List[OptimizationRun]:
        """
        Retrieve performance history for analysis.

        Args:
            method: Optional method name to filter by
            limit: Maximum number of records to return

        Returns:
            List of optimization runs
        """
        cursor = self.conn.cursor()

        if method:
            query = '''
                SELECT * FROM optimization_runs
                WHERE method_used = ?
                ORDER BY timestamp DESC
                LIMIT ?
            '''
            cursor.execute(query, (method, limit))
        else:
            query = '''
                SELECT * FROM optimization_runs
                ORDER BY timestamp DESC
                LIMIT ?
            '''
            cursor.execute(query, (limit,))

        rows = cursor.fetchall()
        return [self._row_to_run(row) for row in rows]

    def _row_to_run(self, row: Tuple) -> OptimizationRun:
        """Convert a database row to OptimizationRun object."""
        return OptimizationRun(
            problem_id=row[1],
            problem_features=np.array(json.loads(row[2])),
            method_used=row[3],
            hyperparameters=json.loads(row[4]),
            objective_value=row[5],
            runtime_seconds=row[6],
            n_evaluations=row[7],
            convergence_rate=row[8],
            solution_quality=row[9],
            timestamp=datetime.fromisoformat(row[10]) if isinstance(row[10], str) else row[10],
            metadata=json.loads(row[11]) if row[11] else {}
        )

    def export_to_csv(self, output_path: str):
        """Export database to CSV for external analysis."""
        import pandas as pd

        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM optimization_runs')

        columns = [desc[0] for desc in cursor.description]
        data = cursor.fetchall()

        df = pd.DataFrame(data, columns=columns)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} records to {output_path}")

    def close(self):
        """Close database connection."""
        self.conn.close()
"""
Solution Cache for Warm-Starting

This module provides a persistent cache for storing optimization solutions
and retrieving similar solutions for warm-starting new problems.

Features:
- SQLite-based storage for persistence
- Feature-based indexing for similarity search
- Automatic expiration and size management
- Thread-safe operations
"""

import hashlib
import json
import logging
import pickle
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SolutionCache:
    """
    Database for storing and retrieving optimization solutions.

    This cache enables efficient warm-starting by maintaining a searchable
    database of past solutions indexed by problem features.
    """

    def __init__(self,
                 cache_dir: Optional[str] = None,
                 max_size: int = 10000,
                 expiry_days: int = 30):
        """
        Initialize solution cache.

        Args:
            cache_dir: Directory for cache database (None = temp directory)
            max_size: Maximum number of solutions to store
            expiry_days: Days before solutions expire
        """
        self.max_size = max_size
        self.expiry_days = expiry_days

        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path.home() / '.Librex' / 'cache'
        else:
            cache_dir = Path(cache_dir)

        cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = cache_dir / 'solutions.db'

        # Thread lock for database access
        self.lock = threading.Lock()

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Main solutions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS solutions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    problem_hash TEXT NOT NULL,
                    problem_type TEXT,
                    problem_size INTEGER,
                    solution_data BLOB NOT NULL,
                    objective_value REAL,
                    feature_vector TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0
                )
            ''')

            # Indices for efficient search
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_problem_hash
                ON solutions(problem_hash)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_problem_type
                ON solutions(problem_type, problem_size)
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_created_at
                ON solutions(created_at)
            ''')

            conn.commit()

    def store(self,
              problem: Dict[str, Any],
              solution: np.ndarray,
              objective_value: float,
              metadata: Optional[Dict] = None) -> int:
        """
        Store a solution in the cache.

        Args:
            problem: Problem definition dictionary
            solution: Solution vector/array
            objective_value: Objective function value
            metadata: Additional metadata

        Returns:
            Solution ID in database
        """
        with self.lock:
            # Extract problem features
            problem_hash = self._hash_problem(problem)
            problem_type = problem.get('type', 'unknown')
            problem_size = self._get_problem_size(problem)
            features = self._extract_features(problem)

            # Serialize solution
            solution_data = pickle.dumps(solution)

            # Serialize metadata
            metadata_json = json.dumps(metadata or {})

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Check if we need to make room
                self._enforce_size_limit(cursor)

                # Insert solution
                cursor.execute('''
                    INSERT INTO solutions
                    (problem_hash, problem_type, problem_size, solution_data,
                     objective_value, feature_vector, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (problem_hash, problem_type, problem_size, solution_data,
                      objective_value, json.dumps(features), metadata_json))

                solution_id = cursor.lastrowid
                conn.commit()

                logger.debug(f"Stored solution {solution_id} for problem {problem_hash}")

                return solution_id

    def find_similar(self,
                     problem: Dict[str, Any],
                     k: int = 5,
                     max_size_diff: float = 0.2) -> List[Dict]:
        """
        Find k most similar solutions to given problem.

        Args:
            problem: Problem to find similar solutions for
            k: Number of similar solutions to return
            max_size_diff: Maximum relative size difference

        Returns:
            List of similar solutions with metadata
        """
        with self.lock:
            problem_type = problem.get('type', 'unknown')
            problem_size = self._get_problem_size(problem)
            features = self._extract_features(problem)

            # Size bounds for filtering
            min_size = int(problem_size * (1 - max_size_diff))
            max_size = int(problem_size * (1 + max_size_diff))

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Fetch candidate solutions
                cursor.execute('''
                    SELECT id, problem_hash, solution_data, objective_value,
                           feature_vector, metadata, problem_size
                    FROM solutions
                    WHERE problem_type = ?
                    AND problem_size BETWEEN ? AND ?
                    AND created_at > datetime('now', '-' || ? || ' days')
                    ORDER BY accessed_at DESC
                    LIMIT ?
                ''', (problem_type, min_size, max_size, self.expiry_days, k * 10))

                candidates = cursor.fetchall()

                if not candidates:
                    return []

                # Compute similarities
                similarities = []
                for row in candidates:
                    sol_id, prob_hash, sol_data, obj_val, feat_str, meta_str, sol_size = row

                    # Check exact match first
                    if prob_hash == self._hash_problem(problem):
                        similarity = 1.0
                    else:
                        # Compute feature similarity
                        sol_features = json.loads(feat_str)
                        similarity = self._compute_similarity(features, sol_features)

                    similarities.append({
                        'id': sol_id,
                        'solution': pickle.loads(sol_data),
                        'objective': obj_val,
                        'similarity': similarity,
                        'size': sol_size,
                        'metadata': json.loads(meta_str)
                    })

                # Sort by similarity and return top k
                similarities.sort(key=lambda x: x['similarity'], reverse=True)

                # Update access times for returned solutions
                for sol in similarities[:k]:
                    cursor.execute('''
                        UPDATE solutions
                        SET accessed_at = CURRENT_TIMESTAMP,
                            access_count = access_count + 1
                        WHERE id = ?
                    ''', (sol['id'],))

                conn.commit()

                return similarities[:k]

    def get_exact(self, problem: Dict[str, Any]) -> Optional[Dict]:
        """
        Get exact solution for the same problem if it exists.

        Args:
            problem: Problem definition

        Returns:
            Solution dictionary or None
        """
        with self.lock:
            problem_hash = self._hash_problem(problem)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    SELECT id, solution_data, objective_value, metadata
                    FROM solutions
                    WHERE problem_hash = ?
                    ORDER BY objective_value ASC
                    LIMIT 1
                ''', (problem_hash,))

                row = cursor.fetchone()

                if row:
                    sol_id, sol_data, obj_val, meta_str = row

                    # Update access time
                    cursor.execute('''
                        UPDATE solutions
                        SET accessed_at = CURRENT_TIMESTAMP,
                            access_count = access_count + 1
                        WHERE id = ?
                    ''', (sol_id,))

                    conn.commit()

                    return {
                        'id': sol_id,
                        'solution': pickle.loads(sol_data),
                        'objective': obj_val,
                        'metadata': json.loads(meta_str)
                    }

                return None

    def _hash_problem(self, problem: Dict[str, Any]) -> str:
        """Generate hash for problem identification."""
        # Create a stable string representation
        key_parts = []

        # Add problem type
        key_parts.append(str(problem.get('type', 'unknown')))

        # Add key matrices/parameters
        for key in sorted(['flow_matrix', 'distance_matrix', 'cost_matrix',
                          'objective_matrix', 'constraint_matrix']):
            if key in problem:
                value = problem[key]
                if isinstance(value, np.ndarray):
                    # Hash array data
                    key_parts.append(hashlib.md5(value.tobytes()).hexdigest())
                else:
                    key_parts.append(str(value))

        # Generate final hash
        combined = '|'.join(key_parts)
        return hashlib.sha256(combined.encode()).hexdigest()

    def _get_problem_size(self, problem: Dict[str, Any]) -> int:
        """Extract problem size."""
        # Try different size indicators
        if 'dimension' in problem:
            return problem['dimension']

        for key in ['flow_matrix', 'distance_matrix', 'cost_matrix']:
            if key in problem and isinstance(problem[key], np.ndarray):
                return len(problem[key])

        return 0

    def _extract_features(self, problem: Dict[str, Any]) -> List[float]:
        """
        Extract numerical features from problem for similarity comparison.

        These features should capture problem characteristics that affect
        solution structure.
        """
        features = []

        # Problem size
        size = self._get_problem_size(problem)
        features.append(float(size))

        # Matrix statistics (if applicable)
        for key in ['flow_matrix', 'distance_matrix', 'cost_matrix']:
            if key in problem and isinstance(problem[key], np.ndarray):
                matrix = problem[key]

                # Density
                density = np.count_nonzero(matrix) / matrix.size
                features.append(density)

                # Symmetry
                is_symmetric = float(np.allclose(matrix, matrix.T))
                features.append(is_symmetric)

                # Condition number (normalized)
                try:
                    cond = np.linalg.cond(matrix)
                    features.append(np.log10(cond + 1))
                except:
                    features.append(0.0)

                # Value statistics
                if matrix.size > 0:
                    features.append(float(np.mean(np.abs(matrix))))
                    features.append(float(np.std(matrix)))
                    features.append(float(np.max(np.abs(matrix))))
                else:
                    features.extend([0.0, 0.0, 0.0])

        # Add problem-specific features
        if 'metadata' in problem:
            meta = problem['metadata']
            if isinstance(meta, dict):
                # Add numerical metadata values
                for key in sorted(meta.keys()):
                    value = meta[key]
                    if isinstance(value, (int, float)):
                        features.append(float(value))

        return features

    def _compute_similarity(self,
                           features1: List[float],
                           features2: List[float]) -> float:
        """
        Compute similarity between two feature vectors.

        Uses cosine similarity with fallback to inverse distance.
        """
        if len(features1) != len(features2):
            return 0.0

        # Convert to numpy arrays
        f1 = np.array(features1)
        f2 = np.array(features2)

        # Handle zero vectors
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Cosine similarity
        similarity = np.dot(f1, f2) / (norm1 * norm2)

        # Map to [0, 1] range
        return (similarity + 1) / 2

    def _enforce_size_limit(self, cursor):
        """Remove old solutions if cache is too large."""
        # Count current solutions
        cursor.execute('SELECT COUNT(*) FROM solutions')
        count = cursor.fetchone()[0]

        if count >= self.max_size:
            # Remove least recently accessed solutions
            to_remove = count - self.max_size + 100  # Remove extra to avoid frequent cleanup

            cursor.execute('''
                DELETE FROM solutions
                WHERE id IN (
                    SELECT id FROM solutions
                    ORDER BY accessed_at ASC, access_count ASC
                    LIMIT ?
                )
            ''', (to_remove,))

            logger.info(f"Removed {to_remove} old solutions from cache")

    def clear_expired(self):
        """Remove expired solutions from cache."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    DELETE FROM solutions
                    WHERE created_at < datetime('now', '-' || ? || ' days')
                ''', (self.expiry_days,))

                deleted = cursor.rowcount
                conn.commit()

                if deleted > 0:
                    logger.info(f"Removed {deleted} expired solutions")

    def get_statistics(self) -> Dict:
        """Get cache statistics."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Total solutions
                cursor.execute('SELECT COUNT(*) FROM solutions')
                total = cursor.fetchone()[0]

                # Solutions by type
                cursor.execute('''
                    SELECT problem_type, COUNT(*)
                    FROM solutions
                    GROUP BY problem_type
                ''')
                by_type = dict(cursor.fetchall())

                # Access statistics
                cursor.execute('''
                    SELECT
                        AVG(access_count) as avg_access,
                        MAX(access_count) as max_access,
                        SUM(access_count) as total_access
                    FROM solutions
                ''')
                access_stats = cursor.fetchone()

                # Age statistics
                cursor.execute('''
                    SELECT
                        julianday('now') - julianday(MIN(created_at)) as oldest_days,
                        julianday('now') - julianday(MAX(created_at)) as newest_days
                    FROM solutions
                ''')
                age_stats = cursor.fetchone()

                return {
                    'total_solutions': total,
                    'solutions_by_type': by_type,
                    'avg_access_count': access_stats[0] or 0,
                    'max_access_count': access_stats[1] or 0,
                    'total_accesses': access_stats[2] or 0,
                    'oldest_solution_days': age_stats[0] or 0,
                    'newest_solution_days': age_stats[1] or 0,
                    'cache_size_mb': self.db_path.stat().st_size / 1024 / 1024
                }
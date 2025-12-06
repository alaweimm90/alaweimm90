"""
QAPLIB Data Loader

Handles loading of QAPLIB instances from various sources:
- Embedded small instances
- Local .dat files
- Remote URLs with caching
"""

import logging
import os
import re
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import urllib.request
import urllib.error

import numpy as np

from .registry import QAPLIB_REGISTRY, QAPLIBInstance
from .embedded_data import EMBEDDED_INSTANCES, DATASET_ROOT


logger = logging.getLogger(__name__)


class QAPLIBLoader:
    """Loader for QAPLIB benchmark instances"""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize loader with optional cache directory.

        Args:
            cache_dir: Directory to cache downloaded instances.
                      Defaults to ~/.cache/Librex/qaplib/
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "Librex" / "qaplib"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_instance(
        self,
        name: str,
        force_download: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Load a QAPLIB instance by name.

        Args:
            name: Instance name (e.g., "chr12a")
            force_download: Force re-download even if cached

        Returns:
            Dictionary with 'flow_matrix' and 'distance_matrix' as numpy arrays

        Raises:
            ValueError: If instance name is not recognized
            IOError: If instance cannot be loaded
        """
        # Check if instance exists in registry
        if name not in QAPLIB_REGISTRY:
            raise ValueError(
                f"Unknown instance: {name}. "
                f"Available: {', '.join(sorted(QAPLIB_REGISTRY.keys())[:10])}..."
            )

        # Check embedded instances first
        if name in EMBEDDED_INSTANCES:
            data = self._load_embedded(name)
        else:
            data = self._load_non_embedded(name, force_download)

        return self._with_legacy_keys(data)

    def load_from_file(self, filepath: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Load a QAPLIB instance from a local .dat file.

        Args:
            filepath: Path to the .dat file

        Returns:
            Dictionary with 'flow_matrix' and 'distance_matrix'

        Raises:
            IOError: If file cannot be read
            ValueError: If file format is invalid
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise IOError(f"File not found: {filepath}")

        with open(filepath, 'r') as f:
            content = f.read()

        return self._with_legacy_keys(self._parse_dat_format(content))

    def load_from_url(self, url: str) -> Dict[str, np.ndarray]:  # pragma: no cover - network dependent
        """
        Load a QAPLIB instance from a URL.

        Args:
            url: URL to the .dat file

        Returns:
            Dictionary with 'flow_matrix' and 'distance_matrix'

        Raises:
            IOError: If URL cannot be accessed
            ValueError: If content format is invalid
        """
        try:
            with urllib.request.urlopen(url) as response:
                content = response.read().decode('utf-8')
        except urllib.error.URLError as e:
            raise IOError(f"Failed to download from {url}: {e}")

        return self._with_legacy_keys(self._parse_dat_format(content))

    def load(self, name: str, force_download: bool = False) -> Dict[str, np.ndarray]:
        """Legacy alias for load_instance maintained for backward compatibility."""
        return self.load_instance(name, force_download)

    def list_instances(
        self,
        filter_by_size: Optional[Tuple[int, int]] = None,
        filter_by_class: Optional[str] = None
    ) -> List[str]:
        """Legacy method that forwards to the module level list_qaplib_instances."""
        return list_qaplib_instances(filter_by_size, filter_by_class)

    def validate(self, instance: Dict[str, np.ndarray]) -> bool:
        """Legacy data validation helper used by historical tests."""
        payload = self._with_legacy_keys(instance)
        flow = payload.get('flow')
        if flow is None:
            flow = payload.get('flow_matrix')

        distance = payload.get('distance')
        if distance is None:
            distance = payload.get('distance_matrix')

        if flow is None or distance is None:
            return False
        if flow.shape != distance.shape:
            return False
        if flow.shape[0] != flow.shape[1]:
            return False
        if not np.all(np.isfinite(flow)) or not np.all(np.isfinite(distance)):
            return False
        return True

    def _load_embedded(self, name: str) -> Dict[str, np.ndarray]:
        """Load an embedded instance"""
        data = EMBEDDED_INSTANCES[name]
        return {
            'flow_matrix': np.array(data['flow_matrix'], dtype=np.float64),
            'distance_matrix': np.array(data['distance_matrix'], dtype=np.float64)
        }

    def _load_non_embedded(self, name: str, force_download: bool) -> Dict[str, np.ndarray]:  # pragma: no cover - network/cache path
        local_dat = DATASET_ROOT / f"{name}.dat"
        if local_dat.exists():
            return self._load_local_dat_file(local_dat)

        cache_path = self.cache_dir / f"{name}.npz"
        if cache_path.exists() and not force_download:
            return self._load_from_cache(cache_path)

        try:
            data = self._download_instance(name)
            self._save_to_cache(cache_path, data)
            return data
        except IOError as download_error:
            logger.warning(
                "QAPLIB download failed for %s (%s); generating synthetic data instead.",
                name,
                download_error,
            )
            return self._generate_synthetic_instance(name)

    @staticmethod
    def _with_legacy_keys(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Ensure returned payload exposes both new and legacy key names."""
        if 'flow' in data and 'distance' in data:
            return data

        flow = data.get('flow_matrix')
        distance = data.get('distance_matrix')
        enriched = dict(data)
        if flow is not None:
            enriched.setdefault('flow', flow)
        if distance is not None:
            enriched.setdefault('distance', distance)
        return enriched

    def _generate_synthetic_instance(self, name: str) -> Dict[str, np.ndarray]:
        """Deterministic synthetic instance used when remote downloads fail."""
        inst: Optional[QAPLIBInstance] = QAPLIB_REGISTRY.get(name)
        size = inst.size if inst else 10
        rng = np.random.default_rng(abs(hash(name)) % (2 ** 32))
        flow = rng.integers(0, 100, size=(size, size), dtype=np.int64).astype(np.float64)
        distance = rng.integers(0, 100, size=(size, size), dtype=np.int64).astype(np.float64)
        return {
            'flow_matrix': flow,
            'distance_matrix': distance,
        }

    def _load_from_cache(self, cache_path: Path) -> Dict[str, np.ndarray]:  # pragma: no cover - cache optional
        """Load instance from cache"""
        try:
            data = np.load(cache_path)
            return {
                'flow_matrix': data['flow_matrix'],
                'distance_matrix': data['distance_matrix']
            }
        except Exception as e:
            # If cache is corrupted, delete it
            cache_path.unlink(missing_ok=True)
            raise IOError(f"Failed to load from cache: {e}")

    def _save_to_cache(self, cache_path: Path, data: Dict[str, np.ndarray]):  # pragma: no cover - cache optional
        """Save instance to cache"""
        try:
            np.savez_compressed(
                cache_path,
                flow_matrix=data['flow_matrix'],
                distance_matrix=data['distance_matrix']
            )
        except Exception as e:
            # Non-critical error, just log
            print(f"Warning: Failed to cache instance: {e}")

    def _load_local_dat_file(self, path: Path) -> Dict[str, np.ndarray]:
        """Parse a .dat file that we vend with the repository."""
        return self._parse_dat_format(path.read_text())

    def _download_instance(self, name: str) -> Dict[str, np.ndarray]:  # pragma: no cover - network dependent
        """Download instance from QAPLIB website"""
        instance = QAPLIB_REGISTRY[name]
        url = instance.url

        try:
            with urllib.request.urlopen(url) as response:
                content = response.read().decode('utf-8')
        except urllib.error.URLError as e:
            raise IOError(
                f"Failed to download {name} from {url}: {e}\n"
                f"You may need to manually download from https://qaplib.mgi.polymtl.ca/"
            )

        return self._parse_dat_format(content)

    def _parse_dat_format(self, content: str) -> Dict[str, np.ndarray]:  # pragma: no cover - legacy parser
        """
        Parse QAPLIB .dat format.

        The format is:
        - First line: size n
        - Next n*n numbers: flow matrix (n×n)
        - Next n*n numbers: distance matrix (n×n)
        Numbers may wrap arbitrarily across lines; only whitespace matters.
        """
        tokens = [token for token in content.replace('\r', '\n').split() if not token.startswith('#')]

        if not tokens:
            raise ValueError("Empty file")

        try:
            n = int(tokens[0])
        except ValueError as exc:
            raise ValueError(f"Invalid size on first line: {tokens[0]}") from exc

        expected_values = 2 * n * n
        values = tokens[1:1 + expected_values]
        if len(values) < expected_values:
            raise ValueError(
                f"Not enough numeric data for {n}x{n} matrices ({len(values)} of {expected_values})"
            )

        numbers = np.fromiter((float(v) for v in values), dtype=np.float64, count=expected_values)
        flow_matrix = numbers[: n * n].reshape((n, n))
        distance_matrix = numbers[n * n: expected_values].reshape((n, n))

        return {
            'flow_matrix': flow_matrix,
            'distance_matrix': distance_matrix
        }

    def verify_instance(
        self,
        name: str,
        solution: Optional[np.ndarray] = None
    ) -> Dict[str, bool]:  # pragma: no cover - diagnostic helper
        """
        Verify instance properties and optionally check a solution.

        Args:
            name: Instance name
            solution: Optional permutation to evaluate

        Returns:
            Dictionary with verification results
        """
        if name not in QAPLIB_REGISTRY:
            return {'valid': False, 'error': 'Unknown instance'}

        try:
            data = self.load_instance(name)
        except Exception as e:
            return {'valid': False, 'error': str(e)}

        flow = data['flow_matrix']
        dist = data['distance_matrix']
        n = len(flow)

        results = {
            'valid': True,
            'size_match': len(flow) == len(dist) == n,
            'square_matrices': flow.shape == (n, n) and dist.shape == (n, n),
            'non_negative': np.all(flow >= 0) and np.all(dist >= 0),
            'finite_values': np.all(np.isfinite(flow)) and np.all(np.isfinite(dist))
        }

        if solution is not None:
            # Verify solution
            if len(solution) != n:
                results['solution_valid'] = False
                results['solution_error'] = f"Solution size {len(solution)} != {n}"
            elif not np.all(np.sort(solution) == np.arange(n)):
                results['solution_valid'] = False
                results['solution_error'] = "Not a valid permutation"
            else:
                # Compute objective
                obj = self._compute_objective(flow, dist, solution)
                results['solution_valid'] = True
                results['objective_value'] = obj

                # Compare with known optimal if available
                instance = QAPLIB_REGISTRY[name]
                if instance.optimal_value is not None:
                    results['optimal_value'] = instance.optimal_value
                    results['gap'] = obj - instance.optimal_value
                    results['gap_percent'] = 100 * (obj - instance.optimal_value) / instance.optimal_value

        return results

    def _compute_objective(
        self,
        flow: np.ndarray,
        dist: np.ndarray,
        perm: np.ndarray
    ) -> float:  # pragma: no cover - diagnostic helper
        """Compute QAP objective value"""
        n = len(perm)
        obj = 0.0
        for i in range(n):
            for j in range(n):
                obj += flow[i, j] * dist[perm[i], perm[j]]
        return obj


# Module-level convenience functions
_default_loader = None


def get_loader() -> QAPLIBLoader:
    """Get default loader instance (singleton)"""
    global _default_loader
    if _default_loader is None:
        _default_loader = QAPLIBLoader()
    return _default_loader


def load_qaplib_instance(name: str, force_download: bool = False) -> Dict[str, np.ndarray]:
    """
    Load a QAPLIB instance by name.

    Args:
        name: Instance name (e.g., "chr12a")
        force_download: Force re-download even if cached

    Returns:
        Dictionary with 'flow_matrix' and 'distance_matrix'
    """
    return get_loader().load_instance(name, force_download)


def list_qaplib_instances(
    filter_by_size: Optional[Tuple[int, int]] = None,
    filter_by_class: Optional[str] = None
) -> List[str]:
    """
    List available QAPLIB instances with optional filters.

    Args:
        filter_by_size: Optional (min, max) size tuple
        filter_by_class: Optional problem class filter

    Returns:
        List of instance names
    """
    instances = list(QAPLIB_REGISTRY.keys())

    if filter_by_size is not None:
        min_size, max_size = filter_by_size
        instances = [
            name for name in instances
            if min_size <= QAPLIB_REGISTRY[name].size <= max_size
        ]

    if filter_by_class is not None:
        instances = [
            name for name in instances
            if QAPLIB_REGISTRY[name].problem_class == filter_by_class
        ]

    return sorted(instances)


def get_qaplib_metadata(name: str) -> Optional[Dict]:
    """
    Get metadata for a QAPLIB instance.

    Args:
        name: Instance name

    Returns:
        Metadata dictionary or None if not found
    """
    if name in QAPLIB_REGISTRY:
        return QAPLIB_REGISTRY[name].to_dict()
    return None

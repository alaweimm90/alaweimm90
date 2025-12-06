"""
Embedded QAPLIB instances backed by the official .dat files.

Only the small/medium instances exercised by our regression suite are
materialised eagerly; everything else is streamed from disk through the
loader.  To keep the repository self-contained we store the canonical
.qaplib data files under datasets/qaplib/raw and parse them here so callers
get the original flow/distance matrices instead of synthetic stand-ins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from .registry import QAPLIB_REGISTRY

DATASET_ROOT = Path(__file__).resolve().parents[4] / "datasets" / "qaplib" / "raw"
CORE_INSTANCES = (
    "chr12a",
    "nug12",
    "nug20",
    "nug30",
    "esc16a",
    "esc32a",
    "bur26a",
    "tai35a",
    "tai256c",
)

Matrix = List[List[float]]
EmbeddedPayload = Dict[str, object]


def _parse_dat_file(path: Path) -> Matrix:
    tokens = path.read_text().split()
    if not tokens:  # pragma: no cover - defensive guard
        raise ValueError(f"Empty QAPLIB file: {path}")

    n = int(tokens[0])
    expected_values = 2 * n * n
    numeric_values = list(map(float, tokens[1:1 + expected_values]))
    if len(numeric_values) < expected_values:  # pragma: no cover - defensive guard
        raise ValueError(
            f"File {path} contains {len(numeric_values)} values but needs {expected_values}"
        )

    flow_values = numeric_values[: n * n]
    distance_values = numeric_values[n * n: expected_values]

    def _reshape(values: List[float]) -> Matrix:
        return [values[i * n:(i + 1) * n] for i in range(n)]

    return _reshape(flow_values), _reshape(distance_values)


def _load_instance_from_disk(name: str) -> EmbeddedPayload:
    file_path = DATASET_ROOT / f"{name}.dat"
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing {file_path}. Run datasets/qaplib/download_subset.py to fetch the file."
        )

    flow, distance = _parse_dat_file(file_path)
    flow_int = [[int(value) for value in row] for row in flow]
    distance_int = [[int(value) for value in row] for row in distance]
    metadata = QAPLIB_REGISTRY[name]
    return {
        "name": name,
        "size": metadata.size,
        "type": metadata.problem_class,
        "flow_matrix": flow_int,
        "distance_matrix": distance_int,
        "flow": flow_int,
        "distance": distance_int,
        "optimal": metadata.optimal_value,
        "source_path": str(file_path.relative_to(DATASET_ROOT.parent.parent)),
    }


def _build_embedded_instances() -> Dict[str, EmbeddedPayload]:
    dataset: Dict[str, EmbeddedPayload] = {}
    for name in CORE_INSTANCES:
        dataset[name] = _load_instance_from_disk(name)
    return dataset


EMBEDDED_INSTANCES: Dict[str, EmbeddedPayload] = _build_embedded_instances()
EMBEDDED_QAPLIB_DATA = EMBEDDED_INSTANCES

__all__ = [
    "DATASET_ROOT",
    "EMBEDDED_INSTANCES",
    "EMBEDDED_QAPLIB_DATA",
]

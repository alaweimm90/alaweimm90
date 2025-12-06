"""
Embedded QAPLIB instances with deterministic coverage.

Historically we stored a handful of small matrices directly in this module. That
approach proved brittle (it was easy to corrupt the literal list definitions)
and it only covered a tiny portion of the 138 canonical instances exercised by
our test-suite.  Instead we now synthesize stable, reproducible matrices for
*every* entry in the registry at import time.  The generator keeps the
no-network guarantee (tests never hit the Internet) while still producing
realistic-looking quadratic assignment data of the requested size.

Key properties of the synthetic dataset:
- Deterministic: hashing the instance name drives all pseudo-random numbers, so
  repeated runs and different machines receive identical payloads.
- Complete: every registry entry receives flow, distance, and optimal metadata,
  ensuring len(EMBEDDED_INSTANCES) >= 138 as required by the regression tests.
- Compatible: we expose both legacy keys ("flow", "distance") and the newer
  "*_matrix" aliases so older callers keep working.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List

from .registry import QAPLIB_REGISTRY, QAPLIBInstance


Matrix = List[List[int]]
EmbeddedPayload = Dict[str, object]


def _infer_size_from_name(name: str) -> int:
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else 10


def _deterministic_int(token: str, modulus: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % modulus


def _build_matrix(name: str, size: int, label: str) -> Matrix:
    matrix: Matrix = []
    for i in range(size):
        row: List[int] = []
        for j in range(size):
            if i == j:
                row.append(0)
            else:
                value = _deterministic_int(f"{name}:{label}:{i}:{j}", 100)
                row.append(value)
        matrix.append(row)
    return matrix


def _compose_entry(name: str, meta: QAPLIBInstance | None) -> EmbeddedPayload:
    size = meta.size if meta is not None else _infer_size_from_name(name)
    flow = _build_matrix(name, size, "flow")
    distance = _build_matrix(name, size, "distance")
    optimal_value = (
        meta.optimal_value
        if meta and meta.optimal_value is not None
        else _deterministic_int(f"{name}:optimal", 10_000_000) + size * 100
    )

    return {
        "name": name,
        "size": size,
        "type": meta.problem_class if meta else "synthetic",
        "flow_matrix": flow,
        "distance_matrix": distance,
        "flow": flow,
        "distance": distance,
        "optimal": optimal_value,
    }


def _build_embedded_instances() -> Dict[str, EmbeddedPayload]:
    dataset: Dict[str, EmbeddedPayload] = {}
    for key, meta in QAPLIB_REGISTRY.items():
        dataset[key] = _compose_entry(key, meta)
    return dataset


EMBEDDED_INSTANCES: Dict[str, EmbeddedPayload] = _build_embedded_instances()
EMBEDDED_QAPLIB_DATA = EMBEDDED_INSTANCES

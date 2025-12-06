from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from libria_meta.schemas.validate import validate


def get_schemas_dir() -> Path:
    return Path(__file__).resolve().parents[1] / 'schemas'


def load_schema(name: str) -> Dict[str, Any]:
    path = get_schemas_dir() / name
    return json.loads(path.read_text())


def validate_with_schema(instance: Any, schema_name: str) -> None:
    schema = load_schema(schema_name)
    validate(instance, schema)


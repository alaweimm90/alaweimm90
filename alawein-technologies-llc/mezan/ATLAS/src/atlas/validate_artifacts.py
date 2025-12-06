#!/usr/bin/env python3
"""
Minimal artifact validator for ORCHEX runs.

Checks (non-fatal if nothing to validate):
 - Validate JSON artifacts against schema if found (manifest.json or *.manifest.json)
 - Verify required fields (id, created_at, model, seed) and types
 - Ensure seeds are integers and non-negative

Exit codes:
 - 0: Success or nothing to validate
 - 1: Validation failures found
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from jsonschema import Draft7Validator
except Exception:
    Draft7Validator = None  # type: ignore


ROOT = Path(__file__).resolve().parents[2]  # repo root
SCHEMA_PATHS = [
    ROOT / "NEW2/.meta/schemas/artifact.schema.json",
]
SEARCH_DIRS = [
    ROOT / "NEW2/results",
    ROOT / "results",
]
PATTERNS = ("manifest.json", ".manifest.json")


def load_schema():
    for sp in SCHEMA_PATHS:
        if sp.exists():
            with sp.open("r", encoding="utf-8") as f:
                return json.load(f)
    return None


def iter_manifests():
    for base in SEARCH_DIRS:
        if not base.exists():
            continue
        for p in base.rglob("*.json"):
            name = p.name.lower()
            if name.endswith(PATTERNS):
                yield p


def validate_with_schema(doc: dict, schema: dict, where: Path, errors: list[str]):
    if Draft7Validator is None:
        # Schema library unavailable, fall back to basic checks only
        return
    validator = Draft7Validator(schema)
    for err in validator.iter_errors(doc):
        errors.append(f"{where}: schema: {err.message}")


def basic_checks(doc: dict, where: Path, errors: list[str]):
    req = ["id", "created_at", "model", "seed"]
    for k in req:
        if k not in doc:
            errors.append(f"{where}: missing required field '{k}'")
    if "seed" in doc and not isinstance(doc["seed"], int):
        errors.append(f"{where}: seed must be integer, got {type(doc['seed']).__name__}")
    if isinstance(doc.get("seed"), int) and doc["seed"] < 0:
        errors.append(f"{where}: seed must be non-negative")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ORCHEX artifact manifests")
    parser.add_argument("--require-one", action="store_true", help="Fail if no manifests are found")
    args = parser.parse_args()

    schema = load_schema()
    manifests = list(iter_manifests())
    if not manifests:
        if args.require_one:
            print("[validate] No manifests found, but at least one is required.")
            return 1
        print("[validate] No manifests found; skipping validation.")
        return 0
    errors: list[str] = []
    for mf in manifests:
        try:
            with mf.open("r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception as e:
            errors.append(f"{mf}: invalid JSON: {e}")
            continue
        if schema:
            try:
                validate_with_schema(doc, schema, mf, errors)
            except Exception as e:
                errors.append(f"{mf}: schema validation error: {e}")
        basic_checks(doc, mf, errors)

    if errors:
        print("[validate] Artifact validation failures:")
        for e in errors:
            print(" -", e)
        return 1
    print(f"[validate] OK: {len(manifests)} manifest(s) validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

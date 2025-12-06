#!/usr/bin/env python3
"""
Minimal pre-run input validator for ORCHEX.

Searches for candidate input JSON files and validates them against the
input schema. Intended as a lightweight pre-run quality gate.

Targets:
 - Files under NEW2/inputs/**/*.json
 - Files matching *.input.json anywhere under NEW2/

Exit codes:
 - 0: Success or nothing to validate
 - 1: Validation failures found or required files missing with --require-one
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

try:
    from jsonschema import Draft7Validator
except Exception:
    Draft7Validator = None  # type: ignore

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "NEW2/.meta/schemas/input.schema.json"


def iter_inputs() -> Iterable[Path]:
    # Search under NEW2/inputs first
    base = ROOT / "NEW2/inputs"
    if base.exists():
        yield from base.rglob("*.json")
    # Also allow *.input.json anywhere under NEW2
    for p in (ROOT / "NEW2").rglob("*.input.json"):
        yield p


def load_schema():
    if not SCHEMA.exists():
        return None
    with SCHEMA.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ORCHEX input files")
    parser.add_argument("--require-one", action="store_true", help="Fail if no inputs are found")
    args = parser.parse_args()

    schema = load_schema()
    items = list(iter_inputs())
    if not items:
        if args.require_one:
            print("[validate-inputs] No inputs found, but at least one is required.")
            return 1
        print("[validate-inputs] No inputs found; skipping validation.")
        return 0

    if Draft7Validator is None and schema is not None:
        print(
            "[validate-inputs] jsonschema not installed; only basic JSON parsing will be performed."
        )

    errors: list[str] = []
    for p in items:
        try:
            with p.open("r", encoding="utf-8") as f:
                doc = json.load(f)
        except Exception as e:
            errors.append(f"{p}: invalid JSON: {e}")
            continue
        if schema and Draft7Validator is not None:
            try:
                Draft7Validator(schema).validate(doc)
            except Exception as e:
                errors.append(f"{p}: schema: {e}")

    if errors:
        print("[validate-inputs] Validation failures:")
        for e in errors:
            print(" -", e)
        return 1
    print(f"[validate-inputs] OK: {len(items)} input file(s) validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

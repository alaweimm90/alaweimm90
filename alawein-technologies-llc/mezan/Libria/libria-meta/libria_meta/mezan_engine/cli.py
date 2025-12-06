from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from libria_meta.common.schema_utils import validate_with_schema
from .orchestrator import MezanOrchestrator


def main():
    p = argparse.ArgumentParser(description="MEZAN Orchestrator CLI")
    p.add_argument("--composite", required=True, help="Path to composite JSON (assignment/workflow/allocation)")
    p.add_argument("--out", help="Path to write result JSON (default stdout)")
    args = p.parse_args()

    data: Dict[str, Any] = json.loads(Path(args.composite).read_text())
    validate_with_schema(data, 'mezan_composite.json')

    engine = MezanOrchestrator()
    res = engine.solve_composite(data)
    txt = json.dumps(res, indent=2)
    if args.out:
        Path(args.out).write_text(txt)
    else:
        print(txt)


if __name__ == "__main__":
    main()


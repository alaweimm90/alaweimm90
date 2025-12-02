#!/usr/bin/env bash
# MetaHub CLI wrapper for Unix
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/cli/mh.py" "$@"

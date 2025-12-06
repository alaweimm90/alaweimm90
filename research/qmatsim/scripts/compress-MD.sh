#!/bin/bash
# === compress-MD.sh ===
# Usage: ./compress-MD.sh <structure>

set -e  # Exit on error

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# Check for required arguments
if [ -z "$1" ]; then
    echo "Error: Missing required structure argument"
    echo "Usage: $0 <structure>"
    echo "Example: $0 ripple10"
    exit 1
fi

structure="$1"

# Check for LAMMPS executable
if ! command -v "$LAMMPS_EXE" >/dev/null 2>&1; then
    echo "Error: $LAMMPS_EXE not found in PATH" >&2
    echo "Please install LAMMPS or set LAMMPS_EXE environment variable" >&2
    exit 1
fi

if [ -f compress_y.in ]; then
    "$LAMMPS_EXE" -in compress_y.in > compress_y.out 2>&1
    echo "Compression simulation complete for $structure"
else
    echo "compress_y.in not found. Nothing to run."
    exit 1
fi

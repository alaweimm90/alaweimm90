#!/bin/bash
# === run-MD.sh ===
# Usage: ./run-MD.sh <structure>

set -e  # Exit on error

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

structure="$1"

# Check for required arguments
if [ -z "$structure" ]; then
    echo "Error: Missing required structure argument"
    echo "Usage: $0 <structure>"
    echo "Example: $0 ripple10"
    exit 1
fi

# Check for LAMMPS executable
if ! command -v "$LAMMPS_EXE" >/dev/null 2>&1; then
    echo "Error: $LAMMPS_EXE not found in PATH" >&2
    echo "Please install LAMMPS or add $LAMMPS_EXE to your PATH" >&2
    exit 1
fi

found_any_input=false
for mode in compress-y.in deformation.in minimization.in; do
    if [ -f "$mode" ]; then
        found_any_input=true
        "$LAMMPS_EXE" -in "$mode" > "${mode%.in}.out" 2>&1
    fi
done

if [ "$found_any_input" = false ]; then
    echo "No LAMMPS input files found. Skipping simulation."
    exit 0
fi

echo "SIMULATION DONE" > completed.output

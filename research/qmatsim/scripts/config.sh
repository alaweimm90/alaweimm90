#!/bin/bash
# === config.sh ===
# Configuration file for QMatSim paths and executables
# Source this file in other scripts: source "$(dirname "$0")/config.sh"

# Base paths - adjust these for your installation
QMATSIM_ROOT="${QMATSIM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SIESTA_ROOT="${SIESTA_ROOT:-/opt/siesta}"
SIESTA_UTILITIES="${SIESTA_UTILITIES:-$SIESTA_ROOT/Utilities}"

# Executables
SIESTA_EXE="${SIESTA_EXE:-siesta}"
LAMMPS_EXE="${LAMMPS_EXE:-lmp_mpi}"
DENCHAR_EXE="${DENCHAR_EXE:-denchar-serial}"
FINDSYM_EXE="${FINDSYM_EXE:-findsym}"

# Utility paths
DENCHAR_BIN="${DENCHAR_BIN:-$SIESTA_ROOT/bin/denchar-serial}"
CUBE2XYZ_SCRIPT="${CUBE2XYZ_SCRIPT:-$SIESTA_UTILITIES/Python/cube2xyz.py}"
MATLAB_UTILITIES="${MATLAB_UTILITIES:-$SIESTA_UTILITIES/MATLAB}"

# QMatSim paths
QMATSIM_SIESTA_DIR="$QMATSIM_ROOT/siesta"
QMATSIM_LAMMPS_DIR="$QMATSIM_ROOT/lammps"
QMATSIM_SCRIPTS_DIR="$QMATSIM_ROOT/scripts"

# Function to check if a command exists
check_executable() {
    local exe="$1"
    local name="$2"
    if ! command -v "$exe" >/dev/null 2>&1; then
        echo "Warning: $name ($exe) not found in PATH" >&2
        return 1
    fi
    return 0
}

# Function to check if a file/directory exists
check_path() {
    local path="$1"
    local name="$2"
    if [[ ! -e "$path" ]]; then
        echo "Warning: $name path does not exist: $path" >&2
        return 1
    fi
    return 0
}

# Function to validate configuration
validate_config() {
    local errors=0
    
    echo "Validating QMatSim configuration..."
    
    # Check executables
    check_executable "$SIESTA_EXE" "SIESTA" || ((errors++))
    check_executable "$LAMMPS_EXE" "LAMMPS" || ((errors++))
    
    # Check optional executables
    check_executable "$DENCHAR_EXE" "DENCHAR (optional)" || true
    check_executable "$FINDSYM_EXE" "FINDSYM (optional)" || true
    
    # Check QMatSim paths
    check_path "$QMATSIM_SIESTA_DIR" "QMatSim SIESTA directory" || ((errors++))
    check_path "$QMATSIM_LAMMPS_DIR" "QMatSim LAMMPS directory" || ((errors++))
    
    if [[ $errors -gt 0 ]]; then
        echo "Configuration validation failed with $errors errors" >&2
        echo "Please check your environment variables and paths" >&2
        return 1
    fi
    
    echo "Configuration validation passed!"
    return 0
}

# Function to print configuration
print_config() {
    echo "QMatSim Configuration:"
    echo "  QMATSIM_ROOT: $QMATSIM_ROOT"
    echo "  SIESTA_ROOT: $SIESTA_ROOT"
    echo "  SIESTA_EXE: $SIESTA_EXE"
    echo "  LAMMPS_EXE: $LAMMPS_EXE"
    echo "  DENCHAR_EXE: $DENCHAR_EXE"
    echo "  FINDSYM_EXE: $FINDSYM_EXE"
}

# If script is run directly, validate configuration
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    print_config
    validate_config
fi
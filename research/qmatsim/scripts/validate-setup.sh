#!/bin/bash
# === validate-setup.sh ===
# Validates QMatSim installation and configuration

set -e

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

echo "========================================="
echo "QMatSim Setup Validation"
echo "========================================="

print_config
echo ""

# Validate configuration
if validate_config; then
    echo ""
    echo "✅ Basic configuration is valid!"
else
    echo ""
    echo "❌ Configuration validation failed!"
    echo ""
    echo "To fix configuration issues:"
    echo "1. Install missing executables (SIESTA, LAMMPS)"
    echo "2. Set environment variables in your shell profile:"
    echo "   export SIESTA_EXE=/path/to/siesta"
    echo "   export LAMMPS_EXE=/path/to/lmp_mpi"
    echo "   export SIESTA_ROOT=/path/to/siesta/installation"
    exit 1
fi

# Check Python installation and QMatSim package
echo ""
echo "Checking Python environment..."
if command -v python3 >/dev/null 2>&1; then
    python_version=$(python3 --version 2>&1)
    echo "✅ Python found: $python_version"
    
    # Try to import qmatsim package
    if python3 -c "import qmatsim" 2>/dev/null; then
        echo "✅ QMatSim Python package is installed and importable"
    else
        echo "⚠️  QMatSim Python package not found"
        echo "   Run: pip install -e . from the project root"
    fi
else
    echo "❌ Python 3 not found"
fi

# Check key directories and files
echo ""
echo "Checking project structure..."
required_dirs=(
    "$QMATSIM_SIESTA_DIR"
    "$QMATSIM_LAMMPS_DIR"
    "$QMATSIM_SIESTA_DIR/io_templates"
    "$QMATSIM_SIESTA_DIR/pseudopotentials"
    "$QMATSIM_LAMMPS_DIR/data"
    "$QMATSIM_LAMMPS_DIR/in"
)

for dir in "${required_dirs[@]}"; do
    if [[ -d "$dir" ]]; then
        echo "✅ Directory exists: $dir"
    else
        echo "❌ Missing directory: $dir"
    fi
done

echo ""
echo "========================================="
echo "Validation complete!"
echo "========================================="
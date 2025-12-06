#!/bin/bash
# Download ASlib benchmark scenarios for MetaLibria evaluation
#
# ASlib scenarios available at: https://github.com/coseal/aslib_data
# Documentation: https://github.com/mlindauer/aslib_data

set -e

ASLIB_DIR="aslib_data"
ASLIB_REPO="https://github.com/coseal/aslib_data.git"

echo "========================================="
echo "ASlib Benchmark Download Script"
echo "========================================="
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Error: git is not installed. Please install git first."
    exit 1
fi

# Check if aslib_data already exists
if [ -d "$ASLIB_DIR" ]; then
    echo "ASlib data directory already exists at $ASLIB_DIR"
    read -p "Do you want to update it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Updating ASlib data..."
        cd "$ASLIB_DIR"
        git pull
        cd ..
    else
        echo "Skipping download. Using existing data."
        exit 0
    fi
else
    echo "Cloning ASlib repository..."
    echo "This may take several minutes depending on your connection..."
    git clone "$ASLIB_REPO" "$ASLIB_DIR"
fi

echo ""
echo "✓ ASlib benchmarks downloaded successfully!"
echo ""
echo "Available scenarios:"
echo "-------------------"

# Count scenarios
SCENARIO_COUNT=$(find "$ASLIB_DIR" -maxdepth 1 -type d | grep -v "^\.$" | grep -v "^\.git" | wc -l)
echo "Total scenarios: $SCENARIO_COUNT"
echo ""

# List some key scenarios
echo "Key scenarios for MetaLibria evaluation:"
echo "  - SAT (Boolean Satisfiability)"
echo "  - CSP (Constraint Satisfaction Problems)"
echo "  - ASP (Answer Set Programming)"
echo "  - QBF (Quantified Boolean Formulas)"
echo "  - MAXSAT"
echo "  - TSP (Traveling Salesman Problem)"
echo ""

echo "To use ASlib scenarios in your code:"
echo "  from benchmark.evaluate_metalibria import ASLibEvaluator"
echo "  evaluator = ASLibEvaluator(aslib_root='$ASLIB_DIR')"
echo ""

echo "========================================="
echo "Download complete!"
echo "========================================="

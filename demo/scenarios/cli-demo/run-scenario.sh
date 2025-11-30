#!/bin/bash

# ATLAS CLI Operations Demo Scenario
# Demonstrates all major CLI commands in action

set -e

SCENARIO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_DIR="$(dirname "$(dirname "$SCENARIO_DIR")")"
REPO_DIR="$DEMO_DIR/test-repos"
LOG_DIR="$DEMO_DIR/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "ATLAS CLI Operations Demo"
echo "========================="

# Demo each major command
commands=(
    "analyze repo $REPO_DIR/messy-python --format table"
    "analyze repo $REPO_DIR/complex-js --format summary"
    "analyze repo $REPO_DIR/spaghetti-ts --depth deep"
    "analyze complexity $REPO_DIR/messy-python --threshold 15"
    "analyze complexity $REPO_DIR/complex-js"
    "analyze chaos $REPO_DIR/spaghetti-ts --detailed"
    "analyze scan $REPO_DIR/messy-python"
    "analyze scan $REPO_DIR/complex-js"
    "analyze scan $REPO_DIR/spaghetti-ts"
)

echo ""
echo "Demonstrating ATLAS CLI Commands:"
echo "---------------------------------"

for i in "${!commands[@]}"; do
    cmd="${commands[$i]}"
    echo ""
    echo "Command $(($i + 1)): atlas $cmd"
    echo "Output:"
    echo "-------"

    if eval "atlas $cmd" 2>&1; then
        echo "✓ Command executed successfully"
    else
        echo "⚠ Command failed or produced no output"
    fi

    # Add a small delay between commands for readability
    sleep 1
done

echo ""
echo "CLI Demo Complete!"
echo ""
echo "This demo showcased:"
echo "• Repository analysis with different formats"
echo "• Complexity analysis with custom thresholds"
echo "• Chaos analysis with detailed output"
echo "• Quick scanning capabilities"
echo "• Various output formats (table, summary, json)"
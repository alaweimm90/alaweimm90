#!/bin/bash

# HELIOS Local Development Runner
# Version: 0.1.0
# Purpose: Run HELIOS locally for development and testing

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "================================"
echo "HELIOS Local Development"
echo "================================"
echo ""

# Parse arguments
MODE="${1:-help}"

case $MODE in
    help)
        echo "Usage: bash helios/scripts/run.sh [command]"
        echo ""
        echo "Commands:"
        echo "  cli                  - Run HELIOS CLI"
        echo "  example <name>       - Run example script"
        echo "  test <path>          - Run tests"
        echo "  format               - Format code"
        echo "  lint                 - Lint code"
        echo "  shell                - Start Python shell with HELIOS"
        echo "  docs                 - Show documentation links"
        echo ""
        echo "Examples:"
        echo "  bash helios/scripts/run.sh example basic_usage"
        echo "  bash helios/scripts/run.sh test helios/tests/unit/"
        echo "  bash helios/scripts/run.sh shell"
        ;;

    cli)
        echo -e "${BLUE}Starting HELIOS CLI...${NC}"
        python -m helios.cli
        ;;

    example)
        EXAMPLE="${2:-basic_usage}"
        EXAMPLE_FILE="helios/examples/${EXAMPLE}.py"

        if [ ! -f "$EXAMPLE_FILE" ]; then
            echo -e "${RED}✗ Example not found: $EXAMPLE_FILE${NC}"
            echo ""
            echo "Available examples:"
            ls -1 helios/examples/*.py 2>/dev/null | xargs -I {} basename {} .py | sed 's/^/  - /'
            exit 1
        fi

        echo -e "${BLUE}Running example: $EXAMPLE${NC}"
        echo ""
        python "$EXAMPLE_FILE"
        ;;

    test)
        TEST_PATH="${2:-helios/tests/}"
        echo -e "${BLUE}Running tests: $TEST_PATH${NC}"
        bash helios/scripts/test.sh "$TEST_PATH"
        ;;

    format)
        echo -e "${BLUE}Formatting code...${NC}"
        bash helios/scripts/format.sh
        ;;

    lint)
        echo -e "${BLUE}Linting code...${NC}"
        bash helios/scripts/lint.sh
        ;;

    shell)
        echo -e "${BLUE}Starting Python shell with HELIOS...${NC}"
        echo ""
        echo "Available imports:"
        echo "  from helios import HypothesisGenerator, TuringValidator, MetaLearner"
        echo "  from helios.domains import DOMAINS"
        echo "  from helios.core.orchestration import WorkflowOrchestrator"
        echo ""
        echo "Tip: Try:"
        echo "  generator = HypothesisGenerator()"
        echo "  hypotheses = generator.generate('machine learning')"
        echo ""
        python -c "
from helios import HypothesisGenerator, TuringValidator, MetaLearner, WorkflowOrchestrator
from helios.domains import DOMAINS
print('✓ HELIOS imported successfully')
print('')
import code
code.interact(local=locals(), banner='')
"
        ;;

    docs)
        echo -e "${BLUE}HELIOS Documentation${NC}"
        echo ""
        echo "Main documentation:"
        echo "  - PROJECT.md - Project overview & vision"
        echo "  - STRUCTURE.md - Directory organization"
        echo "  - CONTRIBUTING.md - Contribution guide"
        echo ""
        echo "Package documentation:"
        echo "  - helios/README.md - Package overview"
        echo "  - helios/docs/ARCHITECTURE.md - System design"
        echo "  - helios/docs/GETTING_STARTED.md - Setup guide"
        echo "  - helios/docs/API.md - API reference"
        echo "  - helios/docs/DOMAINS.md - Domain guide"
        echo ""
        echo "View files with:"
        echo "  cat PROJECT.md"
        echo "  cat helios/docs/ARCHITECTURE.md"
        echo ""
        ;;

    *)
        echo -e "${RED}✗ Unknown command: $MODE${NC}"
        echo ""
        bash "$0" help
        exit 1
        ;;
esac

echo ""

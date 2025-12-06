#!/bin/bash

# Build script for all frontend applications
# Usage: ./build-all.sh [command]
# Commands: install, dev, build, start, test

set -e

APPS=("ghost-researcher" "scientific-tinder" "chaos-engine")
COMMAND=${1:-install}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Frontend Applications Build Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Function to run command for each app
run_command() {
    local app=$1
    local cmd=$2

    echo -e "${YELLOW}[$app] Running: $cmd${NC}"
    cd "$app"

    case "$cmd" in
        install)
            npm install
            ;;
        dev)
            # Run in background with different ports
            case "$app" in
                ghost-researcher)
                    PORT=3000 npm run dev &
                    ;;
                scientific-tinder)
                    PORT=3001 npm run dev &
                    ;;
                chaos-engine)
                    PORT=3002 npm run dev &
                    ;;
            esac
            ;;
        build)
            npm run build
            ;;
        start)
            # Run production servers
            case "$app" in
                ghost-researcher)
                    PORT=3000 npm start &
                    ;;
                scientific-tinder)
                    PORT=3001 npm start &
                    ;;
                chaos-engine)
                    PORT=3002 npm start &
                    ;;
            esac
            ;;
        test)
            npm test
            ;;
        lint)
            npm run lint
            ;;
        format)
            npm run format
            ;;
        *)
            echo -e "${RED}Unknown command: $cmd${NC}"
            exit 1
            ;;
    esac

    cd ..
    echo -e "${GREEN}✓ [$app] Completed: $cmd${NC}"
    echo ""
}

# Main execution
case "$COMMAND" in
    install|build|test|lint|format)
        for app in "${APPS[@]}"; do
            run_command "$app" "$COMMAND"
        done
        echo -e "${GREEN}All applications processed successfully!${NC}"
        ;;
    dev)
        echo -e "${YELLOW}Starting all development servers...${NC}"
        for app in "${APPS[@]}"; do
            run_command "$app" "$COMMAND"
        done
        echo -e "${GREEN}All development servers started!${NC}"
        echo -e "${GREEN}Ghost Researcher: http://localhost:3000${NC}"
        echo -e "${GREEN}Scientific Tinder: http://localhost:3001${NC}"
        echo -e "${GREEN}Chaos Engine: http://localhost:3002${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop all servers${NC}"
        wait
        ;;
    start)
        echo -e "${YELLOW}Starting all production servers...${NC}"
        for app in "${APPS[@]}"; do
            run_command "$app" "$COMMAND"
        done
        echo -e "${GREEN}All production servers started!${NC}"
        echo -e "${GREEN}Ghost Researcher: http://localhost:3000${NC}"
        echo -e "${GREEN}Scientific Tinder: http://localhost:3001${NC}"
        echo -e "${GREEN}Chaos Engine: http://localhost:3002${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop all servers${NC}"
        wait
        ;;
    stop)
        echo -e "${YELLOW}Stopping all servers...${NC}"
        pkill -f "next dev" || true
        pkill -f "next start" || true
        echo -e "${GREEN}All servers stopped${NC}"
        ;;
    clean)
        echo -e "${YELLOW}Cleaning all applications...${NC}"
        for app in "${APPS[@]}"; do
            echo -e "${YELLOW}[$app] Cleaning...${NC}"
            cd "$app"
            rm -rf node_modules .next out dist
            cd ..
            echo -e "${GREEN}✓ [$app] Cleaned${NC}"
        done
        echo -e "${GREEN}All applications cleaned!${NC}"
        ;;
    *)
        echo -e "${RED}Usage: $0 [install|dev|build|start|test|lint|format|stop|clean]${NC}"
        echo ""
        echo "Commands:"
        echo "  install - Install dependencies for all apps"
        echo "  dev     - Start all development servers"
        echo "  build   - Build all applications for production"
        echo "  start   - Start all production servers"
        echo "  test    - Run tests for all applications"
        echo "  lint    - Lint all applications"
        echo "  format  - Format code in all applications"
        echo "  stop    - Stop all running servers"
        echo "  clean   - Clean all build artifacts and node_modules"
        exit 1
        ;;
esac
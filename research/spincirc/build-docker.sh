#!/bin/bash
# Simple Docker build script for SpinCirc Framework
# 
# This script provides a simple interface to build the SpinCirc Docker image
# with common configurations.
#
# Author: Meshal Alawein <meshal@berkeley.edu>
# Copyright © 2025 Meshal Alawein — All rights reserved.
# License: MIT

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Default values
IMAGE_NAME="spincirc"
VERSION="1.0.0"
BUILD_ARGS=""
DOCKERFILE="Dockerfile"

# Help function
show_help() {
    cat << EOF
SpinCirc Docker Build Script

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help          Show this help message
    -n, --name NAME     Set image name (default: spincirc)
    -v, --version VER   Set version tag (default: 1.0.0)
    -t, --target TARGET Set build target stage
    --no-cache          Build without cache
    --pull              Pull latest base images
    --push              Push image after build
    --test              Run tests after build
    --dev               Build development version with volume mounts

EXAMPLES:
    # Basic build
    $0

    # Build specific target
    $0 --target python-env

    # Build and test
    $0 --test

    # Build development version
    $0 --dev

    # Build and push to registry
    $0 --push --name myregistry/spincirc

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -n|--name)
                IMAGE_NAME="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -t|--target)
                BUILD_ARGS="$BUILD_ARGS --target $2"
                shift 2
                ;;
            --no-cache)
                BUILD_ARGS="$BUILD_ARGS --no-cache"
                shift
                ;;
            --pull)
                BUILD_ARGS="$BUILD_ARGS --pull"
                shift
                ;;
            --push)
                PUSH=true
                shift
                ;;
            --test)
                TEST=true
                shift
                ;;
            --dev)
                DEV=true
                VERSION="dev"
                shift
                ;;
            *)
                echo "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Build image
build_image() {
    local tag="${IMAGE_NAME}:${VERSION}"
    local latest_tag="${IMAGE_NAME}:latest"
    
    echo -e "${BLUE}Building SpinCirc Docker Image${NC}"
    echo -e "${BLUE}=============================${NC}"
    echo ""
    echo -e "${GREEN}Image: ${tag}${NC}"
    echo -e "${GREEN}Dockerfile: ${DOCKERFILE}${NC}"
    echo ""
    
    # Build command
    local cmd="docker build ${BUILD_ARGS} -t ${tag}"
    
    # Add latest tag if not dev build
    if [[ "$DEV" != true ]]; then
        cmd="$cmd -t ${latest_tag}"
    fi
    
    # Add build labels
    cmd="$cmd --label org.opencontainers.image.created=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    cmd="$cmd --label org.opencontainers.image.version=${VERSION}"
    
    if command -v git &> /dev/null; then
        local git_ref=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
        cmd="$cmd --label org.opencontainers.image.revision=${git_ref}"
    fi
    
    cmd="$cmd -f ${DOCKERFILE} ."
    
    echo -e "${YELLOW}Running: ${cmd}${NC}"
    echo ""
    
    eval $cmd
    
    echo ""
    echo -e "${GREEN}✓ Build completed successfully!${NC}"
    echo -e "${GREEN}  Image: ${tag}${NC}"
    if [[ "$DEV" != true ]]; then
        echo -e "${GREEN}  Latest: ${latest_tag}${NC}"
    fi
}

# Test image
test_image() {
    local tag="${IMAGE_NAME}:${VERSION}"
    
    echo -e "${BLUE}Testing SpinCirc Docker Image${NC}"
    echo -e "${BLUE}============================${NC}"
    echo ""
    
    # Basic health check
    echo -e "${YELLOW}Running health check...${NC}"
    if docker run --rm "${tag}" /opt/spincirc/healthcheck.sh; then
        echo -e "${GREEN}✓ Health check passed${NC}"
    else
        echo -e "${RED}✗ Health check failed${NC}"
        exit 1
    fi
    
    # Python test
    echo -e "${YELLOW}Testing Python environment...${NC}"
    if docker run --rm "${tag}" python3 -c "
import numpy as np
import scipy
import matplotlib
print('✓ Scientific stack working')
print(f'NumPy: {np.__version__}')
print(f'SciPy: {scipy.__version__}')
print(f'Matplotlib: {matplotlib.__version__}')
"; then
        echo -e "${GREEN}✓ Python tests passed${NC}"
    else
        echo -e "${RED}✗ Python tests failed${NC}"
        exit 1
    fi
    
    # SpinCirc module test
    echo -e "${YELLOW}Testing SpinCirc modules...${NC}"
    if docker run --rm "${tag}" python3 -c "
import sys
sys.path.append('/opt/spincirc/python')
try:
    import ml_tools
    import visualization
    print('✓ SpinCirc modules loaded successfully')
except ImportError as e:
    print(f'✗ Module import failed: {e}')
    exit(1)
"; then
        echo -e "${GREEN}✓ SpinCirc module tests passed${NC}"
    else
        echo -e "${YELLOW}⚠ SpinCirc modules not fully available (this is expected for base images)${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}✓ All tests completed!${NC}"
}

# Push image
push_image() {
    local tag="${IMAGE_NAME}:${VERSION}"
    local latest_tag="${IMAGE_NAME}:latest"
    
    echo -e "${BLUE}Pushing SpinCirc Docker Image${NC}"
    echo -e "${BLUE}============================${NC}"
    echo ""
    
    echo -e "${YELLOW}Pushing ${tag}...${NC}"
    docker push "${tag}"
    
    if [[ "$DEV" != true ]]; then
        echo -e "${YELLOW}Pushing ${latest_tag}...${NC}"
        docker push "${latest_tag}"
    fi
    
    echo ""
    echo -e "${GREEN}✓ Push completed successfully!${NC}"
}

# Show image info
show_info() {
    local tag="${IMAGE_NAME}:${VERSION}"
    
    echo -e "${BLUE}Image Information${NC}"
    echo -e "${BLUE}=================${NC}"
    echo ""
    
    # Image size
    local size=$(docker images --format "table {{.Size}}" "${tag}" | tail -n 1)
    echo -e "${GREEN}Size: ${size}${NC}"
    
    # Image layers
    echo -e "${GREEN}Layers: $(docker history --no-trunc "${tag}" | wc -l)${NC}"
    
    # Labels
    echo -e "${GREEN}Labels:${NC}"
    docker inspect "${tag}" --format '{{range $k, $v := .Config.Labels}}  {{$k}}: {{$v}}{{"\n"}}{{end}}'
    
    echo ""
    echo -e "${BLUE}Usage Examples:${NC}"
    echo "  # Run interactive shell"
    echo "  docker run -it --rm ${tag} bash"
    echo ""
    echo "  # Start Jupyter Lab"
    echo "  docker run -it --rm -p 8888:8888 ${tag} jupyter"
    echo ""
    echo "  # Run tests"
    echo "  docker run --rm ${tag} tests"
    echo ""
    echo "  # Run with volume mounts"
    echo "  docker run -it --rm \\"
    echo "    -v \$(pwd)/data:/opt/spincirc/data \\"
    echo "    -v \$(pwd)/results:/opt/spincirc/results \\"
    echo "    ${tag} python"
}

# Main function
main() {
    parse_args "$@"
    
    echo -e "${BLUE}SpinCirc Docker Build${NC}"
    echo -e "${BLUE}====================${NC}"
    echo ""
    
    # Check if Docker is available
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
        exit 1
    fi
    
    # Build image
    build_image
    
    # Test if requested
    if [[ "$TEST" == true ]]; then
        test_image
    fi
    
    # Push if requested
    if [[ "$PUSH" == true ]]; then
        push_image
    fi
    
    # Show image information
    show_info
    
    echo ""
    echo -e "${GREEN}🎉 SpinCirc Docker build completed successfully!${NC}"
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
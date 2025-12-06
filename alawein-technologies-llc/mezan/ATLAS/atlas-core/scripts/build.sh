#!/bin/bash
# Build script for MEZAN ATLAS Docker images

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REGISTRY=${DOCKER_REGISTRY:-""}
IMAGE_NAME=${IMAGE_NAME:-"mezan-atlas"}
VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "dev")
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Parse arguments
BUILD_TARGET="production"
PUSH=false
NO_CACHE=false
PLATFORMS=""

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --target TARGET    Build target (production|development|tester)"
    echo "  -p, --push             Push image to registry"
    echo "  -n, --no-cache         Build without cache"
    echo "  -r, --registry REGISTRY Docker registry URL"
    echo "  -v, --version VERSION  Image version tag"
    echo "  -m, --multi-platform   Build for multiple platforms (linux/amd64,linux/arm64)"
    echo "  -h, --help             Show this help message"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            BUILD_TARGET="$2"
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        -n|--no-cache)
            NO_CACHE=true
            shift
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -m|--multi-platform)
            PLATFORMS="--platform linux/amd64,linux/arm64"
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Construct image tag
if [ -n "$REGISTRY" ]; then
    FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}"
else
    FULL_IMAGE_NAME="${IMAGE_NAME}"
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}MEZAN ATLAS Docker Build${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Image: ${FULL_IMAGE_NAME}"
echo "Version: ${VERSION}"
echo "Target: ${BUILD_TARGET}"
echo "Git Commit: ${GIT_COMMIT}"
echo "Build Date: ${BUILD_DATE}"
echo ""

# Check Docker daemon
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}Error: Docker daemon is not running${NC}"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "Dockerfile" ]; then
    echo -e "${RED}Error: Dockerfile not found. Run this script from atlas-core directory${NC}"
    exit 1
fi

# Build arguments
BUILD_ARGS=(
    --build-arg "VERSION=${VERSION}"
    --build-arg "BUILD_DATE=${BUILD_DATE}"
    --build-arg "GIT_COMMIT=${GIT_COMMIT}"
    --target "${BUILD_TARGET}"
    --tag "${FULL_IMAGE_NAME}:${VERSION}"
    --tag "${FULL_IMAGE_NAME}:latest"
    --label "org.opencontainers.image.version=${VERSION}"
    --label "org.opencontainers.image.created=${BUILD_DATE}"
    --label "org.opencontainers.image.revision=${GIT_COMMIT}"
    --file Dockerfile
)

# Add platform flag if specified
if [ -n "$PLATFORMS" ]; then
    BUILD_ARGS+=($PLATFORMS)
    # Enable buildx for multi-platform builds
    docker buildx create --use --name atlas-builder 2>/dev/null || true
    BUILD_CMD="docker buildx build"
    if [ "$PUSH" = true ]; then
        BUILD_ARGS+=(--push)
    fi
else
    BUILD_CMD="docker build"
fi

# Add no-cache flag if specified
if [ "$NO_CACHE" = true ]; then
    BUILD_ARGS+=(--no-cache)
fi

# Add build context
BUILD_ARGS+=(.)

# Run the build
echo -e "${YELLOW}Building Docker image...${NC}"
echo "Command: ${BUILD_CMD} ${BUILD_ARGS[*]}"
echo ""

if $BUILD_CMD "${BUILD_ARGS[@]}"; then
    echo -e "${GREEN}✓ Docker image built successfully${NC}"
else
    echo -e "${RED}✗ Docker build failed${NC}"
    exit 1
fi

# Push to registry if requested (and not already done by buildx)
if [ "$PUSH" = true ] && [ -z "$PLATFORMS" ]; then
    echo ""
    echo -e "${YELLOW}Pushing image to registry...${NC}"

    docker push "${FULL_IMAGE_NAME}:${VERSION}"
    docker push "${FULL_IMAGE_NAME}:latest"

    echo -e "${GREEN}✓ Image pushed to registry${NC}"
fi

# Run security scan
echo ""
echo -e "${YELLOW}Running security scan...${NC}"
if command -v trivy &> /dev/null; then
    trivy image --severity HIGH,CRITICAL "${FULL_IMAGE_NAME}:${VERSION}"
else
    echo "Trivy not installed, skipping security scan"
    echo "Install with: brew install trivy (macOS) or check https://github.com/aquasecurity/trivy"
fi

# Print image size
echo ""
echo -e "${GREEN}Image Information:${NC}"
docker images "${FULL_IMAGE_NAME}:${VERSION}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

# Run basic tests if test target was built
if [ "$BUILD_TARGET" = "tester" ]; then
    echo ""
    echo -e "${YELLOW}Running tests in container...${NC}"
    docker run --rm "${FULL_IMAGE_NAME}:${VERSION}" pytest --version
fi

# Generate SBOM (Software Bill of Materials) if syft is available
if command -v syft &> /dev/null; then
    echo ""
    echo -e "${YELLOW}Generating SBOM...${NC}"
    syft "${FULL_IMAGE_NAME}:${VERSION}" -o json > "sbom-${VERSION}.json"
    echo -e "${GREEN}✓ SBOM saved to sbom-${VERSION}.json${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Build Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "To run the container:"
echo "  docker run -p 8080:8080 ${FULL_IMAGE_NAME}:${VERSION}"
echo ""
echo "To run with docker-compose:"
echo "  docker-compose up -d"
echo ""
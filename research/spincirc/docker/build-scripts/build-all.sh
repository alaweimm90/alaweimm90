#!/bin/bash
# SpinCirc Docker Build Script - Build All Variants
# 
# This script builds multiple variants of the SpinCirc Docker image
# for different use cases and architectures.
#
# Author: Meshal Alawein <meshal@berkeley.edu>
# Copyright © 2025 Meshal Alawein — All rights reserved.
# License: MIT

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REGISTRY="spincirc"
VERSION="1.0.0"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Build variants
VARIANTS=(
    "base:base"
    "python:python-env" 
    "matlab:matlab-env"
    "full:final"
)

# Architectures to build for
ARCHITECTURES=(
    "linux/amd64"
    "linux/arm64"
)

echo -e "${BLUE}========================================"
echo "SpinCirc Docker Build Script"
echo -e "========================================${NC}"
echo ""
echo -e "${GREEN}Version: ${VERSION}${NC}"
echo -e "${GREEN}Build Date: ${BUILD_DATE}${NC}"
echo -e "${GREEN}VCS Reference: ${VCS_REF}${NC}"
echo ""

# Function to print status
print_status() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] $1${NC}"
}

# Function to print success
print_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] ✓ $1${NC}"
}

# Function to print error
print_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ✗ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v git &> /dev/null; then
        print_error "Git is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker buildx
    if ! docker buildx version &> /dev/null; then
        print_error "Docker buildx is not available"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Setup buildx
setup_buildx() {
    print_status "Setting up Docker buildx..."
    
    # Create builder instance
    docker buildx create --name spincirc-builder --use --bootstrap 2>/dev/null || true
    
    print_success "Buildx setup complete"
}

# Build single architecture
build_single_arch() {
    local variant_name=$1
    local target_stage=$2
    local arch=$3
    local tag="${REGISTRY}:${variant_name}-${VERSION}-$(echo $arch | tr '/' '-')"
    
    print_status "Building ${variant_name} for ${arch}..."
    
    docker buildx build \
        --platform="${arch}" \
        --target="${target_stage}" \
        --tag="${tag}" \
        --label="org.opencontainers.image.created=${BUILD_DATE}" \
        --label="org.opencontainers.image.version=${VERSION}" \
        --label="org.opencontainers.image.revision=${VCS_REF}" \
        --label="org.opencontainers.image.variant=${variant_name}" \
        --cache-from="type=local,src=/tmp/.buildx-cache" \
        --cache-to="type=local,dest=/tmp/.buildx-cache-new,mode=max" \
        --load \
        .
    
    print_success "Built ${tag}"
}

# Build multi-architecture
build_multi_arch() {
    local variant_name=$1
    local target_stage=$2
    local tag="${REGISTRY}:${variant_name}-${VERSION}"
    local latest_tag="${REGISTRY}:${variant_name}-latest"
    
    print_status "Building multi-arch ${variant_name}..."
    
    # Build for all architectures and push to registry
    docker buildx build \
        --platform="$(IFS=','; echo "${ARCHITECTURES[*]}")" \
        --target="${target_stage}" \
        --tag="${tag}" \
        --tag="${latest_tag}" \
        --label="org.opencontainers.image.created=${BUILD_DATE}" \
        --label="org.opencontainers.image.version=${VERSION}" \
        --label="org.opencontainers.image.revision=${VCS_REF}" \
        --label="org.opencontainers.image.variant=${variant_name}" \
        --cache-from="type=local,src=/tmp/.buildx-cache" \
        --cache-to="type=local,dest=/tmp/.buildx-cache-new,mode=max" \
        --push \
        .
    
    print_success "Built multi-arch ${tag}"
}

# Build all variants
build_variants() {
    local build_type=$1
    
    for variant in "${VARIANTS[@]}"; do
        IFS=':' read -r variant_name target_stage <<< "$variant"
        
        if [[ "$build_type" == "single" ]]; then
            for arch in "${ARCHITECTURES[@]}"; do
                build_single_arch "$variant_name" "$target_stage" "$arch"
            done
        else
            build_multi_arch "$variant_name" "$target_stage"
        fi
    done
}

# Test images
test_images() {
    print_status "Testing built images..."
    
    for variant in "${VARIANTS[@]}"; do
        IFS=':' read -r variant_name target_stage <<< "$variant"
        local tag="${REGISTRY}:${variant_name}-${VERSION}-linux-amd64"
        
        print_status "Testing ${tag}..."
        
        # Basic smoke test
        if docker run --rm "${tag}" python3 -c "print('Python OK')" &>/dev/null; then
            print_success "Python test passed for ${variant_name}"
        else
            print_error "Python test failed for ${variant_name}"
        fi
        
        # Health check test
        if docker run --rm "${tag}" /opt/spincirc/healthcheck.sh &>/dev/null; then
            print_success "Health check passed for ${variant_name}"
        else
            print_error "Health check failed for ${variant_name}"
        fi
    done
}

# Generate image manifest
generate_manifest() {
    print_status "Generating image manifest..."
    
    local manifest_file="image-manifest.json"
    
    cat > "$manifest_file" << EOF
{
  "schemaVersion": 2,
  "mediaType": "application/vnd.docker.distribution.manifest.list.v2+json",
  "manifests": [
EOF

    local first=true
    for variant in "${VARIANTS[@]}"; do
        IFS=':' read -r variant_name target_stage <<< "$variant"
        
        if [[ "$first" == false ]]; then
            echo "    ," >> "$manifest_file"
        fi
        first=false
        
        cat >> "$manifest_file" << EOF
    {
      "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
      "size": 0,
      "digest": "sha256:placeholder",
      "platform": {
        "architecture": "amd64",
        "os": "linux"
      },
      "annotations": {
        "org.opencontainers.image.variant": "${variant_name}",
        "org.opencontainers.image.version": "${VERSION}"
      }
    }
EOF
    done
    
    echo "  ]" >> "$manifest_file"
    echo "}" >> "$manifest_file"
    
    print_success "Generated ${manifest_file}"
}

# Cleanup
cleanup() {
    print_status "Cleaning up..."
    
    # Move cache
    if [[ -d "/tmp/.buildx-cache-new" ]]; then
        rm -rf /tmp/.buildx-cache
        mv /tmp/.buildx-cache-new /tmp/.buildx-cache
    fi
    
    # Remove builder
    docker buildx rm spincirc-builder 2>/dev/null || true
    
    print_success "Cleanup complete"
}

# Main execution
main() {
    local build_type="${1:-single}"
    local test_mode="${2:-true}"
    
    print_status "Starting build process (type: ${build_type})..."
    
    check_prerequisites
    setup_buildx
    
    case "$build_type" in
        "single")
            build_variants "single"
            if [[ "$test_mode" == "true" ]]; then
                test_images
            fi
            ;;
        "multi")
            build_variants "multi"
            generate_manifest
            ;;
        "test")
            test_images
            exit 0
            ;;
        *)
            print_error "Unknown build type: $build_type"
            echo "Usage: $0 [single|multi|test] [true|false]"
            exit 1
            ;;
    esac
    
    cleanup
    
    print_success "Build process completed successfully!"
    echo ""
    echo -e "${GREEN}Available images:${NC}"
    for variant in "${VARIANTS[@]}"; do
        IFS=':' read -r variant_name target_stage <<< "$variant"
        echo "  ${REGISTRY}:${variant_name}-${VERSION}"
        echo "  ${REGISTRY}:${variant_name}-latest"
    done
}

# Handle script arguments
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    trap cleanup EXIT
    main "$@"
fi
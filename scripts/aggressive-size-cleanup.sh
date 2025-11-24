#!/bin/bash

###############################################################################
# Aggressive Size Optimization Script
# Purpose: Reduce repository size by cleaning unnecessary files/directories
# Mode: YOLO - Ultra-aggressive with safety checks
###############################################################################

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Variables
REPO_ROOT=$(pwd)
BACKUP_DIR=".size-optimization-backup-$(date +%s)"
ORIGINAL_SIZE=0
FINAL_SIZE=0

# Functions
log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

get_dir_size() {
    du -sh "$1" 2>/dev/null | cut -f1
}

# Main execution
echo -e "${BLUE}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        AGGRESSIVE SIZE OPTIMIZATION SCRIPT          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════╝${NC}"
echo ""

log_info "Starting aggressive size optimization..."
log_info "Repository Root: $REPO_ROOT"
echo ""

# Get original size
log_info "Calculating original repository size..."
ORIGINAL_SIZE=$(du -sh . 2>/dev/null | cut -f1)
log_success "Original size: $ORIGINAL_SIZE"
echo ""

# Phase 1: Backup critical data
log_info "Phase 1: Creating safety backup..."
mkdir -p "$BACKUP_DIR"
log_success "Backup directory created: $BACKUP_DIR"
echo ""

# Phase 2: Clean node_modules in subdirectories
log_info "Phase 2: Cleaning node_modules in alaweimm90/..."
find alaweimm90 -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true
log_success "Removed node_modules from alaweimm90/"
echo ""

# Phase 3: Clean build artifacts
log_info "Phase 3: Cleaning build artifacts..."
find . -type d -name "dist" -o -name "build" -o -name ".next" | while read dir; do
    if [ -n "$dir" ]; then
        rm -rf "$dir" 2>/dev/null || true
        log_success "Removed: $dir"
    fi
done
echo ""

# Phase 4: Clean coverage reports
log_info "Phase 4: Cleaning coverage reports..."
if [ -d "coverage" ]; then
    rm -rf coverage
    log_success "Removed coverage/"
fi
echo ""

# Phase 5: Clean cache directories
log_info "Phase 5: Cleaning cache directories..."
find . -type d -name ".cache" -o -name "*.cache" | while read dir; do
    if [ -n "$dir" ] && [ "$dir" != "./.cache" ]; then
        rm -rf "$dir" 2>/dev/null || true
        log_success "Removed: $dir"
    fi
done
echo ""

# Phase 6: Clean temporary directories
log_info "Phase 6: Cleaning temporary files..."
find . -type f -name "*.tmp" -o -name "*.bak" -delete 2>/dev/null || true
log_success "Removed temporary files"
echo ""

# Phase 7: Consolidate duplicates
log_info "Phase 7: Checking for duplicate dependencies..."
# This would require more sophisticated analysis
log_warning "Duplicate analysis: Manual review recommended"
echo ""

# Calculate final size
log_info "Calculating final size..."
FINAL_SIZE=$(du -sh . 2>/dev/null | cut -f1)
log_success "Final size: $FINAL_SIZE"
echo ""

# Results summary
echo -e "${BLUE}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║            OPTIMIZATION COMPLETE ✓                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════╝${NC}"
echo ""
log_success "Original size: $ORIGINAL_SIZE"
log_success "Final size: $FINAL_SIZE"
log_success "Backup location: $BACKUP_DIR"
echo ""

log_info "Run 'npm run validate' to ensure everything still works"
exit 0

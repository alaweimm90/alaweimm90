#!/bin/bash

# HELIOS Development Environment Setup Script
# Version: 0.1.0
# Purpose: Automate development environment setup

set -e  # Exit on error

echo "================================"
echo "HELIOS Development Setup"
echo "================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${BLUE}[1/6]${NC} Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Require Python 3.8+
required_version="3.8"
if [[ "$python_version" < "$required_version" ]]; then
    echo -e "${RED}✗ Python 3.8+ required (found $python_version)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}"
echo ""

# Create virtual environment
echo -e "${BLUE}[2/6]${NC} Creating virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}ℹ Virtual environment already exists${NC}"
fi
echo ""

# Activate virtual environment
echo -e "${BLUE}[3/6]${NC} Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo -e "${BLUE}[4/6]${NC} Upgrading pip..."
pip install --upgrade pip setuptools wheel >/dev/null 2>&1
echo -e "${GREEN}✓ Pip upgraded${NC}"
echo ""

# Install package with all dependencies
echo -e "${BLUE}[5/6]${NC} Installing HELIOS with all dependencies..."
cd "$(dirname "$0")/.."
cd ..
pip install -e ".[all,dev]" >/dev/null 2>&1
echo -e "${GREEN}✓ HELIOS installed${NC}"
echo ""

# Verify installation
echo -e "${BLUE}[6/6]${NC} Verifying installation..."
python -c "import helios; print(f'HELIOS version: {helios.__version__ if hasattr(helios, \"__version__\") else \"0.1.0\"}')" 2>/dev/null
echo -e "${GREEN}✓ Installation verified${NC}"
echo ""

# Final messages
echo "================================"
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Run tests: bash helios/scripts/test.sh"
echo "3. Format code: bash helios/scripts/format.sh"
echo "4. Check examples: cd helios/examples && python basic_usage.py"
echo ""
echo "For more info, see:"
echo "- docs/GETTING_STARTED.md - Setup guide"
echo "- docs/ARCHITECTURE.md - System design"
echo ""

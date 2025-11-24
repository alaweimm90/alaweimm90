#!/bin/bash
# PiecesOS Integration Setup Script

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔌 PiecesOS Integration Setup${NC}"
echo -e "${BLUE}============================${NC}"
echo ""

# Check if PiecesOS is already installed
if command -v pieces >/dev/null 2>&1; then
    echo -e "${GREEN}✅ PiecesOS CLI already installed${NC}"
    pieces --version
else
    echo -e "${BLUE}📦 Installing PiecesOS CLI...${NC}"

    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Detected Linux"
        curl -fsSL https://pieces.app/install.sh | sh
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Detected macOS"
        if command -v brew >/dev/null 2>&1; then
            brew install pieces
        else
            echo -e "${YELLOW}⚠️  Homebrew not found. Please install from https://pieces.app${NC}"
            exit 1
        fi
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        echo "Detected Windows"
        echo -e "${YELLOW}Please install PiecesOS from: https://pieces.app/downloads${NC}"
        echo -e "${YELLOW}Then run this script again.${NC}"
        exit 1
    else
        echo -e "${RED}❌ Unsupported OS: $OSTYPE${NC}"
        exit 1
    fi

    echo -e "${GREEN}✅ PiecesOS CLI installed${NC}"
fi

# Create configuration
echo ""
echo -e "${BLUE}⚙️  Configuring PiecesOS for monorepo...${NC}"

mkdir -p .pieces

cat > .pieces/config.json << 'EOF'
{
  "name": "GitHub Monorepo",
  "type": "workspace",
  "settings": {
    "contextPersistence": {
      "enabled": true,
      "autoSave": true,
      "saveInterval": 300
    },
    "codeCapture": {
      "enabled": true,
      "fileTypes": [".ts", ".tsx", ".js", ".jsx", ".py", ".md"],
      "excludePatterns": ["node_modules/**", "dist/**", "build/**", ".git/**"]
    },
    "suggestions": {
      "enabled": true,
      "contextAware": true
    },
    "search": {
      "enabled": true,
      "indexWorkspace": true
    }
  },
  "integrations": {
    "vscode": {
      "enabled": true
    },
    "git": {
      "enabled": true,
      "captureCommits": true
    }
  }
}
EOF

echo -e "${GREEN}✅ Configuration created at .pieces/config.json${NC}"

# Add to .gitignore
if ! grep -q ".pieces" .gitignore 2>/dev/null; then
    echo ".pieces/" >> .gitignore
    echo -e "${GREEN}✅ Added .pieces/ to .gitignore${NC}"
fi

# Initialize workspace
echo ""
echo -e "${BLUE}🚀 Initializing PiecesOS workspace...${NC}"

if command -v pieces >/dev/null 2>&1; then
    pieces init --workspace . --config .pieces/config.json || true
    echo -e "${GREEN}✅ Workspace initialized${NC}"
fi

# Instructions
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ PiecesOS Integration Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}📚 Usage:${NC}"
echo "  pieces save --context 'working on feature X'  - Save current context"
echo "  pieces load --context 'feature X'             - Load saved context"
echo "  pieces search 'jwt implementation'            - Search code snippets"
echo "  pieces list                                     - List saved contexts"
echo ""
echo -e "${BLUE}💡 Tips:${NC}"
echo "  - PiecesOS will auto-save context every 5 minutes"
echo "  - Code snippets are captured automatically"
echo "  - Install the VSCode extension for enhanced features"
echo ""

#!/bin/bash
# Augment CLI Integration Setup Script

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔌 Augment CLI Integration Setup${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Check if Augment CLI is installed
if command -v augment >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Augment CLI already installed${NC}"
    augment --version
else
    echo -e "${BLUE}📦 Installing Augment CLI...${NC}"

    if command -v npm >/dev/null 2>&1; then
        npm install -g @augment/cli 2>/dev/null || {
            echo -e "${YELLOW}⚠️  Augment CLI installation requires authentication${NC}"
            echo -e "${YELLOW}   Please install manually or contact Augment for access${NC}"
            exit 0
        }
        echo -e "${GREEN}✅ Augment CLI installed${NC}"
    else
        echo -e "${RED}❌ npm not found. Please install Node.js first.${NC}"
        exit 1
    fi
fi

# Create configuration
echo ""
echo -e "${BLUE}⚙️  Configuring Augment for monorepo...${NC}"

mkdir -p .augment

cat > .augment/config.yml << 'EOF'
# Augment CLI Configuration
workspace:
  name: GitHub Monorepo
  type: monorepo
  language: typescript

features:
  codeReview:
    enabled: true
    autoReview: false  # Manual trigger only
    rules:
      - security
      - performance
      - bestPractices
      - typeScript

  documentation:
    enabled: true
    autoGenerate: false
    style: jsdoc
    includeExamples: true

  analytics:
    enabled: true
    trackProductivity: true
    teamMode: true

  suggestions:
    enabled: true
    contextAware: true
    languages:
      - typescript
      - javascript
      - python

integrations:
  github:
    enabled: true
    autoPR: false
    reviewComments: true

  ci:
    enabled: true
    provider: github-actions

  vscode:
    enabled: true

excludePatterns:
  - "node_modules/**"
  - "dist/**"
  - "build/**"
  - ".git/**"
  - "coverage/**"
EOF

echo -e "${GREEN}✅ Configuration created at .augment/config.yml${NC}"

# Add to .gitignore
if ! grep -q ".augment" .gitignore 2>/dev/null; then
    echo ".augment/" >> .gitignore
    echo ".augment/cache/" >> .gitignore
    echo -e "${GREEN}✅ Added .augment/ to .gitignore${NC}"
fi

# Initialize workspace
echo ""
echo -e "${BLUE}🚀 Initializing Augment workspace...${NC}"

if command -v augment >/dev/null 2>&1; then
    augment init --workspace . --config .augment/config.yml 2>/dev/null || {
        echo -e "${YELLOW}⚠️  Manual initialization may be required${NC}"
    }
    echo -e "${GREEN}✅ Workspace initialized${NC}"
fi

# Instructions
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Augment CLI Integration Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${BLUE}📚 Usage:${NC}"
echo "  augment review pr --number 123     - Review a pull request"
echo "  augment docs generate --path ./src - Generate documentation"
echo "  augment analytics --team           - View team analytics"
echo "  augment suggest                     - Get AI suggestions"
echo ""
echo -e "${BLUE}💡 Tips:${NC}"
echo "  - Configure your API key: augment config set api-key YOUR_KEY"
echo "  - Enable team mode for shared insights"
echo "  - Install the VSCode extension for inline suggestions"
echo ""

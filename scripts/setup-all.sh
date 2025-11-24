#!/bin/bash

# Complete MCP & Agent setup script for the monorepo

set -e

echo "🚀 Complete MCP & Agent Setup"
echo "=============================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Step 1: Install dependencies
echo -e "${BLUE}Step 1: Installing dependencies...${NC}"
pnpm install
echo -e "${GREEN}✓ Dependencies installed${NC}\n"

# Step 2: Build packages
echo -e "${BLUE}Step 2: Building packages...${NC}"
pnpm run build
echo -e "${GREEN}✓ Packages built${NC}\n"

# Step 3: Set up MCP
echo -e "${BLUE}Step 3: Setting up MCP configuration...${NC}"
npx ts-node ./scripts/setup-mcp.ts
echo -e "${GREEN}✓ MCP configured${NC}\n"

# Step 4: Set up Agents
echo -e "${BLUE}Step 4: Setting up agents and orchestration...${NC}"
npx ts-node ./scripts/setup-agents.ts
echo -e "${GREEN}✓ Agents configured${NC}\n"

# Step 5: Run type check
echo -e "${BLUE}Step 5: Running type checks...${NC}"
pnpm run type-check
echo -e "${GREEN}✓ Type checks passed${NC}\n"

echo -e "${GREEN}✨ All setup complete!${NC}"
echo ""
echo "Next steps:"
echo "1. Review .claude/mcp-config.json"
echo "2. Review .claude/agents.json"
echo "3. Install additional MCP servers as needed"
echo "4. Run tests: pnpm test"
echo ""

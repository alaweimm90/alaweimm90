#!/bin/bash

# Blockchain Setup Script for Automation Platform
# Integrates blockchain audit trail with existing automation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BLOCKCHAIN_DIR="$PROJECT_ROOT/automation/blockchain"

echo "🔗 Setting up blockchain integration..."

# Check if blockchain directory exists
if [ ! -d "$BLOCKCHAIN_DIR" ]; then
    echo "❌ Blockchain directory not found at $BLOCKCHAIN_DIR"
    exit 1
fi

cd "$BLOCKCHAIN_DIR"

# Install dependencies
echo "📦 Installing blockchain dependencies..."
npm install

# Setup environment
if [ ! -f ".env" ]; then
    echo "⚙️ Creating environment configuration..."
    cp .env.example .env
    echo "📝 Please edit .env file with your configuration"
fi

# Start local blockchain (if not running)
echo "🚀 Starting local blockchain..."
if ! pgrep -f "hardhat node" > /dev/null; then
    npx hardhat node &
    HARDHAT_PID=$!
    echo "Started Hardhat node with PID: $HARDHAT_PID"
    sleep 5
fi

# Compile contracts
echo "🔨 Compiling smart contracts..."
npx hardhat compile

# Deploy contracts to local network
echo "🚀 Deploying contracts..."
npx hardhat run scripts/deploy.js --network localhost

# Run health check
echo "🏥 Running health check..."
node scripts/health-check.js

# Integration with existing automation
echo "🔗 Integrating with existing automation..."

# Add blockchain health check to main health check
MAIN_HEALTH_CHECK="$PROJECT_ROOT/.automation/scripts/health-check.sh"
if [ -f "$MAIN_HEALTH_CHECK" ]; then
    if ! grep -q "blockchain" "$MAIN_HEALTH_CHECK"; then
        echo "" >> "$MAIN_HEALTH_CHECK"
        echo "# Blockchain health check" >> "$MAIN_HEALTH_CHECK"
        echo "echo \"🔗 Checking blockchain integration...\"" >> "$MAIN_HEALTH_CHECK"
        echo "cd automation/blockchain && node scripts/health-check.js" >> "$MAIN_HEALTH_CHECK"
    fi
fi

# Add blockchain to pre-commit hook
PRE_COMMIT_HOOK="$PROJECT_ROOT/.automation/hooks/pre-commit"
if [ -f "$PRE_COMMIT_HOOK" ]; then
    if ! grep -q "blockchain" "$PRE_COMMIT_HOOK"; then
        echo "" >> "$PRE_COMMIT_HOOK"
        echo "# Blockchain contract validation" >> "$PRE_COMMIT_HOOK"
        echo "if [ -d \"automation/blockchain\" ]; then" >> "$PRE_COMMIT_HOOK"
        echo "    cd automation/blockchain" >> "$PRE_COMMIT_HOOK"
        echo "    npx hardhat compile || exit 1" >> "$PRE_COMMIT_HOOK"
        echo "    npm test || exit 1" >> "$PRE_COMMIT_HOOK"
        echo "    cd ../.." >> "$PRE_COMMIT_HOOK"
        echo "fi" >> "$PRE_COMMIT_HOOK"
    fi
fi

echo "✅ Blockchain integration setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Edit automation/blockchain/.env with your configuration"
echo "2. Run 'bash .automation/scripts/health-check.sh' to verify setup"
echo "3. Start using blockchain audit trail in your workflows"
echo ""
echo "🔧 Usage examples:"
echo "  node -e \"const b = require('./automation/blockchain'); b.logWorkflowExecution({id: 'test'});\""
echo "  bash .automation/scripts/health-check.sh --blockchain"
#!/bin/bash

# Blockchain Integration Script for Automation Platform

set -e

echo "🔗 Integrating blockchain with automation platform..."

# Add blockchain health check to main health check
MAIN_HEALTH_CHECK=".automation/scripts/health-check.sh"
if [ -f "$MAIN_HEALTH_CHECK" ] && ! grep -q "blockchain" "$MAIN_HEALTH_CHECK"; then
    echo "" >> "$MAIN_HEALTH_CHECK"
    echo "# Blockchain health check" >> "$MAIN_HEALTH_CHECK"
    echo "echo \"🔗 Checking blockchain integration...\"" >> "$MAIN_HEALTH_CHECK"
    echo "if [ -d \"automation/blockchain\" ]; then" >> "$MAIN_HEALTH_CHECK"
    echo "    cd automation/blockchain && node scripts/health-check.js && cd ../.." >> "$MAIN_HEALTH_CHECK"
    echo "fi" >> "$MAIN_HEALTH_CHECK"
fi

# Add blockchain to pre-commit hook
PRE_COMMIT_HOOK=".automation/hooks/pre-commit"
if [ -f "$PRE_COMMIT_HOOK" ] && ! grep -q "blockchain" "$PRE_COMMIT_HOOK"; then
    echo "" >> "$PRE_COMMIT_HOOK"
    echo "# Blockchain validation" >> "$PRE_COMMIT_HOOK"
    echo "if [ -d \"automation/blockchain\" ]; then" >> "$PRE_COMMIT_HOOK"
    echo "    echo \"🔗 Validating blockchain integration...\"" >> "$PRE_COMMIT_HOOK"
    echo "    cd automation/blockchain && node scripts/health-check.js && cd ../.." >> "$PRE_COMMIT_HOOK"
    echo "fi" >> "$PRE_COMMIT_HOOK"
fi

echo "✅ Blockchain integration complete!"
echo ""
echo "🔧 Usage:"
echo "  node automation/blockchain/scripts/health-check.js"
echo "  node automation/blockchain/scripts/compliance-check.js"
echo ""
echo "📋 Integration points:"
echo "  - Security monitoring with immutable audit trails"
echo "  - Automated compliance validation"
echo "  - Workflow execution logging"
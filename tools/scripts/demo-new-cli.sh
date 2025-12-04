#!/bin/bash

# Demo script to showcase the new unified CLI system

echo "🚀 DEMONSTRATING YOUR NEW UNIFIED CLI SYSTEM"
echo "============================================="
echo ""

echo "📌 Before: 66 confusing npm scripts"
echo "📌 After: 1 beautiful unified CLI"
echo ""

echo "Let's explore your new tools..."
echo ""

# Main help
echo "1️⃣ Main Commands:"
echo "   $ meta --help"
npx tsx meta-cli.ts --help | head -15
echo ""

# AI tools
echo "2️⃣ AI Tools (replaces 38 npm scripts):"
echo "   $ meta ai --help"
npx tsx meta-cli.ts ai --help | head -12
echo ""

# Development tools
echo "3️⃣ Development Tools:"
echo "   $ meta dev --help"
npx tsx meta-cli.ts dev --help | head -10
echo ""

echo "4️⃣ Example: Running tests"
echo "   Old way: npm run test:run"
echo "   New way: meta dev test"
echo ""

echo "5️⃣ Example: AI cache stats"
echo "   Old way: npm run ai:cache:stats"
echo "   New way: meta ai cache stats"
echo ""

echo "✨ BENEFITS:"
echo "   • Discoverable with --help at every level"
echo "   • Logical command hierarchy"
echo "   • No more searching through package.json"
echo "   • Professional CLI experience"
echo ""

echo "📚 Full documentation: CLI_MIGRATION_GUIDE.md"
echo "🎯 Try it now: npx tsx meta-cli.ts --help"
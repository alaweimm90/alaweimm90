#!/bin/bash
#
# MagLogic Benchmark Runner
# Uses MetaHub benchmarking system (hub-spoke pattern)
#

set -e

echo "╔═══════════════════════════════════════════════════════╗"
echo "║     MagLogic Benchmarks (Hub-Spoke Pattern)          ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")/.."

HUB_CLI="../../../.metaHub/clis/bench"

if [ ! -f "$HUB_CLI" ]; then
    echo "❌ Error: Hub CLI not found"
    exit 1
fi

chmod +x "$HUB_CLI"

echo "🚀 Running MagLogic benchmarks..."
"$HUB_CLI" config config/benchmarks.yaml

echo ""
echo "📊 Generating visualizations..."
"$HUB_CLI" visualize results/

echo ""
echo "✅ MagLogic benchmarks complete!"
echo "📁 Results: results/"

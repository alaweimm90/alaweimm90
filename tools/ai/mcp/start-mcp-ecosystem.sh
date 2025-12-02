#!/bin/bash
# MCP Ecosystem Startup Script
# Part of AI Tools Cheatsheet v2.0

set -e

echo "🚀 Starting MCP Ecosystem..."
echo "================================"

# Check if npx is available
if ! command -v npx &> /dev/null; then
    echo "❌ npx not found. Please install Node.js first."
    exit 1
fi

# Create log directory
LOG_DIR="$HOME/.ai_tools/logs/mcp"
mkdir -p "$LOG_DIR"

# Function to start an MCP server
start_mcp() {
    local name=$1
    local command=$2
    local port=$3

    echo "Starting $name on port $port..."
    $command > "$LOG_DIR/${name}.log" 2>&1 &
    echo $! > "$LOG_DIR/${name}.pid"
    echo "  ✅ $name started (PID: $!)"
}

# Core MCPs
echo ""
echo "📦 Starting Core MCP Servers..."
start_mcp "filesystem" "npx -y @anthropic/mcp-server-filesystem $HOME/projects" 3001
start_mcp "git" "npx -y @anthropic/mcp-server-git" 3002

# Enhanced MCPs (if available)
echo ""
echo "🧠 Starting Enhanced MCP Servers..."
if command -v python3 &> /dev/null; then
    echo "  ℹ️  Python available - custom MCPs can be started"
fi

echo ""
echo "================================"
echo "✅ MCP Ecosystem started successfully!"
echo ""
echo "📊 Status:"
echo "  • Logs: $LOG_DIR"
echo "  • PIDs saved to: $LOG_DIR/*.pid"
echo ""
echo "🛑 To stop all MCPs, run:"
echo "  ~/.ai_tools/scripts/stop-mcp-ecosystem.sh"

#!/bin/bash
# Stop MCP Ecosystem Script
# Part of AI Tools Cheatsheet v2.0

set -e

echo "🛑 Stopping MCP Ecosystem..."
echo "================================"

LOG_DIR="$HOME/.ai_tools/logs/mcp"

# Stop all MCPs by reading PID files
for pidfile in "$LOG_DIR"/*.pid; do
    if [ -f "$pidfile" ]; then
        name=$(basename "$pidfile" .pid)
        pid=$(cat "$pidfile")

        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $name (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
            echo "  ✅ $name stopped"
        else
            echo "  ⚠️  $name was not running"
        fi

        rm -f "$pidfile"
    fi
done

echo ""
echo "================================"
echo "✅ All MCP servers stopped!"

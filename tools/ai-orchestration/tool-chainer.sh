#!/bin/bash
# Enhanced Tool Chainer v2.3
# AI Tools - Pipeline execution

PIPES_DIR="$HOME/.ai_tools/pipes"
mkdir -p "$PIPES_DIR"

run_pipe() {
    local pipe=$1
    [ ! -f "$PIPES_DIR/$pipe.yaml" ] && { echo "Pipe not found: $pipe"; return 1; }
    echo "Running pipe: $pipe"
    # Parse YAML and execute steps
    grep "^  - " "$PIPES_DIR/$pipe.yaml" | sed 's/^  - //' | while read cmd; do
        echo "→ $cmd"
        eval "$cmd"
    done
}

create_pipe() {
    local name=$1
    cat > "$PIPES_DIR/$name.yaml" << EOF
name: $name
steps:
  - echo "Step 1"
  - echo "Step 2"
EOF
    echo "✓ Created: $name (edit: $PIPES_DIR/$name.yaml)"
}

list_pipes() { echo "Pipes:"; ls "$PIPES_DIR"/*.yaml 2>/dev/null | xargs -I{} basename {} .yaml | sed 's/^/  /'; }

case "$1" in
    run) run_pipe "$2" ;;
    create) create_pipe "$2" ;;
    list) list_pipes ;;
    *) echo "Usage: ai-pipe <run|create|list>" ;;
esac

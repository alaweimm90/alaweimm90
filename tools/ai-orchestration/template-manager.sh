#!/bin/bash
# Enhanced Template Manager v2.3
# AI Tools - Reusable prompt templates

TEMPLATES_DIR="$HOME/.ai_tools/templates"
mkdir -p "$TEMPLATES_DIR/custom"

save_template() {
    local name=$1; shift
    echo "$*" > "$TEMPLATES_DIR/custom/$name.txt"
    echo "✓ Saved: $name"
}

use_template() {
    local name=$1; shift
    local tmpl=$(cat "$TEMPLATES_DIR/custom/$name.txt" 2>/dev/null || cat "$TEMPLATES_DIR/$name.txt" 2>/dev/null)
    [ -z "$tmpl" ] && { echo "Template not found: $name"; return 1; }
    for arg in "$@"; do tmpl="${tmpl/\{\}/$arg}"; done
    echo "$tmpl"
}

list_templates() {
    echo "Templates:"
    ls "$TEMPLATES_DIR"/*.txt "$TEMPLATES_DIR/custom"/*.txt 2>/dev/null | xargs -I{} basename {} .txt | sed 's/^/  /'
}

case "$1" in
    save) save_template "$2" "${@:3}" ;;
    use) use_template "$2" "${@:3}" ;;
    list) list_templates ;;
    *) echo "Usage: ai-template <save|use|list>" ;;
esac

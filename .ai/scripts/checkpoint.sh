#!/bin/bash
# Enhanced Checkpoint System v2.3
# AI Tools - Git-based state management with auto-restore

CHECKPOINT_DIR="$HOME/.ai_tools/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'

create_checkpoint() {
    local name=${1:-"checkpoint_$(date +%Y%m%d_%H%M%S)"}
    git rev-parse --git-dir > /dev/null 2>&1 || { echo -e "${RED}Not a git repo${NC}"; return 1; }
    
    # Stash current state
    git stash push -m "ai-checkpoint: $name" 2>/dev/null
    local stash_ref=$(git stash list | grep "ai-checkpoint: $name" | head -1 | cut -d: -f1)
    
    # Also create a tag
    git tag -f "ai-ckpt-$name" HEAD 2>/dev/null
    
    # Record metadata
    echo "{\"name\":\"$name\",\"ref\":\"$stash_ref\",\"commit\":\"$(git rev-parse HEAD)\",\"time\":\"$(date -Iseconds)\"}" >> "$CHECKPOINT_DIR/history.json"
    
    echo -e "${GREEN}✓${NC} Checkpoint: $name"
    [ -n "$stash_ref" ] && git stash pop --quiet
}

list_checkpoints() {
    echo -e "${CYAN}📌 Checkpoints${NC}"
    git tag -l "ai-ckpt-*" 2>/dev/null | while read tag; do
        local name=${tag#ai-ckpt-}
        local date=$(git log -1 --format=%ci "$tag" 2>/dev/null | cut -d' ' -f1)
        echo "  $name ($date)"
    done
    echo ""
    echo "Stashes:"
    git stash list 2>/dev/null | grep "ai-checkpoint" | head -5
}

restore_checkpoint() {
    local name=$1
    [ -z "$name" ] && { echo "Usage: ai-undo restore <name>"; return 1; }
    
    if git tag -l "ai-ckpt-$name" | grep -q .; then
        git checkout "ai-ckpt-$name" 2>/dev/null
        echo -e "${GREEN}✓${NC} Restored to: $name"
    else
        echo -e "${RED}Checkpoint not found: $name${NC}"
    fi
}

auto_checkpoint() {
    # Create checkpoint before risky operations
    create_checkpoint "auto_$(date +%H%M%S)"
}

case "$1" in
    create|save) create_checkpoint "$2" ;;
    list|ls) list_checkpoints ;;
    restore) restore_checkpoint "$2" ;;
    auto) auto_checkpoint ;;
    *) echo "Usage: ai-undo <create|list|restore|auto>" ;;
esac

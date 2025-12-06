#!/bin/bash
# Enhanced Parallel Executor v2.3
# AI Tools - Git Worktree-based parallel execution with smart merge

WORKTREE_DIR="$HOME/.ai_tools/worktrees"
PARALLEL_LOG="$HOME/.ai_tools/logs/parallel.log"
MERGE_STRATEGY_FILE="$HOME/.ai_tools/config/merge_strategy.yaml"

mkdir -p "$WORKTREE_DIR" "$(dirname "$PARALLEL_LOG")" "$(dirname "$MERGE_STRATEGY_FILE")"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${BLUE}[parallel]${NC} $1"; echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $1" >> "$PARALLEL_LOG"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }

check_git() {
    git rev-parse --git-dir > /dev/null 2>&1 || { error "Not in git repo"; exit 1; }
}

# Create worktree for a task
create_worktree() {
    local branch=$1 task_id=$2
    local worktree_path="$WORKTREE_DIR/$task_id"
    
    [ -d "$worktree_path" ] && { warn "Worktree exists: $task_id"; return 1; }
    
    git worktree add -b "$branch" "$worktree_path" 2>/dev/null || \
    git worktree add "$worktree_path" -b "$branch" HEAD 2>/dev/null
    
    log "Created worktree: $task_id → $worktree_path"
    echo "$worktree_path"
}

# Execute task in worktree
execute_in_worktree() {
    local worktree_path=$1 tool=$2 prompt=$3 task_id=$4
    
    log "[$task_id] Starting: $tool"
    
    (
        cd "$worktree_path" || exit 1
        local start=$(date +%s)
        
        # Execute with timeout
        timeout 600 $tool "$prompt" > "$worktree_path/.task_output.log" 2>&1
        local exit_code=$?
        
        local duration=$(($(date +%s) - start))
        echo "$exit_code" > "$worktree_path/.task_exit_code"
        echo "$duration" > "$worktree_path/.task_duration"
        
        # Auto-commit changes
        if [ $exit_code -eq 0 ]; then
            git add -A 2>/dev/null
            git commit -m "AI Task: $prompt" --allow-empty 2>/dev/null
        fi
        
        exit $exit_code
    ) &
    
    echo $!
}

# Run parallel tasks from config
run_parallel() {
    local config_file=$1
    check_git
    
    [ ! -f "$config_file" ] && { error "Config not found: $config_file"; exit 1; }
    
    echo ""
    echo "⚡ Parallel Execution Engine v2.3"
    echo "=================================="
    echo ""
    
    local pids=() task_ids=() worktrees=()
    local main_branch=$(git branch --show-current)
    
    log "Main branch: $main_branch"
    log "Reading tasks from: $config_file"
    echo ""
    
    # Parse and launch tasks
    while IFS='|' read -r branch tool prompt || [ -n "$branch" ]; do
        [ -z "$branch" ] || [[ "$branch" == \#* ]] && continue
        
        local task_id="task_$(date +%s%N | cut -c1-13)_$RANDOM"
        local worktree_path=$(create_worktree "$branch" "$task_id")
        
        [ -z "$worktree_path" ] && continue
        
        local pid=$(execute_in_worktree "$worktree_path" "$tool" "$prompt" "$task_id")
        
        pids+=("$pid")
        task_ids+=("$task_id")
        worktrees+=("$worktree_path")
        
        echo -e "  ${CYAN}[$task_id]${NC} $branch → $tool"
    done < "$config_file"
    
    echo ""
    log "Launched ${#pids[@]} parallel tasks"
    echo ""
    
    # Wait for all tasks with progress
    local completed=0 failed=0
    for i in "${!pids[@]}"; do
        local pid=${pids[$i]}
        local task_id=${task_ids[$i]}
        local worktree=${worktrees[$i]}
        
        echo -n "  Waiting for $task_id... "
        wait "$pid" 2>/dev/null
        
        local exit_code=$(cat "$worktree/.task_exit_code" 2>/dev/null || echo "1")
        local duration=$(cat "$worktree/.task_duration" 2>/dev/null || echo "?")
        
        if [ "$exit_code" = "0" ]; then
            echo -e "${GREEN}✓${NC} (${duration}s)"
            ((completed++))
        else
            echo -e "${RED}✗${NC} (exit: $exit_code)"
            ((failed++))
        fi
    done
    
    echo ""
    log "Completed: $completed | Failed: $failed"
    
    # Suggest next steps
    echo ""
    echo "Next steps:"
    echo "  ai-parallel status     - View task details"
    echo "  ai-parallel merge      - Merge all to $main_branch"
    echo "  ai-parallel cleanup    - Remove worktrees"
}

# Smart merge with conflict detection
smart_merge() {
    local target_branch=${1:-$(git branch --show-current)}
    check_git
    
    echo ""
    echo "🔀 Smart Merge"
    echo "=============="
    echo ""
    
    local worktrees=$(git worktree list --porcelain | grep "^worktree" | grep "$WORKTREE_DIR" | cut -d' ' -f2)
    
    [ -z "$worktrees" ] && { log "No worktrees to merge"; return; }
    
    git checkout "$target_branch" 2>/dev/null
    
    local merged=0 conflicts=0
    while IFS= read -r wt; do
        [ -z "$wt" ] && continue
        
        local branch=$(git -C "$wt" branch --show-current 2>/dev/null)
        [ -z "$branch" ] && continue
        
        echo -n "  Merging $branch... "
        
        # Try fast-forward first
        if git merge --ff-only "$branch" 2>/dev/null; then
            echo -e "${GREEN}✓ fast-forward${NC}"
            ((merged++))
        # Try regular merge
        elif git merge --no-edit "$branch" 2>/dev/null; then
            echo -e "${GREEN}✓ merged${NC}"
            ((merged++))
        # Conflict detected
        else
            echo -e "${RED}✗ conflict${NC}"
            git merge --abort 2>/dev/null
            ((conflicts++))
            
            # Save conflict info
            echo "$branch" >> "$WORKTREE_DIR/.conflicts"
        fi
    done <<< "$worktrees"
    
    echo ""
    log "Merged: $merged | Conflicts: $conflicts"
    
    if [ $conflicts -gt 0 ]; then
        echo ""
        warn "Some branches had conflicts. Options:"
        echo "  1) Manually resolve: git merge <branch>"
        echo "  2) Use AI to resolve: ai-route 'Resolve merge conflict'"
        echo "  3) Skip conflicting: ai-parallel merge --skip-conflicts"
    fi
}

# Show status of parallel tasks
show_status() {
    echo ""
    echo "📊 Parallel Task Status"
    echo "======================="
    echo ""
    
    local worktrees=$(git worktree list 2>/dev/null | grep "$WORKTREE_DIR")
    
    [ -z "$worktrees" ] && { log "No active worktrees"; return; }
    
    while IFS= read -r line; do
        local path=$(echo "$line" | awk '{print $1}')
        local branch=$(echo "$line" | grep -oP '\[\K[^\]]+')
        
        [ -z "$path" ] && continue
        
        local exit_code=$(cat "$path/.task_exit_code" 2>/dev/null || echo "running")
        local duration=$(cat "$path/.task_duration" 2>/dev/null || echo "-")
        
        local status_color=$YELLOW
        local status_text="running"
        
        if [ "$exit_code" = "0" ]; then
            status_color=$GREEN; status_text="success"
        elif [ "$exit_code" != "running" ]; then
            status_color=$RED; status_text="failed"
        fi
        
        printf "  %-30s ${status_color}%-8s${NC} %ss\n" "$branch" "$status_text" "$duration"
    done <<< "$worktrees"
}

# List worktrees
list_worktrees() {
    echo ""
    echo "📁 Active Worktrees"
    echo "==================="
    echo ""
    git worktree list 2>/dev/null | grep -v "^$(git rev-parse --show-toplevel)" || echo "  None"
}

# Cleanup worktrees
cleanup_worktrees() {
    local force=${1:-false}
    
    echo ""
    log "Cleaning up worktrees..."
    
    local count=0
    for wt in "$WORKTREE_DIR"/task_*; do
        [ -d "$wt" ] || continue
        
        local branch=$(git -C "$wt" branch --show-current 2>/dev/null)
        
        git worktree remove "$wt" --force 2>/dev/null && {
            [ -n "$branch" ] && git branch -D "$branch" 2>/dev/null
            ((count++))
        }
    done
    
    rm -f "$WORKTREE_DIR/.conflicts"
    
    success "Removed $count worktrees"
}

# View task output
view_output() {
    local task_pattern=$1
    
    for wt in "$WORKTREE_DIR"/*"$task_pattern"*; do
        [ -d "$wt" ] || continue
        
        echo ""
        echo "=== Output: $(basename "$wt") ==="
        cat "$wt/.task_output.log" 2>/dev/null || echo "(no output)"
    done
}

show_help() {
    echo "Parallel Executor v2.3 - Git Worktree Engine"
    echo ""
    echo "Usage: ai-parallel <command> [options]"
    echo ""
    echo "Commands:"
    echo "  run <config>      Run parallel tasks from config"
    echo "  status            Show task status"
    echo "  list              List active worktrees"
    echo "  merge [branch]    Smart merge all to branch"
    echo "  output [pattern]  View task output"
    echo "  cleanup           Remove all worktrees"
    echo ""
    echo "Config format (one per line):"
    echo "  branch-name|tool-command|prompt"
    echo ""
    echo "Example config:"
    echo "  feature-api|cline-auto|Create REST API"
    echo "  feature-db|aider-auto|Add migrations"
}

case "$1" in
    run) run_parallel "$2" ;;
    status) show_status ;;
    list) list_worktrees ;;
    merge) smart_merge "$2" ;;
    output) view_output "$2" ;;
    cleanup) cleanup_worktrees "$2" ;;
    -h|--help|"") show_help ;;
    *) error "Unknown: $1"; show_help ;;
esac

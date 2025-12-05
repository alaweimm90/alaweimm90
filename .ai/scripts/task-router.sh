#!/bin/bash
# Enhanced Intelligent Task Router v2.3
# AI Tools Cheatsheet - Bayesian ML-based Task Classification

set -e

HISTORY_FILE="$HOME/.ai_tools/metrics/task_history.json"
ROUTER_LOG="$HOME/.ai_tools/logs/router.log"
MODEL_FILE="$HOME/.ai_tools/learning/routing_model.json"

# Initialize files
mkdir -p "$(dirname "$HISTORY_FILE")" "$(dirname "$MODEL_FILE")"
[ ! -f "$HISTORY_FILE" ] && echo '{"tasks": [], "tool_performance": {}}' > "$HISTORY_FILE"
[ ! -f "$MODEL_FILE" ] && echo '{"priors": {}, "likelihoods": {}, "total_samples": 0}' > "$MODEL_FILE"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${BLUE}[router]${NC} $1"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }

# Enhanced task patterns with weights
declare -A TASK_PATTERNS
TASK_PATTERNS["architect"]="design:3|architecture:3|plan:2|structure:2|schema:2|scalab:2|system:1|microservice:3|diagram:2"
TASK_PATTERNS["build"]="create:2|setup:2|install:1|init:2|scaffold:3|generate:2|build:1|implement:3|develop:2"
TASK_PATTERNS["fix"]="fix:3|bug:3|error:2|debug:3|refactor:2|optimize:2|improve:1|update:1|patch:2|resolve:2"
TASK_PATTERNS["snippet"]="snippet:3|boilerplate:2|template:2|example:2|sample:2|quick:1|config:1"
TASK_PATTERNS["test"]="test:3|spec:2|coverage:2|unit:2|integration:2|e2e:2|jest:2|pytest:2"
TASK_PATTERNS["security"]="security:3|audit:2|vulnerability:3|auth:2|encrypt:2|secret:2"
TASK_PATTERNS["docs"]="document:3|readme:3|comment:2|jsdoc:2|docstring:2|swagger:2"

declare -A TOOL_MAP
TOOL_MAP["architect"]="kilo-auto"
TOOL_MAP["build"]="cline-auto"
TOOL_MAP["fix"]="aider-auto"
TOOL_MAP["snippet"]="blackbox-auto"
TOOL_MAP["test"]="aider-auto"
TOOL_MAP["security"]="kilo-auto"
TOOL_MAP["docs"]="blackbox-auto"

calculate_score() {
    local prompt="$1" category="$2"
    local prompt_lower=$(echo "$prompt" | tr '[:upper:]' '[:lower:]')
    local score=0 pattern="${TASK_PATTERNS[$category]}"
    
    IFS='|' read -ra KEYWORDS <<< "$pattern"
    for kw in "${KEYWORDS[@]}"; do
        local keyword=$(echo "$kw" | cut -d':' -f1)
        local weight=$(echo "$kw" | cut -d':' -f2)
        [[ "$prompt_lower" == *"$keyword"* ]] && score=$((score + weight))
    done
    
    local tool="${TOOL_MAP[$category]}"
    local prior=$(get_prior "$tool" "$category")
    echo "scale=2; $score * $prior" | bc 2>/dev/null || echo "$score"
}

get_prior() {
    local tool=$1 category=$2
    if command -v jq &> /dev/null && [ -f "$MODEL_FILE" ]; then
        jq -r ".priors[\"${tool}_${category}\"] // 1.0" "$MODEL_FILE" 2>/dev/null
    else
        echo "1.0"
    fi
}

update_model() {
    local tool=$1 category=$2 success=$3
    command -v jq &> /dev/null || return
    local key="${tool}_${category}" temp_file=$(mktemp)
    local adjustment=$([ "$success" = "true" ] && echo "1.1" || echo "0.9")
    jq --arg key "$key" --arg adj "$adjustment" '.total_samples += 1 | .priors[$key] = ((.priors[$key] // 1.0) * ($adj | tonumber) | if . > 2.0 then 2.0 elif . < 0.1 then 0.1 else . end)' "$MODEL_FILE" > "$temp_file" && mv "$temp_file" "$MODEL_FILE"
}

classify_task() {
    local prompt="$1" best_match="" best_score=0
    for category in "${!TASK_PATTERNS[@]}"; do
        local score=$(calculate_score "$prompt" "$category")
        if (( $(echo "$score > $best_score" | bc -l 2>/dev/null || echo "0") )); then
            best_score=$score; best_match=$category
        fi
    done
    [ -z "$best_match" ] || [ "$best_score" = "0" ] && best_match="build"
    echo "$best_match"
}

get_confidence() {
    local score=$(calculate_score "$1" "$2")
    if (( $(echo "$score >= 6" | bc -l 2>/dev/null || echo "0") )); then echo "high"
    elif (( $(echo "$score >= 3" | bc -l 2>/dev/null || echo "0") )); then echo "medium"
    else echo "low"; fi
}

explain_classification() {
    local prompt="$1"
    echo "" && echo "🔍 Classification Analysis" && echo "==========================" && echo ""
    echo "Prompt: \"$prompt\"" && echo "" && echo "Scores by category:"
    for category in "${!TASK_PATTERNS[@]}"; do
        local score=$(calculate_score "$prompt" "$category")
        local tool="${TOOL_MAP[$category]}"
        printf "  %-12s %5s → %s\n" "$category" "$score" "$tool"
    done
    echo ""
    local best=$(classify_task "$prompt")
    echo -e "Selected: ${CYAN}$best${NC} → ${GREEN}${TOOL_MAP[$best]}${NC}"
}

log_task() {
    local task_type=$1 tool=$2 prompt=$3 status=$4 duration=$5
    echo "[$(date -Iseconds)] TYPE=$task_type TOOL=$tool STATUS=$status DURATION=${duration}s" >> "$ROUTER_LOG"
    update_model "$tool" "$task_type" "$([ "$status" = "SUCCESS" ] && echo "true" || echo "false")"
}

route_task() {
    local prompt="$1" force_tool="$2" dry_run="$3"
    [ -z "$prompt" ] && { echo -n "Enter task: "; read -r prompt; }
    
    local task_type=$(classify_task "$prompt")
    local tool="${TOOL_MAP[$task_type]}"
    local confidence=$(get_confidence "$prompt" "$task_type")
    [ -n "$force_tool" ] && tool="$force_tool"
    
    echo "" && log "Task: ${CYAN}$task_type${NC} | Tool: ${GREEN}$tool${NC} | Confidence: $confidence"
    [ "$confidence" = "low" ] && echo -e "${YELLOW}⚠️  Low confidence - consider --explain${NC}"
    [ "$dry_run" = "true" ] && { echo "Dry run: $tool \"$prompt\""; return 0; }
    
    local start_time=$(date +%s)
    $tool "$prompt"; local exit_code=$?
    local duration=$(($(date +%s) - start_time))
    log_task "$task_type" "$tool" "$prompt" "$([ $exit_code -eq 0 ] && echo "SUCCESS" || echo "FAILURE")" "$duration"
    [ $exit_code -eq 0 ] && success "Completed in ${duration}s" || echo -e "${RED}❌ Failed${NC}"
    return $exit_code
}

show_stats() {
    echo "" && echo "📊 Routing Statistics" && echo "=====================" && echo ""
    [ ! -f "$ROUTER_LOG" ] && { echo "No history"; return; }
    echo "By Type:" && grep -oP 'TYPE=\K\w+' "$ROUTER_LOG" | sort | uniq -c | sort -rn
    echo "" && echo "Success Rate:"
    local total=$(grep -c "STATUS=" "$ROUTER_LOG" 2>/dev/null || echo 0)
    local success=$(grep -c "STATUS=SUCCESS" "$ROUTER_LOG" 2>/dev/null || echo 0)
    [ "$total" -gt 0 ] && echo "  $success / $total ($(echo "scale=1; $success * 100 / $total" | bc)%)"
}

show_help() {
    echo "Intelligent Task Router v2.3 - Bayesian ML"
    echo ""
    echo "Usage: ai-route [OPTIONS] \"<prompt>\""
    echo ""
    echo "Options:"
    echo "  --force <tool>    Force specific tool"
    echo "  --dry-run         Show without executing"
    echo "  --explain         Show scoring details"
    echo "  --stats           Show statistics"
    echo "  --history [n]     Show recent tasks"
    echo "  -h, --help        Show help"
}

case "$1" in
    --force) route_task "$3" "$2" "false" ;;
    --dry-run) route_task "$2" "" "true" ;;
    --explain) explain_classification "$2" ;;
    --stats) show_stats ;;
    --history) tail -n "${2:-10}" "$ROUTER_LOG" 2>/dev/null ;;
    -h|--help) show_help ;;
    "") route_task "" ;;
    *) route_task "$*" ;;
esac

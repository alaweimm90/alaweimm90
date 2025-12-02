#!/bin/bash
# Enhanced Self-Improving System v2.3
# AI Tools - ML-based learning with pattern recognition

LEARNING_DIR="$HOME/.ai_tools/learning"
MODELS_DIR="$LEARNING_DIR/models"
mkdir -p "$MODELS_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'

log() { echo -e "${BLUE}[learn]${NC} $1"; }
success() { echo -e "${GREEN}✓ $1${NC}"; }
warn() { echo -e "${YELLOW}! $1${NC}"; }

ROUTING_MODEL="$LEARNING_DIR/routing_model.json"
PATTERNS_FILE="$MODELS_DIR/patterns.json"
FEEDBACK_FILE="$LEARNING_DIR/feedback.json"

init_learning() {
    [ -f "$ROUTING_MODEL" ] || echo '{"version":"2.3","priors":{"code":0.25,"debug":0.25,"docs":0.25,"test":0.25},"task_history":[],"feedback_count":0}' > "$ROUTING_MODEL"
    [ -f "$PATTERNS_FILE" ] || echo '{"successful_patterns":[],"tool_affinity":{}}' > "$PATTERNS_FILE"
    [ -f "$FEEDBACK_FILE" ] || echo '{"entries":[]}' > "$FEEDBACK_FILE"
}

record_outcome() {
    local task="$1" tool="$2" category="$3" suc="$4" dur="${5:-0}"
    init_learning
    local tmp=$(mktemp)
    jq --arg t "$task" --arg tool "$tool" --arg c "$category" --argjson s "$suc" --argjson d "$dur" \
       '.task_history += [{"task":$t,"tool":$tool,"category":$c,"success":$s,"duration":$d}] | .task_history = .task_history[-100:] | .feedback_count += 1' \
       "$ROUTING_MODEL" > "$tmp" && mv "$tmp" "$ROUTING_MODEL"
    log "Recorded: $category → $tool (success: $suc)"
}

train_model() {
    init_learning
    log "Training model..."
    local count=$(jq '.task_history | length' "$ROUTING_MODEL")
    [ "$count" -lt 5 ] && { warn "Need 5+ samples (have $count)"; return; }
    local tmp=$(mktemp)
    jq '.task_history | group_by(.category) | map({key:.[0].category, value:(length/100)}) | from_entries' "$ROUTING_MODEL" > "$tmp.p"
    [ -s "$tmp.p" ] && jq --slurpfile p "$tmp.p" '.priors = $p[0] | .last_training = (now|todate)' "$ROUTING_MODEL" > "$tmp" && mv "$tmp" "$ROUTING_MODEL"
    rm -f "$tmp.p"
    success "Model trained on $count samples"
}

show_stats() {
    init_learning
    echo ""; echo "Learning Stats"; echo "=============="
    echo "Samples: $(jq '.task_history | length' "$ROUTING_MODEL")"
    echo "Feedback: $(jq '.feedback_count' "$ROUTING_MODEL")"
    echo "Priors:"; jq -r '.priors | to_entries[] | "  \(.key): \(.value*100|floor)%"' "$ROUTING_MODEL"
}

record_feedback() {
    local r=$1 c="$2"; init_learning
    local tmp=$(mktemp)
    jq --argjson r "$r" --arg c "$c" '.entries += [{"rating":$r,"comment":$c}]' "$FEEDBACK_FILE" > "$tmp" && mv "$tmp" "$FEEDBACK_FILE"
    success "Feedback recorded (rating: $r)"
}

auto_improve() { log "Auto-improving..."; train_model; success "Complete"; }

case "$1" in
    record) record_outcome "$2" "$3" "$4" "$5" "$6" ;;
    feedback) record_feedback "$2" "$3" ;;
    train) train_model ;;
    stats) show_stats ;;
    auto) auto_improve ;;
    init) init_learning && success "Initialized" ;;
    *) echo "Usage: ai-learn <record|feedback|train|stats|auto|init>" ;;
esac

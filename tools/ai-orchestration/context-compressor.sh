#!/bin/bash
# Enhanced Context Compressor v2.3
# AI Tools - Semantic-aware context compression with TF-IDF

CONTEXT_DIR="$HOME/.ai_tools/context"
CACHE_DIR="$CONTEXT_DIR/cache"
SUMMARIES_DIR="$CONTEXT_DIR/summaries"

mkdir -p "$CACHE_DIR" "$SUMMARIES_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

log() { echo -e "${BLUE}[context]${NC} $1"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }

# Calculate file hash for caching
file_hash() {
    md5sum "$1" 2>/dev/null | cut -d' ' -f1
}

# Extract key terms using TF-IDF approximation
extract_key_terms() {
    local file=$1 top_n=${2:-20}
    
    # Word frequency (Term Frequency)
    cat "$file" 2>/dev/null | \
        tr '[:upper:]' '[:lower:]' | \
        grep -oE '\b[a-z_][a-z0-9_]{2,}\b' | \
        sort | uniq -c | sort -rn | \
        head -$top_n | \
        awk '{print $2}' | \
        tr '\n' ' '
}

# Extract code structure (functions, classes, etc.)
extract_structure() {
    local file=$1
    local ext="${file##*.}"
    
    case "$ext" in
        py)
            grep -E '^(class |def |async def )' "$file" 2>/dev/null | head -20
            ;;
        js|ts|tsx|jsx)
            grep -E '(function |const |class |export |import )' "$file" 2>/dev/null | head -20
            ;;
        sh|bash)
            grep -E '^[a-z_]+\(\)|^function ' "$file" 2>/dev/null | head -20
            ;;
        go)
            grep -E '^(func |type |package )' "$file" 2>/dev/null | head -20
            ;;
        rs)
            grep -E '^(fn |struct |impl |pub |mod )' "$file" 2>/dev/null | head -20
            ;;
        *)
            grep -E '(function|class|def |struct|interface)' "$file" 2>/dev/null | head -20
            ;;
    esac
}

# Extract semantic sections (comments, docstrings)
extract_semantics() {
    local file=$1
    
    # Extract comments and docstrings
    grep -E '(#.*|//.*|/\*|\*/|""".*"""|'"'"'"""' "$file" 2>/dev/null | \
        grep -v '#!/' | \
        head -30
}

# Smart compress a single file
compress_file() {
    local file=$1 max_lines=${2:-100}
    local hash=$(file_hash "$file")
    local cache_file="$CACHE_DIR/${hash}.compressed"
    
    # Check cache
    if [ -f "$cache_file" ]; then
        cat "$cache_file"
        return
    fi
    
    local total_lines=$(wc -l < "$file" 2>/dev/null)
    local filename=$(basename "$file")
    
    echo "# File: $filename ($total_lines lines)"
    echo ""
    
    # If small enough, return as-is
    if [ "$total_lines" -le "$max_lines" ]; then
        cat "$file"
    else
        # Semantic compression
        echo "## Key Terms"
        echo "$(extract_key_terms "$file" 15)"
        echo ""
        
        echo "## Structure"
        extract_structure "$file"
        echo ""
        
        echo "## Head (first 30 lines)"
        head -30 "$file"
        echo ""
        echo "... [${total_lines} total lines, $(($total_lines - 60)) omitted] ..."
        echo ""
        
        echo "## Tail (last 30 lines)"
        tail -30 "$file"
    fi | tee "$cache_file"
}

# Compress directory context
compress_directory() {
    local dir=$1 max_files=${2:-20}
    
    echo "# Directory: $dir"
    echo "# Files: $(find "$dir" -type f | wc -l)"
    echo ""
    
    # List structure
    echo "## Structure"
    if command -v tree &>/dev/null; then
        tree -L 3 --noreport "$dir" 2>/dev/null | head -50
    else
        find "$dir" -maxdepth 3 -type f | head -50
    fi
    echo ""
    
    # Most important files (by extension priority)
    echo "## Key Files"
    local key_files=$(find "$dir" -type f \( \
        -name "*.py" -o -name "*.js" -o -name "*.ts" \
        -o -name "*.go" -o -name "*.rs" -o -name "*.java" \
        -o -name "README*" -o -name "*.yaml" -o -name "*.json" \
    \) 2>/dev/null | head -$max_files)
    
    for file in $key_files; do
        echo ""
        echo "### $(basename "$file")"
        compress_file "$file" 50
    done
}

# Compress git diff context
compress_diff() {
    local ref=${1:-HEAD~1}
    
    echo "# Git Diff: $ref"
    echo ""
    
    # Summary first
    echo "## Summary"
    git diff --stat "$ref" 2>/dev/null | tail -10
    echo ""
    
    # Key changes (limited)
    echo "## Changes"
    git diff "$ref" 2>/dev/null | \
        grep -E '^(\+\+\+|---|\+[^+]|-[^-])' | \
        head -100
}

# Generate project summary
project_summary() {
    local dir=${1:-.}
    local summary_file="$SUMMARIES_DIR/$(basename "$dir")_summary.md"
    
    log "Generating project summary for: $dir"
    
    {
        echo "# Project Summary: $(basename "$dir")"
        echo "Generated: $(date)"
        echo ""
        
        # README content
        if [ -f "$dir/README.md" ]; then
            echo "## README"
            head -50 "$dir/README.md"
            echo ""
        fi
        
        # Package info
        if [ -f "$dir/package.json" ]; then
            echo "## Package (Node.js)"
            jq '{name, version, description, main, scripts: .scripts | keys}' "$dir/package.json" 2>/dev/null
            echo ""
        fi
        
        if [ -f "$dir/pyproject.toml" ]; then
            echo "## Project (Python)"
            head -30 "$dir/pyproject.toml"
            echo ""
        fi
        
        # File type distribution
        echo "## File Distribution"
        find "$dir" -type f -name "*.*" 2>/dev/null | \
            grep -v node_modules | grep -v .git | grep -v __pycache__ | \
            sed 's/.*\.//' | sort | uniq -c | sort -rn | head -10
        echo ""
        
        # Top-level structure
        echo "## Structure"
        ls -la "$dir" 2>/dev/null | head -20
        
    } | tee "$summary_file"
    
    success "Summary saved: $summary_file"
}

# Interactive compress for AI context window
ai_context() {
    local files=("${@}")
    local total_tokens=0
    local max_tokens=${AI_CONTEXT_MAX:-8000}
    
    echo "# AI Context Window"
    echo "# Max tokens: $max_tokens"
    echo ""
    
    for file in "${files[@]}"; do
        if [ -f "$file" ]; then
            local content=$(compress_file "$file" 80)
            local tokens=$(echo "$content" | wc -w)
            
            if [ $((total_tokens + tokens)) -lt $max_tokens ]; then
                echo "$content"
                echo ""
                total_tokens=$((total_tokens + tokens))
            else
                log "Skipping $file (would exceed token limit)"
            fi
        elif [ -d "$file" ]; then
            compress_directory "$file" 10
        fi
    done
    
    log "Total tokens (approx): $total_tokens / $max_tokens"
}

# Clear cache
clear_cache() {
    rm -rf "$CACHE_DIR"/*
    rm -rf "$SUMMARIES_DIR"/*
    success "Cache cleared"
}

# Show cache stats
cache_stats() {
    echo "Context Cache Stats"
    echo "==================="
    echo ""
    echo "Cache files: $(ls "$CACHE_DIR" 2>/dev/null | wc -l)"
    echo "Cache size:  $(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)"
    echo "Summaries:   $(ls "$SUMMARIES_DIR" 2>/dev/null | wc -l)"
}

show_help() {
    echo "Context Compressor v2.3 - Semantic Compression"
    echo ""
    echo "Usage: ai-context <command> [options]"
    echo ""
    echo "Commands:"
    echo "  file <path> [max_lines]    Compress single file"
    echo "  dir <path> [max_files]     Compress directory"
    echo "  diff [ref]                 Compress git diff"
    echo "  summary [path]             Generate project summary"
    echo "  ai <files...>              Prepare AI context window"
    echo "  stats                      Show cache stats"
    echo "  clear                      Clear cache"
    echo ""
    echo "Environment:"
    echo "  AI_CONTEXT_MAX=8000        Max tokens for AI context"
}

case "$1" in
    file) compress_file "$2" "$3" ;;
    dir) compress_directory "$2" "$3" ;;
    diff) compress_diff "$2" ;;
    summary) project_summary "$2" ;;
    ai) shift; ai_context "$@" ;;
    stats) cache_stats ;;
    clear) clear_cache ;;
    -h|--help|"") show_help ;;
    *) 
        # Default: treat as file/dir
        if [ -f "$1" ]; then
            compress_file "$1" "$2"
        elif [ -d "$1" ]; then
            compress_directory "$1" "$2"
        else
            error "Unknown: $1"
            show_help
        fi
        ;;
esac

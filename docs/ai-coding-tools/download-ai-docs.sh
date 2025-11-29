#!/bin/bash
# ============================================================================
# AI Coding Tools Documentation Downloader (Git Bash Edition)
# Downloads documentation for ALL major AI coding assistants and IDEs
# ============================================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  AI Coding Tools Documentation Downloader     ${NC}"
echo -e "${BLUE}  Git Bash Edition                             ${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Function to clone repos
clone_docs() {
    local name=$1
    local repo=$2
    local dir="$BASE_DIR/$name"

    echo -e "${GREEN}[$(date +%H:%M:%S)] Cloning $name...${NC}"
    echo -e "${YELLOW}  Repository: $repo${NC}"
    echo -e "${YELLOW}  Destination: $dir${NC}"

    if [ -d "$dir" ]; then
        echo -e "${YELLOW}  Directory exists, pulling latest...${NC}"
        cd "$dir" && git pull --quiet 2>/dev/null && cd - > /dev/null
    else
        mkdir -p "$(dirname "$dir")"
        git clone --quiet --depth 1 "$repo" "$dir" 2>/dev/null || echo -e "${YELLOW}  Warning: Clone may have issues${NC}"
    fi

    echo -e "${GREEN}  Done: $name${NC}"
    echo ""
}

# ============================================================================
# Clone GitHub-based Documentation
# ============================================================================

echo -e "${BLUE}=== Cloning GitHub-based Documentation ===${NC}"
echo ""

# Continue.dev
clone_docs "01-major-assistants/continue-dev" "https://github.com/continuedev/continue.git"

# Aider
clone_docs "02-terminal-cli/aider" "https://github.com/paul-gauthier/aider.git"

# Shell GPT
clone_docs "02-terminal-cli/shell-gpt" "https://github.com/TheR1D/shell_gpt.git"

# AI Shell
clone_docs "02-terminal-cli/ai-shell" "https://github.com/BuilderIO/ai-shell.git"

# GPT Engineer
clone_docs "05-specialized-tools/gpt-engineer" "https://github.com/gpt-engineer-org/gpt-engineer.git"

# Sweep AI
clone_docs "05-specialized-tools/sweep-ai" "https://github.com/sweepai/sweep.git"

# Mentat
clone_docs "05-specialized-tools/mentat" "https://github.com/AbanteAI/mentat.git"

# CodeGPT
clone_docs "05-specialized-tools/codegpt" "https://github.com/appleboy/CodeGPT.git"

# Ollama
clone_docs "06-open-source-llms/ollama" "https://github.com/ollama/ollama.git"

# Jan AI
clone_docs "06-open-source-llms/jan-ai" "https://github.com/janhq/jan.git"

# LocalAI
clone_docs "06-open-source-llms/local-ai" "https://github.com/mudler/LocalAI.git"

# AutoGPT
clone_docs "08-resources/autogpt" "https://github.com/Significant-Gravitas/AutoGPT.git"

# LangChain
clone_docs "08-resources/langchain" "https://github.com/langchain-ai/langchain.git"

# LlamaIndex
clone_docs "08-resources/llamaindex" "https://github.com/run-llama/llama_index.git"

echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}  Documentation Download Complete!              ${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo -e "${YELLOW}Location: $BASE_DIR${NC}"

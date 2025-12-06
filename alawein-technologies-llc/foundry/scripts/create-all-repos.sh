#!/bin/bash

# ATLAS Repository Creator - Create All Repos
# Creates all ATLAS product repositories at once
#
# Usage: ./create-all-repos.sh [phase]
# Example:
#   ./create-all-repos.sh phase1  # Creates Phase 1 repos only
#   ./create-all-repos.sh all     # Creates ALL repos

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PHASE=${1:-phase1}

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}🚀 ATLAS Mass Repository Creator${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Define repositories by phase
declare -A PHASE1_REPOS=(
    ["nightmare-mode"]="atlas-nightmare-validator"
    ["chaos-engine"]="atlas-chaos-engine"
    ["research-prison"]="atlas-interrogator"
)

declare -A PHASE2_REPOS=(
    ["atlas-core"]="atlas-core"
)

declare -A PHASE3_REPOS=(
    # Add when templates ready
)

# Function to create a single repo
create_repo() {
    local template=$1
    local repo_name=$2

    echo -e "${YELLOW}Creating ${repo_name} from ${template}...${NC}"

    if ./create-repo.sh "$template" "$repo_name"; then
        echo -e "${GREEN}✅ ${repo_name} created successfully${NC}"
        return 0
    else
        echo -e "${RED}❌ Failed to create ${repo_name}${NC}"
        return 1
    fi
}

# Create based on phase
case $PHASE in
    phase1)
        echo "Creating Phase 1 repositories (Quick Wins)..."
        echo ""
        for template in "${!PHASE1_REPOS[@]}"; do
            create_repo "$template" "${PHASE1_REPOS[$template]}"
            echo ""
        done
        ;;

    phase2)
        echo "Creating Phase 2 repositories (Platform)..."
        echo ""
        for template in "${!PHASE2_REPOS[@]}"; do
            create_repo "$template" "${PHASE2_REPOS[$template]}"
            echo ""
        done
        ;;

    all)
        echo "Creating ALL repositories..."
        echo ""

        echo "=== Phase 1: Quick Wins ==="
        for template in "${!PHASE1_REPOS[@]}"; do
            create_repo "$template" "${PHASE1_REPOS[$template]}"
            echo ""
        done

        echo "=== Phase 2: Platform ==="
        for template in "${!PHASE2_REPOS[@]}"; do
            create_repo "$template" "${PHASE2_REPOS[$template]}"
            echo ""
        done
        ;;

    *)
        echo "Usage: ./create-all-repos.sh [phase1|phase2|all]"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✅ Repository creation complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Created repositories:"
case $PHASE in
    phase1)
        for repo in "${PHASE1_REPOS[@]}"; do
            echo "  - $repo"
        done
        ;;
    phase2)
        for repo in "${PHASE2_REPOS[@]}"; do
            echo "  - $repo"
        done
        ;;
    all)
        for repo in "${PHASE1_REPOS[@]}" "${PHASE2_REPOS[@]}"; do
            echo "  - $repo"
        done
        ;;
esac

echo ""
echo "Next steps:"
echo "  1. cd into each repository"
echo "  2. Setup environment variables (.env files)"
echo "  3. Start development: docker-compose up"
echo "  4. Create GitHub repos and push"
echo ""
echo -e "${CYAN}Happy building! 🚀${NC}"

#!/usr/bin/env bash
# census-promote.sh — Convert orphan projects into proper repos

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== Portfolio Census: Promotion Tool ===${NC}"
echo ""

# Verify we have inventory.json with orphans
if [ ! -f "inventory.json" ]; then
    echo -e "${RED}✗ inventory.json not found. Run census.sh first.${NC}"
    exit 1
fi

# Extract orphans
ORPHANS=$(jq -r '.projects_without_repo[]? | "\(.org),\(.project_number),\(.reason)"' inventory.json)
DRAFTS=$(jq -r '.orphan_drafts[]? | "\(.org),\(.project),\(.draft_title)"' inventory.json)

if [ -z "$ORPHANS" ] && [ -z "$DRAFTS" ]; then
    echo -e "${GREEN}✓ No orphan projects found. Portfolio is clean!${NC}"
    exit 0
fi

echo -e "${YELLOW}Orphan Projects Found:${NC}"
echo ""

if [ -n "$ORPHANS" ]; then
    echo -e "${RED}Projects without repo links:${NC}"
    echo "$ORPHANS" | nl
    echo ""
fi

if [ -n "$DRAFTS" ]; then
    echo -e "${RED}Orphan DraftIssues:${NC}"
    echo "$DRAFTS" | nl
    echo ""
fi

# Interactive promotion
echo -e "${YELLOW}Promotion decisions:${NC}"
echo ""

# Helper function to convert title to repo name
title_to_repo() {
    local title="$1"
    local prefix="$2"
    echo "$title" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g;s/^-|-$//g' | xargs -I {} echo "${prefix}-{}"
}

# Helper function to create repo from template
create_repo_from_template() {
    local org="$1"
    local repo="$2"
    local template="$3"

    echo -e "${BLUE}  Creating: $repo${NC}"

    # Create repo
    if gh repo create "$org/$repo" --public --confirm 2>/dev/null; then
        echo -e "${GREEN}  ✓ Repo created${NC}"

        # Clone template
        if gh repo clone "alaweimm90/$template" "/tmp/$repo" 2>/dev/null; then
            cd "/tmp/$repo"

            # Update metadata
            cat > .meta/repo.yaml <<EOF
type: $(echo "$template" | sed 's/^template-//')
language: mixed
description: "Promoted from orphan project"
docs_profile: minimal
criticality_tier: 3
owner: "@$org"
created_date: "$(date +'%Y-%m-%d')"
last_updated: "$(date +'%Y-%m-%d')"
EOF

            # Update CODEOWNERS
            cat > .github/CODEOWNERS <<EOF
* @$org
EOF

            # Commit and push
            git add .meta/repo.yaml .github/CODEOWNERS
            git commit -m "chore: initialize from template"
            git push -u origin main 2>/dev/null

            echo -e "${GREEN}  ✓ Initialized from template: $template${NC}"

            cd - > /dev/null
            rm -rf "/tmp/$repo"
        else
            echo -e "${RED}  ✗ Failed to clone template${NC}"
        fi
    else
        echo -e "${RED}  ✗ Failed to create repo${NC}"
    fi
}

# Process orphan drafts
if [ -n "$DRAFTS" ]; then
    while IFS="," read -r org project draft_title; do
        echo -e "${YELLOW}Project: $project (Org: $org)${NC}"
        echo -e "  Draft: $draft_title"

        # Infer type from title
        if echo "$draft_title" | grep -iq "research\|paper\|notebook"; then
            prefix="paper"
            template="template-research"
        elif echo "$draft_title" | grep -iq "library\|lib"; then
            prefix="lib"
            template="template-python-lib"
        elif echo "$draft_title" | grep -iq "tool\|cli"; then
            prefix="tool"
            template="template-python-lib"
        elif echo "$draft_title" | grep -iq "adapter\|provider\|integration"; then
            prefix="adapter"
            template="template-python-lib"
        else
            prefix="lib"
            template="template-python-lib"
        fi

        # Suggest repo name
        repo_name=$(title_to_repo "$draft_title" "$prefix")
        echo -e "  ${YELLOW}Suggested repo: $repo_name${NC}"

        # Ask for confirmation
        read -p "  Create? (y/n/custom_name): " choice
        case "$choice" in
            y)
                create_repo_from_template "$org" "$repo_name" "$template"
                ;;
            n)
                echo -e "${YELLOW}  Skipped${NC}"
                ;;
            *)
                if [ -n "$choice" ]; then
                    create_repo_from_template "$org" "$choice" "$template"
                fi
                ;;
        esac

        echo ""
    done <<< "$DRAFTS"
fi

echo -e "${BLUE}=== Promotion Complete ===${NC}"
echo -e "${YELLOW}Run: bash census.sh again to verify all orphans are resolved${NC}"

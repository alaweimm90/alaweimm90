#!/usr/bin/env bash
# Portfolio Census — Complete sweep of repos, projects, gists, packages
# Outputs: consolidated inventory.json with project orphan detection

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
GH_ORGS="${GH_ORGS:-alaweimm90 AlaweinOS alaweimm90-science alaweimm90-business alaweimm90-tools}"
GH_USER="${GH_USER:-alaweimm90}"
TS=$(date +'%Y%m%d-%H%M%S')
OUT="outputs/$TS"

echo -e "${BLUE}=== Portfolio Census Suite ===${NC}"
echo -e "${BLUE}Timestamp: $TS${NC}"
echo -e "${BLUE}Output directory: $OUT${NC}"
echo ""

# Verify auth
echo -e "${YELLOW}Checking GitHub authentication...${NC}"
gh auth status || { echo "Not authenticated. Run: gh auth login"; exit 1; }
echo -e "${GREEN}✓ Authenticated${NC}"
echo ""

# Create output directory
mkdir -p "$OUT"

# =============================================================================
# STEP 1: Gather all repos (public, private, archived)
# =============================================================================
echo -e "${BLUE}[1/6] Scanning repositories...${NC}"
for ORG in $GH_ORGS; do
  echo -e "${YELLOW}  Org: $ORG${NC}"
  gh api -H "Accept: application/vnd.github+json" \
    "/orgs/$ORG/repos?per_page=100&type=all" --paginate \
    > "$OUT/$ORG.repos.json" 2>/dev/null && \
    echo -e "${GREEN}  ✓ Repos: $(jq -r 'length' "$OUT/$ORG.repos.json")${NC}" || \
    echo -e "${RED}  ✗ Failed to fetch repos${NC}"
done
echo ""

# =============================================================================
# STEP 2: Gather organization-level Projects (v2)
# =============================================================================
echo -e "${BLUE}[2/6] Scanning organization Projects (v2)...${NC}"
for ORG in $GH_ORGS; do
  echo -e "${YELLOW}  Org: $ORG${NC}"
  gh api graphql \
    --raw-field query='
query($login:String!,$endCursor:String){
  organization(login:$login){
    projectsV2(first:100, after:$endCursor){
      nodes{
        id
        title
        url
        number
        public
        repository { nameWithOwner }
      }
      pageInfo{hasNextPage endCursor}
    }
  }
}' \
    -F login="$ORG" --paginate 2>/dev/null \
    > "$OUT/$ORG.projects.json" && \
    COUNT=$(jq -r '.data.organization.projectsV2.nodes | length' "$OUT/$ORG.projects.json" 2>/dev/null || echo "0") && \
    echo -e "${GREEN}  ✓ Projects: $COUNT${NC}" || \
    echo -e "${RED}  ✗ Failed to fetch projects${NC}"
done
echo ""

# =============================================================================
# STEP 3: Gather user-level Projects (v2)
# =============================================================================
echo -e "${BLUE}[3/6] Scanning user Projects (v2)...${NC}"
echo -e "${YELLOW}  User: $GH_USER${NC}"
gh api graphql \
  --raw-field query='
query($login:String!,$endCursor:String){
  user(login:$login){
    projectsV2(first:100, after:$endCursor){
      nodes{
        id
        title
        url
        number
        public
      }
      pageInfo{hasNextPage endCursor}
    }
  }
}' \
  -F login="$GH_USER" --paginate 2>/dev/null \
  > "$OUT/user.projects.json" && \
  COUNT=$(jq -r '.data.user.projectsV2.nodes | length' "$OUT/user.projects.json" 2>/dev/null || echo "0") && \
  echo -e "${GREEN}  ✓ User projects: $COUNT${NC}" || \
  echo -e "${RED}  ✗ Failed to fetch user projects${NC}"
echo ""

# =============================================================================
# STEP 4: Scan project items (deep GraphQL) for orphans
# =============================================================================
echo -e "${BLUE}[4/6] Scanning project items for orphan detection...${NC}"
for ORG in $GH_ORGS; do
  echo -e "${YELLOW}  Org: $ORG${NC}"
  gh api graphql \
    --raw-field query='
query($login:String!,$endCursor:String){
  organization(login:$login){
    projectsV2(first:50, after:$endCursor){
      nodes{
        id
        title
        number
        url
        items(first:200){
          nodes{
            id
            content{
              __typename
              ... on Issue{
                title
                repository{
                  name
                  owner{ login }
                }
                url
              }
              ... on PullRequest{
                title
                repository{
                  name
                  owner{ login }
                }
                url
              }
              ... on DraftIssue{
                title
              }
            }
          }
        }
      }
      pageInfo{hasNextPage endCursor}
    }
  }
}' \
    -F login="$ORG" --paginate 2>/dev/null \
    > "$OUT/$ORG.project_items.json" && \
    echo -e "${GREEN}  ✓ Project items fetched${NC}" || \
    echo -e "${RED}  ✗ Failed to fetch project items${NC}"
done
echo ""

# =============================================================================
# STEP 5: Gather gists and packages
# =============================================================================
echo -e "${BLUE}[5/6] Scanning gists and packages...${NC}"
echo -e "${YELLOW}  Gists for: $GH_USER${NC}"
gh api "/users/$GH_USER/gists?per_page=100" --paginate 2>/dev/null \
  > "$OUT/user.gists.json" && \
  COUNT=$(jq -r 'length' "$OUT/user.gists.json" 2>/dev/null || echo "0") && \
  echo -e "${GREEN}  ✓ Gists: $COUNT${NC}" || \
  echo -e "${RED}  ✗ Failed to fetch gists${NC}"

for ORG in $GH_ORGS; do
  echo -e "${YELLOW}  Packages for: $ORG${NC}"
  gh api "/orgs/$ORG/packages?per_page=100" --paginate 2>/dev/null \
    > "$OUT/$ORG.packages.json" || \
    echo -e "${RED}  ✗ No packages or API error${NC}"
done
echo ""

# =============================================================================
# STEP 6: Consolidate and analyze
# =============================================================================
echo -e "${BLUE}[6/6] Consolidating inventory...${NC}"

python3 - "$OUT" << 'PYTHON'
import json
import glob
import sys
import os
from pathlib import Path
from collections import defaultdict

OUT_DIR = sys.argv[1]

# Initialize consolidated structure
consolidated = {
    "timestamp": os.path.basename(OUT_DIR),
    "organizations": list(os.environ.get("GH_ORGS", "").split()),
    "user": os.environ.get("GH_USER", ""),
    "repos_by_org": {},
    "projects_v2_by_org": {},
    "projects_without_repo": [],
    "orphan_drafts": [],
    "gists": [],
    "packages": []
}

# Load repos
for org_file in glob.glob(f"{OUT_DIR}/*.repos.json"):
    org = Path(org_file).stem.replace(".repos", "")
    try:
        repos = json.load(open(org_file, "r", encoding="utf-8"))
        consolidated["repos_by_org"][org] = {
            "count": len(repos),
            "repos": [{"name": r["name"], "full_name": r["full_name"], "url": r["html_url"], "private": r["private"], "archived": r["archived"]} for r in repos]
        }
    except Exception as e:
        print(f"Error loading {org_file}: {e}", file=sys.stderr)

# Load projects v2
project_repo_map = {}  # Track which projects have repos
for proj_file in glob.glob(f"{OUT_DIR}/*.projects.json"):
    try:
        data = json.load(open(proj_file, "r", encoding="utf-8"))
        org = Path(proj_file).stem.replace(".projects", "")
        if org.startswith("user"):
            org = "user"

        nodes = data.get("data", {}).get("organization", {}).get("projectsV2", {}).get("nodes", []) or \
                data.get("data", {}).get("user", {}).get("projectsV2", {}).get("nodes", [])

        if not nodes:
            continue

        consolidated["projects_v2_by_org"][org] = {
            "count": len(nodes),
            "projects": [{"title": p["title"], "number": p["number"], "url": p["url"], "public": p.get("public")} for p in nodes]
        }

        # Map projects to repos
        for p in nodes:
            project_key = f"{org}/{p['number']}"
            project_repo_map[project_key] = []
    except Exception as e:
        print(f"Error loading {proj_file}: {e}", file=sys.stderr)

# Load and analyze project items
for items_file in glob.glob(f"{OUT_DIR}/*.project_items.json"):
    try:
        data = json.load(open(items_file, "r", encoding="utf-8"))
        org = Path(items_file).stem.replace(".project_items", "")

        projects = data.get("data", {}).get("organization", {}).get("projectsV2", {}).get("nodes", [])

        for p in projects:
            project_key = f"{org}/{p['number']}"
            items = p.get("items", {}).get("nodes", [])

            for item in items:
                content = item.get("content", {})
                content_type = content.get("__typename")

                # Track repos referenced
                if content_type in ("Issue", "PullRequest"):
                    repo_owner = content.get("repository", {}).get("owner", {}).get("login", "")
                    repo_name = content.get("repository", {}).get("name", "")
                    if repo_owner and repo_name:
                        project_repo_map[project_key].append(f"{repo_owner}/{repo_name}")

                # Flag orphans
                elif content_type == "DraftIssue":
                    consolidated["orphan_drafts"].append({
                        "org": org,
                        "project": p["title"],
                        "project_url": p["url"],
                        "draft_title": content.get("title", ""),
                        "reason": "DraftIssue with no repo"
                    })
    except Exception as e:
        print(f"Error loading {items_file}: {e}", file=sys.stderr)

# Identify projects with no repo linkage
for proj_key, repos in project_repo_map.items():
    if not repos:
        org, proj_num = proj_key.rsplit("/", 1)
        consolidated["projects_without_repo"].append({
            "org": org,
            "project_number": proj_num,
            "reason": "No Issue/PR/repo references found in project items"
        })

# Load gists
try:
    gists = json.load(open(f"{OUT_DIR}/user.gists.json", "r", encoding="utf-8"))
    consolidated["gists"] = {
        "count": len(gists),
        "gists": [{"id": g["id"], "description": g.get("description", ""), "url": g["html_url"], "public": g["public"]} for g in gists]
    }
except Exception:
    pass

# Summary stats
summary = {
    "total_repos": sum(v["count"] for v in consolidated["repos_by_org"].values()),
    "total_projects_v2": sum(v["count"] for v in consolidated["projects_v2_by_org"].values()),
    "orphan_projects": len(consolidated["projects_without_repo"]),
    "orphan_drafts": len(consolidated["orphan_drafts"]),
    "total_gists": consolidated.get("gists", {}).get("count", 0)
}

# Write consolidated inventory
out_path = os.path.join(OUT_DIR, "consolidated.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(consolidated, f, indent=2)

# Also update root inventory.json if it exists
root_inv = "inventory.json"
if os.path.exists(root_inv):
    try:
        prior = json.load(open(root_inv, "r", encoding="utf-8"))
        prior.update(consolidated)
        with open(root_inv, "w", encoding="utf-8") as f:
            json.dump(prior, f, indent=2)
    except:
        pass
else:
    with open(root_inv, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2)

# Print summary
print(f"\n✓ Consolidated: {out_path}")
print(f"✓ Root inventory: {root_inv}")
print(f"\n{json.dumps(summary, indent=2)}")

PYTHON

echo -e "${GREEN}✓ Consolidation complete${NC}"
echo ""

# =============================================================================
# Final report
# =============================================================================
echo -e "${BLUE}=== Census Complete ===${NC}"
echo -e "${GREEN}✓ All data saved to: $OUT/${NC}"
echo -e "${GREEN}✓ Consolidated inventory: ./inventory.json${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Review inventory.json for 'projects_without_repo' and 'orphan_drafts'"
echo "  2. Run: census-promote.sh to convert orphans into repos"
echo "  3. Run: OPA policies to enforce no future orphans"
echo ""

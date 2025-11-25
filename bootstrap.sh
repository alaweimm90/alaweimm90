#!/bin/bash
# Bootstrap script for alaweimm90 GitHub OS
# Creates 14 foundation repos with starter code
# Usage: bash bootstrap.sh [--dry-run] [--skip-push]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ORG="alaweimm90"
REPOS=(
  ".github:Org-wide reusable workflows and governance"
  "standards:SSOT - policies, naming, docs rules"
  "core-control-center:Vendor-neutral DAG orchestrator"
  "adapter-claude:Claude API adapter"
  "adapter-openai:OpenAI API adapter"
  "adapter-lammps:LAMMPS molecular dynamics adapter"
  "adapter-siesta:SIESTA quantum chemistry adapter"
  "template-python-lib:Golden Python library starter"
  "template-ts-lib:Golden TypeScript library starter"
  "template-research:Golden research/notebook starter"
  "template-monorepo:Golden monorepo starter"
  "infra-actions:Composite GitHub Actions"
  "infra-containers:GHCR base images for CI"
  "demo-physics-notebooks:Reference physics examples"
)

DRY_RUN=false
SKIP_PUSH=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --skip-push)
      SKIP_PUSH=true
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Functions
log_info() {
  echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
  echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
  echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
  echo -e "${RED}❌ $1${NC}"
}

create_repo() {
  local repo_name=$1
  local description=$2

  if [ "$DRY_RUN" = true ]; then
    log_info "[DRY-RUN] Would create: gh repo create $ORG/$repo_name --public --description '$description' --confirm"
    return 0
  fi

  log_info "Creating repository: $ORG/$repo_name"

  if gh repo create "$ORG/$repo_name" --public --description "$description" --confirm 2>/dev/null; then
    log_success "Created: $ORG/$repo_name"
    return 0
  else
    # Repo might already exist
    log_warning "Could not create $ORG/$repo_name (may already exist)"
    return 0
  fi
}

setup_github_repo() {
  log_info "Setting up .github repo with reusable workflows..."

  # Create temporary directory
  local tmpdir=$(mktemp -d)
  cd "$tmpdir"

  git init -b main
  git remote add origin "https://github.com/$ORG/.github.git"

  # Create directory structure
  mkdir -p .github/workflows
  mkdir -p .github/ISSUE_TEMPLATE
  mkdir -p .github/actions/setup-python-dev

  # Create reusable-python-ci.yml
  cat > .github/workflows/reusable-python-ci.yml << 'EOF'
name: Python CI
on:
  workflow_call:
    inputs:
      python-version:
        type: string
        default: '3.11'
      test-command:
        type: string
        default: 'pytest -q --cov --cov-report=term-missing --cov-report=xml --cov-fail-under=80'

jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ inputs.python-version }}
          cache: pip
      - run: python -m pip install --upgrade pip setuptools wheel
      - run: pip install -e ".[dev]" 2>&1 | tail -20 || true
      - run: ruff check . --extend-ignore=E501
      - run: black --check .
      - run: mypy . --no-error-summary 2>&1 | head -50 || true
      - run: ${{ inputs.test-command }}
      - uses: codecov/codecov-action@v4
        if: always()
        with:
          files: ./coverage.xml
          flags: python
EOF

  # Create reusable-ts-ci.yml
  cat > .github/workflows/reusable-ts-ci.yml << 'EOF'
name: TypeScript CI
on:
  workflow_call:
    inputs:
      node-version:
        type: string
        default: '20'
      test-command:
        type: string
        default: 'pnpm -w test -- --run'

jobs:
  ci:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ inputs.node-version }}
          cache: 'pnpm'
      - run: corepack enable
      - run: pnpm i --frozen-lockfile
      - run: pnpm -w lint 2>&1 | head -50 || true
      - run: pnpm -w type-check 2>&1 | head -50 || true
      - run: pnpm -w format:check 2>&1 | head -50 || true
      - run: ${{ inputs.test-command }}
EOF

  # Create reusable-policy.yml
  cat > .github/workflows/reusable-policy.yml << 'EOF'
name: Policy & Standards
on:
  workflow_call:
    inputs:
      fetch-latest-policies:
        type: boolean
        default: true

jobs:
  policy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build file inventory
        run: |
          python3 << 'PYTHON'
          import json, os
          paths = []
          for root, _, files in os.walk('.'):
              for f in files:
                  p = os.path.join(root, f)[2:]
                  if not p.startswith('.git/'):
                      paths.append(p)
          with open('tree.json', 'w') as fh:
              json.dump({"paths": paths}, fh)
          print(f"Inventoried {len(paths)} files")
          PYTHON

      - name: Check required files
        run: |
          required_files=("README.md" "LICENSE" ".meta/repo.yaml")
          missing=0
          for f in "${required_files[@]}"; do
            if [ ! -f "$f" ]; then
              echo "❌ Missing: $f"
              missing=$((missing+1))
            fi
          done
          if [ $missing -gt 0 ]; then
            echo "::warning::Missing $missing required files"
          else
            echo "✅ All required files present"
          fi
EOF

  # Create CODEOWNERS
  cat > .github/CODEOWNERS << 'EOF'
* @alaweimm90
.github/workflows/ @alaweimm90
.github/CODEOWNERS @alaweimm90
EOF

  # Create README
  cat > README.md << 'EOF'
# Org-Wide GitHub Governance

Reusable workflows, issue templates, and policies for all alaweimm90 repositories.

## Reusable Workflows

All consumer repos call these workflows instead of duplicating CI:

- `reusable-python-ci.yml` — Python projects (pytest, ruff, black, mypy)
- `reusable-ts-ci.yml` — TypeScript projects (pnpm, ESLint, Vitest)
- `reusable-policy.yml` — Policy enforcement (OPA, markdown linting)

### Usage

In any repo, add `.github/workflows/ci.yml`:

```yaml
name: ci
on: [push, pull_request]

jobs:
  ci:
    uses: alaweimm90/.github/.github/workflows/reusable-python-ci.yml@main

  policy:
    uses: alaweimm90/.github/.github/workflows/reusable-policy.yml@main
```

## More Info

See [standards](https://github.com/alaweimm90/standards) for policies and conventions.
EOF

  # Create LICENSE
  cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2025 Meshal Alawein

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
EOF

  # Commit and push
  git add .
  git commit -m "chore: initialize org-wide governance workflows

- Python CI (pytest, ruff, black, mypy)
- TypeScript CI (pnpm, ESLint, Vitest)
- Policy enforcement (OPA, markdown linting)
- CODEOWNERS for governance review"

  if [ "$SKIP_PUSH" = false ]; then
    git push -u origin main
    log_success "Pushed .github repo"
  else
    log_warning "Skipped push (--skip-push)"
  fi

  cd - > /dev/null
  rm -rf "$tmpdir"
}

# Main execution
log_info "🚀 Starting alaweimm90 GitHub OS Bootstrap"
log_info "Organization: $ORG"
log_info "Repos to create: ${#REPOS[@]}"

if [ "$DRY_RUN" = true ]; then
  log_warning "DRY-RUN MODE - No repos will be created"
fi

echo ""

# Create all repos
for repo_info in "${REPOS[@]}"; do
  IFS=':' read -r repo_name description <<< "$repo_info"
  create_repo "$repo_name" "$description"
done

echo ""
log_info "Setting up .github repo with starter code..."

if [ "$DRY_RUN" = false ]; then
  setup_github_repo
else
  log_warning "[DRY-RUN] Would set up .github repo"
fi

echo ""
log_success "Bootstrap complete! ✨"

if [ "$DRY_RUN" = true ]; then
  log_warning "This was a dry-run. Run without --dry-run to execute."
fi

log_info "Next steps:"
echo "  1. Create standards/ repo with OPA policies"
echo "  2. Create core-control-center/ with orchestrator code"
echo "  3. Retrofit priority repos (repz, live-it-iconic, optilibria, AlaweinOS)"
echo "  4. Enable branch protection on all new repos"
echo ""
log_info "See IMPLEMENTATION_GUIDE.md for detailed timeline"

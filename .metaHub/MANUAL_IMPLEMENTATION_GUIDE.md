# Manual Implementation Guide - Meta GitHub Governance Phase 1

This guide provides step-by-step manual instructions for implementing all 5 governance tools.

---

## Prerequisites

Before starting, ensure you have:
- [ ] Git installed and configured
- [ ] Docker and Docker Compose installed
- [ ] Node.js 20+ installed
- [ ] GitHub account with repository access
- [ ] Text editor (VS Code, vim, etc.)

---

## Part 1: Backstage Developer Portal

### Step 1.1: Create Backstage Dockerfile

```bash
# Create directory
mkdir -p .metaHub/backstage

# Create Dockerfile
cat > .metaHub/backstage/Dockerfile << 'EOF'
FROM node:20-alpine AS base
WORKDIR /app
RUN apk add --no-cache python3 make g++ curl

FROM base AS builder
WORKDIR /app
COPY package.json ./
RUN npm install

FROM node:20-alpine AS production
WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules
COPY package.json ./
COPY server.js ./
COPY app-config.yaml ./
COPY catalog-info.yaml ./

RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nodejs

RUN chown -R nodejs:nodejs /app
USER nodejs

EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:3000/healthcheck || exit 1

CMD ["node", "server.js"]
EOF
```

### Step 1.2: Create Backstage Server

```bash
cat > .metaHub/backstage/server.js << 'EOF'
const express = require('express');
const fs = require('fs');
const yaml = require('js-yaml');
const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());

// Health check endpoint
app.get('/healthcheck', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

// Catalog entities endpoint (YAML format)
app.get('/api/catalog/entities', (req, res) => {
  try {
    const catalogData = yaml.load(fs.readFileSync('/app/catalog-info.yaml', 'utf8'));
    res.yaml(catalogData);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Services endpoint (JSON format)
app.get('/api/services', (req, res) => {
  try {
    const serviceData = JSON.parse(fs.readFileSync('/app/service-catalog.json', 'utf8'));
    res.json(serviceData);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    name: 'Backstage Developer Portal',
    version: '1.0.0',
    endpoints: {
      healthcheck: '/healthcheck',
      catalog: '/api/catalog/entities',
      services: '/api/services'
    }
  });
});

app.listen(PORT, () => {
  console.log(`Backstage portal listening on port ${PORT}`);
});
EOF
```

### Step 1.3: Create Package.json

```bash
cat > .metaHub/backstage/package.json << 'EOF'
{
  "name": "backstage-portal",
  "version": "1.0.0",
  "description": "Lightweight Backstage developer portal",
  "main": "server.js",
  "scripts": {
    "start": "node server.js"
  },
  "dependencies": {
    "express": "^4.18.2",
    "js-yaml": "^4.1.0"
  },
  "engines": {
    "node": ">=20.0.0"
  }
}
EOF
```

### Step 1.4: Create Catalog Configuration

```bash
cat > .metaHub/backstage/catalog-info.yaml << 'EOF'
apiVersion: backstage.io/v1alpha1
kind: System
metadata:
  name: multi-org-platform
  description: Multi-organization monorepo platform
spec:
  owner: platform-team
  domain: infrastructure

---
apiVersion: backstage.io/v1alpha1
kind: Component
metadata:
  name: simcore
  description: SimCore service - React TypeScript application
  tags:
    - react
    - typescript
    - frontend
spec:
  type: service
  lifecycle: production
  owner: alawein-os
  system: multi-org-platform

---
apiVersion: backstage.io/v1alpha1
kind: Component
metadata:
  name: repz
  description: Repz service - Node.js backend
  tags:
    - nodejs
    - backend
spec:
  type: service
  lifecycle: production
  owner: alawein-os
  system: multi-org-platform
EOF
```

Add remaining 10 services following the same pattern.

### Step 1.5: Create App Configuration

```bash
cat > .metaHub/backstage/app-config.yaml << 'EOF'
app:
  title: Multi-Org Platform
  baseUrl: http://localhost:3030

backend:
  baseUrl: http://localhost:7007
  database:
    client: better-sqlite3
    connection: ':memory:'

catalog:
  locations:
    - type: file
      target: ../../.metaHub/backstage/catalog-info.yaml
EOF
```

### Step 1.6: Update docker-compose.yml

Add this service to your `docker-compose.yml`:

```yaml
  backstage:
    build:
      context: ./.metaHub/backstage
      dockerfile: Dockerfile
    container_name: backstage-portal
    ports:
      - "3030:3000"
    restart: unless-stopped
    environment:
      - NODE_ENV=development
      - PORT=3000
    volumes:
      - ./.metaHub/backstage/app-config.yaml:/app/app-config.yaml:ro
      - ./.metaHub/backstage/catalog-info.yaml:/app/catalog-info.yaml:ro
      - ./.metaHub/service-catalog.json:/app/service-catalog.json:ro
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:3000/healthcheck || exit 1"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 30s
    networks:
      - default

networks:
  default:
    name: multi-org-network
    driver: bridge
```

### Step 1.7: Build and Start Backstage

```bash
# Build the container
docker compose build backstage

# Start the service
docker compose up -d backstage

# Verify it's running
docker compose ps backstage
curl http://localhost:3030/healthcheck
```

---

## Part 2: Renovate Dependency Automation

### Step 2.1: Create Renovate Configuration

```bash
cat > .metaHub/renovate.json << 'EOF'
{
  "$schema": "https://docs.renovatebot.com/renovate-schema.json",
  "extends": ["config:base"],
  "schedule": ["after 10pm every weekday", "every weekend"],
  "timezone": "America/New_York",
  "prConcurrentLimit": 5,
  "prHourlyLimit": 2,
  "automerge": false,
  "platformAutomerge": false,
  "rebaseWhen": "behind-base-branch",
  "semanticCommits": "enabled",
  "commitMessagePrefix": "chore(deps):",
  "packageRules": [
    {
      "matchPackagePatterns": ["^@yourorg/", "workspace:"],
      "groupName": "Internal workspace packages",
      "automerge": true,
      "minimumReleaseAge": "0 days"
    },
    {
      "matchDepTypes": ["devDependencies"],
      "automerge": true,
      "minimumReleaseAge": "3 days"
    },
    {
      "matchUpdateTypes": ["major"],
      "automerge": false,
      "labels": ["major-update", "review-required"]
    }
  ],
  "vulnerabilityAlerts": {
    "enabled": true,
    "labels": ["security", "priority-high"],
    "minimumReleaseAge": "0 days"
  }
}
EOF
```

### Step 2.2: Create Renovate Workflow

```bash
mkdir -p .github/workflows

cat > .github/workflows/renovate.yml << 'EOF'
name: Renovate Dependency Updates

on:
  schedule:
    - cron: '0 */3 * * *'
  workflow_dispatch:

jobs:
  renovate:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Run Renovate
        uses: renovatebot/github-action@v40.3.2
        with:
          configurationFile: .metaHub/renovate.json
          token: ${{ secrets.RENOVATE_TOKEN }}
        env:
          LOG_LEVEL: debug
EOF
```

---

## Part 3: OpenSSF Scorecard Security

### Step 3.1: Create Scorecard Workflow

```bash
cat > .github/workflows/scorecard.yml << 'EOF'
name: OpenSSF Scorecard Security Analysis

on:
  schedule:
    - cron: '30 1 * * 6'
  push:
    branches: [main, master]
  workflow_dispatch:

permissions:
  security-events: write
  id-token: write
  contents: read
  actions: read

jobs:
  analysis:
    name: Scorecard analysis
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        with:
          persist-credentials: false

      - name: Run analysis
        uses: ossf/scorecard-action@v2.4.0
        with:
          results_file: results.sarif
          results_format: sarif
          publish_results: true

      - name: Upload SARIF
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: results.sarif

      - name: Store results
        run: |
          mkdir -p .metaHub/security/scorecard/history
          cp results.sarif .metaHub/security/scorecard/history/scorecard-$(date +%Y%m%d).sarif
EOF
```

### Step 3.2: Create Security Policy

```bash
cat > SECURITY.md << 'EOF'
# Security Policy

## Reporting Vulnerabilities

Please report security vulnerabilities to: security@yourorg.com

## OpenSSF Scorecard

This repository is monitored by OpenSSF Scorecard, which performs 18 automated security checks weekly.

### Checks Performed

1. Binary-Artifacts
2. Branch-Protection
3. CI-Tests
4. Code-Review
5. Dependency-Update-Tool
6. Security-Policy
7. SAST
8. Token-Permissions

And 10 more...

View latest results: GitHub Security Tab → Code Scanning
EOF
```

### Step 3.3: Create Results Directory

```bash
mkdir -p .metaHub/security/scorecard/history
touch .metaHub/security/scorecard/history/.gitkeep
```

---

## Part 4: Enhanced CI/CD Pipelines

### Step 4.1: Create Matrix Build Workflow

```bash
cat > .github/workflows/ci-matrix-build.yml << 'EOF'
name: CI - Matrix Build with Intelligent Caching

on:
  push:
    branches: [main, master, develop]
  pull_request:
    branches: [main, master, develop]

jobs:
  detect-changes:
    runs-on: ubuntu-latest
    outputs:
      matrix: ${{ steps.set-matrix.outputs.matrix }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 2

      - name: Detect changed services
        id: set-matrix
        run: |
          CHANGED_FILES=$(git diff --name-only HEAD~1 HEAD)
          SERVICES='[]'

          if echo "$CHANGED_FILES" | grep -qE '^apps/ai-agent-demo/'; then
            SERVICES=$(echo "$SERVICES" | jq '. + ["ai-agent-demo"]')
          fi

          echo "matrix=$SERVICES" >> $GITHUB_OUTPUT

  build-and-scan:
    needs: detect-changes
    if: needs.detect-changes.outputs.matrix != '[]'
    runs-on: ubuntu-latest
    strategy:
      matrix:
        service: ${{ fromJson(needs.detect-changes.outputs.matrix) }}

    steps:
      - uses: actions/checkout@v4

      - uses: docker/setup-buildx-action@v3

      - name: Build Docker image
        run: docker compose build ${{ matrix.service }}

      - name: Trivy vulnerability scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: ${{ matrix.service }}:latest
          format: sarif
          output: trivy-results.sarif

      - name: Upload SARIF
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: trivy-results.sarif
EOF
```

### Step 4.2: Create Turborepo CI Workflow

```bash
cat > .github/workflows/turbo-ci.yml << 'EOF'
name: Turborepo CI with Remote Caching

on:
  push:
    branches: [main, master, develop]
  pull_request:

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: '20'

      - uses: pnpm/action-setup@v4
        with:
          version: 8

      - name: Cache Turbo
        uses: actions/cache@v4
        with:
          path: .turbo
          key: ${{ runner.os }}-turbo-${{ github.sha }}
          restore-keys: |
            ${{ runner.os }}-turbo-

      - run: pnpm install

      - run: pnpm turbo build --cache-dir=.turbo
EOF
```

---

## Part 5: OPA Policy Enforcement

### Step 5.1: Create Repository Structure Policy

```bash
mkdir -p .metaHub/policies

cat > .metaHub/policies/repo-structure.rego << 'EOF'
package repo_structure

allowed_roots := {
    ".github", ".metaHub", ".config", "apps", "packages",
    "alaweimm90", "ops", "scripts", "templates", "docs"
}

deny[msg] {
    input.file.path
    parts := split(input.file.path, "/")
    root := parts[0]
    not allowed_roots[root]
    msg := sprintf("File '%s' violates repository structure", [input.file.path])
}

forbidden_patterns := {
    ".DS_Store", "*.log", "node_modules", ".env"
}

deny[msg] {
    input.file.path
    pattern := forbidden_patterns[_]
    contains(input.file.path, pattern)
    msg := sprintf("Forbidden file pattern: %s", [input.file.path])
}
EOF
```

### Step 5.2: Create Docker Security Policy

```bash
cat > .metaHub/policies/docker-security.rego << 'EOF'
package docker_security

deny[msg] {
    input.dockerfile
    not contains(input.dockerfile, "USER ")
    msg := "Dockerfile must include USER directive to run as non-root user"
}

deny[msg] {
    input.dockerfile
    not contains(input.dockerfile, "HEALTHCHECK")
    msg := "Dockerfile must include HEALTHCHECK directive"
}

deny[msg] {
    input.dockerfile
    contains(input.dockerfile, ":latest")
    msg := "FROM directive must not use ':latest' tag"
}
EOF
```

### Step 5.3: Create Pre-commit Hook

```bash
cat > .metaHub/policies/pre-commit-opa.sh << 'EOF'
#!/bin/bash
set -e

# Check if OPA is installed
if ! command -v opa &> /dev/null; then
    echo "OPA not found. Installing..."
    curl -L -o /tmp/opa https://openpolicyagent.org/downloads/latest/opa_linux_amd64
    chmod +x /tmp/opa
    sudo mv /tmp/opa /usr/local/bin/opa
fi

# Get staged files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM)

for FILE in $STAGED_FILES; do
    echo "Checking: $FILE"

    cat > /tmp/opa-input.json <<EOF
{"file": {"path": "$FILE", "size": $(stat -c%s "$FILE" 2>/dev/null || stat -f%z "$FILE")}}
EOF

    RESULT=$(opa eval -d .metaHub/policies/repo-structure.rego \
             -i /tmp/opa-input.json "data.repo_structure.deny" -f json)

    DENIALS=$(echo "$RESULT" | jq -r '.result[0].expressions[0].value // []')

    if [ "$DENIALS" != "[]" ]; then
        echo "Policy violation in $FILE"
        echo "$DENIALS"
        exit 1
    fi
done

echo "All policy checks passed"
exit 0
EOF

chmod +x .metaHub/policies/pre-commit-opa.sh
```

### Step 5.4: Create Policies README

```bash
cat > .metaHub/policies/README.md << 'EOF'
# OPA Policy Enforcement

## Installation

```bash
# Install OPA
brew install opa jq  # macOS
# OR
curl -L -o /usr/local/bin/opa https://openpolicyagent.org/downloads/latest/opa_linux_amd64
chmod +x /usr/local/bin/opa

# Install pre-commit hook
ln -sf ../../.metaHub/policies/pre-commit-opa.sh .git/hooks/pre-commit
```

## Usage

The hook runs automatically on `git commit`.

## Testing

```bash
# Test manually
.metaHub/policies/pre-commit-opa.sh
```
EOF
```

---

## Verification Steps

### 1. Verify Backstage

```bash
docker compose ps backstage
curl http://localhost:3030/healthcheck
curl http://localhost:3030/api/services
```

Expected: Healthy status, service catalog JSON

### 2. Verify Renovate Configuration

```bash
# Check syntax
cat .metaHub/renovate.json | jq .
```

Expected: Valid JSON, no errors

### 3. Verify Workflows

```bash
# Check workflow files exist
ls -la .github/workflows/
```

Expected: renovate.yml, scorecard.yml, ci-matrix-build.yml, turbo-ci.yml

### 4. Verify OPA Policies

```bash
# Test OPA installation
opa version

# Format policies
opa fmt -w .metaHub/policies/

# Test policy
.metaHub/policies/pre-commit-opa.sh
```

Expected: OPA version displayed, policies formatted, hook runs successfully

---

## Git Commit Steps

```bash
# Add all files
git add .metaHub/
git add .github/workflows/
git add SECURITY.md
git add docker-compose.yml

# Verify what will be committed
git status

# Create commit
git commit -m "feat(governance): implement Phase 1 meta GitHub governance

- Backstage developer portal
- Renovate dependency automation
- OpenSSF Scorecard security
- Enhanced CI/CD pipelines
- OPA policy enforcement

Files: 19 created, 1 modified"

# Push to GitHub
git push origin master
```

---

## Post-Push Configuration

### 1. Add GitHub Secrets

Go to: `https://github.com/yourusername/yourrepo/settings/secrets/actions`

Add:
- **RENOVATE_TOKEN**: GitHub PAT with `repo` scope
- **TURBO_TOKEN** (optional): Vercel token for remote cache

### 2. Verify Workflows Activated

Go to: `https://github.com/yourusername/yourrepo/actions`

Check that all 4 workflows appear and can be triggered.

### 3. Install Local OPA Hook

```bash
ln -sf ../../.metaHub/policies/pre-commit-opa.sh .git/hooks/pre-commit
```

---

## Troubleshooting

### Backstage won't build
```bash
docker compose build --no-cache backstage
docker compose logs backstage
```

### Renovate workflow fails
- Verify `RENOVATE_TOKEN` is set
- Check token has `repo` scope
- Review workflow run logs

### OPA hook fails
```bash
# Check OPA installed
which opa

# Check jq installed
which jq

# Test manually
.metaHub/policies/pre-commit-opa.sh
```

---

## Complete File Checklist

- [ ] `.metaHub/backstage/Dockerfile`
- [ ] `.metaHub/backstage/server.js`
- [ ] `.metaHub/backstage/package.json`
- [ ] `.metaHub/backstage/catalog-info.yaml`
- [ ] `.metaHub/backstage/app-config.yaml`
- [ ] `.metaHub/renovate.json`
- [ ] `.github/workflows/renovate.yml`
- [ ] `.github/workflows/scorecard.yml`
- [ ] `.github/workflows/ci-matrix-build.yml`
- [ ] `.github/workflows/turbo-ci.yml`
- [ ] `.metaHub/policies/repo-structure.rego`
- [ ] `.metaHub/policies/docker-security.rego`
- [ ] `.metaHub/policies/pre-commit-opa.sh`
- [ ] `.metaHub/policies/README.md`
- [ ] `SECURITY.md`
- [ ] `docker-compose.yml` (modified)

---

That's it! Follow these steps manually to implement all 5 governance tools.

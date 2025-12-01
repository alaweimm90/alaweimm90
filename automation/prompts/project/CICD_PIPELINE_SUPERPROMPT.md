---
name: "CI/CD Pipeline Superprompt"
version: "1.0"
category: "project"
tags: ["cicd", "devops", "automation", "deployment", "github-actions"]
created: "2024-11-30"
---

# CI/CD Pipeline Superprompt

## Purpose

Comprehensive CI/CD pipeline design and implementation framework for automated building, testing, and deployment across multiple environments.

---

## System Prompt

```
You are a DevOps Engineer and CI/CD Architect with expertise in:
- GitHub Actions, GitLab CI, Azure DevOps, Jenkins
- Container orchestration (Docker, Kubernetes)
- Infrastructure as Code (Terraform, Pulumi)
- Security scanning and compliance automation
- Multi-environment deployment strategies
- Release management and versioning

Your mission is to design CI/CD pipelines that:
1. Provide fast, reliable feedback loops
2. Enforce quality gates and security standards
3. Enable safe, automated deployments
4. Support rollback and recovery procedures
5. Maintain audit trails and compliance
```

---

## Pipeline Architecture

### Standard Pipeline Stages

```yaml
pipeline_stages:
  1_validate:
    - lint_code
    - type_check
    - format_check
    - dependency_audit
    
  2_build:
    - compile_source
    - build_artifacts
    - generate_docs
    - create_containers
    
  3_test:
    - unit_tests
    - integration_tests
    - security_scan
    - performance_tests
    
  4_quality_gates:
    - coverage_check
    - sonar_analysis
    - license_compliance
    - vulnerability_scan
    
  5_deploy:
    - deploy_staging
    - smoke_tests
    - deploy_production
    - health_checks
    
  6_release:
    - tag_release
    - generate_changelog
    - notify_stakeholders
    - update_documentation
```

---

## GitHub Actions Workflows

### Main CI Workflow

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

env:
  NODE_VERSION: '20'
  PYTHON_VERSION: '3.12'

jobs:
  # ============================================
  # Stage 1: Validation
  # ============================================
  validate:
    name: Validate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
          
      - name: Install dependencies
        run: npm ci
        
      - name: Lint
        run: npm run lint
        
      - name: Type check
        run: npm run type-check
        
      - name: Format check
        run: npm run format:check

  # ============================================
  # Stage 2: Security Scan
  # ============================================
  security:
    name: Security Scan
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          severity: 'CRITICAL,HIGH'
          exit-code: '1'
          
      - name: Dependency audit
        run: npm audit --audit-level=high
        
      - name: SAST scan
        uses: github/codeql-action/analyze@v2

  # ============================================
  # Stage 3: Build
  # ============================================
  build:
    name: Build
    needs: [validate]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
          
      - name: Install dependencies
        run: npm ci
        
      - name: Build
        run: npm run build
        
      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: build
          path: dist/
          retention-days: 7

  # ============================================
  # Stage 4: Test
  # ============================================
  test:
    name: Test
    needs: [build]
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: test
        ports:
          - 5432:5432
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
          
      redis:
        image: redis:7
        ports:
          - 6379:6379
          
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
          
      - name: Install dependencies
        run: npm ci
        
      - name: Run unit tests
        run: npm run test:unit -- --coverage
        
      - name: Run integration tests
        run: npm run test:integration
        env:
          DATABASE_URL: postgres://postgres:test@localhost:5432/test
          REDIS_URL: redis://localhost:6379
          
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage/lcov.info
          fail_ci_if_error: true

  # ============================================
  # Stage 5: E2E Tests
  # ============================================
  e2e:
    name: E2E Tests
    needs: [build]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
          
      - name: Install dependencies
        run: npm ci
        
      - name: Install Playwright
        run: npx playwright install --with-deps
        
      - name: Download build
        uses: actions/download-artifact@v4
        with:
          name: build
          path: dist/
          
      - name: Run E2E tests
        run: npm run test:e2e
        
      - name: Upload test results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: playwright-report
          path: playwright-report/

  # ============================================
  # Stage 6: Quality Gate
  # ============================================
  quality-gate:
    name: Quality Gate
    needs: [test, security]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          
      - name: SonarCloud Scan
        uses: SonarSource/sonarcloud-github-action@master
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
          
      - name: Check quality gate
        run: |
          # Custom quality gate checks
          echo "Checking coverage threshold..."
          echo "Checking code smells..."
          echo "Checking security hotspots..."
```

### Deployment Workflow

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      environment:
        description: 'Deployment environment'
        required: true
        default: 'staging'
        type: choice
        options:
          - staging
          - production

jobs:
  # ============================================
  # Build Container
  # ============================================
  build-container:
    name: Build Container
    runs-on: ubuntu-latest
    outputs:
      image_tag: ${{ steps.meta.outputs.tags }}
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
        
      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
          
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=sha,prefix=
            type=ref,event=branch
            type=semver,pattern={{version}}
            
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  # ============================================
  # Deploy to Staging
  # ============================================
  deploy-staging:
    name: Deploy to Staging
    needs: [build-container]
    runs-on: ubuntu-latest
    environment:
      name: staging
      url: https://staging.example.com
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          kubeconfig: ${{ secrets.KUBE_CONFIG_STAGING }}
          
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/app \
            app=${{ needs.build-container.outputs.image_tag }} \
            -n staging
          kubectl rollout status deployment/app -n staging
          
      - name: Run smoke tests
        run: |
          curl -f https://staging.example.com/health || exit 1
          npm run test:smoke -- --env=staging

  # ============================================
  # Deploy to Production
  # ============================================
  deploy-production:
    name: Deploy to Production
    needs: [deploy-staging]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment:
      name: production
      url: https://example.com
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure kubectl
        uses: azure/k8s-set-context@v3
        with:
          kubeconfig: ${{ secrets.KUBE_CONFIG_PRODUCTION }}
          
      - name: Blue-Green Deployment
        run: |
          # Deploy to green environment
          kubectl apply -f k8s/production-green.yaml
          kubectl rollout status deployment/app-green -n production
          
          # Run health checks
          ./scripts/health-check.sh green
          
          # Switch traffic
          kubectl patch service app -p '{"spec":{"selector":{"version":"green"}}}'
          
          # Scale down blue
          kubectl scale deployment/app-blue --replicas=0 -n production
          
      - name: Notify deployment
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {
              "text": "🚀 Deployed to production: ${{ github.sha }}"
            }
        env:
          SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK }}
```

### Release Workflow

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

permissions:
  contents: write
  packages: write

jobs:
  release:
    name: Create Release
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          
      - name: Generate changelog
        id: changelog
        uses: orhun/git-cliff-action@v2
        with:
          config: cliff.toml
          args: --latest --strip header
          
      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          body: ${{ steps.changelog.outputs.content }}
          draft: false
          prerelease: ${{ contains(github.ref, 'alpha') || contains(github.ref, 'beta') }}
          files: |
            dist/*.tar.gz
            dist/*.zip
            
      - name: Publish to npm
        run: |
          echo "//registry.npmjs.org/:_authToken=${{ secrets.NPM_TOKEN }}" > ~/.npmrc
          npm publish --access public
```

---

## Deployment Strategies

```yaml
deployment_strategies:
  rolling_update:
    description: "Gradual replacement of instances"
    use_case: "Standard deployments with zero downtime"
    config:
      maxSurge: 25%
      maxUnavailable: 25%
      
  blue_green:
    description: "Two identical environments, instant switch"
    use_case: "Critical applications requiring instant rollback"
    steps:
      - Deploy to inactive environment
      - Run validation tests
      - Switch traffic
      - Keep old environment for rollback
      
  canary:
    description: "Gradual traffic shift to new version"
    use_case: "High-risk changes requiring validation"
    stages:
      - 5% traffic for 10 minutes
      - 25% traffic for 30 minutes
      - 50% traffic for 1 hour
      - 100% traffic
      
  feature_flags:
    description: "Deploy code, enable features separately"
    use_case: "Decoupling deployment from release"
    tools: ["LaunchDarkly", "Unleash", "Flagsmith"]
```

---

## Environment Configuration

```yaml
environments:
  development:
    auto_deploy: true
    approval_required: false
    retention: 7 days
    
  staging:
    auto_deploy: true
    approval_required: false
    retention: 30 days
    smoke_tests: required
    
  production:
    auto_deploy: false
    approval_required: true
    approvers: ["@team-leads"]
    retention: 90 days
    smoke_tests: required
    rollback_enabled: true
```

---

## Execution Phases

### Phase 1: Foundation
- [ ] Set up repository structure
- [ ] Configure CI workflow for validation
- [ ] Add build and test stages
- [ ] Configure artifact storage

### Phase 2: Quality Gates
- [ ] Integrate code coverage
- [ ] Add security scanning
- [ ] Configure SonarCloud
- [ ] Set up dependency auditing

### Phase 3: Deployment
- [ ] Configure staging deployment
- [ ] Implement blue-green for production
- [ ] Add smoke tests
- [ ] Set up monitoring integration

### Phase 4: Release Management
- [ ] Automate versioning
- [ ] Generate changelogs
- [ ] Configure release workflows
- [ ] Set up notifications

---

*Last updated: 2024-11-30*

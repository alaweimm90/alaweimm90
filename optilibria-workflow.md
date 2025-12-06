# Fullstack Development Workflow

**Project:** {{project_name}}
**Stack:** {{tech_stack}}

---

## Phase 1: Architecture

---
name: 'Monorepo & Architecture Superprompt'
version: '1.0'
category: 'project'
tags: ['monorepo', 'architecture', 'modular', 'dependencies', 'organization']
created: '2024-11-30'
---

# Monorepo & Architecture Superprompt

## Purpose

Comprehensive framework for organizing, structuring, and managing monorepo architectures with modular design, dependency handling, and scalable project organization.

---

## System Prompt

```
You are a Software Architect and Monorepo Expert with expertise in:
- Monorepo tooling (Nx, Turborepo, Lerna, Bazel)
- Modular architecture and domain-driven design
- Dependency management and versioning
- Build optimization and caching
- Code sharing and package boundaries
- CI/CD optimization for monorepos

Your mission is to design architectures that:
1. Enable code sharing while maintaining boundaries
2. Optimize build and test performance
3. Support independent deployability
4. Maintain clear ownership and responsibility
5. Scale with team and codebase growth
```

---

## Monorepo Structure

### Standard Layout

```
monorepo/
├── .github/
│   ├── workflows/           # CI/CD workflows
│   ├── CODEOWNERS           # Ownership rules
│   └── PULL_REQUEST_TEMPLATE.md
├── apps/                    # Deployable applications
│   ├── web/                 # Web application
│   ├── api/                 # API service
│   ├── admin/               # Admin dashboard
│   └── mobile/              # Mobile app
├── packages/                # Shared packages
│   ├── ui/                  # UI component library
│   ├── utils/               # Utility functions
│   ├── config/              # Shared configurations
│   ├── types/               # TypeScript types
│   └── api-client/          # API client SDK
├── libs/                    # Domain libraries
│   ├── auth/                # Authentication logic
│   ├── users/               # User management
│   ├── billing/             # Billing domain
│   └── analytics/           # Analytics domain
├── tools/                   # Build and dev tools
│   ├── scripts/             # Automation scripts
│   ├── generators/          # Code generators
│   └── plugins/             # Custom plugins
├── docs/                    # Documentation
├── infrastructure/          # IaC and deployment
│   ├── terraform/
│   ├── k8s/
│   └── docker/
├── package.json             # Root package.json
├── nx.json                  # Nx configuration
├── turbo.json               # Turborepo config
├── tsconfig.base.json       # Base TS config
└── README.md
```

---

## Nx Configuration

### nx.json

```json
{
  "$schema": "./node_modules/nx/schemas/nx-schema.json",
  "namedInputs": {
    "default": ["{projectRoot}/**/*", "sharedGlobals"],
    "production": [
      "default",
      "!{projectRoot}/**/?(*.)+(spec|test).[jt]s?(x)?(.snap)",
      "!{projectRoot}/tsconfig.spec.json",
      "!{projectRoot}/.eslintrc.json"
    ],
    "sharedGlobals": ["{workspaceRoot}/tsconfig.base.json"]
  },
  "targetDefaults": {
    "build": {
      "dependsOn": ["^build"],
      "inputs": ["production", "^production"],
      "cache": true
    },
    "test": {
      "inputs": ["default", "^production", "{workspaceRoot}/jest.preset.js"],
      "cache": true
    },
    "lint": {
      "inputs": ["default", "{workspaceRoot}/.eslintrc.json"],
      "cache": true
    },
    "e2e": {
      "inputs": ["default", "^production"],
      "cache": true
    }
  },
  "parallel": 3,
  "cacheDirectory": ".nx/cache",
  "defaultBase": "main",
  "plugins": [
    {
      "plugin": "@nx/eslint/plugin",
      "options": {
        "targetName": "lint"
      }
    },
    {
      "plugin": "@nx/vite/plugin",
      "options": {
        "buildTargetName": "build",
        "testTargetName": "test"
      }
    }
  ],
  "generators": {
    "@nx/react": {
      "application": {
        "style": "tailwind",
        "linter": "eslint",
        "bundler": "vite"
      },
      "component": {
        "style": "tailwind"
      },
      "library": {
        "style": "tailwind",
        "linter": "eslint",
        "unitTestRunner": "vitest"
      }
    }
  }
}
```

### project.json (App)

```json
{
  "name": "web",
  "$schema": "../../node_modules/nx/schemas/project-schema.json",
  "sourceRoot": "apps/web/src",
  "projectType": "application",
  "tags": ["scope:web", "type:app"],
  "targets": {
    "build": {
      "executor": "@nx/vite:build",
      "outputs": ["{options.outputPath}"],
      "options": {
        "outputPath": "dist/apps/web"
      }
    },
    "serve": {
      "executor": "@nx/vite:dev-server",
      "options": {
        "buildTarget": "web:build"
      }
    },
    "test": {
      "executor": "@nx/vite:test",
      "options": {
        "passWithNoTests": true
      }
    },
    "lint": {
      "executor": "@nx/eslint:lint",
      "options": {
        "lintFilePatterns": ["apps/web/**/*.{ts,tsx,js,jsx}"]
      }
    },
    "deploy": {
      "executor": "nx:run-commands",
      "options": {
        "command": "npm run deploy:web"
      },
      "dependsOn": ["build"]
    }
  },
  "implicitDependencies": ["ui", "utils", "api-client"]
}
```

---

## Turborepo Configuration

### turbo.json

```json
{
  "$schema": "https://turbo.build/schema.json",
  "globalDependencies": ["**/.env.*local"],
  "globalEnv": ["NODE_ENV", "CI"],
  "pipeline": {
    "build": {
      "dependsOn": ["^build"],
      "outputs": ["dist/**", ".next/**", "!.next/cache/**"],
      "env": ["NODE_ENV"]
    },
    "test": {
      "dependsOn": ["^build"],
      "outputs": ["coverage/**"],
      "inputs": ["src/**/*.tsx", "src/**/*.ts", "test/**/*.ts"]
    },
    "lint": {
      "outputs": [],
      "cache": true
    },
    "dev": {
      "cache": false,
      "persistent": true
    },
    "deploy": {
      "dependsOn": ["build", "test", "lint"],
      "outputs": []
    },
    "typecheck": {
      "dependsOn": ["^build"],
      "outputs": []
    }
  },
  "remoteCache": {
    "signature": true
  }
}
```

---

## Package Boundaries

### Dependency Rules

```yaml
# .dependency-cruiser.js
dependency_rules:
  apps_to_packages:
    allowed: true
    description: 'Apps can import from packages'

  apps_to_libs:
    allowed: true
    description: 'Apps can import from libs'

  apps_to_apps:
    allowed: false
    description: 'Apps cannot import from other apps'

  packages_to_packages:
    allowed: true
    description: 'Packages can import from other packages'

  packages_to_libs:
    allowed: true
    description: 'Packages can import from libs'

  packages_to_apps:
    allowed: false
    description: 'Packages cannot import from apps'

  libs_to_libs:
    allowed: true
    condition: 'Same domain or shared'

  libs_to_packages:
    allowed: true
    description: 'Libs can use shared packages'

  libs_to_apps:
    allowed: false
    description: 'Libs cannot import from apps'
```

### ESLint Boundaries

```javascript
// .eslintrc.js
module.exports = {
  plugins: ['@nx/enforce-module-boundaries'],
  rules: {
    '@nx/enforce-module-boundaries': [
      'error',
      {
        enforceBuildableLibDependency: true,
        allow: [],
        depConstraints: [
          // Apps can depend on anything except other apps
          {
            sourceTag: 'type:app',
            onlyDependOnLibsWithTags: ['type:lib', 'type:util', 'type:ui'],
          },
          // UI components can only depend on utils
          {
            sourceTag: 'type:ui',
            onlyDependOnLibsWithTags: ['type:util'],
          },
          // Domain libs can depend on utils and other domain libs
          {
            sourceTag: 'type:lib',
            onlyDependOnLibsWithTags: ['type:lib', 'type:util'],
          },
          // Utils cannot depend on anything except other utils
          {
            sourceTag: 'type:util',
            onlyDependOnLibsWithTags: ['type:util'],
          },
          // Scope-based constraints
          {
            sourceTag: 'scope:web',
            onlyDependOnLibsWithTags: ['scope:web', 'scope:shared'],
          },
          {
            sourceTag: 'scope:api',
            onlyDependOnLibsWithTags: ['scope:api', 'scope:shared'],
          },
        ],
      },
    ],
  },
};
```

---

## Workspace Configuration

### Root package.json

```json
{
  "name": "monorepo",
  "private": true,
  "workspaces": ["apps/*", "packages/*", "libs/*"],
  "scripts": {
    "build": "turbo run build",
    "build:affected": "nx affected --target=build",
    "test": "turbo run test",
    "test:affected": "nx affected --target=test",
    "lint": "turbo run lint",
    "lint:affected": "nx affected --target=lint",
    "dev": "turbo run dev --parallel",
    "clean": "turbo run clean && rm -rf node_modules",
    "format": "prettier --write \"**/*.{ts,tsx,md,json}\"",
    "prepare": "husky install",
    "graph": "nx graph",
    "affected:graph": "nx affected:graph"
  },
  "devDependencies": {
    "@nx/devkit": "^17.0.0",
    "@nx/eslint": "^17.0.0",
    "@nx/js": "^17.0.0",
    "@nx/react": "^17.0.0",
    "@nx/vite": "^17.0.0",
    "@nx/workspace": "^17.0.0",
    "nx": "^17.0.0",
    "turbo": "^1.10.0",
    "typescript": "^5.0.0",
    "prettier": "^3.0.0",
    "eslint": "^8.0.0",
    "husky": "^8.0.0"
  },
  "engines": {
    "node": ">=18.0.0",
    "npm": ">=9.0.0"
  },
  "packageManager": "npm@10.0.0"
}
```

### TypeScript Configuration

```json
// tsconfig.base.json
{
  "compileOnSave": false,
  "compilerOptions": {
    "rootDir": ".",
    "sourceMap": true,
    "declaration": false,
    "moduleResolution": "bundler",
    "emitDecoratorMetadata": true,
    "experimentalDecorators": true,
    "importHelpers": true,
    "target": "ES2022",
    "module": "ESNext",
    "lib": ["ES2022", "DOM"],
    "skipLibCheck": true,
    "skipDefaultLibCheck": true,
    "strict": true,
    "noImplicitAny": true,
    "noImplicitReturns": true,
    "noFallthroughCasesInSwitch": true,
    "forceConsistentCasingInFileNames": true,
    "baseUrl": ".",
    "paths": {
      "@repo/ui": ["packages/ui/src/index.ts"],
      "@repo/ui/*": ["packages/ui/src/*"],
      "@repo/utils": ["packages/utils/src/index.ts"],
      "@repo/utils/*": ["packages/utils/src/*"],
      "@repo/config": ["packages/config/src/index.ts"],
      "@repo/types": ["packages/types/src/index.ts"],
      "@repo/api-client": ["packages/api-client/src/index.ts"],
      "@lib/auth": ["libs/auth/src/index.ts"],
      "@lib/users": ["libs/users/src/index.ts"],
      "@lib/billing": ["libs/billing/src/index.ts"]
    }
  },
  "exclude": ["node_modules", "tmp"]
}
```

---

## CI/CD for Monorepos

### Affected-Based CI

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  setup:
    runs-on: ubuntu-latest
    outputs:
      affected: ${{ steps.affected.outputs.affected }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Derive appropriate SHAs
        uses: nrwl/nx-set-shas@v4

      - name: Get affected projects
        id: affected
        run: |
          AFFECTED=$(npx nx show projects --affected --json)
          echo "affected=$AFFECTED" >> $GITHUB_OUTPUT

  lint:
    needs: setup
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: nrwl/nx-set-shas@v4
      - run: npm ci
      - run: npx nx affected --target=lint --parallel=3

  test:
    needs: setup
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: nrwl/nx-set-shas@v4
      - run: npm ci
      - run: npx nx affected --target=test --parallel=3 --coverage

  build:
    needs: [lint, test]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: nrwl/nx-set-shas@v4
      - run: npm ci
      - run: npx nx affected --target=build --parallel=3

  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    strategy:
      matrix:
        app: [web, api, admin]
    steps:
      - uses: actions/checkout@v4
      - run: npm ci
      - name: Check if affected
        id: check
        run: |
          if npx nx show projects --affected | grep -q "${{ matrix.app }}"; then
            echo "deploy=true" >> $GITHUB_OUTPUT
          fi
      - name: Deploy
        if: steps.check.outputs.deploy == 'true'
        run: npx nx deploy ${{ matrix.app }}
```

---

## Code Generation

### Custom Generator

```typescript
// tools/generators/lib/index.ts
import { Tree, formatFiles, generateFiles, joinPathFragments, names } from '@nx/devkit';

interface LibGeneratorSchema {
  name: string;
  directory?: string;
  tags?: string;
}

export default async function (tree: Tree, schema: LibGeneratorSchema) {
  const { name, directory, tags } = schema;
  const projectName = names(name).fileName;
  const projectRoot = directory
    ? joinPathFragments('libs', directory, projectName)
    : joinPathFragments('libs', projectName);

  generateFiles(tree, joinPathFragments(__dirname, './files'), projectRoot, {
    ...names(name),
    tmpl: '',
  });

  // Update project.json
  const projectJson = {
    name: projectName,
    $schema: '../../node_modules/nx/schemas/project-schema.json',
    sourceRoot: `${projectRoot}/src`,
    projectType: 'library',
    tags: tags?.split(',') || [],
    targets: {
      build: {
        executor: '@nx/js:tsc',
        outputs: ['{options.outputPath}'],
        options: {
          outputPath: `dist/${projectRoot}`,
          main: `${projectRoot}/src/index.ts`,
          tsConfig: `${projectRoot}/tsconfig.lib.json`,
        },
      },
      test: {
        executor: '@nx/vite:test',
      },
      lint: {
        executor: '@nx/eslint:lint',
      },
    },
  };

  tree.write(joinPathFragments(projectRoot, 'project.json'), JSON.stringify(projectJson, null, 2));

  await formatFiles(tree);
}
```

---

## Execution Phases

### Phase 1: Foundation

- [ ] Initialize monorepo structure
- [ ] Configure Nx/Turborepo
- [ ] Set up TypeScript paths
- [ ] Create base configurations

### Phase 2: Package Structure

- [ ] Define package boundaries
- [ ] Create shared packages
- [ ] Set up domain libraries
- [ ] Configure dependency rules

### Phase 3: Build Optimization

- [ ] Configure caching
- [ ] Set up remote cache
- [ ] Optimize parallel execution
- [ ] Add affected commands

### Phase 4: CI/CD Integration

- [ ] Implement affected-based CI
- [ ] Configure deployment pipelines
- [ ] Set up code generation
- [ ] Add dependency graph visualization

---

_Last updated: 2024-11-30_


---

## Phase 2: Backend Development

## Code Review - {{backend_lang

**Focus Areas:**
- Code quality and maintainability
- Performance optimization
- Security vulnerabilities
- Best practices adherence
}}

**Requirements:**
- RESTful API design
- Database schema
- Authentication

## Testing Requirements

- Minimum coverage: 85%
- Unit tests for all functions
- Integration tests for APIs
- Edge case handling


---

## Phase 3: Frontend Development

## Code Review - {{frontend_lang

**Focus Areas:**
- Code quality and maintainability
- Performance optimization
- Security vulnerabilities
- Best practices adherence
}}

**Requirements:**
- Component architecture
- State management
- Responsive design

---

## Phase 4: Testing & Deployment

[Component 'security-checklist}}

---
name: 'CI/CD Pipeline Superprompt'
version: '1.0'
category: 'project'
tags: ['cicd', 'devops', 'automation', 'deployment', 'github-actions']
created: '2024-11-30'
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
    description: 'Gradual replacement of instances'
    use_case: 'Standard deployments with zero downtime'
    config:
      maxSurge: 25%
      maxUnavailable: 25%

  blue_green:
    description: 'Two identical environments, instant switch'
    use_case: 'Critical applications requiring instant rollback'
    steps:
      - Deploy to inactive environment
      - Run validation tests
      - Switch traffic
      - Keep old environment for rollback

  canary:
    description: 'Gradual traffic shift to new version'
    use_case: 'High-risk changes requiring validation'
    stages:
      - 5% traffic for 10 minutes
      - 25% traffic for 30 minutes
      - 50% traffic for 1 hour
      - 100% traffic

  feature_flags:
    description: 'Deploy code, enable features separately'
    use_case: 'Decoupling deployment from release'
    tools: ['LaunchDarkly', 'Unleash', 'Flagsmith']
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
    approvers: ['@team-leads']
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

_Last updated: 2024-11-30_


---

## Success Criteria

- All tests passing
- {{coverage_target' not found]% code coverage
- Performance benchmarks met
- Security audit passed


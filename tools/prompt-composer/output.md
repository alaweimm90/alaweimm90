# Fullstack Development Workflow

**Project:** EcommerceApp
**Stack:** Next.js + PostgreSQL

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

## Code Review - TypeScript

**Focus Areas:**
- Code quality and maintainability
- Performance optimization
- Security vulnerabilities
- Best practices adherence


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

## Code Review - TypeScript

**Focus Areas:**
- Code quality and maintainability
- Performance optimization
- Security vulnerabilities
- Best practices adherence


**Requirements:**
- Component architecture
- State management
- Responsive design

---

## Phase 4: Testing & Deployment

[Component 'security-checklist}}

{{include:superprompts/cicd-pipeline-setup.md' not found]

---

## Success Criteria

- All tests passing
- 90% code coverage
- Performance benchmarks met
- Security audit passed


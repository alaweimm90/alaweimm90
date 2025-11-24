# Organization-Specific Concerns: Multi-Org Monorepo Guide

**Date**: November 24, 2025
**Focus**: Managing multiple organizations (alaweimm90, alaweimm90-business, etc.) in single monorepo
**Scope**: Configuration, code sharing, dependency management, and isolation strategies

---

## 🏢 ORGANIZATION STRUCTURE IN YOUR MONOREPO

### Current State

Your monorepo contains multiple organizations:

```
/.config/organizations/                    (Archive/Reference)
├── alaweimm90/                           (ACTIVE - Primary)
├── alaweimm90-business/                  (In separate dir)
├── alaweimm90-business-duplicate/        (DUPLICATE - DELETE)
├── alaweimm90-science/                   (Archive)
├── alaweimm90-tools/                     (Archive)
├── AlaweinOS/                            (Archive)
├── MeatheadPhysicist/                    (Archive)
└── [others - 8 more organizations]

ACTIVE WORKSPACE (root monorepo)
├── /alaweimm90/                          (Primary workspace - 708 MB)
├── /packages/                            (Core shared packages - 5)
├── /src/                                 (Root services)
└── /automation/                          (Blockchain specific)
```

**Key Question**: Are these separate organizations that should:

- Share infrastructure? → YES (use core packages)
- Share code? → SOMETIMES (via shared utilities)
- Share configurations? → Mostly no (org-specific)
- Share dependencies? → YES (via pnpm workspace)

---

## 🔧 ORGANIZATION-SPECIFIC CONFIGURATIONS

### Pattern 1: Environment Variables by Organization

**Current Issue**: No org-specific environment handling
**Recommended Structure**:

```
.env.example              (Template)
.env.{org-name}         (Per-organization, IGNORED by git)
.env.{org-name}.enc     (Encrypted, WITH git)
```

**Implementation**:

```bash
# Root .env.example
ORGANIZATION=alaweimm90
LOG_LEVEL=info
DATABASE_URL=postgresql://localhost/alaweimm90_dev
API_GATEWAY_HOST=localhost:3000
FEATURE_FLAGS={"newUI":false}

# For alaweimm90-science
# .env.alaweimm90-science
ORGANIZATION=alaweimm90-science
LOG_LEVEL=debug
DATABASE_URL=postgresql://localhost/science_dev
API_GATEWAY_HOST=science.local:3000
FEATURE_FLAGS={"newUI":true,"experiments":true}
```

**Loading Priority** (in order):

```typescript
// config/load-config.ts
const loadOrgConfig = (orgName: string) => {
  // 1. Try org-specific .env.{org-name}.enc
  let config = loadEncrypted(`.env.${orgName}.enc`);

  // 2. Fall back to local .env.{org-name}
  if (!config) {
    config = loadFromFile(`.env.${orgName}`);
  }

  // 3. Fall back to base .env
  if (!config) {
    config = loadFromFile('.env');
  }

  // 4. Use defaults from .env.example
  config = mergeDefaults(config, loadFromFile('.env.example'));

  return config;
};
```

**Usage in Applications**:

```typescript
// alaweimm90/src/index.ts
const ORG_NAME = 'alaweimm90';
const config = loadOrgConfig(ORG_NAME);

// science-org/src/index.ts
const ORG_NAME = 'alaweimm90-science';
const config = loadOrgConfig(ORG_NAME);
```

---

### Pattern 2: Organization-Specific Plugins/Modules

**Current Issue**: alaweimm90 has hardcoded automation modules, not extendable

**Recommended Structure**:

```
/packages/
  /agent-core/
    - Base agent classes (core)

  /agent-plugins/           (NEW - Optional plugins)
    /finance-plugin/
      ├── FinanceAgent
      ├── FinanceOrchestrator
      └── types.ts

    /healthcare-plugin/
      ├── HealthcareAgent
      └── types.ts

    /manufacturing-plugin/
      ├── ManufacturingAgent
      └── types.ts

/alaweimm90/
  /automation/
    ├── api-gateway/        (Core - always included)
    ├── autonomous/         (Core - always included)
    └── plugins/            (Org-specific - loaded conditionally)
        ├── finance/
        ├── healthcare/
        └── manufacturing/

/alaweimm90-science/       (Different org, different plugins)
  /automation/
    ├── api-gateway/        (Core - same as alaweimm90)
    ├── autonomous/         (Core - same as alaweimm90)
    └── plugins/            (Org-specific - DIFFERENT from alaweimm90)
        ├── research/
        ├── data-analysis/
        └── ml-pipeline/
```

**Plugin Loading Pattern**:

```typescript
// packages/agent-core/src/plugin-loader.ts
import { PluginRegistry } from '@monorepo/agent-core';

const registry = new PluginRegistry();

// Load org-specific plugins
const orgPlugins = config.org.plugins || [];
for (const pluginName of orgPlugins) {
  const plugin = await import(`./plugins/${pluginName}`);
  registry.register(plugin);
}

// Agents use plugins via registry
const financeAgent = registry.get('finance');
```

**Configuration Per Organization**:

```json
{
  "alaweimm90": {
    "organization": "alaweimm90",
    "plugins": ["finance", "healthcare", "manufacturing", "retail", "security-advanced"],
    "environment": "production"
  },
  "alaweimm90-science": {
    "organization": "alaweimm90-science",
    "plugins": ["research", "data-analysis", "ml-pipeline"],
    "environment": "research"
  }
}
```

---

### Pattern 3: Organization-Specific TypeScript Configs

**Current Issue**: Single monolithic tsconfig.json at root

**Recommended Structure**:

```
tsconfig.json                  (Base config)
tsconfig.core.json             (For packages/)
tsconfig.alaweimm90.json       (For alaweimm90/)
tsconfig.alaweimm90-science.json (For alaweimm90-science/)
```

**Base tsconfig.json**:

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "moduleResolution": "node",
    "baseUrl": ".",
    "paths": {
      "@monorepo/*": ["packages/*/src"],
      "@alaweimm90/*": ["alaweimm90/*/src"],
      "@shared/*": ["packages/shared-utils/src"]
    }
  }
}
```

**Organization-Specific (tsconfig.alaweimm90.json)**:

```json
{
  "extends": "./tsconfig.json",
  "compilerOptions": {
    "moduleResolution": "bundler",
    "paths": {
      "@alaweimm90/*": ["alaweimm90/*/src"],
      "@plugins/*": ["alaweimm90/plugins/*/src"]
    }
  },
  "include": ["alaweimm90/**/*", "packages/**/*"],
  "exclude": ["alaweimm90-science/**/*", "node_modules"]
}
```

**Usage in Build Pipeline**:

```json
{
  "name": "root",
  "scripts": {
    "build": "pnpm -r build",
    "build:alaweimm90": "tsc -p tsconfig.alaweimm90.json",
    "build:science": "tsc -p tsconfig.alaweimm90-science.json"
  }
}
```

---

## 📦 CODE SHARING STRATEGIES

### Strategy 1: Shared Code - Packages

**What to Share**: Code used by 2+ organizations

**Location**: `packages/shared-{domain}/`

```
/packages/
  /shared-utils/           (Logging, validation, errors)
  /shared-automation/      (Express middleware, auth)
  /shared-database/        (Connection pooling, migrations)
  /shared-api/             (API client, error handling)
```

**Example - Shared Database Package**:

```typescript
// packages/shared-database/src/index.ts
export interface DatabaseConfig {
  host: string;
  port: number;
  username: string;
  database: string;
  organization: string;
}

export class OrganizationDataSource {
  constructor(private config: DatabaseConfig) {}

  getConnection() {
    // Multi-tenancy support
    return {
      url: `postgres://${this.config.username}@${this.config.host}:${this.config.port}/${this.config.database}`,
      schema: `org_${this.config.organization}`,
    };
  }
}

export { Database, Migration, Query } from './database';
```

**Usage in alaweimm90**:

```typescript
import { OrganizationDataSource } from '@monorepo/shared-database';

const db = new OrganizationDataSource({
  host: 'localhost',
  organization: 'alaweimm90',
  database: 'alaweimm90_dev',
});
```

**Usage in alaweimm90-science**:

```typescript
import { OrganizationDataSource } from '@monorepo/shared-database';

const db = new OrganizationDataSource({
  host: 'localhost',
  organization: 'alaweimm90-science',
  database: 'science_dev',
});
```

---

### Strategy 2: Organization-Specific Code - Modules

**What NOT to Share**: Organization-specific business logic

**Location**: `/{org-name}/src/`

```
/alaweimm90/
  /src/
    /domain/               (Business logic specific to alaweimm90)
      /finance/
      /healthcare/
      /manufacturing/
    /services/             (Organization services)
    /adapters/             (Convert between org format and shared format)

/alaweimm90-science/
  /src/
    /domain/               (Different business logic)
      /research/
      /data-analysis/
    /services/
    /adapters/
```

**Organization Adapter Pattern** (for multi-org compatibility):

```typescript
// alaweimm90/src/adapters/to-shared-api.ts
import { APIRequest, APIResponse } from '@monorepo/shared-api';

export class AlawemmAPIAdapter {
  static toShared(request: AlawemmRequest): APIRequest {
    return {
      organization: 'alaweimm90',
      method: request.method,
      path: request.path,
      body: request.payload, // Different property name!
      headers: request.headers,
    };
  }

  static fromShared(response: APIResponse): AlawemmResponse {
    return {
      code: response.status,
      data: response.body,
      message: response.message,
    };
  }
}

// alaweimm90-science/src/adapters/to-shared-api.ts
export class ScienceAPIAdapter {
  static toShared(request: ScienceRequest): APIRequest {
    return {
      organization: 'alaweimm90-science',
      method: request.operation, // Different property name!
      path: request.endpoint,
      body: request.data,
      headers: request.meta,
    };
  }

  static fromShared(response: APIResponse): ScienceResponse {
    return {
      status: response.status,
      result: response.body,
    };
  }
}
```

---

### Strategy 3: Conditional Imports by Organization

**Pattern for Org-Specific Modules**:

```typescript
// shared-code/src/index.ts
import { type OrganizationConfig } from '@monorepo/shared-utils';

export async function loadOrgSpecific(config: OrganizationConfig) {
  switch (config.organization) {
    case 'alaweimm90':
      return await import('@alaweimm90/domain');
    case 'alaweimm90-science':
      return await import('@alaweimm90-science/domain');
    default:
      throw new Error(`Unknown organization: ${config.organization}`);
  }
}

// In application
const domain = await loadOrgSpecific(config);
const service = domain.createService(config);
```

**Benefits**:

- ✅ Single code path for all organizations
- ✅ Organization-specific behavior injected at runtime
- ✅ Easier to test each organization separately
- ✅ Clear boundaries between shared and org-specific

---

## 🔐 DEPENDENCY MANAGEMENT BY ORGANIZATION

### Challenge: Different Organizations Need Different Versions

**Example Problem**:

```
alaweimm90:
  - Requires: express@^4.18.0 (stable)
  - Requires: React@18 (stable)

alaweimm90-science:
  - Requires: express@^5.0.0-beta (for new features)
  - Requires: React@19 (for experiments)
```

### Solution 1: Workspace Overrides (Current - Recommended)

```json
{
  "name": "root",
  "pnpm": {
    "overrides": {
      "express": "^4.18.0",
      "react": "^18.0.0"
    }
  },
  "dependencies": {
    "express": "^4.18.0",
    "react": "^18.0.0"
  }
}
```

**Pros**: Single dependency version for all
**Cons**: Can't use beta or experimental versions
**Best for**: Stable monorepo focused on compatibility

---

### Solution 2: Multiple package.json Files (Advanced)

**Structure**:

```
/packages/
  /mcp-core/package.json
  /agent-core/package.json
  /alaweimm90/package.json        (inherits root deps)
  /alaweimm90-science/package.json (inherits root deps)

/pnpm-workspace.yaml
packages:
  - 'packages/*'
  - 'alaweimm90'
  - 'alaweimm90-science'  (Added as workspace)
```

**Organization-Specific Dependencies**:

```json
// alaweimm90/package.json
{
  "name": "@alaweimm90/workspace",
  "dependencies": {
    "express": "^4.18.0",    // Override root
    "react": "^18.0.0"
  }
}

// alaweimm90-science/package.json
{
  "name": "@alaweimm90-science/workspace",
  "dependencies": {
    "express": "^5.0.0-beta", // Different version!
    "react": "^19.0.0"
  }
}
```

**Pros**: Different versions per organization
**Cons**: Larger node_modules (2x dependencies), harder to maintain
**Best for**: Organizations with very different stacks

---

### Solution 3: Selective Workspace Inclusion (Recommended for Your Case)

```yaml
# pnpm-workspace.yaml
packages:
  - 'packages/*' # Core packages - ALL orgs use these
  - 'alaweimm90' # Workspace 1
  # - 'alaweimm90-science'   # (If separate, exclude from workspace)
  # - 'alaweimm90-tools'     # (If separate, exclude from workspace)
```

**Separate Monorepos** (if needed):

```
GitHub Repository Structure:
├── /monorepo-core/
│   ├── /packages/
│   │   ├── /mcp-core/
│   │   ├── /agent-core/
│   │   └── [shared packages]
│   └── pnpm-workspace.yaml
│
├── /alaweimm90-monorepo/
│   ├── /alaweimm90/
│   ├── pnpm-workspace.yaml
│   └── (references /monorepo-core for core packages)
│
└── /alaweimm90-science-monorepo/
    ├── /alaweimm90-science/
    ├── pnpm-workspace.yaml
    └── (references /monorepo-core for core packages)
```

**Pros**: True isolation, different versions, independent CI/CD
**Cons**: More complex setup, coordination overhead
**Best for**: Truly independent organizations

---

## 🎯 DEPENDENCY CONFLICT RESOLUTION

### When Different Orgs Require Conflicting Dependencies

**Scenario**:

```
alaweimm90 requires: typescript@^5.0.0
alaweimm90-science requires: typescript@^4.9.0
```

### Option 1: Enforce Single Version (Recommended)

```json
// Root package.json
{
  "pnpm": {
    "overrides": {
      "typescript": "^5.3.0"
    }
  }
}
```

Then alaweimm90-science uses TypeScript 5.3, which is backward compatible with 4.9.

---

### Option 2: Isolate With Virtual Environments

```bash
# Install for alaweimm90
cd alaweimm90
pnpm install --frozen-lockfile

# Install for alaweimm90-science (separate node_modules)
cd alaweimm90-science
pnpm install --frozen-lockfile
```

---

### Option 3: Update Constraints

Talk to alaweimm90-science team:

```
"Hey, we need TypeScript 5.0+ because [reason].
Can you update your code to work with 5.0?"
```

Most libraries support a range of versions anyway.

---

## 📋 ORGANIZATION-SPECIFIC CHECKLIST

### For Each Organization, Define:

- [ ] **Organization Name**: `alaweimm90`, `alaweimm90-science`, etc.
- [ ] **Location**: `/alaweimm90`, `/.config/organizations/alaweimm90-business`, etc.
- [ ] **Status**: Active (in monorepo) vs Archive (reference only)
- [ ] **Core Packages Used**: `@monorepo/mcp-core`, `@monorepo/agent-core`, etc.
- [ ] **Shared Packages**: `@monorepo/shared-database`, `@monorepo/shared-utils`, etc.
- [ ] **Organization-Specific Plugins**: Which plugins enabled
- [ ] **Environment**: `production`, `staging`, `development`, `research`
- [ ] **Dependencies**: Override versions (if any)
- [ ] **Database Schema**: `org_alaweimm90`, `org_science`, etc.
- [ ] **Configuration File**: `.env.alaweimm90`, `.env.science`, etc.

---

## 🔧 TEMPLATE: ORGANIZATION CONFIGURATION

Create `organizations.json` in root:

```json
{
  "organizations": [
    {
      "id": "alaweimm90",
      "name": "Alaweimm 90",
      "status": "active",
      "location": "/alaweimm90",
      "type": "workspace",
      "environment": "production",
      "corePackages": [
        "@monorepo/mcp-core",
        "@monorepo/agent-core",
        "@monorepo/context-provider",
        "@monorepo/workflow-templates"
      ],
      "sharedPackages": [
        "@monorepo/shared-utils",
        "@monorepo/shared-automation",
        "@monorepo/shared-database"
      ],
      "plugins": [
        "api-gateway",
        "autonomous",
        "finance",
        "healthcare",
        "manufacturing",
        "mobile",
        "retail",
        "security-advanced",
        "federated-learning",
        "cloud"
      ],
      "database": {
        "name": "alaweimm90_prod",
        "schema": "org_alaweimm90"
      },
      "config": ".env.alaweimm90"
    },
    {
      "id": "alaweimm90-science",
      "name": "Alaweimm 90 - Science",
      "status": "archived",
      "location": "/.config/organizations/alaweimm90-science",
      "type": "archived",
      "environment": "research",
      "corePackages": ["@monorepo/mcp-core", "@monorepo/agent-core"],
      "sharedPackages": ["@monorepo/shared-utils"],
      "plugins": ["research", "data-analysis"],
      "database": {
        "name": "science_dev",
        "schema": "org_science"
      },
      "config": ".env.science"
    }
  ]
}
```

---

## 🚀 IMPLEMENTATION ROADMAP

### Week 1: Foundation

- [ ] Create organizations.json configuration file
- [ ] Set up environment file templates (.env.example)
- [ ] Document current org structure and status

### Week 2: Shared Infrastructure

- [ ] Create shared-utils package (if not exists)
- [ ] Create shared-database package (if not exists)
- [ ] Create shared-automation package (if not exists)

### Week 3: Organization Isolation

- [ ] Implement plugin loader for conditional loads
- [ ] Create org-specific adapters pattern
- [ ] Document organization boundaries

### Week 4: Configuration Management

- [ ] Implement config loader by organization
- [ ] Add org-specific TypeScript configs
- [ ] Document dependency management strategy

---

## ✅ SUMMARY

### Key Principles for Multi-Org Monorepo

1. **Separate What Differs**: Org-specific code → separate modules
2. **Share What's Common**: Shared code → core packages
3. **Use Dependency Injection**: Don't hardcode org-specific logic
4. **Version Single Core**: Keep core packages at same version
5. **Isolate Configs**: Each org has own .env and config
6. **Document Boundaries**: Clear what each org can do
7. **Plugin Everything**: Make org features pluggable
8. **Test Independently**: Each org tested separately

---

**Status**: ✅ ORGANIZATION CONCERNS DOCUMENTED
**Next**: Documentation Strategy Guide

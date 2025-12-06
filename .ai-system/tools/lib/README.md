# Shared Library

Common utilities and shared functionality for all tools in the meta-governance repository.

## Modules

### config.ts

Configuration management utilities:

```typescript
import { resolveTargetDir, loadConfig, DevOpsConfig } from './config.js';

// Resolve target directory from args/env
const targetDir = resolveTargetDir(process.argv.slice(2));

// Load configuration with defaults
const config = loadConfig('./config.json');

// Parse command-line arguments
const placeholders = parsePlaceholders(args);
const verbose = parseFlag(args, 'verbose', false);
```

### fs.ts

File system utilities for template management:

```typescript
import {
  walk,
  findManifests,
  readJson,
  validateTemplate,
  validateContent,
  copyDirWithReplacements,
  ensureDir,
  DevOpsFS,
} from './fs.js';

// Walk directory tree
const files = walk('./templates');

// Find template manifests
const manifests = findManifests('./templates');

// Validate template
const result = validateTemplate(manifest, templateDir);

// Copy with placeholder replacement
copyDirWithReplacements(src, dest, { '{{NAME}}': 'myapp' });
```

## Design Principles

1. **Single Source of Truth**: All shared utilities live here
2. **Type Safety**: Full TypeScript with exported interfaces
3. **Node.js Native**: Uses `node:` protocol for imports
4. **Zero Dependencies**: No external runtime dependencies

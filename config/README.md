# Configuration Directory

This directory centralizes all configuration files for the meta-governance repository.

## Structure

```
config/
├── README.md           # This file
├── typescript/         # TypeScript configuration
│   ├── base.json      # Base TypeScript config
│   ├── node.json      # Node.js specific config
│   └── build.json     # Build configuration
├── linting/           # Linting and formatting
│   ├── eslint.js      # ESLint configuration
│   ├── prettier.js    # Prettier configuration
│   └── .editorconfig  # Editor configuration
├── ci/                # CI/CD configuration
│   ├── pre-commit.yaml
│   └── workflows/     # GitHub Actions workflows
├── docker/            # Docker configuration
│   ├── Dockerfile
│   └── .dockerignore
└── environment/       # Environment configuration
    ├── .env.example   # Example environment variables
    └── defaults.json  # Default configuration values
```

## Usage

All configuration files should be imported from this central location:

```typescript
// Import TypeScript config
import tsConfig from '@config/typescript/base.json';

// Import environment config
import { defaults } from '@config/environment/defaults.json';
```

## Migration Status

- [ ] TypeScript configs consolidated
- [ ] Linting configs consolidated
- [ ] CI/CD configs consolidated
- [ ] Docker configs consolidated
- [ ] Environment configs consolidated

## Benefits

1. **Single Source of Truth**: All configs in one place
2. **Easy Discovery**: Clear organization by type
3. **Reduced Duplication**: Shared base configs
4. **Better Maintenance**: Centralized updates
5. **Clear Dependencies**: Explicit imports

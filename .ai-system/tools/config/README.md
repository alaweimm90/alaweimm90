# Global Tool Configurations

Shared configs for all projects. Symlink instead of duplicate.

## Available Configs

### JavaScript/TypeScript

- `eslint.config.js` - Linting (ESLint v9 flat config)
- `vitest.config.ts` - Unit testing
- `playwright.config.ts` - E2E testing

### Python

- `ruff.toml` - Linting & formatting

## Usage

### New Project

```bash
cd my-project
ln -s ../../tools/config/eslint.config.js .
ln -s ../../tools/config/vitest.config.ts .
```

### Existing Project

```bash
# Backup original
mv eslint.config.js .backup_eslint.config.js

# Link to global
ln -s ../../tools/config/eslint.config.js .
```

## Override

To override specific rules, create local config that extends global:

```js
// eslint.config.js
import base from '../../tools/config/eslint.config.js';
export default [...base, { rules: { 'no-console': 'off' } }];
```

## Maintenance

Update once, applies to all projects using symlinks.

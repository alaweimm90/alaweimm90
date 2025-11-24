# Development Tools Configuration

This directory consolidates all development tool configurations for the monorepo.

## Directory Structure

```
dev-tools/
├── ide/                    # IDE configurations
│   └── vscode/             # VS Code settings (moved from .vscode/)
├── ai-assistants/          # AI assistant configurations
│   └── claude/             # Claude AI settings (moved from .claude/)
├── linters/                # Linter configurations
│   ├── eslint/             # ESLint configs
│   └── other-linters/      # Other linting tools
├── formatters/             # Code formatter configurations
│   ├── prettier/           # Prettier configs
│   └── other-formatters/   # Other formatting tools
├── security/               # Security tool configurations
│   └── snyk/               # Snyk security configs
├── git-hooks/              # Git hook configurations
│   └── husky/              # Husky git hooks (moved from .husky/)
└── trae-ide/               # Trae IDE configurations (moved from .trae/)
```

## Purpose

This centralized structure provides:

1. **Single Source of Truth**: All dev tool configs in one place
2. **Easy Discovery**: Developers know where to find tool configurations
3. **Consistent Standards**: Shared configs across the monorepo
4. **Clean Root**: Reduces clutter at repository root level
5. **Version Control**: All configs tracked and versioned together

## Usage

### IDE Configurations

IDE-specific settings are stored in `ide/` subdirectories. Symlinks or IDE-specific configuration can reference these locations.

### AI Assistants

AI assistant configurations (Claude, Cursor, etc.) are stored in `ai-assistants/` subdirectories.

### Linters and Formatters

Linting and formatting tool configs are separated for clarity. Update root `package.json` or tool-specific config files to reference these locations.

### Git Hooks

Git hook configurations (Husky, etc.) are stored in `git-hooks/`. Ensure your git hook setup references the correct path.

## Migration Notes

This directory was created during the 2025-11-22 repository restructuring to consolidate scattered configuration files. Original locations:

- `.vscode/` → `ide/vscode/`
- `.claude/` → `ai-assistants/claude/`
- `.trae/` → `trae-ide/`
- `.husky/` → `git-hooks/husky/`

## Maintenance

When adding new development tools:

1. Create an appropriate subdirectory under the relevant category
2. Move or create configuration files in that subdirectory
3. Update this README to document the new tool
4. Update root-level configs to reference the new location


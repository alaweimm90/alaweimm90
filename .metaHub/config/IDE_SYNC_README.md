# IDE Configuration Synchronization

**SSOT Location**: `.metaHub/config/`  
**Last Updated**: 2025-11-24  
**Status**: Active

## Overview

All IDE configurations across the GitHub workspace are synchronized with a Single Source of Truth (SSOT) located in `.metaHub/config/`.

## Configuration Files

### 1. VSCode Settings

- **SSOT**: `.metaHub/config/vscode/settings.json`
- **Synced to**:
  - `.vscode/settings.json` (workspace root)
  - All organization project `.vscode/settings.json` files

### 2. EditorConfig

- **SSOT**: `.metaHub/config/.editorconfig`
- **Synced to**: All project root `.editorconfig` files

### 3. Prettier

- **SSOT**: `.metaHub/config/.prettierrc.json`
- **Synced to**: All project root `.prettierrc` or `.prettierrc.json` files

### 4. ESLint

- **SSOT**: `.metaHub/config/.eslintrc.json`
- **Synced to**: All project root `.eslintrc.json` files

## Synchronization Process

### Automated Sync

Projects should reference the SSOT configs using symbolic links or copy scripts:

```powershell
# Create symlinks (requires admin on Windows)
New-Item -ItemType SymbolicLink -Path ".editorconfig" -Target "..\.metaHub\config\.editorconfig"
New-Item -ItemType SymbolicLink -Path ".prettierrc.json" -Target "..\.metaHub\config\.prettierrc.json"
```

### Manual Sync

If symlinks are not feasible, copy the SSOT files:

```powershell
# Copy SSOT configs to project
Copy-Item ".metaHub\config\.editorconfig" -Destination ".\path\to\project\"
Copy-Item ".metaHub\config\.prettierrc.json" -Destination ".\path\to\project\"
Copy-Item ".metaHub\config\.eslintrc.json" -Destination ".\path\to\project\"
```

## Configuration Standards

### VSCode Settings Highlights

- **Format on Save**: Enabled
- **Default Formatter**: Prettier
- **Auto Fix**: ESLint on save
- **Performance**: Optimized for monorepo
- **Governance**: Enabled

### EditorConfig Standards

- **Charset**: UTF-8
- **Line Ending**: LF (Unix-style)
- **Indent**: 2 spaces (JS/TS), 4 spaces (Python)
- **Trim Trailing**: Enabled
- **Final Newline**: Enabled

### Prettier Standards

- **Print Width**: 100 characters
- **Tab Width**: 2 spaces
- **Semicolons**: Required
- **Quotes**: Single quotes (JS/TS)
- **Trailing Commas**: ES5
- **Arrow Parens**: Avoid when possible

### ESLint Standards

- **Parser**: TypeScript-ESLint for TS files
- **React**: JSX support with hooks
- **Testing**: Jest plugin for test files
- **Rules**: Strict error detection

## Projects Using SSOT

### Organizations

- ✅ **alaweimm90-business**: All projects (repz, benchbarrier, calla-lily-couture, live-it-iconic)
- ✅ **alaweimm90-tools**: All projects
- ✅ **alaweimm90-science**: All projects
- ✅ **AlaweinOS**: All projects
- ✅ **MeatheadPhysicist**: All projects

### Workspace Root

- ✅ `.vscode/settings.json`: Synced with SSOT
- ✅ `config/.editorconfig`: Original comprehensive config

## Maintenance

### Adding New Settings

1. Update SSOT files in `.metaHub/config/`
2. Run sync script or manually copy to projects
3. Commit changes with message: `chore: sync IDE configs from SSOT`

### Project-Specific Overrides

If a project requires specific overrides:

1. Start with SSOT base configuration
2. Add project-specific overrides in separate section
3. Document why override is needed in project README

### Validation

Check config sync status:

```powershell
# Compare project config with SSOT
diff .\.metaHub\config\.editorconfig .\path\to\project\.editorconfig
```

## Benefits

1. ✅ **Consistency**: All projects follow same code style
2. ✅ **Maintenance**: Update once, apply everywhere
3. ✅ **Onboarding**: New developers get consistent environment
4. ✅ **CI/CD**: Automated linting and formatting enforcement
5. ✅ **Quality**: Enforced best practices across all projects

## Troubleshooting

### VSCode Not Picking Up Settings

1. Reload window: `Ctrl+Shift+P` → "Reload Window"
2. Check workspace trust: Settings should apply to trusted workspace
3. Verify `.vscode/settings.json` exists in project root

### EditorConfig Not Working

1. Install EditorConfig extension for VSCode
2. Verify `root = true` in `.editorconfig`
3. Check file is in project root

### Prettier Conflicts with ESLint

1. Ensure `prettier` is in ESLint extends array
2. This disables ESLint formatting rules that conflict with Prettier
3. Both configs are designed to work together

## Related Documentation

- [VSCode Settings Documentation](https://code.visualstudio.com/docs/getstarted/settings)
- [EditorConfig Specification](https://editorconfig.org/)
- [Prettier Options](https://prettier.io/docs/en/options.html)
- [ESLint Configuration](https://eslint.org/docs/latest/user-guide/configuring/)

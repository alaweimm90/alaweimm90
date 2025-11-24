# IDE Unification & Autonomy - Complete Implementation

This document summarizes the complete IDE unification and autonomy setup implemented for VS Code, Trae, Windsurf, Cursor, and other VS Code-compatible editors.

## What Was Implemented

### 1. Team-Safe Baseline (Repo)

**Location**: `.vscode/*` and `.config/vscode/*`

**Files Created/Updated**:
- `.vscode/settings.json` - Team-safe workspace settings
- `.vscode/extensions.json` - Shared extension recommendations including AI assistants
- `.vscode/keybindings.json` - Minimal shared keybindings
- `.config/vscode/settings.json` - Synchronized with workspace settings

**Key Principles**:
- Only project-specific settings (formatting, linting, imports, file excludes)
- No personal preferences (fonts, themes, window behavior)
- No risky automation (auto-approve, auto-submit flags)
- Extensions recommended but behavior stays user-controlled

### 2. User-Level Autonomy System

**User Settings Automation** (`scripts/enable-autonomy.ps1`, `scripts/disable-autonomy.ps1`):
- Enables Copilot inline suggestions across all IDEs
- Activates common auto-apply/autonomous flags for installed assistants:
  - `github.copilot.enable`, `github.copilot.inlineSuggest.enable`
  - `anthropic.claude-code.autoSubmit`, `anthropic.claude-code.bypassConfirmations`
  - `cline.autonomous.enable`
  - `continue.autocomplete.enabled`
  - `augment.vscode-augment.autoApply`
  - `codeium.codeAssistant.enableAutoApply`
- Optional keybinding (Ctrl+Alt+A) to run "Approve Next 50" task
- Creates timestamped backups before any changes

**User Task Automation** (`scripts/add-startup-approvals.ps1`, `scripts/toggle-startup-approvals.ps1`):
- Adds user-level tasks that run on folder open
- Pre-approves next N commands (configurable: 50, 100, 200, etc.)
- Optional "Wrap + Report" task for approval + reporting
- Fully user-specific (not in repo)

### 3. User Settings Repair Utilities

**Problem Solved**: Invalid JSON in user `settings.json` preventing IDE from writing settings changes.

**Scripts Created**:
- `scripts/repair-user-settings-json.ps1` - Heuristic JSON repairs
- `scripts/clean-user-settings-hard.ps1` - Block-level property removal
- `scripts/clean-user-settings-lines.ps1` - Line-based array/property removal

**Issues Fixed**:
- Multi-line unescaped strings in arrays (`kilo-code.allowedCommands`)
- Raw Python code blocks with newlines
- Invalid `$schema` properties
- Malformed `github.copilot.chat.codeGeneration.instructions`
- Trailing commas and comment-stripping edge cases

### 4. Status and Management

**Scripts**:
- `scripts/status-startup-approvals.ps1` - View current auto-approve tasks per IDE
- `scripts/remove-startup-approvals.ps1` - Remove all auto-approve tasks

## Quick Start

### Enable Full Autonomy (User-Only)

```powershell
# Enable AI assistant auto-behaviors + add Ctrl+Alt+A keybinding
pwsh scripts/enable-autonomy.ps1 -AddKeybindings

# Add startup tasks: approve next 200 commands on folder open + generate reports
pwsh scripts/add-startup-approvals.ps1 -ApproveNext 200 -IncludeWrapReport
```

### Disable/Adjust

```powershell
# Disable autonomy (or restore from backup)
pwsh scripts/disable-autonomy.ps1
pwsh scripts/disable-autonomy.ps1 -RestoreFromBackup

# Remove startup tasks
pwsh scripts/remove-startup-approvals.ps1

# Toggle approve count to 300, keep auto-run
pwsh scripts/toggle-startup-approvals.ps1 -ApproveNext 300 -RunOnOpen Enable
```

### Check Status

```powershell
# View current startup tasks
pwsh scripts/status-startup-approvals.ps1
```

## Current Configuration

As of implementation completion:

### Repo Baseline
- **Formatting**: Prettier (single quotes, no semicolons, 100 char width)
- **Linting**: ESLint on save with auto-fix
- **Extensions**: Prettier, ESLint, TypeScript tools, Copilot, Copilot Chat, Claude Code
- **Keybindings**: Ctrl+Shift+F (format), Ctrl+` (toggle terminal)

### User Autonomy (Applied to VS Code, Trae, Windsurf, Cursor)
- **AI Assistants**: Auto-submit and inline suggestions enabled where supported
- **Startup Tasks**: "Approve Next 200" + "Wrap + Report" run on folder open
- **Keybinding**: Ctrl+Alt+A → run "Approve Next 50 (Auto-Approve)" task
- **Settings Backups**: Created before every autonomy toggle

## File Structure

```
GitHub/
├── .vscode/
│   ├── settings.json          # Team baseline (mirrors .config/vscode)
│   ├── extensions.json        # Shared recommendations
│   ├── keybindings.json       # Minimal team bindings
│   └── tasks.json             # Workspace tasks (wrapper tasks)
├── .config/vscode/
│   ├── settings.json          # Team baseline
│   ├── extensions.json        # Shared recommendations
│   └── keybindings.json       # Minimal team bindings
├── scripts/
│   ├── enable-autonomy.ps1              # Enable AI auto-behaviors
│   ├── disable-autonomy.ps1             # Disable or restore
│   ├── add-startup-approvals.ps1        # Add folder-open tasks
│   ├── remove-startup-approvals.ps1     # Remove folder-open tasks
│   ├── toggle-startup-approvals.ps1     # Adjust count & run-on-open
│   ├── status-startup-approvals.ps1     # View current tasks
│   ├── clean-user-settings-lines.ps1    # Fix invalid user JSON
│   ├── clean-user-settings-hard.ps1     # Block-level JSON cleanup
│   └── repair-user-settings-json.ps1    # Heuristic JSON repairs
└── docs/
    ├── IDE_AUTONOMY_SETUP.md            # User guide
    └── IDE_UNIFICATION_COMPLETE.md      # This document
```

## User Profile Locations

Scripts modify these per-IDE files (under `%APPDATA%`):

```
C:\Users\mesha\AppData\Roaming\
├── Code\User\
│   ├── settings.json     # Autonomy flags
│   ├── tasks.json        # Startup tasks
│   └── keybindings.json  # Optional Ctrl+Alt+A
├── Trae\User\
│   ├── settings.json
│   ├── tasks.json
│   └── keybindings.json
├── Windsurf\User\
│   ├── settings.json
│   ├── tasks.json
│   └── keybindings.json
└── Cursor\User\
    ├── settings.json
    ├── tasks.json
    └── keybindings.json
```

## Verification Steps

1. **Check user settings are valid**:
```powershell
$vc = "$env:APPDATA/Code/User/settings.json"
Get-Content -LiteralPath $vc -Raw | ConvertFrom-Json | Out-Null
```

2. **Verify workspace tasks exist**:
```powershell
code .vscode/tasks.json
# Should see: Approve Next 50, Wrap Command, etc.
```

3. **Test autonomy**:
- Open VS Code/Trae/Windsurf/Cursor in this repo
- On folder open, task "Auto-Approve: Approve Next 200" should run silently
- Start typing in a code file; Copilot suggestions should appear inline
- Try an assistant (Claude Code, etc.); edits should apply without extra prompts

4. **Test wrapper tasks**:
```powershell
# Run a workspace task
# Ctrl+Shift+P → Tasks: Run Task → "Approve Next 50 (Auto-Approve)"
# Then run any command requiring approval (e.g., npm install)
```

## Architecture Decisions

### Repo vs User Split

**Repo** (committed, shared):
- Editor behavior that improves code quality (format, lint, imports)
- File/search excludes for performance
- Extension recommendations (non-intrusive)
- Wrapper tasks anyone can run manually

**User** (local, personal):
- Font sizes, themes, layout preferences
- Auto-save delays, minimap, cursor style
- Autonomous AI flags (auto-submit, bypass confirmations)
- Startup tasks with `runOn: folderOpen`

### Why This Approach

1. **Team Safety**: No teammate inherits aggressive automation
2. **Flexibility**: Each developer controls their own risk/autonomy level
3. **Portability**: Repo works in containers, Codespaces, new machines
4. **Transparency**: Scripts log changes and create backups
5. **Reversibility**: Disable scripts restore prior state

## Common Workflows

### New Team Member Setup

```powershell
# Clone repo, open in VS Code
# Workspace settings/extensions load automatically (team baseline)

# Optional: enable personal autonomy
pwsh scripts/enable-autonomy.ps1 -AddKeybindings
pwsh scripts/add-startup-approvals.ps1 -ApproveNext 100
```

### Switch Between Conservative and Aggressive

```powershell
# Conservative (manual approvals, no auto-run)
pwsh scripts/disable-autonomy.ps1
pwsh scripts/remove-startup-approvals.ps1

# Aggressive (auto-approve 200, run on open, wrap+report)
pwsh scripts/enable-autonomy.ps1 -AddKeybindings
pwsh scripts/toggle-startup-approvals.ps1 -ApproveNext 200 -IncludeWrapReport -RunOnOpen Enable
```

### Adjust for a Specific Session

```powershell
# Disable auto-run on open but keep tasks available
pwsh scripts/toggle-startup-approvals.ps1 -RunOnOpen Disable

# Re-enable later
pwsh scripts/toggle-startup-approvals.ps1 -RunOnOpen Enable
```

## Troubleshooting

### "Unable to write into user settings" Error

**Cause**: Invalid JSON in user `settings.json`

**Fix**:
```powershell
# Run line-based cleanup (safest)
pwsh scripts/clean-user-settings-lines.ps1

# Then enable autonomy again
pwsh scripts/enable-autonomy.ps1
```

### Startup Task Not Running

**Check**:
1. Verify task exists: `pwsh scripts/status-startup-approvals.ps1`
2. Ensure `runOn: folderOpen` is set (should show "folderOpen" in status)
3. Reload window: Ctrl+Shift+P → "Developer: Reload Window"

### Settings Changes Don't Persist

**Cause**: JSON syntax error introduced after script run

**Fix**:
1. Open user settings (JSON) in the IDE
2. Look for red squiggles or Problems panel errors
3. Restore from backup: `$env:APPDATA/<IDE>/User/settings.json.backup-YYYYMMDD-HHmmss.json`

## Extension Recommendations (Included in Repo)

- **esbenp.prettier-vscode** - Code formatting
- **dbaeumer.vscode-eslint** - JavaScript/TypeScript linting
- **ms-vscode.vscode-typescript-next** - Latest TypeScript support
- **github.copilot** - AI pair programmer
- **github.copilot-chat** - Chat interface for Copilot
- **anthropic.claude-code** - Claude AI assistant

*Others can be added via `.vscode/extensions.json` without affecting user autonomy.*

## Maintenance

### Update Approve Count Globally

```powershell
pwsh scripts/toggle-startup-approvals.ps1 -ApproveNext 500 -IncludeWrapReport -RunOnOpen Enable
```

### Add New AI Assistant Auto-Enable

Edit `scripts/enable-autonomy.ps1`, add to `$desired` hashtable:
```powershell
'newassistant.autoApply' = $true
```

### Remove Old Backups (Optional)

```powershell
Get-ChildItem "$env:APPDATA\Code\User\*.backup-*.json" | 
  Where-Object LastWriteTime -lt (Get-Date).AddDays(-30) | 
  Remove-Item
```

## Summary

This implementation provides:
- ✅ Unified team-safe baseline across VS Code, Trae, Windsurf, Cursor
- ✅ Per-IDE user autonomy with one-command enable/disable
- ✅ Auto-approve on folder open (configurable count)
- ✅ Wrapper report generation on startup
- ✅ Full backup and restore capability
- ✅ Status visibility for all automation
- ✅ No risky defaults in the repo
- ✅ Easy teammate onboarding

All autonomy is opt-in, user-controlled, and reversible.

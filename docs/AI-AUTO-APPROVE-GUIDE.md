# AI Tools Auto-Approve Configuration Guide

This document details how to configure each AI coding assistant for maximum autonomous operation ("YOLO mode").

## Quick Reference

| Tool | Config Location | Auto-Approve Flag | Status |
|------|-----------------|-------------------|--------|
| **Aider** | `.aider.conf.yml` | `yes-always: true` | Fully supported |
| **Cursor** | `.cursor/settings.json` | `autoApply: true` | Supported |
| **Windsurf** | `.windsurfrules` | In-file settings | Supported |
| **Claude Code** | `.claude/settings.json` | `--dangerously-skip-permissions` | CLI flag |
| **GitHub Copilot** | `.github/copilot-instructions.md` | Context only | No auto-approve |
| **Continue** | `.continue/config.json` | `autoApprove: true` | Experimental |
| **Kilo Code** | `.kilocode/config.yaml` | `auto_approve: true` | Supported |
| **Amazon Q** | `.amazonq/settings.json` | `autoApprove: true` | Limited |
| **Cline** | VS Code settings | Extension settings | Limited |

## Detailed Configuration

### Aider CLI

**Location:** `.aider.conf.yml` (root) or `~/.aider.conf.yml` (global)

```yaml
# Full autonomous mode
yes-always: true
auto-commits: true
model: claude-3-5-sonnet-20241022
```

**CLI Override:**
```bash
aider --yes-always --auto-commits
```

### Cursor IDE

**Location:** `.cursor/settings.json`

```json
{
  "cursor.general.autoApply": true,
  "cursor.chat.autoApplyEdits": true,
  "cursor.composer.autoApply": true,
  "cursor.terminal.autoRun": true
}
```

**Also:** Create `.cursorrules` for project context.

### Windsurf IDE

**Location:** `.windsurfrules` (root)

Include these directives in the rules file:
```
## Preferences
- Auto-apply edits: YES
- Auto-run terminal commands: YES
- Trust workspace: YES
```

### Claude Code CLI

**CLI Flag:**
```bash
claude --dangerously-skip-permissions
```

**Environment Variable:**
```bash
export CLAUDE_SKIP_PERMISSIONS=1
```

**Local Settings:** `.claude/settings.local.json` (auto-generated)

### GitHub Copilot

No true auto-approve - Microsoft policy prevents it.

**Best Option:** Use `.github/copilot-instructions.md` for context.

### Continue.dev

**Location:** `.continue/config.json`

```json
{
  "autoApprove": true,
  "autoRun": true,
  "experimental": {
    "autoApply": true
  }
}
```

### Kilo Code

**Location:** `.kilocode/config.yaml`

```yaml
settings:
  auto_approve: true
  auto_apply: true
  auto_run_commands: true
```

### Amazon Q

**Location:** `.amazonq/settings.json`

```json
{
  "amazonQ.autoApprove": true,
  "amazonQ.autoApplyCodeSuggestions": true
}
```

## Global VS Code Settings

For VS Code-based tools, add to `settings.json`:

```json
{
  "files.autoSave": "afterDelay",
  "files.autoSaveDelay": 1000,
  "editor.formatOnSave": true,
  "workbench.notifications.enabled": false,

  // Copilot
  "github.copilot.enable": {
    "*": true
  },

  // Cline/Roo-Cline
  "cline.bypassApprovals": true,

  // Continue
  "continue.enableAutoApprove": true
}
```

## Environment Variables

Set these in your shell profile (`~/.bashrc`, `~/.zshrc`, PowerShell profile):

```bash
# Aider
export AIDER_YES=1
export AIDER_AUTO_COMMITS=1

# Claude
export CLAUDE_SKIP_PERMISSIONS=1

# OpenAI/Anthropic API Keys
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

## Windows PowerShell Profile

Location: `$PROFILE` (usually `~\Documents\PowerShell\Microsoft.PowerShell_profile.ps1`)

```powershell
$env:AIDER_YES = "1"
$env:AIDER_AUTO_COMMITS = "1"
$env:CLAUDE_SKIP_PERMISSIONS = "1"
```

## Verification

Test that auto-approve is working:

```bash
# Aider
aider --yes-always --message "Add a hello world function to test.py"

# Claude Code
claude --dangerously-skip-permissions -p "Create a test file"
```

## Security Considerations

Auto-approve modes bypass safety prompts. Use with caution:

1. **Only in trusted projects** - Never enable globally on untrusted repos
2. **Version control** - Always have git commits to revert if needed
3. **Sandbox testing** - Test in a branch first
4. **Review changes** - Periodically review what was auto-applied

## Troubleshooting

### Settings Not Applied

1. Check file location (root vs. home directory)
2. Verify JSON/YAML syntax
3. Restart IDE after config changes
4. Check extension is installed and enabled

### Still Getting Prompts

Some tools have hardcoded safety prompts that can't be bypassed:
- GitHub Copilot Chat (Microsoft policy)
- Some VS Code extension actions

Workaround: Use CLI versions (Aider, Claude Code) which have more control.

## Files in This Repository

```
.aider.conf.yml          # Aider config
.cursorrules             # Cursor rules
.windsurfrules           # Windsurf rules
.ai-context.md           # Shared AI context
.cursor/settings.json    # Cursor IDE settings
.continue/config.json    # Continue.dev config
.kilocode/config.yaml    # Kilo Code config
.amazonq/settings.json   # Amazon Q settings
.github/copilot-instructions.md  # Copilot context
```

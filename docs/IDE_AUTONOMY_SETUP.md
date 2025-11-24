# Ide Autonomy Setup

This repo standardizes a team-safe baseline for VS Code–compatible IDEs (VS Code, Trae, Windsurf). Personal and autonomous behaviors live only in user profiles.

- Shared (repo): `.vscode/*` and `.config/vscode/*` contain workspace settings, extension recommendations, and minimal keybindings that are safe for everyone.
- Personal (user): Each IDE’s `%APPDATA%/<IDE>/User/settings.json` contains your font/theme and any autonomous assistant settings.

## What the repo provides

- Formatting, linting, and import hygiene on save (Prettier + ESLint).
- Sensible editor navigation defaults (preview tabs off, stable peek).
- File/watch excludes for heavy folders.
- Extension recommendations (Prettier, ESLint, TypeScript tools, Copilot, Claude Code).
- Minimal keybindings: format document and toggle terminal.

## Enable or disable autonomy (local only)

Two convenience scripts toggle user-level autonomy across VS Code, Trae, Windsurf (and similar VS Code derivatives found under `%APPDATA%`). Both create a timestamped backup of your user `settings.json` before changes.

Enable autonomy:

```powershell
pwsh scripts/enable-autonomy.ps1
# Optional: also add a user keybinding Ctrl+Alt+A to run the workspace task
pwsh scripts/enable-autonomy.ps1 -AddKeybindings
```

Disable autonomy:

```powershell
pwsh scripts/disable-autonomy.ps1
# Or restore from the most recent backup created by the enable script
pwsh scripts/disable-autonomy.ps1 -RestoreFromBackup
```

### What “enable autonomy” does

- Turns on Copilot inline suggestions and common auto-apply/autonomous flags for popular assistants (Claude Code, Cline, Continue, Augment, Codeium) if present.
- Sets a personal auto-save preference (`afterDelay`, 1000ms) in your user profile.
- Optionally adds a user keybinding `Ctrl+Alt+A` to run the task “Approve Next 50 (Auto-Approve)”.

Notes:

- Keys are applied even if the extension isn’t installed; harmless but no effect until installed.
- The scripts remove JSON comments when writing back (backups are made first).

## Using the auto-approve wrapper tasks

The workspace exposes tasks that wrap commands through `.tools/ide-auto-approve.ps1` and generate reports:

- `Wrap Command (Auto-Approve)`
- `Wrap + Report (Auto-Approve)`
- `Approve Next 50 (Auto-Approve)` / `Approve Next N (Auto-Approve)`

Run them via the VS Code “Run Task” UI or bind your own user-level keybinding. These tasks stay in the repo so everyone can use them, but aggressive triggers (like auto-run on folder open) should remain personal if you choose to add them.

## Repo vs user split at a glance

- Keep in repo: formatting & linting defaults, safe editor navigation, file excludes, extension recommendations, minimal shared keybindings.
- Keep in user: font size, theme, notification preferences, panel layout, and any autonomous/auto-apply/auto-submit flags.

If you later add a dev container, the repo workspace files will be picked up by all editors. Keep secrets and risky automation out of the repo; use your user profile instead.

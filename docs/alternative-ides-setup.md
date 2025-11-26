# Alternative IDEs Setup for Auto-Approve and YOLO Mode

This guide provides step-by-step instructions to configure Cursor and Windsurf for fully autonomous, uncensored development workflows without manual approvals.

## Cursor Setup

1. **Download and Install**:
   - Visit [cursor.sh](https://cursor.sh) and download the installer.
   - Run the installer and follow the prompts.

2. **Initial Configuration**:
   - Open Cursor and sign in with your account.
   - Go to Settings (Ctrl+,) and apply the following:

   ```json
   {
     "terminal.integrated.confirmOnExit": "never",
     "terminal.integrated.confirmOnKill": "never",
     "git.confirmSync": false,
     "git.confirmEmptyCommits": false,
     "extensions.autoUpdate": true,
     "security.workspace.trust.enabled": false,
     "security.workspace.trust.startupPrompt": "never",
     "tasks.autoDetect": "on",
     "editor.formatOnSave": true,
     "files.autoSave": "afterDelay",
     "files.autoSaveDelay": 1000,
     "copilot.enable": {
       "*": true
     },
     "claude.enable": true,
     "claude.autoExecute": true,
     "claude.allowAllCommands": true
   }
   ```

3. **Automation Tasks**:
   - Create `.cursorrules` in your project root:

     ```
     - Auto-approve all commands
     - Run builds on save
     - Enable YOLO mode for rapid iteration
     ```

4. **Extensions**:
   - Install via Command Palette: `Cursor: Install Extension`
   - Recommended: GitLens, Code Runner, Remote Containers.

5. **Startup Automation**:
   - Use Cursor's "Composer" for multi-file edits with auto-execution.

## Windsurf Setup

1. **Download and Install**:
   - Visit [windsurf.io](https://windsurf.io) and download.
   - Install and launch.

2. **Configuration**:
   - Open Settings and update:

   ```json
   {
     "terminal.integrated.confirmOnExit": "never",
     "git.confirmSync": false,
     "extensions.autoUpdate": true,
     "security.workspace.trust.enabled": false,
     "tasks.autoDetect": "on",
     "windsurf.yoloMode": true,
     "windsurf.autoExecute": true
   }
   ```

3. **Automation**:
   - Windsurf has built-in YOLO mode. Enable it in settings.
   - Tasks run automatically on folder open.

4. **Extensions**:
   - Similar to VSCode; install automation-focused ones.

## General Tips for All IDEs

- **API Keys**: Set environment variables for AI models (e.g., `ANTHROPIC_API_KEY`).
- **Sandboxing**: Use Docker for isolated execution.
- **Backup**: Ensure Git is configured for rollbacks.
- **Testing**: Start with simple commands to verify auto-execution.

## Global System-Wide Setup

To apply auto-approve settings everywhere on your system, run the provided scripts:

1. **Apply Global Settings**:
   - Run `scripts/apply-global-settings.ps1` (can run as user).
   - This configures VSCode global settings, Git, NPM, PowerShell profile, and environment variables.

2. **Setup System Automation**:
   - Run `scripts/setup-system-automation.ps1` **as administrator** (right-click PowerShell, "Run as administrator").
   - This adds startup tasks, scheduled jobs, registry edits for auto-approve, and firewall rules.
   - If errors occur, the script will warn but continue with available operations.

3. **Manual Global VSCode Settings** (if script fails):
   - Edit `%APPDATA%\Code\User\settings.json` with the settings from the script.

4. **Environment Variables**:
   - Set `AUTO_APPROVE=true` and `YOLO_MODE=enabled` globally.

5. **Restart**: Restart your computer and IDEs for full effect.

**Note**: Some operations require admin privileges. Run PowerShell as administrator for system-level changes.

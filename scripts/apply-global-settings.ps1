# Global Settings Application Script for YOLO Mode
# Run this script to apply auto-approve settings everywhere

# VSCode Global Settings
$vsCodeSettingsPath = "$env:APPDATA\Code\User\settings.json"
$globalSettings = @{
    "terminal.integrated.confirmOnExit" = "never"
    "terminal.integrated.confirmOnKill" = "never"
    "git.confirmSync"                   = $false
    "git.confirmEmptyCommits"           = $false
    "extensions.autoUpdate"             = $true
    "security.workspace.trust.enabled"  = $false
    "tasks.autoDetect"                  = "on"
    "files.autoSave"                    = "afterDelay"
    "files.autoSaveDelay"               = 1000
    "editor.formatOnSave"               = $true
    "copilot.enable"                    = @{
        "*" = $true
    }
    "claude.enable"                     = $true
    "claude.autoExecute"                = $true
    "claude.allowAllCommands"           = $true
}

$globalSettings | ConvertTo-Json | Set-Content -Path $vsCodeSettingsPath -Encoding UTF8
Write-Host "Applied global VSCode settings."

# Git Global Config
git config --global core.autocrlf false
git config --global push.default simple
git config --global pull.rebase false
git config --global init.defaultBranch main
Write-Host "Configured Git global settings."

# NPM Global Config
npm config set fund false
npm config set audit false
npm config set save-exact true
Write-Host "Configured NPM global settings."

# PowerShell Profile for Auto-Execution
$profilePath = $PROFILE
$profileContent = @"
# YOLO Mode Profile
Write-Host "YOLO Mode Active: All commands auto-approved."
"@

if (!(Test-Path $profilePath)) {
    New-Item -Path $profilePath -ItemType File -Force
}
Add-Content -Path $profilePath -Value $profileContent
Write-Host "Updated PowerShell profile."

# Environment Variables
[Environment]::SetEnvironmentVariable("AUTO_APPROVE", "true", "User")
[Environment]::SetEnvironmentVariable("YOLO_MODE", "enabled", "User")
Write-Host "Set environment variables."

Write-Host "Global settings applied successfully. Restart terminals/IDEs for changes to take effect."
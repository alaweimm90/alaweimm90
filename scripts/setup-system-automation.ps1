# System-Wide Automation Setup Script
# Sets up startup automation for YOLO mode

# Check for admin privileges
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Administrator)) {
    Write-Warning "This script requires administrator privileges. Please run as administrator."
    exit 1
}

# Add to Startup Folder
$startupPath = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Startup"
$scriptPath = "$PSScriptRoot\auto-workflow.ps1"
$shortcutPath = "$startupPath\YOLO-AutoWorkflow.lnk"

# Create shortcut (simplified, using PowerShell to run script)
$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = "powershell.exe"
$shortcut.Arguments = "-ExecutionPolicy Bypass -File `"$scriptPath`" -Action all"
$shortcut.WorkingDirectory = Split-Path $scriptPath
$shortcut.Save()

Write-Host "Added auto-workflow to startup."

# Scheduled Task for Daily Automation
$taskName = "YOLO-DailyAutomation"
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -File `"$scriptPath`" -Action commit"
$trigger = New-ScheduledTaskTrigger -Daily -At "18:00"
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
try {
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -RunLevel Highest -Force
    Write-Host "Created daily automation task."
}
catch {
    Write-Warning "Failed to create scheduled task: $_"
}

# Registry Edits for Auto-Approve (Advanced)
$regPath = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Policies\Explorer"
try {
    if (!(Test-Path $regPath)) { New-Item -Path $regPath -Force }
    Set-ItemProperty -Path $regPath -Name "NoRunasInstallPrompt" -Value 1 -Type DWord
    Set-ItemProperty -Path $regPath -Name "NoRunasInstallPromptAllUsers" -Value 1 -Type DWord
    Write-Host "Applied registry settings for auto-approve."
}
catch {
    Write-Warning "Failed to apply registry settings: $_"
}

# Firewall Rules for Development
try {
    New-NetFirewallRule -DisplayName "Allow All Dev Ports" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 3000-9999
    Write-Host "Opened development ports in firewall."
}
catch {
    Write-Warning "Failed to open firewall ports: $_"
}

Write-Host "System-wide automation setup complete. Restart required for some changes."
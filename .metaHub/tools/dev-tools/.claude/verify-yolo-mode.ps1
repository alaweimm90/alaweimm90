#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Verify YOLO Mode Configuration for Claude Code

.DESCRIPTION
    This script validates the YOLO mode configuration, checks JSON syntax,
    counts commands, and verifies documentation files.

.EXAMPLE
    .\verify-yolo-mode.ps1
#>

param(
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

# Colors
$Green = "Green"
$Red = "Red"
$Yellow = "Yellow"
$Cyan = "Cyan"

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor $Cyan
Write-Host "  YOLO Mode Configuration Verification" -ForegroundColor $Cyan
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor $Cyan
Write-Host ""

# Check if config file exists
$configPath = ".claude\settings.local.json"
if (-not (Test-Path $configPath)) {
    Write-Host "❌ Configuration file not found: $configPath" -ForegroundColor $Red
    exit 1
}
Write-Host "✅ Configuration file exists: $configPath" -ForegroundColor $Green

# Validate JSON syntax
try {
    $config = Get-Content $configPath | ConvertFrom-Json
    Write-Host "✅ JSON syntax is valid" -ForegroundColor $Green
}
catch {
    Write-Host "❌ JSON syntax error: $($_.Exception.Message)" -ForegroundColor $Red
    exit 1
}

# Check structure
if (-not $config.permissions) {
    Write-Host "❌ Missing 'permissions' object" -ForegroundColor $Red
    exit 1
}
Write-Host "✅ Configuration structure is valid" -ForegroundColor $Green

# Count commands
$allowCount = $config.permissions.allow.Count
$denyCount = $config.permissions.deny.Count
$askCount = $config.permissions.ask.Count

Write-Host ""
Write-Host "Command Statistics:" -ForegroundColor $Cyan
Write-Host "   Allowed Commands: $allowCount" -ForegroundColor $Green
Write-Host "   Blocked Commands: $denyCount" -ForegroundColor $Red
Write-Host "   Ask Commands: $askCount" -ForegroundColor $Yellow

# Verify minimum commands
if ($allowCount -lt 100) {
    Write-Host "⚠️  Warning: Only $allowCount commands allowed (expected 200+)" -ForegroundColor $Yellow
}

if ($denyCount -lt 10) {
    Write-Host "⚠️  Warning: Only $denyCount commands blocked (expected 15+)" -ForegroundColor $Yellow
}

# Check for dangerous patterns in allow list
$dangerousPatterns = @(
    "rm -rf /",
    "rm -rf /*",
    "shutdown",
    "reboot",
    "Format-Volume",
    "Stop-Computer"
)

$foundDangerous = $false
foreach ($pattern in $dangerousPatterns) {
    $found = $config.permissions.allow | Where-Object { $_ -like "*$pattern*" }
    if ($found) {
        Write-Host "❌ Dangerous pattern found in allow list: $pattern" -ForegroundColor $Red
        $foundDangerous = $true
    }
}

if (-not $foundDangerous) {
    Write-Host "✅ No dangerous patterns in allow list" -ForegroundColor $Green
}

# Check documentation files
Write-Host ""
Write-Host "Documentation Files:" -ForegroundColor $Cyan

$docFiles = @(
    ".claude\YOLO_MODE_CONFIGURATION.md",
    ".claude\YOLO_MODE_QUICK_REFERENCE.md",
    ".claude\README.md"
)

$allDocsExist = $true
foreach ($file in $docFiles) {
    if (Test-Path $file) {
        $size = (Get-Item $file).Length
        Write-Host "   ✅ $file ($size bytes)" -ForegroundColor $Green
    }
    else {
        Write-Host "   ❌ $file (missing)" -ForegroundColor $Red
        $allDocsExist = $false
    }
}

# Analyze command distribution
Write-Host ""
Write-Host "Command Distribution:" -ForegroundColor $Cyan

$bashCommands = $config.permissions.allow | Where-Object { $_ -like "Bash(*" }
$psCommands = $config.permissions.allow | Where-Object { $_ -like "PowerShell(*" }

Write-Host "   Bash Commands: $($bashCommands.Count)" -ForegroundColor $Green
Write-Host "   PowerShell Commands: $($psCommands.Count)" -ForegroundColor $Green

# Check for common commands
Write-Host ""
Write-Host "Common Commands Check:" -ForegroundColor $Cyan

$commonCommands = @(
    "Bash(git:*)",
    "Bash(npm:*)",
    "Bash(node:*)",
    "PowerShell(Get-ChildItem:*)",
    "PowerShell(Get-Content:*)",
    "Bash(docker:*)"
)

foreach ($cmd in $commonCommands) {
    $found = $config.permissions.allow | Where-Object { $_ -eq $cmd }
    if ($found) {
        Write-Host "   ✅ $cmd" -ForegroundColor $Green
    }
    else {
        Write-Host "   ⚠️  $cmd (not found)" -ForegroundColor $Yellow
    }
}

# Verbose output
if ($Verbose) {
    Write-Host ""
    Write-Host "Detailed Command List:" -ForegroundColor $Cyan
    Write-Host ""
    Write-Host "Allowed Commands:" -ForegroundColor $Green
    $config.permissions.allow | ForEach-Object { Write-Host "   $_" }
    Write-Host ""
    Write-Host "Blocked Commands:" -ForegroundColor $Red
    $config.permissions.deny | ForEach-Object { Write-Host "   $_" }
}

# Final summary
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor $Cyan
Write-Host "  Verification Summary" -ForegroundColor $Cyan
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor $Cyan
Write-Host ""

$issues = 0

if ($allowCount -lt 100) { $issues++ }
if ($denyCount -lt 10) { $issues++ }
if ($foundDangerous) { $issues++ }
if (-not $allDocsExist) { $issues++ }

if ($issues -eq 0) {
    Write-Host "✅ All checks passed! YOLO mode is properly configured." -ForegroundColor $Green
    Write-Host ""
    Write-Host "You can now use Claude Code with auto-approved commands." -ForegroundColor $Green
    Write-Host "Run 'git status' or 'npm --version' to test." -ForegroundColor $Green
}
else {
    Write-Host "⚠️  $issues issue(s) found. Please review the output above." -ForegroundColor $Yellow
}

Write-Host ""
Write-Host "Documentation:" -ForegroundColor $Cyan
Write-Host "   Quick Reference: .claude\YOLO_MODE_QUICK_REFERENCE.md" -ForegroundColor $Yellow
Write-Host "   Full Guide: .claude\YOLO_MODE_CONFIGURATION.md" -ForegroundColor $Yellow
Write-Host "   Directory Info: .claude\README.md" -ForegroundColor $Yellow
Write-Host ""

exit 0


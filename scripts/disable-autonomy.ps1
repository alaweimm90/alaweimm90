<#
Disables autonomous/autorun behavior by flipping keys to a conservative state or restoring from
latest backup created by enable-autonomy.ps1.

Usage:
  pwsh scripts/disable-autonomy.ps1
  pwsh scripts/disable-autonomy.ps1 -RestoreFromBackup
#>
[CmdletBinding()]
param(
  [switch]$RestoreFromBackup
)

function Remove-JsonComments {
  param([string]$JsonText)
  # Only remove block comments; leave // intact (e.g., file:// URIs in yaml.schemas)
  $noBlock = [System.Text.RegularExpressions.Regex]::Replace($JsonText, "/\*.*?\*/", '', 'Singleline')
  return $noBlock
}

function Set-JsonValue {
  param(
    [hashtable]$Root,
    [string]$Path,
    $Value
  )
  $parts = $Path -split '\.'
  $cursor = $Root
  for ($i = 0; $i -lt $parts.Length; $i++) {
    $key = $parts[$i]
    if ($i -eq $parts.Length - 1) {
      if ($null -eq $Value) {
        $Root.Remove($key) | Out-Null
      } else {
        $cursor[$key] = $Value
      }
    } else {
      if (-not $cursor.ContainsKey($key) -or -not ($cursor[$key] -is [hashtable])) {
        $cursor[$key] = @{}
      }
      $cursor = $cursor[$key]
    }
  }
}

function Get-IdeSettingsPaths {
  $base = $env:APPDATA
  $targets = @()
  $ideNames = @('Code', 'Code - Insiders', 'VSCodium', 'Trae', 'Windsurf', 'Cursor')
  foreach ($name in $ideNames) {
    $p = Join-Path $base $name
    $userDir = Join-Path $p 'User'
    $settings = Join-Path $userDir 'settings.json'
    if (Test-Path $settings) { $targets += $settings }
  }
  $pattern = Join-Path $base '*\User\settings.json'
  $extra = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue
  foreach ($file in $extra) {
    if (-not ($targets -contains $file.FullName)) { $targets += $file.FullName }
  }
  return $targets | Sort-Object -Unique
}

function Restore-LatestBackup {
  param([string]$SettingsPath)
  $dir = Split-Path -Parent $SettingsPath
  $name = Split-Path -Leaf $SettingsPath
  $pattern = "$name.backup-*.json"
  $backups = Get-ChildItem -Path $dir -Filter $pattern | Sort-Object LastWriteTime -Descending
  if ($backups.Count -gt 0) {
    Copy-Item -LiteralPath $backups[0].FullName -Destination $SettingsPath -Force
    Write-Host "Restored backup: $($backups[0].Name) -> $name" -ForegroundColor Yellow
    return $true
  }
  Write-Warning "No backups found for $SettingsPath"
  return $false
}

function Update-UserSettingsFile {
  param([string]$SettingsPath)
  Write-Host "Updating: $SettingsPath" -ForegroundColor Cyan
  $raw = Get-Content -LiteralPath $SettingsPath -Raw -ErrorAction Stop
  $rawNoComments = Remove-JsonComments $raw
  $parsed = $null
  try { $parsed = ConvertFrom-Json -InputObject $rawNoComments -ErrorAction Stop } catch { throw }
  $obj = @{}
  foreach ($p in $parsed.PSObject.Properties) { $obj[$p.Name] = $p.Value }

  $disable = @{
    'editor.inlineSuggest.enabled'            = $false
    'github.copilot.enable'                   = $true  # keep Copilot enabled but not aggressive
    'github.copilot.inlineSuggest.enable'     = $true
    'github.copilot.editor.enableAutoCompletions' = $true
    'github.copilot.suggestions.enable'       = $true
    'anthropic.claude-code.autoSubmit'        = $false
    'anthropic.claude-code.bypassConfirmations' = $false
    'cline.autonomous.enable'                 = $false
    'continue.autocomplete.enabled'           = $true
    'augment.vscode-augment.autoApply'        = $false
    'codeium.codeAssistant.enableAutoApply'   = $false
    'files.autoSave'                          = 'off'
    'files.autoSaveDelay'                     = 1000
  }
  foreach ($k in $disable.Keys) { Set-JsonValue -Root $obj -Path $k -Value $disable[$k] }

  $json = $obj | ConvertTo-Json -Depth 50
  $json | Set-Content -LiteralPath $SettingsPath -Encoding UTF8
}

$targets = Get-IdeSettingsPaths
if (-not $targets -or $targets.Count -eq 0) {
  Write-Warning "No IDE user settings.json files found under %APPDATA%. Nothing to do."
  exit 0
}

foreach ($settings in $targets) {
  try {
    if ($RestoreFromBackup) {
      if (-not (Restore-LatestBackup -SettingsPath $settings)) {
        Write-Host "Falling back to conservative toggle updates..." -ForegroundColor DarkGray
        Update-UserSettingsFile -SettingsPath $settings
      }
    } else {
      Update-UserSettingsFile -SettingsPath $settings
    }
  } catch {
    Write-Error $_
  }
}

Write-Host "Autonomy disabled for $(($targets | Measure-Object).Count) IDE profile(s)." -ForegroundColor Green

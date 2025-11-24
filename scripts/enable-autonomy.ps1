<#
Enables autonomous/autorun behavior for VS Code–compatible IDEs (VS Code, Trae, Windsurf) by
updating each IDE's user settings.json.

- Backs up settings.json to settings.json.backup-YYYYMMDD-HHmmss.json
- Sets Copilot inline suggestions and common assistant auto-apply flags
- Optionally adds a user keybinding to run the workspace task "Approve Next 50 (Auto-Approve)"

Usage:
  pwsh scripts/enable-autonomy.ps1
  pwsh scripts/enable-autonomy.ps1 -AddKeybindings

Notes:
- This rewrites JSON without comments. VS Code supports JSON with comments, but backups are made.
- Only settings files found under %APPDATA%/*/User/settings.json will be edited.
#>
[CmdletBinding()]
param(
  [switch]$AddKeybindings
)

function Remove-JsonComments {
  param([string]$JsonText)
  $noBlock = [System.Text.RegularExpressions.Regex]::Replace($JsonText, "/\*.*?\*/", '', 'Singleline')
  $noLine = [System.Text.RegularExpressions.Regex]::Replace($noBlock, "(?m)//.*$", '')
  return $noLine
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
      $cursor[$key] = $Value
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
  # Common VS Code family names
  $ideNames = @('Code', 'Code - Insiders', 'VSCodium', 'Trae', 'Windsurf', 'Cursor')
  foreach ($name in $ideNames) {
    $p = Join-Path $base $name
    $userDir = Join-Path $p 'User'
    $settings = Join-Path $userDir 'settings.json'
    if (Test-Path $settings) { $targets += $settings }
  }
  # Fallback: pattern search under %APPDATA%/*/User/settings.json
  $pattern = Join-Path $base '*\User\settings.json'
  $extra = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue
  foreach ($file in $extra) {
    if (-not ($targets -contains $file.FullName)) { $targets += $file.FullName }
  }
  return $targets | Sort-Object -Unique
}

function Backup-File {
  param([string]$Path)
  $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
  $backup = "$Path.backup-$stamp.json"
  Copy-Item -LiteralPath $Path -Destination $backup -Force
  return $backup
}

function Update-UserSettingsFile {
  param([string]$SettingsPath)
  Write-Host "Updating: $SettingsPath" -ForegroundColor Cyan
  $raw = Get-Content -LiteralPath $SettingsPath -Raw -ErrorAction Stop
  $rawNoComments = Remove-JsonComments $raw
  $obj = $null
  try {
    $obj = ConvertFrom-Json -InputObject $rawNoComments -ErrorAction Stop
  } catch {
    throw "Failed to parse JSON: $SettingsPath. See backup for manual fix. Error: $($_.Exception.Message)"
  }
  if (-not ($obj -is [hashtable])) {
    $obj = @{} + $obj.PSObject.Properties | ForEach-Object { @{ ($_.Name) = $_.Value } }
  }
  if (-not ($obj -is [hashtable])) { $obj = @{} }

  # Autonomous/persona keys
  $desired = @{
    'editor.inlineSuggest.enabled'            = $true
    'github.copilot.enable'                   = $true
    'github.copilot.inlineSuggest.enable'     = $true
    'github.copilot.editor.enableAutoCompletions' = $true
    'github.copilot.suggestions.enable'       = $true
    'anthropic.claude-code.autoSubmit'        = $true
    'anthropic.claude-code.bypassConfirmations' = $true
    'cline.autonomous.enable'                 = $true
    'continue.autocomplete.enabled'           = $true
    'augment.vscode-augment.autoApply'        = $true
    'codeium.codeAssistant.enableAutoApply'   = $true
    # Personal auto-save preference
    'files.autoSave'                          = 'afterDelay'
    'files.autoSaveDelay'                     = 1000
  }
  foreach ($k in $desired.Keys) { Set-JsonValue -Root $obj -Path $k -Value $desired[$k] }

  $json = $obj | ConvertTo-Json -Depth 50
  $json | Set-Content -LiteralPath $SettingsPath -Encoding UTF8
}

function Ensure-UserKeybinding {
  param(
    [string]$KeybindingsPath,
    [string]$Command,
    [string]$Key,
    [object]$Args = $null,
    [string]$When = $null
  )
  $arr = @()
  if (Test-Path $KeybindingsPath) {
    $raw = Get-Content -LiteralPath $KeybindingsPath -Raw
    $rawNoComments = Remove-JsonComments $raw
    try { $arr = ConvertFrom-Json $rawNoComments } catch { $arr = @() }
  }
  if (-not ($arr -is [System.Collections.IList])) { $arr = @() }
  $exists = $false
  foreach ($item in $arr) {
    if ($item.command -eq $Command -and $item.key -eq $Key) { $exists = $true; break }
  }
  if (-not $exists) {
    $entry = [ordered]@{ key = $Key; command = $Command }
    if ($Args) { $entry.args = $Args }
    if ($When) { $entry.when = $When }
    $arr = @($arr + $entry)
    ($arr | ConvertTo-Json -Depth 20) | Set-Content -LiteralPath $KeybindingsPath -Encoding UTF8
  }
}

$targets = Get-IdeSettingsPaths
if (-not $targets -or $targets.Count -eq 0) {
  Write-Warning "No IDE user settings.json files found under %APPDATA%. Nothing to do."
  exit 0
}

foreach ($settings in $targets) {
  try {
    $backup = Backup-File -Path $settings
    Write-Host "Backup created: $backup" -ForegroundColor DarkGray
    Update-UserSettingsFile -SettingsPath $settings
    if ($AddKeybindings) {
      $userDir = Split-Path -Parent $settings
      $kb = Join-Path $userDir 'keybindings.json'
      Ensure-UserKeybinding -KeybindingsPath $kb -Command 'workbench.action.tasks.runTask' -Key 'ctrl+alt+a' -Args 'Approve Next 50 (Auto-Approve)'
    }
  } catch {
    Write-Error $_
  }
}

Write-Host "Autonomy enabled for $(($targets | Measure-Object).Count) IDE profile(s)." -ForegroundColor Green

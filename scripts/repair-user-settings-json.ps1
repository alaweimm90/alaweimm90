<#
Attempts to repair common JSON issues in IDE user settings.json files so they can be parsed:
- Removes JSON comments
- Removes problematic properties with multi-line/unescaped content (kilo-code.allowedCommands, github.copilot.chat.codeGeneration.instructions)
- Removes stray $schema lines
- Fixes trailing commas before } or ]
Creates a timestamped backup before changes.
#>
[CmdletBinding()]
param()

function Remove-JsonComments([string]$t){ $a=[System.Text.RegularExpressions.Regex]::Replace($t, "/\*.*?\*/", '', 'Singleline'); [System.Text.RegularExpressions.Regex]::Replace($a, "(?m)//.*$", '') }

function Get-IdeSettingsPaths {
  $base = $env:APPDATA
  $targets = @()
  $ideNames = @('Code', 'Code - Insiders', 'VSCodium', 'Trae', 'Windsurf', 'Cursor', 'BLACKBOXAI')
  foreach ($name in $ideNames) {
    $p = Join-Path $base $name
    $userDir = Join-Path $p 'User'
    $settings = Join-Path $userDir 'settings.json'
    if (Test-Path $settings) { $targets += $settings }
  }
  return $targets | Sort-Object -Unique
}

function Backup-File {
  param([string]$Path)
  $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
  $backup = "$Path.repair-backup-$stamp.json"
  Copy-Item -LiteralPath $Path -Destination $backup -Force
  return $backup
}

function Repair-JsonText([string]$text){
  $t = $text
  # Remove comments first
  $t = Remove-JsonComments $t
  # Remove multi-line problematic arrays/properties
  $t = [System.Text.RegularExpressions.Regex]::Replace($t, '"kilo-code\.allowedCommands"\s*:\s*\[(?:.|\r|\n)*?\]\s*,?', '', 'Singleline')
  $t = [System.Text.RegularExpressions.Regex]::Replace($t, '"github\.copilot\.chat\.codeGeneration\.instructions"\s*:\s*\[(?:.|\r|\n)*?\]\s*,?', '', 'Singleline')
  # Remove $schema entry if present
  $t = [System.Text.RegularExpressions.Regex]::Replace($t, '"\$schema"\s*:\s*".*?"\s*,?', '', 'Singleline')
  # Fix trailing commas before } or ]
  $t = [System.Text.RegularExpressions.Regex]::Replace($t, ',\s*([\}\]])', '$1')
  # Normalize CRLF
  return $t -replace "\r?\n","`r`n"
}

$targets = Get-IdeSettingsPaths
if (-not $targets -or $targets.Count -eq 0) { Write-Host "No user settings found."; exit 0 }

foreach ($p in $targets) {
  try {
    Write-Host "Repairing: $p" -ForegroundColor Cyan
    $backup = Backup-File -Path $p
    Write-Host "Backup: $backup" -ForegroundColor DarkGray
    $raw = Get-Content -LiteralPath $p -Raw
    $fixed = Repair-JsonText $raw
    # Validate
    try {
      $obj = ConvertFrom-Json -InputObject $fixed -ErrorAction Stop
    } catch {
      Write-Warning "Still invalid after heuristics: $p -> manual edit required. Skipping write."
      continue
    }
    ($obj | ConvertTo-Json -Depth 50) | Set-Content -LiteralPath $p -Encoding UTF8
    Write-Host "✅ Repaired and saved: $p" -ForegroundColor Green
  } catch {
    Write-Error $_
  }
}

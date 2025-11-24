<#
Robustly remove problematic properties from IDE user settings.json that cause JSON parse failures.
Targets: kilo-code.allowedCommands, github.copilot.chat.codeGeneration.instructions, $schema, augment.chat.userGuidelines
#>
[CmdletBinding()]
param()

function Remove-JsonComments([string]$t){ $a=[System.Text.RegularExpressions.Regex]::Replace($t, "/\*.*?\*/", '', 'Singleline'); [System.Text.RegularExpressions.Regex]::Replace($a, "(?m)//.*$", '') }

function Escape-Regex([string]$s){ [Regex]::Escape($s) }

function Remove-PropertyBlock {
  param(
    [string]$Text,
    [string]$PropertyName
  )
  $pattern = '"' + (Escape-Regex $PropertyName) + '"\s*:\s*'
  $output = $Text
  while ($true) {
    $m = [Regex]::Match($output, $pattern, [System.Text.RegularExpressions.RegexOptions]::Singleline)
    if (-not $m.Success) { break }
    $start = $m.Index
    $valueStart = $m.Index + $m.Length
    if ($valueStart -ge $output.Length) { break }
    $c = $output[$valueStart]
    $i = $valueStart
    $inString = $false; $esc = $false; $depth = 0
    if ($c -eq '[' -or $c -eq '{') {
      $open = $c; $close = ($c -eq '[') ? ']' : '}'
      while ($i -lt $output.Length) {
        $ch = $output[$i]
        if ($inString) {
          if ($esc) { $esc = $false }
          elseif ($ch -eq '\\') { $esc = $true }
          elseif ($ch -eq '"') { $inString = $false }
        } else {
          if ($ch -eq '"') { $inString = $true }
          elseif ($ch -eq $open) { $depth++ }
          elseif ($ch -eq $close) { $depth--; if ($depth -le 0) { $i++; break } }
        }
        $i++
      }
    } else {
      # Primitive value; advance until comma or closing brace not in string
      while ($i -lt $output.Length) {
        $ch = $output[$i]
        if ($inString) {
          if ($esc) { $esc = $false }
          elseif ($ch -eq '\\') { $esc = $true }
          elseif ($ch -eq '"') { $inString = $false }
        } else {
          if ($ch -eq '"') { $inString = $true }
          elseif ($ch -eq ',' -or $ch -eq '}') { break }
        }
        $i++
      }
    }
    $valueEnd = $i
    # Include trailing whitespace and optional trailing comma
    $j = $valueEnd
    while ($j -lt $output.Length -and [char]::IsWhiteSpace($output[$j])) { $j++ }
    if ($j -lt $output.Length -and $output[$j] -eq ',') { $j++ }
    $output = $output.Remove($start, $j - $start)
  }
  return $output
}

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
  $backup = "$Path.hardclean-backup-$stamp.json"
  Copy-Item -LiteralPath $Path -Destination $backup -Force
  return $backup
}

$targets = Get-IdeSettingsPaths
if (-not $targets -or $targets.Count -eq 0) { Write-Host "No user settings found."; exit 0 }

$propsToRemove = @(
  'kilo-code.allowedCommands',
  'github.copilot.chat.codeGeneration.instructions',
  '$schema',
  'augment.chat.userGuidelines'
)

foreach ($p in $targets) {
  try {
    Write-Host "Cleaning: $p" -ForegroundColor Cyan
    $backup = Backup-File -Path $p
    Write-Host "Backup: $backup" -ForegroundColor DarkGray
    $raw = Get-Content -LiteralPath $p -Raw
    $text = Remove-JsonComments $raw
    foreach ($prop in $propsToRemove) { $text = Remove-PropertyBlock -Text $text -PropertyName $prop }
    # Remove trailing commas before } or ] after deletions
    $text = [Regex]::Replace($text, ',\s*([\}\]])', '$1')
    # Validate
    try {
      $obj = ConvertFrom-Json -InputObject $text -ErrorAction Stop
    } catch {
      Write-Warning "Still invalid after hard clean: $p -> manual edit required."
      continue
    }
    ($obj | ConvertTo-Json -Depth 60) | Set-Content -LiteralPath $p -Encoding UTF8
    Write-Host "✅ Clean and valid: $p" -ForegroundColor Green
  } catch {
    Write-Error $_
  }
}

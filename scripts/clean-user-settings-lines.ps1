[CmdletBinding()]
param()

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
  $backup = "$Path.lines-backup-$stamp.json"
  Copy-Item -LiteralPath $Path -Destination $backup -Force
  return $backup
}

function Remove-ArrayPropertyLines {
  param(
    [string[]]$Lines,
    [string]$PropertyName
  )
  $pattern = '"' + [Regex]::Escape($PropertyName) + '"\s*:\s*\['
  $i = 0
  while ($i -lt $Lines.Length) {
    if ($Lines[$i] -match $pattern) {
      $start = $i
      # advance until a line that looks like end of array: ] or ], possibly with spaces
      $j = $i
      while ($j -lt $Lines.Length -and ($Lines[$j] -notmatch '^\s*\],?\s*$')) { $j++ }
      if ($j -lt $Lines.Length) {
        # Include the closing line and possible trailing comma on that line is handled by JSON parser later
        $end = $j
        # Also remove a trailing comma on the next line if it exists isolated
        # Remove from start..end
        $new = @()
        if ($start -gt 0) { $new += $Lines[0..($start-1)] }
        if ($end + 1 -le $Lines.Length - 1) { $new += $Lines[($end+1)..($Lines.Length-1)] }
        $Lines = $new
        # Restart scan from beginning in case multiple occurrences exist
        $i = 0
        continue
      } else {
        break
      }
    }
    $i++
  }
  return ,$Lines
}

function Remove-LineProperty {
  param(
    [string[]]$Lines,
    [string]$PropertyName
  )
  $pattern = '"' + [Regex]::Escape($PropertyName) + '"\s*:'
  $out = New-Object System.Collections.Generic.List[string]
  for ($i=0; $i -lt $Lines.Length; $i++) {
    $line = $Lines[$i]
    if ($line -match $pattern) {
      # Drop this line; also if the next line is only a comma, drop it
      if ($i + 1 -lt $Lines.Length -and $Lines[$i+1] -match '^\s*,\s*$') { $i++ }
      continue
    }
    $out.Add($line)
  }
  return ,$out.ToArray()
}

$targets = Get-IdeSettingsPaths
foreach ($p in $targets) {
  try {
    Write-Host "Line-cleaning: $p" -ForegroundColor Cyan
    $backup = Backup-File -Path $p
    Write-Host "Backup: $backup" -ForegroundColor DarkGray
    $lines = Get-Content -LiteralPath $p
    $lines = Remove-ArrayPropertyLines -Lines $lines -PropertyName 'kilo-code.allowedCommands'
    $lines = Remove-ArrayPropertyLines -Lines $lines -PropertyName 'github.copilot.chat.codeGeneration.instructions'
    $lines = Remove-LineProperty -Lines $lines -PropertyName '$schema'
    $lines = Remove-LineProperty -Lines $lines -PropertyName 'augment.chat.userGuidelines'
    # Write back and attempt to validate
    $tmp = [System.IO.Path]::GetTempFileName()
    $lines | Set-Content -LiteralPath $tmp -Encoding UTF8
    try {
      $raw = Get-Content -LiteralPath $tmp -Raw
      $obj = ConvertFrom-Json -InputObject $raw -ErrorAction Stop
      ($obj | ConvertTo-Json -Depth 60) | Set-Content -LiteralPath $p -Encoding UTF8
      Remove-Item $tmp -Force
      Write-Host "✅ Clean and valid: $p" -ForegroundColor Green
    } catch {
      Write-Warning "Still invalid after line clean: $p -> manual edit may be required."
      Move-Item -LiteralPath $tmp -Destination "$p.after-lines-attempt.json" -Force
    }
  } catch { Write-Error $_ }
}

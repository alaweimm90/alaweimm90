param([switch]$DryRun,[switch]$FixTitles,[switch]$UpdateIndex,[switch]$RemoveEmpty)
$root = (Resolve-Path (Join-Path $PSScriptRoot ".." )).Path
$docs = Join-Path $root "docs"
if(-not (Test-Path $docs)){ exit 0 }
$files = Get-ChildItem -Path $docs -Filter "*.md" -File
function ToTitleCase([string]$s){
  $name = $s -replace "[-_]+"," "
  $name = $name.Trim()
  $words = ($name -split " ") | ForEach-Object { if($_.Length -gt 0){ $_.Substring(0,1).ToUpper() + $_.Substring(1).ToLower() } }
  ($words -join " ")
}
$index = @()
foreach($f in $files){
  $content = Get-Content -Path $f.FullName -Raw
  $lines = $content -split "`r?`n"
  $first = if($lines.Length -gt 0){ $lines[0] } else { "" }
  $base = [IO.Path]::GetFileNameWithoutExtension($f.Name)
  $titleFromFile = ToTitleCase($base)
  $changed = $false
  if($FixTitles){
    if($first -match '^#\s'){ if(($first -replace '^#\s','') -ne $titleFromFile){ $lines[0] = "# " + $titleFromFile; $changed = $true } }
    else { $lines = @("# " + $titleFromFile) + $lines; $changed = $true }
  }
  if($RemoveEmpty){ $lines = $lines | Where-Object { $_ -notmatch '^\s*$' -or $_ -match '^#\s' } }
  if($changed -and -not $DryRun){ ($lines -join "`n") | Set-Content -Path $f.FullName -Encoding UTF8 }
  $index += @{ name = $f.Name; path = $f.FullName; title = $titleFromFile }
}
if($UpdateIndex){
  $ordered = $index | Sort-Object -Property title
  $lines = @()
  $lines += "# Documentation Index"
  $lines += ""
  foreach($i in $ordered){
    $rel = [IO.Path]::GetFileName($i.path)
    $t = $i.title
    $lines += ("- [" + $t + "](" + $rel + ")")
  }
  if(-not $DryRun){ ($lines -join "`n") | Set-Content -Path (Join-Path $docs "README.md") -Encoding UTF8 }
}

param([string]$LogDir,[switch]$Markdown,[switch]$Open,[string]$OutputPath,[string]$Start,[string]$End,[string]$Keyword,[switch]$ByCategory,[int]$TopN = 10,[switch]$GroupByDay)
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if(-not $LogDir){ $LogDir = Join-Path $repoRoot ".cache/ide-wrapper" }
$files = Get-ChildItem -Path $LogDir -Filter "decisions-*.log" -File -ErrorAction SilentlyContinue
if($Start){ try{ $startDt = [DateTime]::Parse($Start) } catch{ $startDt = $null } } else { $startDt = $null }
if($End){ try{ $endDt = [DateTime]::Parse($End) } catch{ $endDt = $null } } else { $endDt = $null }
if($startDt -or $endDt){
  $files = $files | Where-Object {
    $name = $_.Name
    $dt = $null
    if($name -match '^decisions-(\d{8})-(\d{6})\.log$'){
      $ds = $Matches[1] + '-' + $Matches[2]
      try { $dt = [DateTime]::ParseExact($ds,'yyyyMMdd-HHmmss',$null) } catch { $dt = $null }
    }
    if(-not $dt){ $dt = $_.LastWriteTime }
    if($startDt -and $dt -lt $startDt){ return $false }
    if($endDt -and $dt -gt $endDt){ return $false }
    return $true
  }
}
$summary = @{}
function GetCategory([string]$p){
  if($p -match '(?i)overwrite'){ return 'overwrite' }
  if($p -match '(?i)continue'){ return 'continue' }
  if($p -match '(?i)install|dependency'){ return 'install' }
  if($p -match '(?i)delete|remove'){ return 'delete' }
  if($p -match '(?i)deploy|deployment|proceed'){ return 'deployment' }
  return 'other'
}
$catSummary = @{}
$daySummary = @{}
foreach($f in $files){
  $lines = Get-Content -Path $f -ErrorAction SilentlyContinue
  $fileDate = $f.LastWriteTime.ToString('yyyy-MM-dd')
  for($i=0; $i -lt $lines.Count; $i++){
    if($lines[$i] -like "PROMPT: *"){ $p = $lines[$i].Substring(8).Trim(); $r = ""; if($i + 1 -lt $lines.Count -and $lines[$i+1] -like "RESPONSE: *"){ $r = $lines[$i+1].Substring(10).Trim() }; if($Keyword -and -not ($p -like ("*" + $Keyword + "*"))){ continue }; if(-not $summary.ContainsKey($p)){ $summary[$p] = @{ count = 0; responses = @{} } }; $summary[$p].count += 1; if(-not $summary[$p].responses.ContainsKey($r)){ $summary[$p].responses[$r] = 0 }; $summary[$p].responses[$r] += 1; $cat = GetCategory $p; if(-not $catSummary.ContainsKey($cat)){ $catSummary[$cat] = 0 }; $catSummary[$cat] += 1; if(-not $daySummary.ContainsKey($fileDate)){ $daySummary[$fileDate] = 0 }; $daySummary[$fileDate] += 1 }
  }
}
if(-not $Markdown){
  $out = @()
  foreach($k in $summary.Keys){ $out += [PSCustomObject]@{ prompt = $k; count = $summary[$k].count; responses = ($summary[$k].responses | ConvertTo-Json -Compress) } }
  $out = $out | Sort-Object -Property count -Descending
  if($TopN -gt 0){ $out = $out | Select-Object -First $TopN }
  $out | Format-Table -AutoSize | Out-String | Write-Output
  if($ByCategory){
    Write-Host ""
    Write-Host "By Category"
    $catOut = @()
    foreach($c in $catSummary.Keys){ $catOut += [PSCustomObject]@{ category = $c; count = $catSummary[$c] } }
    $catOut | Sort-Object -Property count -Descending | Format-Table -AutoSize | Out-String | Write-Output
  }
  if($GroupByDay){
    Write-Host ""
    Write-Host "By Day"
    $dayOut = @()
    foreach($d in $daySummary.Keys){ $dayOut += [PSCustomObject]@{ day = $d; count = $daySummary[$d] } }
    $dayOut | Sort-Object -Property day | Format-Table -AutoSize | Out-String | Write-Output
  }
} else {
  $ts = Get-Date -Format "yyyyMMdd-HHmmss"
  if(-not $OutputPath){ $OutputPath = Join-Path $LogDir ("report-" + $ts + ".md") }
  $lines = @()
  $lines += "# IDE Wrapper Decisions"
  $lines += ("Generated: " + (Get-Date).ToString("o"))
  $lines += ""
  $lines += "| Prompt | Count | Responses |"
  $lines += "|--------|-------|-----------|"
  $ordered = ($summary.Keys | Sort-Object -Property { $summary[$_].count } -Descending)
  if($TopN -gt 0){ $ordered = $ordered | Select-Object -First $TopN }
  foreach($k in $ordered){
    $cnt = $summary[$k].count
    $resp = ($summary[$k].responses | ConvertTo-Json -Compress)
    $safePrompt = ($k -replace "\|","\\|")
    $lines += ("| " + $safePrompt + " | " + $cnt + " | " + ($resp -replace "\|","\\|") + " |")
  }
  if($ByCategory){
    $lines += ""
    $lines += "## By Category"
    $lines += "| Category | Count |"
    $lines += "|----------|-------|"
    foreach($c in ($catSummary.Keys | Sort-Object -Property { $catSummary[$_] } -Descending)){
      $lines += ("| " + $c + " | " + $catSummary[$c] + " |")
    }
  }
  if($GroupByDay){
    $lines += ""
    $lines += "## By Day"
    $lines += "| Day | Count |"
    $lines += "|-----|-------|"
    foreach($d in ($daySummary.Keys | Sort-Object)){
      $lines += ("| " + $d + " | " + $daySummary[$d] + " |")
    }
  }
  $content = ($lines -join "`n")
  $content | Set-Content -Path $OutputPath -Encoding UTF8
  Write-Host "Report:" $OutputPath
  $codeCmd = Get-Command code -ErrorAction SilentlyContinue
  if($Open){ if($codeCmd){ try { & code $OutputPath } catch {} } else { try { Start-Process $OutputPath } catch {} } }
}

param(
  [string]$Command,
  [string]$RulesPath,
  [switch]$VerboseLog,
  [int]$ApproveNext = 0
)
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if(-not $RulesPath){ $RulesPath = Join-Path $repoRoot ".config/ide-wrapper.json" }
$cache = Join-Path $repoRoot ".cache"
$logDir = Join-Path $cache "ide-wrapper"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$ts = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $logDir ("session-" + $ts + ".log")
$decisionLogPath = Join-Path $logDir ("decisions-" + $ts + ".log")
$cfg = $null
try { if(Test-Path $RulesPath){ $cfg = Get-Content -Raw -Path $RulesPath | ConvertFrom-Json } } catch { $cfg = $null }
$rules = if($cfg -and $cfg.rules){ $cfg.rules } else { @() }
Start-Transcript -Path $logPath -Force | Out-Null
$global:ConfirmPreference = 'None'
$global:__ApproveRemaining = $ApproveNext
function ApplyForceFlags([string]$cmd){
  $map = if($cfg -and $cfg.forceFlags){ $cfg.forceFlags } else { @() }
  foreach($m in $map){
    $pat = [string]$m.pattern
    $flag = [string]$m.flag
    if($cmd -match $pat){ if($cmd -notmatch [Regex]::Escape($flag)){ $cmd = $cmd + " " + $flag } }
  }
  return $cmd
}
if($Command){ $Command = ApplyForceFlags $Command }
function Read-Host{
  param([string]$Prompt)
  $resp = 'y'
  $forced = $false
  if($global:__ApproveRemaining -gt 0){
    $resp = 'y'
    $global:__ApproveRemaining = [int]$global:__ApproveRemaining - 1
    $forced = $true
  } else {
    foreach($r in $rules){
      $pat = [string]$r.pattern
      if($Prompt -match $pat){
        if([string]$r.response){ $resp = [string]$r.response }
        elseif([string]$r.action -eq 'accept'){ $resp = 'y' }
        elseif([string]$r.action -eq 'reject'){ $resp = 'n' }
        else { $resp = 'y' }
        break
      }
    }
  }
  try {
    $prefsPath = Join-Path $repoRoot ".config/knowledge/prefs.json"
    $prefs = $null
    try { if(Test-Path $prefsPath){ $prefs = Get-Content -Raw -Path $prefsPath | ConvertFrom-Json } } catch { $prefs = $null }
    if(-not $prefs){ $prefs = @{ patterns = @{} } }
    $key = ($Prompt -replace "\s+"," ")
    $patterns = $prefs.patterns
    if($patterns -isnot [hashtable]){ $ht = @{}; foreach($prop in ($patterns.PSObject.Properties)){ $ht[$prop.Name] = $prop.Value }; $patterns = $ht }
    if(-not $patterns.ContainsKey($key)){ $patterns[$key] = @{ count = 0; last = $resp } }
    $patterns[$key].count = [int]$patterns[$key].count + 1
    $patterns[$key].last = $resp
    $prefs.patterns = $patterns
    ($prefs | ConvertTo-Json -Depth 8) | Set-Content -Path $prefsPath -Encoding UTF8
    $learnThreshold = 2
    if($prefs.patterns[$key].count -ge $learnThreshold){
      $cfgPath = $RulesPath
      $cfgContent = $null
      try { if(Test-Path $cfgPath){ $cfgContent = Get-Content -Raw -Path $cfgPath | ConvertFrom-Json } } catch { $cfgContent = $null }
      if(-not $cfgContent){ $cfgContent = @{ rules = @(); forceFlags = @() } }
      $exists = $false
      foreach($rr in $cfgContent.rules){ if([string]$rr.pattern -eq $key){ $exists = $true; break } }
      if(-not $exists){
        $newRule = @{ pattern = $key; action = (if($resp -eq 'y' -or $resp -eq 'yes'){ 'accept' } else { 'reject' }); response = $resp }
        $cfgContent.rules += $newRule
        ($cfgContent | ConvertTo-Json -Depth 8) | Set-Content -Path $cfgPath -Encoding UTF8
      }
    }
  } catch {}
  ("PROMPT: " + $Prompt) | Add-Content -Path $decisionLogPath -Encoding UTF8
  ("RESPONSE: " + $resp) | Add-Content -Path $decisionLogPath -Encoding UTF8
  if($forced){ ("FORCED: true") | Add-Content -Path $decisionLogPath -Encoding UTF8 }
  return $resp
}
if(Test-Path $Command){ & $Command } else { Invoke-Expression $Command }
Stop-Transcript | Out-Null
Write-Host "Log:" $logPath
Write-Host "Decisions:" $decisionLogPath
try {
  $prefsPath = Join-Path $repoRoot ".config/knowledge/prefs.json"
  if(Test-Path $prefsPath){
    $prefs = Get-Content -Raw -Path $prefsPath | ConvertFrom-Json
    $patterns = $prefs.patterns
    if($patterns -isnot [hashtable]){ $ht = @{}; foreach($prop in ($patterns.PSObject.Properties)){ $ht[$prop.Name] = $prop.Value }; $patterns = $ht }
    $cfgPath = $RulesPath
    $cfgContent = $null
    if(Test-Path $cfgPath){ $cfgContent = Get-Content -Raw -Path $cfgPath | ConvertFrom-Json } else { $cfgContent = @{ rules = @(); forceFlags = @() } }
    if(-not $cfgContent.rules){ $cfgContent.rules = @() }
    foreach($k in $patterns.Keys){
      $entry = $patterns[$k]
      if([int]$entry.count -ge 2){
        $exists = $false
        foreach($rr in $cfgContent.rules){ if([string]$rr.pattern -eq $k){ $exists = $true; break } }
        if(-not $exists){
          $resp = [string]$entry.last
          $act = ($resp -eq 'y' -or $resp -eq 'yes') ? 'accept' : 'reject'
          $cfgContent.rules += @{ pattern = $k; action = $act; response = $resp }
        }
      }
    }
    ($cfgContent | ConvertTo-Json -Depth 8) | Set-Content -Path $cfgPath -Encoding UTF8
  }
} catch {}

param(
  [string]$Prompt,
  [int]$Rounds=3,
  [string]$ClaudeCmd,
  [string]$Mode,
  [switch]$DryRun
)
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$cache = Join-Path $repoRoot ".cache"
$logDir = Join-Path $cache "llm-bridge"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$ts = Get-Date -Format "yyyyMMdd-HHmmss"
$transcript = Join-Path $logDir ("claude-" + $ts + ".md")
function WriteLog($text){ $text | Add-Content -Path $transcript -Encoding UTF8 }
$cfgPath = Join-Path $repoRoot ".config/llm-bridge.json"
$cfg = $null
try { if(Test-Path $cfgPath){ $cfg = Get-Content -Raw -Path $cfgPath | ConvertFrom-Json } } catch { $cfg = $null }
if(-not $ClaudeCmd){ $ClaudeCmd = if($cfg){ $cfg.claudeCmd } else { "claude" } }
if(-not $Mode){ $Mode = if($cfg){ $cfg.mode } else { "argue" } }
if($Rounds -lt 1){ $Rounds = if($cfg){ [int]$cfg.rounds } else { 3 } }
if($cfg -and $cfg.agents){ $agents = $cfg.agents } else { $agents = @(@{ name = "claude"; cmd = $ClaudeCmd }) }
$cmd = Get-Command $ClaudeCmd -ErrorAction SilentlyContinue
if($DryRun){ Write-Host "DryRun"; Write-Host "Prompt:"; Write-Host $Prompt; Write-Host "Rounds:" $Rounds; Write-Host "Mode:" $Mode; exit 0 }
$current = $Prompt
for($i=1; $i -le $Rounds; $i++){
  $responses = @()
  foreach($a in $agents){
    $acmd = [string]$a.cmd
    $exists = Get-Command $acmd -ErrorAction SilentlyContinue
    if(-not $exists){ continue }
    $out = $null
    try { $out = & $acmd -p $current } catch { }
    if(-not $out){ try { $out = & $acmd $current } catch { $out = "" } }
    $responses += [PSCustomObject]@{ name = $a.name; text = [string]$out }
  }
  WriteLog ("# Round $i")
  WriteLog ("## Prompt")
  WriteLog $current
  foreach($r in $responses){ WriteLog ("## Response:" + $r.name); WriteLog ($r.text) }
  $aggregate = ($responses | ForEach-Object { $_.text }) -join "`n---`n"
  if($Mode -eq "argue"){
    $current = "Counter-argue concisely and provide actionable corrections for: `n`n" + $aggregate
  } elseif($Mode -eq "qa"){
    $current = "Answer directly and cite assumptions for: `n`n" + $aggregate
  } else {
    $current = "Refine and improve clarity for: `n`n" + $aggregate
  }
}
Write-Host "Transcript:" $transcript

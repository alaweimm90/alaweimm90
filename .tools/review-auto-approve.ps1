param([int]$Max=100)
$root = (Resolve-Path (Join-Path $PSScriptRoot ".." )).Path
function IsCritical($p){
  $crit = @('src/coaching-api','packages/agent-core','packages/workflow-templates','.config','.github')
  foreach($c in $crit){ if($p -like ("$root/" + $c + "*")){ return $true } }
  return $false
}
$changed = & git ls-files -m
$approved = @()
$manual = @()
foreach($f in $changed){
  $full = Join-Path $root $f
  if(IsCritical($full)){ $manual += $f } else { if($approved.Count -lt $Max){ $approved += $f } else { $manual += $f } }
}
$outDir = Join-Path $root ".cache/review-auto-approve"
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
($approved | ConvertTo-Json -Depth 4) | Set-Content -Path (Join-Path $outDir "approved.json") -Encoding UTF8
($manual   | ConvertTo-Json -Depth 4) | Set-Content -Path (Join-Path $outDir "manual.json") -Encoding UTF8
Write-Host ("Approved:" + $approved.Count)
Write-Host ("Manual:" + $manual.Count)

param([int]$Threshold=95)
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$expectedHidden = @(".tools",".config",".cache")
$allowedRoot = @(".github",".git")
$defaultMoved = @(
  ".automation",
  ".dev-tools",
  ".vscode",
  ".meta",
  ".metaHub",
  ".husky",
  ".governance",
  ".organizations",
  ".archives",
  ".turbo",
  ".claude",
  ".devcontainer",
  ".idea",
  ".cursor",
  ".continue",
  ".windsurf",
  ".amazonq",
  ".copilot"
)
$mapPath = Join-Path $repoRoot ".config/consolidation-map.json"
$moved = if(Test-Path $mapPath){ try { (Get-Content -Raw -Path $mapPath | ConvertFrom-Json | ForEach-Object { $_.old }) } catch { $defaultMoved } } else { $defaultMoved }
$validatorCfgPath = Join-Path $repoRoot ".config/validator.json"
$validatorCfg = if(Test-Path $validatorCfgPath){ try { Get-Content -Raw -Path $validatorCfgPath | ConvertFrom-Json } catch { $null } } else { $null }
$allowedRoot = if($validatorCfg -and $validatorCfg.allowedHidden){ $validatorCfg.allowedHidden } else { $allowedRoot }
$presentHidden = Get-ChildItem -Path $repoRoot -Directory -Force | Where-Object { $_.Name -like ".*" } | Select-Object -ExpandProperty Name
$score = 0
$checks = @()
foreach($d in $expectedHidden){
  if(Test-Path (Join-Path $repoRoot $d)){
    $score += 1
    $checks += [PSCustomObject]@{ check = "exists:$d"; ok = $true }
  } else {
    $checks += [PSCustomObject]@{ check = "exists:$d"; ok = $false }
  }
}
$movedPresent = @()
foreach($d in $moved){ if(Test-Path (Join-Path $repoRoot $d)){ $movedPresent += $d } }
foreach($d in $movedPresent){
  $old = Join-Path $repoRoot $d
  $isLink = $false
  try { $isLink = (Get-Item $old -Force).LinkType -ne $null } catch {}
  if($isLink){ $score += 1; $checks += [PSCustomObject]@{ check = "symlink:$d"; ok = $true } } else { $checks += [PSCustomObject]@{ check = "symlink:$d"; ok = $false } }
}
$extraHidden = @()
foreach($name in $presentHidden){
  $isAllowed = $false
  foreach($pattern in $allowedRoot){ if($name -like $pattern){ $isAllowed = $true; break } }
  if(($expectedHidden -notcontains $name) -and (-not $isAllowed) -and ($moved -notcontains $name)){
    $extraHidden += $name
  }
}
$totalChecks = ($expectedHidden.Count + $movedPresent.Count)
$compliance = if($totalChecks -gt 0){ [math]::Round((100.0 * $score) / $totalChecks,2) } else { 0 }
$result = [PSCustomObject]@{ compliance = $compliance; checks = $checks; extraHidden = $extraHidden; threshold = $Threshold; pass = ($compliance -ge $Threshold) }
$result | ConvertTo-Json -Depth 5

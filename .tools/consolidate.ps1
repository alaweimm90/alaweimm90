param([switch]$Rollback,[switch]$DryRun,[string]$MapPath)
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$tools = Join-Path $repoRoot ".tools"
$config = Join-Path $repoRoot ".config"
$cache = Join-Path $repoRoot ".cache"
New-Item -ItemType Directory -Force -Path $tools | Out-Null
New-Item -ItemType Directory -Force -Path $config | Out-Null
New-Item -ItemType Directory -Force -Path $cache | Out-Null
$manifestPath = Join-Path $cache "consolidation-manifest.json"
function SaveManifest($items){ $items | ConvertTo-Json -Depth 5 | Set-Content -Path $manifestPath -Encoding UTF8 }
function LoadManifest(){ if(Test-Path $manifestPath){ Get-Content -Raw -Path $manifestPath | ConvertFrom-Json } else { @() } }
function NewLink($path,$target){ try { New-Item -ItemType SymbolicLink -Path $path -Target $target -Force | Out-Null } catch { try { cmd /c "mklink /J \"$path\" \"$target\"" | Out-Null } catch { } } }
$defaultTargets = @(
    @{ old = ".automation"; new = ".tools/automation" },
    @{ old = ".dev-tools"; new = ".tools/dev-tools" },
    @{ old = ".vscode"; new = ".config/vscode" },
    @{ old = ".meta"; new = ".config/meta" },
    @{ old = ".metaHub"; new = ".config/metaHub" },
    @{ old = ".husky"; new = ".tools/husky" },
    @{ old = ".governance"; new = ".config/governance" },
    @{ old = ".organizations"; new = ".config/organizations" },
    @{ old = ".archives"; new = ".config/archives" },
    @{ old = ".turbo"; new = ".cache/turbo" },
    @{ old = ".claude"; new = ".config/claude" }
)
if(-not $MapPath){ $MapPath = Join-Path $repoRoot ".config/consolidation-map.json" }
$targets = if(Test-Path $MapPath){ try { Get-Content -Raw -Path $MapPath | ConvertFrom-Json } catch { $defaultTargets } } else { $defaultTargets }
if($Rollback){
  $m = LoadManifest
  foreach($entry in $m){
    $oldPath = Join-Path $repoRoot $entry.old
    $newPath = Join-Path $repoRoot $entry.new
    try { if(Test-Path $oldPath){ Remove-Item $oldPath -Force -Recurse } } catch {}
    try { if(Test-Path $newPath){ Move-Item $newPath $oldPath -Force } } catch {}
  }
  try { Remove-Item $manifestPath -Force } catch {}
  exit 0
}
$planned = @()
$ts = Get-Date -Format "yyyyMMdd-HHmmss"
$backupRoot = Join-Path $cache ("backups-" + $ts)
New-Item -ItemType Directory -Force -Path $backupRoot | Out-Null
$manifest = @()
foreach($t in $targets){
  $oldPath = Join-Path $repoRoot $t.old
  if(Test-Path $oldPath){
    $newPath = Join-Path $repoRoot $t.new
    $newDirParent = Split-Path -Parent $newPath
    New-Item -ItemType Directory -Force -Path $newDirParent | Out-Null
    $backupPath = Join-Path $backupRoot ($t.old.TrimStart('.'))
    $isLink = $false
    try { $isLink = (Get-Item $oldPath -Force).LinkType -ne $null } catch {}
    if($DryRun){
      $planned += [PSCustomObject]@{ old = $t.old; new = $t.new; backup = $backupPath; exists = $true }
    } else {
      if(-not $isLink){
        try { Copy-Item $oldPath $backupPath -Recurse -Force } catch {}
        try { Move-Item $oldPath $newPath -Force } catch {}
        NewLink $oldPath $newPath
      }
      $manifest += [PSCustomObject]@{ old = $t.old; new = $t.new; backup = (Resolve-Path $backupPath -ErrorAction SilentlyContinue).Path }
    }
  }
}
if($DryRun){ $planned | ConvertTo-Json -Depth 5 } else { SaveManifest $manifest }

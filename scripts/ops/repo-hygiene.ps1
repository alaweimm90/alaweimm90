Param(
  [switch]$DeleteEmpty,
  [switch]$DryRun
)

$allowed = @(
  '.config','ops','.github','.metaHub','.vscode','scripts','automation','docs','alaweimm90','.husky','.claude'
)

$root = Get-Location
$entries = Get-ChildItem -Path $root -Force
$report = @()

foreach ($e in $entries) {
  if ($e.PSIsContainer) {
    if ($allowed -notcontains $e.Name) {
      $report += @{ type='dir'; name=$e.Name; action='review' }
    }
    if ($DeleteEmpty) {
      $empty = -not (Get-ChildItem -Path $e.FullName -Force | Where-Object { $_.Name -ne '.gitkeep' })
      if ($empty) {
        if (-not $DryRun) { Remove-Item -Recurse -Force $e.FullName }
        $report += @{ type='dir'; name=$e.Name; action='deleted-empty' }
      }
    }
  } else {
    if ($e.Extension -in @('.log','.tmp','.bak')) {
      if (-not $DryRun) { Remove-Item -Force $e.FullName }
      $report += @{ type='file'; name=$e.Name; action='deleted-junk' }
    }
  }
}

$report | ForEach-Object { Write-Output ("{0} {1} {2}" -f $_.type, $_.name, $_.action) }

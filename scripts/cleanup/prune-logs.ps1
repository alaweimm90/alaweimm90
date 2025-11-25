Param(
  [string]$Root = (Get-Location),
  [int]$DaysToKeep = 14,
  [string[]]$IncludePatterns = @('*.log','*.tmp','*.bak'),
  [string[]]$AdditionalTargets = @(),
  [switch]$DryRun
)

$targets = @(
  "docs/reports/security",
  "docs/reports/health-check",
  ".metaHub/tools/automation/logs",
  "alaweimm90/automation/logs"
)
$targets += $AdditionalTargets

$deadline = (Get-Date).AddDays(-$DaysToKeep)
$deleted = @()
$candidates = @()

foreach ($rel in $targets) {
  $path = Join-Path $Root $rel
  if (-not (Test-Path $path)) { continue }

  Get-ChildItem -Path $path -Recurse -File -Include $IncludePatterns |
    Where-Object { $_.LastWriteTime -lt $deadline } |
    ForEach-Object {
      if ($DryRun) { $candidates += $_.FullName }
      else { Remove-Item -Force $_.FullName; $deleted += $_.FullName }
    }

  $auditPath = Join-Path $path "audit"
  if (Test-Path $auditPath) {
    Get-ChildItem -Path $auditPath -Recurse -File -Include *.json |
      Where-Object { $_.LastWriteTime -lt $deadline } |
      ForEach-Object {
        if ($DryRun) { $candidates += $_.FullName }
        else { Remove-Item -Force $_.FullName; $deleted += $_.FullName }
      }
  }
}

Write-Output ("Retention days: {0}" -f $DaysToKeep)
Write-Output ("Dry run: {0}" -f ([bool]$DryRun))
Write-Output ("Candidates: {0}" -f $candidates.Count)
Write-Output ("Deleted: {0}" -f $deleted.Count)

if ($DryRun -and $candidates.Count -gt 0) {
  $candidates | ForEach-Object { Write-Output $_ }
}

# Create MEZAN folder structure and copy core docs

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $root "..")

# Create directories
$paths = @(
  "MEZAN",
  "MEZAN\docs\branding",
  "MEZAN\docs\research",
  "MEZAN\docs\benchmarking",
  "MEZAN\docs\orchestration"
)
foreach ($p in $paths) {
  if (-not (Test-Path $p)) { New-Item -ItemType Directory -Path $p | Out-Null }
}

# Copy 4 core files
if (Test-Path "docs\MASTER_PROJECT_INDEX.md") { Copy-Item "docs\MASTER_PROJECT_INDEX.md" "MEZAN\MASTER_PROJECT_INDEX.md" -Force }
if (Test-Path "docs\FINAL_PROJECT_VALIDATION_REPORT.md") { Copy-Item "docs\FINAL_PROJECT_VALIDATION_REPORT.md" "MEZAN\FINAL_PROJECT_VALIDATION_REPORT.md" -Force }
if (Test-Path "docs\COMPREHENSIVE_PROJECT_HANDOFF_FOR_SIDER_AI.md") { Copy-Item "docs\COMPREHENSIVE_PROJECT_HANDOFF_FOR_SIDER_AI.md" "MEZAN\COMPREHENSIVE_PROJECT_HANDOFF_FOR_SIDER_AI.md" -Force }
if (Test-Path "FILE_MANIFEST.md") { Copy-Item "FILE_MANIFEST.md" "MEZAN\FILE_MANIFEST.md" -Force }

Write-Host "MEZAN structure created and core files copied."

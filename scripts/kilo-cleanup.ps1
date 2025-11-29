# KILO Radical Simplification Cleanup Script
# WARNING: This script will DELETE files. Ensure you have committed all changes to git first.

param(
    [switch]$DryRun = $false,
    [switch]$Force = $false
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }
function Write-Info { Write-Host $args -ForegroundColor Cyan }

# Statistics tracking
$script:FilesDeleted = 0
$script:DirsDeleted = 0
$script:BytesFreed = 0

function Get-DirectorySize {
    param([string]$Path)
    if (Test-Path $Path) {
        return (Get-ChildItem -Path $Path -Recurse -File | Measure-Object -Property Length -Sum).Sum
    }
    return 0
}

function Remove-SafelyWithStats {
    param(
        [string]$Path,
        [string]$Reason
    )
    
    if (-not (Test-Path $Path)) {
        Write-Warning "  [SKIP] $Path (not found)"
        return
    }
    
    $size = Get-DirectorySize $Path
    $fileCount = (Get-ChildItem -Path $Path -Recurse -File).Count
    
    if ($DryRun) {
        Write-Info "  [DRY-RUN] Would delete: $Path"
        Write-Info "            Files: $fileCount, Size: $([math]::Round($size/1MB, 2)) MB"
        Write-Info "            Reason: $Reason"
    } else {
        Write-Warning "  [DELETE] $Path"
        Write-Warning "           Files: $fileCount, Size: $([math]::Round($size/1MB, 2)) MB"
        Write-Warning "           Reason: $Reason"
        
        Remove-Item -Path $Path -Recurse -Force
        $script:FilesDeleted += $fileCount
        $script:DirsDeleted += 1
        $script:BytesFreed += $size
        Write-Success "  [OK] Deleted successfully"
    }
}

# Safety check
Write-Info "=" * 80
Write-Info "KILO RADICAL SIMPLIFICATION CLEANUP"
Write-Info "=" * 80
Write-Info ""

if (-not $Force) {
    Write-Warning "This script will DELETE files permanently!"
    Write-Warning "Ensure you have committed all changes to git."
    Write-Info ""
    $confirm = Read-Host "Type 'DELETE' to continue"
    if ($confirm -ne "DELETE") {
        Write-Error "Aborted by user"
        exit 1
    }
}

if ($DryRun) {
    Write-Info "DRY RUN MODE - No files will be deleted"
    Write-Info ""
}

# PHASE 1: DELETE MIGRATION ARCHIVE
Write-Info "Phase 1: Deleting Migration Archive"
Write-Info "-" * 80
Remove-SafelyWithStats "docs/migration-archive" "Historical data belongs in git history"
Write-Info ""

# PHASE 2: DELETE ARCHIVE DIRECTORIES
Write-Info "Phase 2: Deleting Archive Directories"
Write-Info "-" * 80
Remove-SafelyWithStats "docs/archive" "Old documentation, use git history"
Write-Info ""

# PHASE 3: DELETE INFRASTRUCTURE (move to templates first if needed)
Write-Info "Phase 3: Cleaning Infrastructure Directory"
Write-Info "-" * 80
Write-Warning "NOTE: Review infrastructure/ before deletion - some may need to move to templates/"
Remove-SafelyWithStats "tools/infrastructure/ansible" "Should be in templates if needed"
Remove-SafelyWithStats "tools/infrastructure/gitops" "Should be in templates if needed"
Remove-SafelyWithStats "tools/infrastructure/terraform/environments" "Environment-specific configs not needed in meta repo"
Write-Info ""

# PHASE 4: CLEAN UP CONSOLE.LOG AND PRINT STATEMENTS
Write-Info "Phase 4: Removing Debug Statements"
Write-Info "-" * 80

if ($DryRun) {
    Write-Info "  [DRY-RUN] Would remove console.log/print statements from:"
    $files = Get-ChildItem -Recurse -Include *.ts,*.js,*.py | 
        Where-Object { $_.FullName -notmatch 'node_modules|\.git|templates' } |
        Select-String -Pattern 'console\.log|print\(' |
        Select-Object -ExpandProperty Path -Unique
    $files | ForEach-Object { Write-Info "    - $_" }
    Write-Info "  Total files: $($files.Count)"
} else {
    Write-Warning "  [CLEAN] Removing console.log and print statements..."
    $cleaned = 0
    Get-ChildItem -Recurse -Include *.ts,*.js,*.py | 
        Where-Object { $_.FullName -notmatch 'node_modules|\.git|templates' } | 
        ForEach-Object {
            $content = Get-Content $_.FullName -Raw
            $newContent = $content -replace '^\s*console\.log\([^)]*\);\s*$', '' -replace '^\s*print\([^)]*\)\s*$', ''
            if ($content -ne $newContent) {
                Set-Content $_.FullName -Value $newContent -NoNewline
                $cleaned++
                Write-Info "    Cleaned: $($_.Name)"
            }
        }
    Write-Success "  [OK] Cleaned $cleaned files"
}
Write-Info ""

# PHASE 5: STANDARDIZE YAML EXTENSIONS
Write-Info "Phase 5: Standardizing YAML Extensions (.yml → .yaml)"
Write-Info "-" * 80

$ymlFiles = Get-ChildItem -Recurse -Filter *.yml | 
    Where-Object { $_.FullName -notmatch 'node_modules|\.git' }

if ($DryRun) {
    Write-Info "  [DRY-RUN] Would rename $($ymlFiles.Count) .yml files to .yaml"
    $ymlFiles | Select-Object -First 10 | ForEach-Object { Write-Info "    - $($_.Name)" }
    if ($ymlFiles.Count -gt 10) {
        Write-Info "    ... and $($ymlFiles.Count - 10) more"
    }
} else {
    Write-Warning "  [RENAME] Renaming .yml files to .yaml..."
    $renamed = 0
    $ymlFiles | ForEach-Object {
        $newName = $_.Name -replace '\.yml$', '.yaml'
        $newPath = Join-Path $_.DirectoryName $newName
        if (-not (Test-Path $newPath)) {
            Rename-Item $_.FullName -NewName $newName
            $renamed++
        }
    }
    Write-Success "  [OK] Renamed $renamed files"
}
Write-Info ""

# PHASE 6: REMOVE TODO/FIXME COMMENTS
Write-Info "Phase 6: Reporting TODO/FIXME Comments"
Write-Info "-" * 80

$todoFiles = Get-ChildItem -Recurse -Include *.ts,*.js,*.py | 
    Where-Object { $_.FullName -notmatch 'node_modules|\.git|templates' } |
    Select-String -Pattern 'TODO|FIXME|HACK|XXX' |
    Group-Object Path

Write-Info "  Found TODO/FIXME comments in $($todoFiles.Count) files:"
$todoFiles | Select-Object -First 10 | ForEach-Object {
    Write-Info "    - $($_.Name): $($_.Count) comments"
}
if ($todoFiles.Count -gt 10) {
    Write-Info "    ... and $($todoFiles.Count - 10) more files"
}
Write-Warning "  ACTION REQUIRED: Review and fix or delete these comments manually"
Write-Info ""

# SUMMARY
Write-Info "=" * 80
Write-Info "CLEANUP SUMMARY"
Write-Info "=" * 80

if ($DryRun) {
    Write-Info "DRY RUN - No changes made"
    Write-Info "Run with -DryRun:`$false to apply changes"
} else {
    Write-Success "Directories Deleted: $script:DirsDeleted"
    Write-Success "Files Deleted: $script:FilesDeleted"
    Write-Success "Space Freed: $([math]::Round($script:BytesFreed/1MB, 2)) MB"
    Write-Info ""
    Write-Success "✅ Cleanup complete!"
    Write-Info ""
    Write-Warning "NEXT STEPS:"
    Write-Info "1. Review changes: git status"
    Write-Info "2. Test the application"
    Write-Info "3. Commit changes: git add -A && git commit -m 'KILO: Phase 1 cleanup'"
    Write-Info "4. Continue with Phase 2 (Consolidation)"
}

Write-Info "=" * 80
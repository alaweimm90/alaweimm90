# Quick Workspace Cleanup Script
# Removes empty folders, cache, large files, and duplicates
# Created: 2025-11-24

param(
    [switch]$Execute = $false
)

$rootPath = "c:\Users\mesha\Desktop\GitHub"
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$logFile = Join-Path $rootPath "cleanup-log-$timestamp.txt"

function Log($msg) {
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $msg"
    Write-Host $line
    $line | Out-File -FilePath $logFile -Append
}

Log "=== WORKSPACE CLEANUP $(if($Execute){'EXECUTING'}else{'DRY RUN'}) ==="

# Statistics
$stats = @{Removed=0; SpaceFreed=0}

# =====================================
# 1. REMOVE CACHE DIRECTORIES
# =====================================
Log "`n--- Removing Cache/Build Directories ---"

$cacheDirs = @(
    'node_modules',
    '.next',
    '.turbo',
    'dist',
    'build',
    'coverage',
    '__pycache__',
    '.pytest_cache',
    '.cache'
)

Get-ChildItem -Path $rootPath -Directory -Recurse -Force -ErrorAction SilentlyContinue |
    Where-Object { $cacheDirs -contains $_.Name } |
    ForEach-Object {
        $size = (Get-ChildItem $_.FullName -Recurse -File -Force -ErrorAction SilentlyContinue |
                 Measure-Object -Property Length -Sum -ErrorAction SilentlyContinue).Sum
        $sizeMB = [math]::Round($size/1MB, 2)

        if ($Execute) {
            Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
            Log "✓ Removed $($_.Name): $sizeMB MB"
            $stats.Removed++
            $stats.SpaceFreed += $size
        } else {
            Log "[DRY RUN] Would remove $($_.FullName): $sizeMB MB"
        }
    }

# =====================================
# 2. REMOVE LARGE FILES (>50MB)
# =====================================
Log "`n--- Finding Files Over 50MB ---"

$largeFiles = Get-ChildItem -Path $rootPath -File -Recurse -Force -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Length -gt 50MB -and
        $_.FullName -notlike '*\.git\objects\*' -and
        $_.FullName -notlike '*\.metaHub\archives\*'
    }

Log "Found $($largeFiles.Count) large files"

$largeFiles | ForEach-Object {
    $sizeMB = [math]::Round($_.Length/1MB, 2)
    Log "[REVIEW] $($_.FullName): $sizeMB MB"
}

# =====================================
# 3. FIND DUPLICATES IN BACKUP FOLDERS
# =====================================
Log "`n--- Checking for Duplicates in Backup Folders ---"

$backupFolders = @('.cache\backups-*', '.config\organizations')

foreach ($pattern in $backupFolders) {
    $folders = Get-ChildItem -Path $rootPath -Directory -Filter $pattern -Recurse -Force -ErrorAction SilentlyContinue

    foreach ($folder in $folders) {
        if (Test-Path $folder.FullName) {
            $size = (Get-ChildItem $folder.FullName -Recurse -File -Force -ErrorAction SilentlyContinue |
                     Measure-Object -Property Length -Sum -ErrorAction SilentlyContinue).Sum
            $sizeGB = [math]::Round($size/1GB, 2)

            if ($Execute -and $sizeGB -gt 0.1) {
                Remove-Item $folder.FullName -Recurse -Force -ErrorAction SilentlyContinue
                Log "✓ Removed backup folder: $($folder.Name) - $sizeGB GB"
                $stats.Removed++
                $stats.SpaceFreed += $size
            } else {
                Log "[BACKUP] $($folder.FullName): $sizeGB GB"
            }
        }
    }
}

# =====================================
# 4. REMOVE EMPTY DIRECTORIES
# =====================================
Log "`n--- Removing Empty Directories ---"

$emptyCount = 0
for ($pass = 1; $pass -le 3; $pass++) {
    $emptyDirs = @(Get-ChildItem -Path $rootPath -Directory -Recurse -Force -ErrorAction SilentlyContinue |
        Where-Object {
            (Get-ChildItem $_.FullName -Force -ErrorAction SilentlyContinue).Count -eq 0
        })

    if ($emptyDirs.Count -eq 0) { break }

    Log "Pass ${pass}: Found $($emptyDirs.Count) empty directories"

    if ($Execute) {
        $emptyDirs | ForEach-Object {
            Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
            $emptyCount++
        }
    }
}

if ($Execute) {
    Log "✓ Removed $emptyCount empty directories"
}

# =====================================
# SUMMARY
# =====================================
Log "`n=== CLEANUP SUMMARY ==="
Log "Items Removed: $($stats.Removed)"
Log "Space Freed: $([math]::Round($stats.SpaceFreed/1GB, 2)) GB"
Log "Log saved to: $logFile"

if (-not $Execute) {
    Log "`n⚠️  DRY RUN COMPLETE - No files were deleted"
    Log "To execute cleanup, run: .\scripts\quick-cleanup.ps1 -Execute"
}

Log "=== DONE ==="

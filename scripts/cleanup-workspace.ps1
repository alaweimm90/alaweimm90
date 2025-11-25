# Workspace Cleanup Script
# Removes empty folders, cache files, files > 50MB, duplicates, and corrupt files
# Created: 2025-11-24

param(
    [switch]$DryRun = $true,
    [switch]$Force = $false
)

$ErrorActionPreference = "SilentlyContinue"
$rootPath = "c:\Users\mesha\Desktop\GitHub"
$logFile = Join-Path $rootPath "cleanup-log-$(Get-Date -Format 'yyyyMMdd-HHmmss').txt"

function Write-Log {
    param($Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    Write-Host $logMessage
    Add-Content -Path $logFile -Value $logMessage
}

function Get-DirectorySize {
    param($Path)
    $size = (Get-ChildItem -Path $Path -Recurse -File -Force -ErrorAction SilentlyContinue |
             Measure-Object -Property Length -Sum).Sum
    return $size
}

Write-Log "=== WORKSPACE CLEANUP STARTED ==="
Write-Log "Mode: $(if($DryRun){'DRY RUN (no files will be deleted)'}else{'LIVE - FILES WILL BE DELETED'})"
Write-Log "Root Path: $rootPath"

$stats = @{
    EmptyDirsRemoved = 0
    CacheDirsRemoved = 0
    LargeFilesRemoved = 0
    DuplicatesRemoved = 0
    CorruptFilesRemoved = 0
    SpaceFreed = 0
}

# =====================================================
# 1. REMOVE CACHE AND BUILD DIRECTORIES
# =====================================================
Write-Log "`n--- STEP 1: Removing Cache/Build Directories ---"

$cachePatterns = @(
    'node_modules',
    '.next',
    '.turbo',
    'dist',
    'build',
    'coverage',
    '__pycache__',
    '.pytest_cache',
    '.mypy_cache',
    '.tox',
    '.eggs',
    '*.egg-info',
    '.sass-cache',
    '.parcel-cache'
)

# Safe directories to preserve
$preservePaths = @(
    '*\.git\*',
    '*\.github\*',
    '*\.metaHub\*',
    '*\.vscode\*'
)

$cacheDirs = Get-ChildItem -Path $rootPath -Recurse -Directory -Force -ErrorAction SilentlyContinue |
    Where-Object {
        $dirName = $_.Name
        $fullPath = $_.FullName
        $shouldRemove = $false

        foreach ($pattern in $cachePatterns) {
            if ($dirName -like $pattern) {
                $shouldRemove = $true
                break
            }
        }

        # Don't remove if in preserved paths
        if ($shouldRemove) {
            foreach ($preserve in $preservePaths) {
                if ($fullPath -like $preserve) {
                    $shouldRemove = $false
                    break
                }
            }
        }

        $shouldRemove
    }

Write-Log "Found $($cacheDirs.Count) cache/build directories"

foreach ($dir in $cacheDirs) {
    try {
        $size = Get-DirectorySize -Path $dir.FullName
        $sizeMB = [math]::Round($size / 1MB, 2)

        if (-not $DryRun) {
            Remove-Item -Path $dir.FullName -Recurse -Force -ErrorAction Stop
            Write-Log "✓ Removed: $($dir.FullName) ($sizeMB MB)"
            $stats.CacheDirsRemoved++
            $stats.SpaceFreed += $size
        } else {
            Write-Log "[DRY RUN] Would remove: $($dir.FullName) ($sizeMB MB)"
        }
    } catch {
        Write-Log "✗ Failed to remove: $($dir.FullName) - $($_.Exception.Message)"
    }
}

# =====================================================
# 2. REMOVE FILES OVER 50MB
# =====================================================
Write-Log "`n--- STEP 2: Removing Files Over 50MB ---"

# Exclude git pack files and essential large files
$excludePatterns = @(
    '*\.git\objects\pack\*',
    '*\.metaHub\archives\*'
)

$largeFiles = Get-ChildItem -Path $rootPath -Recurse -File -Force -ErrorAction SilentlyContinue |
    Where-Object {
        $_.Length -gt 50MB -and
        -not ($excludePatterns | Where-Object { $_.FullName -like $_ })
    }

Write-Log "Found $($largeFiles.Count) files over 50MB (excluding git archives)"

foreach ($file in $largeFiles) {
    $sizeMB = [math]::Round($file.Length / 1MB, 2)

    if (-not $DryRun -and $Force) {
        try {
            Remove-Item -Path $file.FullName -Force -ErrorAction Stop
            Write-Log "✓ Removed: $($file.FullName) ($sizeMB MB)"
            $stats.LargeFilesRemoved++
            $stats.SpaceFreed += $file.Length
        } catch {
            Write-Log "✗ Failed to remove: $($file.FullName) - $($_.Exception.Message)"
        }
    } else {
        Write-Log "[MANUAL REVIEW NEEDED] Large file: $($file.FullName) ($sizeMB MB)"
    }
}

# =====================================================
# 3. FIND AND REMOVE DUPLICATE FILES
# =====================================================
Write-Log "`n--- STEP 3: Finding Duplicate Files ---"

# Focus on common duplicate areas
$duplicateSearchPaths = @(
    '.cache',
    '.archives',
    '.config\organizations'
)

$duplicates = @{}
$filesScanned = 0

foreach ($searchPath in $duplicateSearchPaths) {
    $fullPath = Join-Path $rootPath $searchPath
    if (Test-Path $fullPath) {
        Write-Log "Scanning for duplicates in: $searchPath"

        Get-ChildItem -Path $fullPath -Recurse -File -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Length -gt 1MB } | # Only check files > 1MB
            ForEach-Object {
                $filesScanned++
                $hash = (Get-FileHash -Path $_.FullName -Algorithm MD5 -ErrorAction SilentlyContinue).Hash

                if ($hash) {
                    if (-not $duplicates.ContainsKey($hash)) {
                        $duplicates[$hash] = @()
                    }
                    $duplicates[$hash] += $_
                }
            }
    }
}

Write-Log "Scanned $filesScanned files for duplicates"

$duplicateGroups = $duplicates.GetEnumerator() | Where-Object { $_.Value.Count -gt 1 }
Write-Log "Found $($duplicateGroups.Count) duplicate file groups"

foreach ($group in $duplicateGroups) {
    $files = $group.Value | Sort-Object LastWriteTime -Descending
    $keepFile = $files[0] # Keep the newest
    $removeFiles = $files[1..($files.Count - 1)]

    Write-Log "`nDuplicate group (Hash: $($group.Key)):"
    Write-Log "  KEEPING: $($keepFile.FullName)"

    foreach ($file in $removeFiles) {
        $sizeMB = [math]::Round($file.Length / 1MB, 2)

        if (-not $DryRun) {
            try {
                Remove-Item -Path $file.FullName -Force -ErrorAction Stop
                Write-Log "  ✓ Removed duplicate: $($file.FullName) ($sizeMB MB)"
                $stats.DuplicatesRemoved++
                $stats.SpaceFreed += $file.Length
            } catch {
                Write-Log "  ✗ Failed to remove: $($file.FullName)"
            }
        } else {
            Write-Log "  [DRY RUN] Would remove: $($file.FullName) ($sizeMB MB)"
        }
    }
}

# =====================================================
# 4. FIND AND REMOVE CORRUPT FILES
# =====================================================
Write-Log "`n--- STEP 4: Checking for Corrupt Files ---"

$corruptPatterns = @(
    '*.json',
    '*.js',
    '*.ts',
    '*.py',
    '*.md'
)

$corruptFiles = @()

foreach ($pattern in $corruptPatterns) {
    $files = Get-ChildItem -Path $rootPath -Recurse -Filter $pattern -File -Force -ErrorAction SilentlyContinue |
        Select-Object -First 1000 # Limit for performance

    foreach ($file in $files) {
        try {
            # Test if file is readable
            $null = Get-Content -Path $file.FullName -TotalCount 1 -ErrorAction Stop

            # For JSON files, try to parse
            if ($file.Extension -eq '.json') {
                $null = Get-Content -Path $file.FullName -Raw | ConvertFrom-Json -ErrorAction Stop
            }
        } catch {
            $corruptFiles += $file
            Write-Log "Potentially corrupt: $($file.FullName) - $($_.Exception.Message)"
        }
    }
}

Write-Log "Found $($corruptFiles.Count) potentially corrupt files"

foreach ($file in $corruptFiles) {
    $sizeMB = [math]::Round($file.Length / 1MB, 2)

    if (-not $DryRun -and $Force) {
        try {
            # Backup corrupt files before removing
            $backupPath = Join-Path $rootPath ".cleanup-backup\corrupt\"
            if (-not (Test-Path $backupPath)) {
                New-Item -Path $backupPath -ItemType Directory -Force | Out-Null
            }

            Copy-Item -Path $file.FullName -Destination $backupPath -Force
            Remove-Item -Path $file.FullName -Force -ErrorAction Stop
            Write-Log "✓ Backed up and removed: $($file.FullName)"
            $stats.CorruptFilesRemoved++
            $stats.SpaceFreed += $file.Length
        } catch {
            Write-Log "✗ Failed to handle corrupt file: $($file.FullName)"
        }
    } else {
        Write-Log "[MANUAL REVIEW NEEDED] Corrupt file: $($file.FullName)"
    }
}

# =====================================================
# 5. REMOVE EMPTY DIRECTORIES
# =====================================================
Write-Log "`n--- STEP 5: Removing Empty Directories ---"

# Remove empty dirs multiple times to handle nested empties
for ($i = 1; $i -le 3; $i++) {
    $emptyDirs = Get-ChildItem -Path $rootPath -Recurse -Directory -Force -ErrorAction SilentlyContinue |
        Where-Object {
            (Get-ChildItem $_.FullName -Force -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0
        }

    Write-Log "Pass ${i}: Found $($emptyDirs.Count) empty directories"

    foreach ($dir in $emptyDirs) {
        if (-not $DryRun) {
            try {
                Remove-Item -Path $dir.FullName -Force -ErrorAction Stop
                $stats.EmptyDirsRemoved++
            } catch {
                # Silently skip if can't remove
            }
        }
    }

    if ($emptyDirs.Count -eq 0) { break }
}

Write-Log "Total empty directories removed: $($stats.EmptyDirsRemoved)"

# =====================================================
# SUMMARY
# =====================================================
Write-Log "`n=== CLEANUP SUMMARY ==="
Write-Log "Cache/Build Directories Removed: $($stats.CacheDirsRemoved)"
Write-Log "Large Files Removed (>50MB): $($stats.LargeFilesRemoved)"
Write-Log "Duplicate Files Removed: $($stats.DuplicatesRemoved)"
Write-Log "Corrupt Files Removed: $($stats.CorruptFilesRemoved)"
Write-Log "Empty Directories Removed: $($stats.EmptyDirsRemoved)"
Write-Log "Total Space Freed: $([math]::Round($stats.SpaceFreed / 1GB, 2)) GB"
Write-Log "`nLog file saved to: $logFile"

if ($DryRun) {
    Write-Log "`n⚠️  THIS WAS A DRY RUN - NO FILES WERE DELETED"
    Write-Log "To execute cleanup, run: .\cleanup-workspace.ps1 -DryRun:`$false"
    Write-Log "To also remove large files and corrupt files, add: -Force"
}

Write-Log "`n=== CLEANUP COMPLETED ==="

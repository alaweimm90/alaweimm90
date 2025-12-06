# Rename Optilibria to Equilibria
# This script renames all occurrences of "optilibria" to "equilibria" in the codebase

$rootPath = "c:\Users\mesha\Desktop\GitHub"

Write-Host "=== Optilibria → Equilibria Rename Script ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Rename the main directory
$oldDir = "$rootPath\organizations\AlaweinOS\Optilibria"
$newDir = "$rootPath\organizations\AlaweinOS\Equilibria"

if (Test-Path $oldDir) {
    Write-Host "Step 1: Renaming directory..." -ForegroundColor Yellow
    Write-Host "  $oldDir → $newDir"
    # Rename-Item -Path $oldDir -NewName "Equilibria" -WhatIf
    Write-Host "  [DRY RUN] Would rename directory" -ForegroundColor Gray
}

# Step 2: Rename the Python package directory inside
$oldPkg = "$newDir\optilibria"
$newPkg = "$newDir\equilibria"

if (Test-Path $oldPkg) {
    Write-Host ""
    Write-Host "Step 2: Renaming Python package..." -ForegroundColor Yellow
    Write-Host "  $oldPkg → $newPkg"
    # Rename-Item -Path $oldPkg -NewName "equilibria" -WhatIf
    Write-Host "  [DRY RUN] Would rename package" -ForegroundColor Gray
}

# Step 3: Find all files containing "optilibria" (case-insensitive)
Write-Host ""
Write-Host "Step 3: Finding files with 'optilibria'..." -ForegroundColor Yellow

$extensions = @("*.py", "*.md", "*.yaml", "*.yml", "*.json", "*.toml", "*.txt", "*.rst", "*.ini", "*.cfg")
$excludeDirs = @("node_modules", ".git", "__pycache__", ".venv", "venv", "dist", "build", ".archive")

$filesToUpdate = @()

foreach ($ext in $extensions) {
    $files = Get-ChildItem -Path $rootPath -Filter $ext -Recurse -ErrorAction SilentlyContinue |
        Where-Object {
            $exclude = $false
            foreach ($dir in $excludeDirs) {
                if ($_.FullName -like "*\$dir\*") { $exclude = $true; break }
            }
            -not $exclude
        } |
        Where-Object { (Get-Content $_.FullName -Raw -ErrorAction SilentlyContinue) -match "optilibria" }

    $filesToUpdate += $files
}

Write-Host "  Found $($filesToUpdate.Count) files to update" -ForegroundColor Green

# Step 4: Show files that would be updated
Write-Host ""
Write-Host "Step 4: Files to update:" -ForegroundColor Yellow
$filesToUpdate | ForEach-Object {
    $relativePath = $_.FullName.Replace($rootPath, ".")
    Write-Host "  $relativePath" -ForegroundColor Gray
}

# Step 5: Perform replacements (dry run)
Write-Host ""
Write-Host "Step 5: Replacement patterns:" -ForegroundColor Yellow
Write-Host "  'Optilibria' → 'Equilibria'" -ForegroundColor Gray
Write-Host "  'optilibria' → 'equilibria'" -ForegroundColor Gray
Write-Host "  'OPTILIBRIA' → 'EQUILIBRIA'" -ForegroundColor Gray

Write-Host ""
Write-Host "=== DRY RUN COMPLETE ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "To execute the rename, uncomment the Rename-Item and add:" -ForegroundColor Yellow
Write-Host '  (Get-Content $file.FullName) -replace "optilibria", "equilibria" | Set-Content $file.FullName' -ForegroundColor Gray
Write-Host ""
Write-Host "Or run manually:" -ForegroundColor Yellow
Write-Host "  1. Rename directory: organizations\AlaweinOS\Optilibria → Equilibria" -ForegroundColor Gray
Write-Host "  2. Rename package: Equilibria\optilibria → equilibria" -ForegroundColor Gray
Write-Host "  3. Find/replace in IDE: optilibria → equilibria (case-sensitive)" -ForegroundColor Gray
Write-Host "  4. Update pyproject.toml, setup.py, __init__.py" -ForegroundColor Gray
Write-Host "  5. Update all imports" -ForegroundColor Gray

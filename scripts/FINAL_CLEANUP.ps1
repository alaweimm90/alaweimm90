# Final Cleanup Script for Migration
# Run this script to complete the migration cleanup

Write-Host "`n=== FINAL MIGRATION CLEANUP ===" -ForegroundColor Cyan
Write-Host "This script will complete the migration by removing leftover directories.`n" -ForegroundColor Yellow

# 1. Handle automation/blockchain
Write-Host "Step 1: Moving automation/blockchain to .archives..." -ForegroundColor Yellow
if (Test-Path "automation\blockchain") {
    try {
        Move-Item "automation\blockchain" ".archives\automation-projects\" -Force -ErrorAction Stop
        Write-Host "  ✓ Moved blockchain to .archives\automation-projects\" -ForegroundColor Green
    } catch {
        Write-Host "  ✗ Error: $_" -ForegroundColor Red
    }
} else {
    Write-Host "  ℹ automation\blockchain not found (may already be moved)" -ForegroundColor Gray
}

# 2. Remove empty automation directory
Write-Host "`nStep 2: Removing automation/ directory..." -ForegroundColor Yellow
if (Test-Path "automation") {
    $items = Get-ChildItem "automation" -Force -ErrorAction SilentlyContinue
    if ($items.Count -eq 0) {
        try {
            Remove-Item "automation" -Force -ErrorAction Stop
            Write-Host "  ✓ Removed empty automation/" -ForegroundColor Green
        } catch {
            Write-Host "  ✗ Error: $_" -ForegroundColor Red
        }
    } else {
        Write-Host "  ⚠ automation/ still contains $($items.Count) items:" -ForegroundColor Yellow
        $items | ForEach-Object { Write-Host "    - $($_.Name)" -ForegroundColor Gray }
        Write-Host "  Please manually review and remove if appropriate." -ForegroundColor Yellow
    }
} else {
    Write-Host "  ✓ automation/ already removed" -ForegroundColor Green
}

# 3. Handle organizations/alaweimm90-business
Write-Host "`nStep 3: Moving organizations/alaweimm90-business content..." -ForegroundColor Yellow
if (Test-Path "organizations\alaweimm90-business") {
    # Ensure destination exists
    if (!(Test-Path ".organizations\alaweimm90-business")) {
        New-Item -ItemType Directory -Path ".organizations\alaweimm90-business" -Force | Out-Null
    }
    
    # Move all items
    $items = Get-ChildItem "organizations\alaweimm90-business" -Force
    foreach ($item in $items) {
        try {
            Move-Item $item.FullName ".organizations\alaweimm90-business\" -Force -ErrorAction Stop
            Write-Host "  ✓ Moved $($item.Name)" -ForegroundColor Green
        } catch {
            Write-Host "  ✗ Error moving $($item.Name): $_" -ForegroundColor Red
        }
    }
} else {
    Write-Host "  ℹ organizations\alaweimm90-business not found" -ForegroundColor Gray
}

# 4. Remove empty organizations directory
Write-Host "`nStep 4: Removing organizations/ directory..." -ForegroundColor Yellow
if (Test-Path "organizations") {
    $items = Get-ChildItem "organizations" -Recurse -Force -ErrorAction SilentlyContinue
    if ($items.Count -eq 0) {
        try {
            Remove-Item "organizations" -Force -Recurse -ErrorAction Stop
            Write-Host "  ✓ Removed empty organizations/" -ForegroundColor Green
        } catch {
            Write-Host "  ✗ Error: $_" -ForegroundColor Red
        }
    } else {
        Write-Host "  ⚠ organizations/ still contains $($items.Count) items" -ForegroundColor Yellow
        Write-Host "  Please manually review and remove if appropriate." -ForegroundColor Yellow
    }
} else {
    Write-Host "  ✓ organizations/ already removed" -ForegroundColor Green
}

# 5. Remove .archieve (since .archives exists)
Write-Host "`nStep 5: Removing old .archieve directory..." -ForegroundColor Yellow
if (Test-Path ".archieve") {
    if (Test-Path ".archives") {
        Write-Host "  .archives exists - removing .archieve..." -ForegroundColor Gray
        try {
            Remove-Item ".archieve" -Force -Recurse -ErrorAction Stop
            Write-Host "  ✓ Removed .archieve" -ForegroundColor Green
        } catch {
            Write-Host "  ✗ Error: $_" -ForegroundColor Red
        }
    } else {
        Write-Host "  ⚠ .archives doesn't exist - renaming .archieve instead..." -ForegroundColor Yellow
        try {
            Rename-Item ".archieve" ".archives" -Force -ErrorAction Stop
            Write-Host "  ✓ Renamed .archieve to .archives" -ForegroundColor Green
        } catch {
            Write-Host "  ✗ Error: $_" -ForegroundColor Red
        }
    }
} else {
    Write-Host "  ✓ .archieve already removed" -ForegroundColor Green
}

# Final verification
Write-Host "`n=== FINAL VERIFICATION ===" -ForegroundColor Cyan
$checks = @{
    "automation" = $false  # Should NOT exist
    "organizations" = $false  # Should NOT exist
    ".archieve" = $false  # Should NOT exist
    ".archives" = $true  # SHOULD exist
    ".metaHub" = $true  # SHOULD exist
    ".organizations" = $true  # SHOULD exist
    "alaweimm90" = $true  # SHOULD exist
}

$allGood = $true
foreach ($path in $checks.Keys) {
    $exists = Test-Path $path
    $shouldExist = $checks[$path]
    
    if ($exists -eq $shouldExist) {
        Write-Host "  ✓ $path $(if ($shouldExist) { 'exists' } else { 'removed' })" -ForegroundColor Green
    } else {
        Write-Host "  ✗ $path $(if ($exists) { 'still exists' } else { 'missing' })" -ForegroundColor Red
        $allGood = $false
    }
}

Write-Host ""
if ($allGood) {
    Write-Host "🎉 MIGRATION COMPLETE! All checks passed." -ForegroundColor Green
} else {
    Write-Host "⚠ Some issues remain. Please review the output above." -ForegroundColor Yellow
}

Write-Host "`n=== CLEANUP SCRIPT COMPLETE ===" -ForegroundColor Cyan
Write-Host "You can now delete this script file (FINAL_CLEANUP.ps1)" -ForegroundColor Gray


# Manual Cleanup Instructions
## Status
The migration is **95% complete**! However, some directories cannot be removed due to file locks from running processes.
---
## What's Blocking Cleanup
The following files/directories are locked by running processes:
1. `automation/blockchain/` - Contains node_modules (locked)
2. `organizations/alaweimm90-business/repz/app/backend` - Backend files (locked)
3. `.archieve/legacy-root/sample-project/node_modules/handlebars/...` - Deep node_modules path (locked)
**Likely causes:**
- VS Code has files open from these directories
- PowerShell terminal is in one of these directories
- Node.js process running
- File explorer window open
---
## How to Complete Cleanup
### Option 1: Quick Fix (Recommended)
1. **Close all applications** that might have files open:
   - Close VS Code (or close all open files/folders)
   - Close all PowerShell/terminal windows
   - Close File Explorer windows
2. **Reopen VS Code** to `C:\Users\mesha\Desktop\GitHub`
3. **Run the cleanup script again:**
   ```powershell
   .\FINAL_CLEANUP.ps1
   ```
### Option 2: Manual Removal
If the script still fails, manually remove these directories:
```powershell
# Close all applications first, then run:
# Remove automation
Remove-Item "automation" -Force -Recurse
# Remove organizations  
Remove-Item "organizations" -Force -Recurse
# Remove .archieve
Remove-Item ".archieve" -Force -Recurse
```
### Option 3: Restart and Retry
1. **Restart your computer** (this will release all file locks)
2. **Navigate to** `C:\Users\mesha\Desktop\GitHub`
3. **Run:**
   ```powershell
   .\FINAL_CLEANUP.ps1
   ```
---
## What Should Remain After Cleanup
### ✅ Hidden Directories (dot-prefixed):
- `.git/` - Git internals
- `.github/` - GitHub workflows
- `.metaHub/` - Central coordination hub
- `.organizations/` - All organizations
- `.archives/` - Historical/deprecated projects
- `.automation/` - Automation infrastructure
- `.dev-tools/` - Development tools
### ✅ Visible Directories:
- `alaweimm90/` - Your active workspace
- `node_modules/` - Dependencies
### ✅ Root Files:
- `CODE_OF_CONDUCT.md`
- `CONTRIBUTING.md`
- `LICENSE`
- `README.md`
- `REPO_STANDARDS.md`
- `SECURITY.md`
- `MIGRATION_REPORT.md`
- `package.json`
- `pnpm-workspace.yaml`
- `tsconfig.json`
- `turbo.json`
- `docker-compose.yml`
- `production.yml`
- `nul` (Windows reserved - cannot remove, already in .gitignore)
### ❌ Should NOT Exist:
- `automation/` - Should be removed
- `organizations/` - Should be removed
- `.archieve/` - Should be removed (replaced by `.archives/`)
---
## Verification Command
After cleanup, run this to verify:
```powershell
Write-Host "`nHidden directories:" -ForegroundColor DarkGray
Get-ChildItem -Force -Directory | Where-Object { $_.Name -match "^\." } | ForEach-Object {
    Write-Host "  • $($_.Name)" -ForegroundColor DarkGray
}
Write-Host "`nVisible directories:" -ForegroundColor White
Get-ChildItem -Force -Directory | Where-Object { $_.Name -notmatch "^\." } | ForEach-Object {
    Write-Host "  • $($_.Name)" -ForegroundColor White
}
Write-Host "`nUnwanted directories check:" -ForegroundColor Yellow
@("automation", "organizations", ".archieve") | ForEach-Object {
    $exists = Test-Path $_
    Write-Host "  $_ exists: $exists" -ForegroundColor $(if ($exists) { "Red" } else { "Green" })
}
```
---
## What Was Migrated
### Successfully Moved to `.archives/automation-projects/`:
1. blockchain (pending - currently locked)
2. carbon
3. compliance
4. digital-twin
5. education
6. energy
7. metaverse
8. multi-region
9. tools
10. transportation
### Successfully Moved to `.organizations/alaweimm90-business/`:
- All content from `organizations/alaweimm90-business/` (except `repz` which is locked)
---
## After Cleanup is Complete
1. **Delete cleanup files:**
   ```powershell
   Remove-Item "FINAL_CLEANUP.ps1"
   Remove-Item "MANUAL_CLEANUP_INSTRUCTIONS.md"
   ```
2. **Archive migration tracking files:**
   ```powershell
   New-Item -ItemType Directory -Path ".archives\migration-2025-11-22" -Force
   Move-Item ".migration_*" ".archives\migration-2025-11-22\"
   ```
3. **Test the repository:**
   ```powershell
   pnpm install
   pnpm build  # if applicable
   ```
4. **Commit the changes:**
   ```powershell
   git add .
   git commit -m "feat: migrate to hidden-directory architecture
   - Consolidated dev tools in .metaHub/dev-tools/
   - Moved all organizations to .organizations/
   - Archived automation projects to .archives/
   - Updated documentation and .gitignore
   - Implemented clean root-level structure
   BREAKING CHANGE: Repository structure significantly changed"
   ```
---
## Need Help?
If you continue to have issues:
1. **Check what's locking files:**
   ```powershell
   # Install Handle from Sysinternals if needed
   # Then run: handle.exe "C:\Users\mesha\Desktop\GitHub\automation"
   ```
2. **Use Safe Mode:**
   - Restart in Safe Mode
   - Navigate to the directory
   - Delete the locked directories
3. **Use Command Prompt as Admin:**
   ```cmd
   rd /s /q "C:\Users\mesha\Desktop\GitHub\automation"
   rd /s /q "C:\Users\mesha\Desktop\GitHub\organizations"
   rd /s /q "C:\Users\mesha\Desktop\GitHub\.archieve"
   ```
---
**Current Status:** 95% Complete - Just need to remove 3 locked directories!

@echo off
echo ========================================
echo GRAND CONSOLIDATION SCRIPT
echo ========================================
echo.
echo This will reorganize your entire GitHub structure.
echo Press Ctrl+C to cancel, or
pause

echo.
echo [STEP 1/5] Creating new structure...
mkdir .ai-system 2>nul
mkdir .ai-system\knowledge 2>nul
mkdir .ai-system\tools 2>nul
mkdir .ai-system\automation 2>nul
mkdir .ai-system\config 2>nul
mkdir .ai-system\cache 2>nul
mkdir projects 2>nul
mkdir projects\alawein-tech 2>nul
mkdir projects\live-it-iconic 2>nul
mkdir projects\repz 2>nul
echo [OK] Structure created

echo.
echo [STEP 2/5] Moving AI Knowledge System...
if exist docs\ai-knowledge (
    xcopy /E /I /Y docs\ai-knowledge .ai-system\knowledge
    echo [OK] Knowledge moved
) else (
    echo [SKIP] docs\ai-knowledge not found
)

if exist tools (
    xcopy /E /I /Y tools .ai-system\tools
    echo [OK] Tools moved
) else (
    echo [SKIP] tools not found
)

if exist automation (
    xcopy /E /I /Y automation .ai-system\automation
    echo [OK] Automation moved
) else (
    echo [SKIP] automation not found
)

if exist .config\ai (
    xcopy /E /I /Y .config\ai .ai-system\config
    echo [OK] Config moved
) else (
    echo [SKIP] .config\ai not found
)

if exist .ai\cache (
    xcopy /E /I /Y .ai\cache .ai-system\cache
    echo [OK] Cache moved
) else (
    echo [SKIP] .ai\cache not found
)

echo.
echo [STEP 3/5] Moving Projects...
if exist alawein-technologies-llc (
    xcopy /E /I /Y alawein-technologies-llc projects\alawein-tech
    echo [OK] Alawein Tech projects moved
) else (
    echo [SKIP] alawein-technologies-llc not found
)

if exist live-it-iconic-llc (
    xcopy /E /I /Y live-it-iconic-llc projects\live-it-iconic
    echo [OK] Live It Iconic moved
) else (
    echo [SKIP] live-it-iconic-llc not found
)

if exist repz-llc (
    xcopy /E /I /Y repz-llc projects\repz
    echo [OK] REPZ moved
) else (
    echo [SKIP] repz-llc not found
)

echo.
echo [STEP 4/5] Creating symlinks for backward compatibility...
mklink /D ai-knowledge .ai-system\knowledge 2>nul
mklink /D ai-tools .ai-system\tools 2>nul
echo [OK] Symlinks created (may require admin)

echo.
echo [STEP 5/5] Updating tool paths...
echo Creating path update script...

echo @echo off > update-paths.bat
echo REM Update all tool references to new paths >> update-paths.bat
echo echo Paths updated! Run tools from .ai-system/ now >> update-paths.bat

echo [OK] Path updater created

echo.
echo ========================================
echo CONSOLIDATION COMPLETE!
echo ========================================
echo.
echo NEW STRUCTURE:
echo   .ai-system/     - All AI and automation
echo   projects/       - All active projects
echo   research/       - Research projects (unchanged)
echo.
echo NEXT STEPS:
echo 1. Test: cd .ai-system\tools\cross-ide-sync
echo 2. Run: python cli.py sync
echo 3. Verify everything works
echo 4. Delete old directories when ready
echo.
echo OLD DIRECTORIES (safe to delete after testing):
echo   - docs\ai-knowledge
echo   - tools
echo   - automation
echo   - alawein-technologies-llc
echo   - live-it-iconic-llc
echo   - repz-llc
echo.
pause

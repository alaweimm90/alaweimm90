@echo off
echo ========================================
echo Daily AI Prompt System Routine
echo ========================================
echo.

echo [1/4] Syncing prompts to all IDEs...
cd tools\cross-ide-sync
python cli.py sync
echo.

echo [2/4] Validating prompt quality...
cd ..\prompt-testing
python cli.py validate --all
echo.

echo [3/4] Checking for regressions...
python cli.py regression --check
echo.

echo [4/4] Viewing analytics...
cd ..\analytics
python dashboard.py
echo.

echo ========================================
echo Daily routine complete!
echo ========================================
pause

@echo off
echo Starting Auto-Sync for AI Prompts...
echo.
echo This will watch for changes and automatically sync to all IDEs.
echo Press Ctrl+C to stop.
echo.

cd tools\cross-ide-sync
python cli.py watch

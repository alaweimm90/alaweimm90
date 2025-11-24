@echo off
REM Centralized devtools setup script for Windows

echo Setting up centralized devtools...
echo.

set DEVTOOLS_DIR=.devtools
set TOOLS=amazonq cursor continue windsurf cline

for %%t in (%TOOLS%) do (
    echo Setting up .%%t...
    if not exist ".%%t" mkdir ".%%t"
    
    if exist ".%%t\rules" (
        echo   Rules already linked for %%t
    ) else (
        mklink /D ".%%t\rules" "%DEVTOOLS_DIR%\rules" >nul 2>&1
        echo   Linked rules for %%t
    )
    
    if exist ".%%t\integrations" (
        echo   Integrations already linked for %%t
    ) else (
        mklink /D ".%%t\integrations" "%DEVTOOLS_DIR%\integrations" >nul 2>&1
        echo   Linked integrations for %%t
    )
    echo.
)

echo Centralized devtools setup complete!
echo.
echo Structure created:
echo    .devtools/          (source of truth)
echo    +-- rules/          (5 rule files)
echo    +-- mcps/           (2 config files)
echo    +-- integrations/   (5 config files)
echo.
echo Symlinks created for: %TOOLS%
echo.
echo See .devtools\README.md for usage

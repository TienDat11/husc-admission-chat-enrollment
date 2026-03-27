@echo off
REM ========================================================================
REM  Fix Missing Dependencies - Quick Install
REM ========================================================================
REM  This script installs missing dependencies without re-running full setup
REM ========================================================================

setlocal enabledelayedexpansion

set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "RESET=[0m"

echo.
echo %BLUE%========================================================================%RESET%
echo %BLUE%   Fixing Missing Dependencies%RESET%
echo %BLUE%========================================================================%RESET%
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo %RED%[ERROR]%RESET% Virtual environment not found
    echo %RED%[ERROR]%RESET% Please run setup first or create venv manually
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
echo %BLUE%[INFO]%RESET% Activating virtual environment...
call venv\Scripts\activate.bat
if !errorlevel! neq 0 (
    echo %RED%[ERROR]%RESET% Failed to activate virtual environment
    pause
    exit /b 1
)
echo %GREEN%[OK]%RESET% Virtual environment activated
echo.

REM Install missing packages
echo %BLUE%[STEP 1]%RESET% Installing critical missing packages...
echo.

echo %YELLOW%Installing jsonschema...%RESET%
pip install jsonschema
if !errorlevel! neq 0 (
    echo %RED%[ERROR]%RESET% Failed to install jsonschema
    pause
    exit /b 1
)
echo %GREEN%[OK]%RESET% jsonschema installed
echo.

REM Install/update all dependencies from requirements.txt
echo %BLUE%[STEP 2]%RESET% Installing/updating all dependencies from requirements.txt...
echo.

pip install -r requirements.txt
if !errorlevel! neq 0 (
    echo %RED%[ERROR]%RESET% Failed to install requirements.txt
    pause
    exit /b 1
)

echo.
echo %GREEN%========================================================================%RESET%
echo %GREEN%   Dependencies Fixed!%RESET%
echo %GREEN%========================================================================%RESET%
echo.
echo %GREEN%Installed:%RESET%
echo   - jsonschema (for JSONL validation)
echo   - All other dependencies from requirements.txt
echo.
echo %BLUE%Next Steps:%RESET%
echo   1. Re-run %GREEN%setup_data.bat%RESET% to continue data ingestion
echo   2. Or manually run the pipeline step by step
echo.
pause
exit /b 0

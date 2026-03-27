@echo off
REM ========================================================================
REM  Apply Corrected Configuration
REM ========================================================================
REM  This script backs up the old .env and applies the corrected version
REM ========================================================================

setlocal enabledelayedexpansion

set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "RESET=[0m"

echo.
echo %BLUE%========================================================================%RESET%
echo %BLUE%   Applying Corrected Configuration%RESET%
echo %BLUE%========================================================================%RESET%
echo.

REM Check if corrected config exists
if not exist ".env.correct" (
    echo %RED%[ERROR]%RESET% Cannot find .env.correct file
    pause
    exit /b 1
)

REM Backup existing .env
if exist ".env" (
    echo %YELLOW%[INFO]%RESET% Backing up current .env to .env.backup...
    copy /Y ".env" ".env.backup" >nul
    echo %GREEN%[OK]%RESET% Backup created: .env.backup
    echo.
)

REM Apply corrected config
echo %BLUE%[INFO]%RESET% Applying corrected configuration...
copy /Y ".env.correct" ".env" >nul
echo %GREEN%[OK]%RESET% Configuration applied!
echo.

echo %GREEN%Key Changes:%RESET%
echo   - USE_QDRANT:    true  -^> false
echo   - EMBEDDING_DIM: 768   -^> 384
echo   - FORCE_RAG_ONLY: true -^> false
echo.

echo %BLUE%Next Steps:%RESET%
echo   1. Run %GREEN%setup_data.bat%RESET% to ingest your data
echo   2. Run %GREEN%run_lab.bat%RESET% to start the server
echo.
pause
exit /b 0

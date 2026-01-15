@echo off
REM ========================================================================
REM  2025 RAG Lab - Production Launcher with Preflight Checks
REM ========================================================================
REM  Performs strict health checks before starting the server.
REM  Steps:
REM    1. Activate/create virtual environment
REM    2. Run preflight checks (config, connectivity, data)
REM    3. Start Uvicorn API server
REM    4. Auto-open browser to Swagger UI
REM ========================================================================

setlocal enabledelayedexpansion

REM Set colors for output
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "RESET=[0m"

echo.
echo %BLUE%========================================================================%RESET%
echo %BLUE%   2025 RAG Lab - Startup Sequence%RESET%
echo %BLUE%========================================================================%RESET%
echo.

REM ========================================================================
REM Step 1: Environment Validation
REM ========================================================================

echo %BLUE%[STEP 1]%RESET% Validating environment...
echo.

REM Check if we're in the correct directory
if not exist "src\main.py" (
    echo %RED%[ERROR]%RESET% Cannot find src\main.py
    echo %RED%[ERROR]%RESET% Please run this script from the rag2025 directory
    echo.
    pause
    exit /b 1
)

echo %GREEN%[OK]%RESET% Found src\main.py - correct directory
echo.

REM ========================================================================
REM Step 2: Detect Python and Setup Virtual Environment
REM ========================================================================

echo %BLUE%[STEP 2]%RESET% Setting up Python environment...
echo.

set "PYTHON_CMD="

REM Try different Python commands
python --version >nul 2>&1
if !errorlevel! equ 0 (
    set "PYTHON_CMD=python"
    goto :PythonFound
)

py --version >nul 2>&1
if !errorlevel! equ 0 (
    set "PYTHON_CMD=py"
    goto :PythonFound
)

python3 --version >nul 2>&1
if !errorlevel! equ 0 (
    set "PYTHON_CMD=python3"
    goto :PythonFound
)

echo %RED%[ERROR]%RESET% Python not found!
echo %RED%[ERROR]%RESET% Install Python 3.10+ from python.org
echo.
pause
exit /b 1

:PythonFound
echo %GREEN%[OK]%RESET% Using Python: %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

REM Check/Create Virtual Environment
if not exist "venv\Scripts\activate.bat" (
    echo %YELLOW%[WARNING]%RESET% Virtual environment not found
    echo %BLUE%[INFO]%RESET% Creating virtual environment...
    %PYTHON_CMD% -m venv venv
    if !errorlevel! neq 0 (
        echo %RED%[ERROR]%RESET% Failed to create virtual environment
        pause
        exit /b 1
    )
    echo %GREEN%[OK]%RESET% Virtual environment created
    echo.
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

REM Install/Update dependencies
echo %BLUE%[INFO]%RESET% Checking dependencies...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
if !errorlevel! neq 0 (
    echo %RED%[ERROR]%RESET% Failed to install dependencies
    pause
    exit /b 1
)
echo %GREEN%[OK]%RESET% Dependencies ready
echo.

REM ========================================================================
REM Step 3: Run Preflight Checks (STRICT)
REM ========================================================================

echo %BLUE%[STEP 3]%RESET% Running preflight checks...
echo.

python scripts\preflight_check.py
if !errorlevel! neq 0 (
    echo.
    echo %RED%========================================================================%RESET%
    echo %RED%   PREFLIGHT CHECKS FAILED%RESET%
    echo %RED%========================================================================%RESET%
    echo.
    echo %RED%Server will NOT start. Please fix the errors above.%RESET%
    echo.
    pause
    exit /b 1
)

echo %GREEN%========================================================================%RESET%
echo %GREEN%   PREFLIGHT CHECKS PASSED%RESET%
echo %GREEN%========================================================================%RESET%
echo.

REM ========================================================================
REM Step 4: Start Uvicorn Server
REM ========================================================================

echo %BLUE%[STEP 4]%RESET% Starting API server...
echo.
echo %GREEN%Server URL:%RESET% http://localhost:8000
echo %GREEN%Swagger UI:%RESET% http://localhost:8000/docs
echo.
echo %YELLOW%Press Ctrl+C to stop the server%RESET%
echo.

REM Auto-open browser after 5 seconds (in background)
start /B cmd /c "timeout /t 5 /nobreak >nul && start http://localhost:8000/docs"

REM Start server (logs output naturally to console)
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

REM If server exits normally
echo.
echo %BLUE%[INFO]%RESET% Server stopped
echo.
pause
exit /b 0

@echo off
REM ========================================================================
REM  2025 RAG Lab - GraphRAG Knowledge Graph Builder
REM ========================================================================
REM  Builds a knowledge graph from chunked documents using LLM-based NER.
REM
REM  Usage:
REM    build_graph.bat                  -- incremental (default, safe to re-run)
REM    build_graph.bat --full           -- full rebuild from scratch
REM    build_graph.bat --dry-run        -- load chunks only, skip LLM/NER calls
REM    build_graph.bat --limit 20       -- process only first 20 chunks (test)
REM
REM  Prerequisites:
REM    - Run setup_data.bat first to generate data\chunked\chunked_2.jsonl
REM    - Set GEMINI_API_KEY (or other LLM key) in .env
REM ========================================================================

setlocal enabledelayedexpansion

REM Parse arguments
set "GRAPH_MODE=--incremental"
set "EXTRA_ARGS="

:ParseArgs
if "%~1"=="" goto :ArgsDone
if /i "%~1"=="--full"     ( set "GRAPH_MODE=" & shift & goto :ParseArgs )
if /i "%~1"=="--dry-run"  ( set "GRAPH_MODE=--dry-run" & shift & goto :ParseArgs )
if /i "%~1"=="--limit"    ( set "EXTRA_ARGS=--limit %~2" & shift & shift & goto :ParseArgs )
shift
goto :ParseArgs
:ArgsDone

REM Set colors for output
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "RESET=[0m"

echo.
echo %BLUE%========================================================================%RESET%
echo %BLUE%   2025 RAG Lab - GraphRAG Knowledge Graph Builder%RESET%
echo %BLUE%========================================================================%RESET%
echo %YELLOW%   Mode: %GRAPH_MODE% %EXTRA_ARGS%%RESET%
echo.

REM ========================================================================
REM Step 0: Environment Check
REM ========================================================================

echo %BLUE%[STEP 0]%RESET% Checking environment...
echo.

REM Check correct directory
if not exist "src\main.py" (
    echo %RED%[ERROR]%RESET% Cannot find src\main.py
    echo %RED%[ERROR]%RESET% Please run this script from the rag2025 directory
    echo.
    pause
    exit /b 1
)

REM Check chunked data exists
if not exist "data\chunked" (
    echo %RED%[ERROR]%RESET% data\chunked directory not found
    echo %RED%[ERROR]%RESET% Please run setup_data.bat first
    pause
    exit /b 1
)

REM Check at least one chunked file
set "FOUND_CHUNKS=0"
for %%f in (data\chunked\chunked_*.jsonl) do set "FOUND_CHUNKS=1"
if "!FOUND_CHUNKS!"=="0" (
    echo %RED%[ERROR]%RESET% No chunked_*.jsonl files found in data\chunked\
    echo %RED%[ERROR]%RESET% Please run setup_data.bat first to generate chunks
    pause
    exit /b 1
)
echo %GREEN%[OK]%RESET% Chunked data found in data\chunked\

REM Check build_graph.py exists
if not exist "scripts\build_graph.py" (
    echo %RED%[ERROR]%RESET% scripts\build_graph.py not found
    pause
    exit /b 1
)
echo %GREEN%[OK]%RESET% scripts\build_graph.py found

REM Activate virtual environment
if not exist "venv\Scripts\activate.bat" (
    echo %RED%[ERROR]%RESET% Virtual environment not found
    echo %RED%[ERROR]%RESET% Please run setup_lab.bat first
    pause
    exit /b 1
)

echo %BLUE%[INFO]%RESET% Activating virtual environment...
call venv\Scripts\activate.bat
echo %GREEN%[OK]%RESET% Virtual environment activated
echo.

REM Detect Python
set "PYTHON_CMD="
python --version >nul 2>&1
if !errorlevel! equ 0 ( set "PYTHON_CMD=python" & goto :PythonFound )
py --version >nul 2>&1
if !errorlevel! equ 0 ( set "PYTHON_CMD=py" & goto :PythonFound )
python3 --version >nul 2>&1
if !errorlevel! equ 0 ( set "PYTHON_CMD=python3" & goto :PythonFound )
echo %RED%[ERROR]%RESET% Python not found!
pause
exit /b 1

:PythonFound
echo %GREEN%[OK]%RESET% Using Python: %PYTHON_CMD%
echo.

REM Check .env for API key (warn only, don't block dry-run)
if not exist ".env" (
    echo %YELLOW%[WARNING]%RESET% .env file not found - LLM calls may fail
) else (
    findstr /i "RAMCLOUDS_API_KEY\|GEMINI_API_KEY\|OPENAI_API_KEY\|GROQ_API_KEY" .env >nul 2>&1
    if !errorlevel! neq 0 (
        echo %YELLOW%[WARNING]%RESET% No LLM API key found in .env
        echo %YELLOW%[WARNING]%RESET% NER extraction will fail without an API key
        echo %YELLOW%[INFO]%RESET%    Use --dry-run to test without API calls
    ) else (
        echo %GREEN%[OK]%RESET% API key found in .env
    )
)
echo.

REM Create graph output directory
if not exist "data\graph" mkdir data\graph
echo %GREEN%[OK]%RESET% Output directory: data\graph
echo.

REM ========================================================================
REM Step 1: Build Knowledge Graph
REM ========================================================================

echo %BLUE%========================================================================%RESET%
echo %BLUE%[STEP 1]%RESET% Building GraphRAG knowledge graph...
echo %BLUE%========================================================================%RESET%
echo.

REM Show existing graph info if incremental
if "%GRAPH_MODE%"=="--incremental" (
    if exist "data\graph\knowledge_graph.graphml" (
        echo %YELLOW%[INFO]%RESET% Existing graph found - running in incremental mode
        echo %YELLOW%[INFO]%RESET% Only NEW chunks will be processed
    ) else (
        echo %YELLOW%[INFO]%RESET% No existing graph - will do full build
    )
    echo.
)

echo Running: %PYTHON_CMD% scripts\build_graph.py %GRAPH_MODE% %EXTRA_ARGS%
echo.

%PYTHON_CMD% scripts\build_graph.py %GRAPH_MODE% %EXTRA_ARGS%

if !errorlevel! neq 0 (
    echo.
    echo %RED%[ERROR]%RESET% GraphRAG build failed!
    echo %RED%[ERROR]%RESET% Please check the errors above
    echo.
    echo %YELLOW%[TIP]%RESET% Common fixes:
    echo    - Check API key in .env
    echo    - Try: build_graph.bat --dry-run
    echo    - Try: build_graph.bat --limit 5
    echo.
    pause
    exit /b 1
)

echo.
echo %GREEN%========================================================================%RESET%
echo %GREEN%   GRAPHRAG BUILD COMPLETE!%RESET%
echo %GREEN%========================================================================%RESET%
echo.
echo %GREEN%Output files:%RESET%
echo    - data\graph\knowledge_graph.graphml
echo    - data\graph\entity_index.json
echo.
echo %BLUE%Next Steps:%RESET%
echo    1. Run %GREEN%run_lab.bat%RESET% to start the API server
echo    2. Open http://localhost:8000/docs to test GraphRAG queries
echo.
pause
exit /b 0

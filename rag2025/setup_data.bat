@echo off
REM ========================================================================
REM  2025 RAG Lab - Automated Data Ingestion Pipeline
REM ========================================================================
REM  This script automates the complete data ingestion workflow:
REM    1. Normalize raw JSONL data
REM    2. Validate JSONL data
REM    3. Chunk documents with adaptive profiles
REM    4. Build vector index with embeddings
REM ========================================================================

setlocal enabledelayedexpansion

REM Set colors for output
set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "RESET=[0m"

REM Runtime mode + input path
set "RUN_MODE="
set "INPUT_PATH="

if "%~1"=="" (
    set "RUN_MODE=INCREMENTAL"
) else if /I "%~1"=="FULL" (
    set "RUN_MODE=FULL"
    set "INPUT_PATH=%~2"
) else if /I "%~1"=="INCREMENTAL" (
    set "RUN_MODE=INCREMENTAL"
    set "INPUT_PATH=%~2"
) else if /I "%~1"=="LIGHT" (
    set "RUN_MODE=LIGHT"
    set "INPUT_PATH=%~2"
) else (
    REM Backward compatibility: first arg is input path
    set "RUN_MODE=INCREMENTAL"
    set "INPUT_PATH=%~1"
)

if /I not "%RUN_MODE%"=="FULL" if /I not "%RUN_MODE%"=="INCREMENTAL" if /I not "%RUN_MODE%"=="LIGHT" (
    echo %RED%[ERROR]%RESET% Invalid mode: %RUN_MODE%
    echo %RED%[ERROR]%RESET% Usage: setup_data.bat [FULL^|INCREMENTAL^|LIGHT] [input_path]
    echo.
    pause
    exit /b 1
)

echo.
echo %BLUE%========================================================================%RESET%
echo %BLUE%   2025 RAG Lab - Data Ingestion Pipeline%RESET%
echo %BLUE%========================================================================%RESET%
echo.

REM ========================================================================
REM Step 0: Environment Setup
REM ========================================================================

echo %BLUE%[STEP 0]%RESET% Setting up environment...
echo %BLUE%[INFO]%RESET% Mode: %RUN_MODE%
echo.

REM Check if we're in the correct directory
if not exist "src\main.py" (
    echo %RED%[ERROR]%RESET% Cannot find src\main.py
    echo %RED%[ERROR]%RESET% Please run this script from the rag2025 directory
    echo.
    pause
    exit /b 1
)

REM Detect Python
set "PYTHON_CMD="

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
pause
exit /b 1

:PythonFound
echo %GREEN%[OK]%RESET% Using Python: %PYTHON_CMD%
echo.

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

REM Ensure dependencies are installed (including lancedb)
if /I "%RUN_MODE%"=="LIGHT" (
    echo %YELLOW%[WARNING]%RESET% LIGHT mode: skipping dependency reinstall to reduce disk usage
) else (
    echo %BLUE%[INFO]%RESET% Installing/updating dependencies...
    python -m pip install --quiet --upgrade pip
    python -m pip install --quiet -r requirements.txt
    if !errorlevel! neq 0 (
        echo %RED%[ERROR]%RESET% Failed to install dependencies
        pause
        exit /b 1
    )
    echo %GREEN%[OK]%RESET% Dependencies ready
)

REM Load .env values if present
if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
        if not "%%A"=="" if /i not "%%A:~0,1"=="#" set "%%A=%%B"
    )
    echo %GREEN%[OK]%RESET% Loaded .env values
)
echo.

REM Create data directories
echo %BLUE%[INFO]%RESET% Creating data directories...
if not exist "data" mkdir data
if not exist "data\raw" mkdir data\raw
if not exist "data\normalized" mkdir data\normalized
if not exist "data\validated" mkdir data\validated
if not exist "data\chunked" mkdir data\chunked
if not exist "data\graph" mkdir data\graph
if not exist "data\lancedb" mkdir data\lancedb
if not exist "index" mkdir index
echo %GREEN%[OK]%RESET% Directories ready
echo.

REM Check for input path (supports file or directory)
if "%INPUT_PATH%"=="" set "INPUT_PATH=data\raw"
if /I "%RUN_MODE%"=="LIGHT" if "%INPUT_PATH%"=="data\raw" if not exist "%INPUT_PATH%" set "INPUT_PATH=."

if not exist "%INPUT_PATH%" (
    echo %YELLOW%[WARNING]%RESET% Input path not found: %INPUT_PATH%
    echo %YELLOW%[INFO]%RESET% Looking for fallback locations...

    if exist "data\raw" (
        set "INPUT_PATH=data\raw"
        echo %GREEN%[OK]%RESET% Found: data\raw
    ) else if exist "..\2.jsonl" (
        set "INPUT_PATH=..\2.jsonl"
        echo %GREEN%[OK]%RESET% Found: ..\2.jsonl
    ) else if exist "2.jsonl" (
        set "INPUT_PATH=2.jsonl"
        echo %GREEN%[OK]%RESET% Found: 2.jsonl
    ) else (
        echo %RED%[ERROR]%RESET% Cannot find input path
        echo %RED%[ERROR]%RESET% Provide a file/folder path as second argument
        echo               Example: setup_data.bat INCREMENTAL data\raw
        echo.
        pause
        exit /b 1
    )
)

echo %GREEN%[OK]%RESET% Input path found: %INPUT_PATH%
echo.

if /I "%RUN_MODE%"=="LIGHT" goto :LightModeOnly

REM ========================================================================
REM Step 1: Normalize Raw Data
REM ========================================================================

echo %BLUE%========================================================================%RESET%
echo %BLUE%[STEP 1]%RESET% Normalizing raw data...
echo %BLUE%========================================================================%RESET%
echo.

set "NORMALIZED_FILE=data\normalized\normalized_2.jsonl"

echo Running: python src\normalize_data.py "%INPUT_PATH%" "%NORMALIZED_FILE%"
echo.

python src\normalize_data.py "%INPUT_PATH%" "%NORMALIZED_FILE%"

if !errorlevel! neq 0 (
    echo.
    echo %RED%[ERROR]%RESET% Normalization failed!
    echo %RED%[ERROR]%RESET% Please check the errors above
    pause
    exit /b 1
)

echo.
echo %GREEN%[OK]%RESET% Normalization complete: %NORMALIZED_FILE%
echo.

REM ========================================================================
REM Step 2: Validate JSONL
REM ========================================================================

echo %BLUE%========================================================================%RESET%
echo %BLUE%[STEP 2]%RESET% Validating JSONL data...
echo %BLUE%========================================================================%RESET%
echo.

set "SCHEMA_FILE=config\rag_chunk_schema.json"
set "VALIDATED_FILE=data\validated\validated_2.jsonl"

REM Check if schema file exists
if not exist "%SCHEMA_FILE%" (
    echo %RED%[ERROR]%RESET% Schema file not found: %SCHEMA_FILE%
    echo %RED%[ERROR]%RESET% Please ensure the schema file exists
    pause
    exit /b 1
)

echo Running: python src\validate_jsonl.py "%NORMALIZED_FILE%" "%SCHEMA_FILE%" "%VALIDATED_FILE%"
echo.

python src\validate_jsonl.py "%NORMALIZED_FILE%" "%SCHEMA_FILE%" "%VALIDATED_FILE%"

if !errorlevel! neq 0 (
    echo.
    echo %RED%[ERROR]%RESET% Validation failed!
    echo %RED%[ERROR]%RESET% Please check the errors above and fix the input file
    pause
    exit /b 1
)

echo.
echo %GREEN%[OK]%RESET% Validation complete: %VALIDATED_FILE%
echo.

REM ========================================================================
REM Step 3: Chunk Documents
REM ========================================================================

echo %BLUE%========================================================================%RESET%
echo %BLUE%[STEP 3]%RESET% Chunking documents...
echo %BLUE%========================================================================%RESET%
echo.

set "CHUNKED_FILE=data\chunked\chunked_2.jsonl"
set "CHUNK_PROFILES=config\chunk_profiles.yaml"
set "CHUNK_PROFILE=auto"

REM Check if chunk profiles file exists
if not exist "%CHUNK_PROFILES%" (
    echo %RED%[ERROR]%RESET% Chunk profiles file not found: %CHUNK_PROFILES%
    echo %RED%[ERROR]%RESET% Please ensure the chunk profiles file exists
    pause
    exit /b 1
)

echo Running: python src\chunker.py "%VALIDATED_FILE%" "%CHUNKED_FILE%" "%CHUNK_PROFILES%" %CHUNK_PROFILE%
echo.

python src\chunker.py "%VALIDATED_FILE%" "%CHUNKED_FILE%" "%CHUNK_PROFILES%" %CHUNK_PROFILE%

if !errorlevel! neq 0 (
    echo.
    echo %RED%[ERROR]%RESET% Chunking failed!
    echo %RED%[ERROR]%RESET% Please check the errors above
    pause
    exit /b 1
)

echo.
echo %GREEN%[OK]%RESET% Chunking complete: %CHUNKED_FILE%
echo.

REM Normalize all chunked files to canonical object schema
echo %BLUE%[INFO]%RESET% Normalizing all chunked_*.jsonl files...
python src\normalize_chunks.py "data\chunked" --in-place
if !errorlevel! neq 0 (
    echo.
    echo %RED%[ERROR]%RESET% Chunk normalization failed!
    pause
    exit /b 1
)
echo %GREEN%[OK]%RESET% Chunk files normalized

echo.

REM ========================================================================
REM Step 4: Build LanceDB Vector Store
REM ========================================================================

echo %BLUE%========================================================================%RESET%
echo %BLUE%[STEP 4]%RESET% Building LanceDB vector store...
echo %BLUE%========================================================================%RESET%
echo.

echo Running: python scripts\ingest_lancedb.py
if /I "%RUN_MODE%"=="INCREMENTAL" echo %BLUE%[INFO]%RESET% Incremental ingest enabled (--incremental)
echo.

if /I "%RUN_MODE%"=="INCREMENTAL" (
    python scripts\ingest_lancedb.py --incremental
) else (
    python scripts\ingest_lancedb.py
)

if !errorlevel! neq 0 (
    echo.
    echo %RED%[ERROR]%RESET% LanceDB ingestion failed!
    echo %RED%[ERROR]%RESET% Please check the errors above
    pause
    exit /b 1
)

echo.
echo %GREEN%[OK]%RESET% LanceDB vector store ready
echo.

REM ========================================================================
REM Step 5: Build Knowledge Graph
REM ========================================================================

echo %BLUE%========================================================================%RESET%
echo %BLUE%[STEP 5]%RESET% Building GraphRAG knowledge graph...
echo %BLUE%========================================================================%RESET%
echo.

echo Running: python scripts\build_graph.py
if /I "%RUN_MODE%"=="INCREMENTAL" echo %BLUE%[INFO]%RESET% Incremental graph build enabled (--incremental)
echo.

if /I "%RUN_MODE%"=="INCREMENTAL" (
    python scripts\build_graph.py --incremental
) else (
    python scripts\build_graph.py
)

if !errorlevel! neq 0 (
    echo.
    echo %RED%[ERROR]%RESET% Graph build failed!
    echo %RED%[ERROR]%RESET% Please check RAMCLOUDS_API_KEY / GROQ_API_KEY and the errors above
    pause
    exit /b 1
)

echo.
echo %GREEN%[OK]%RESET% Knowledge graph ready
echo.

REM ========================================================================
REM Step 6: Post-setup verification
REM ========================================================================

echo %BLUE%========================================================================%RESET%
echo %BLUE%[STEP 6]%RESET% Verifying LanceDB + graph artifacts...
echo %BLUE%========================================================================%RESET%
echo.

python scripts\check_lancedb.py
if !errorlevel! neq 0 (
    echo.
    echo %RED%[ERROR]%RESET% LanceDB verification failed!
    pause
    exit /b 1
)

if not exist "data\graph\knowledge_graph.graphml" (
    echo %RED%[ERROR]%RESET% Graph file missing: data\graph\knowledge_graph.graphml
    pause
    exit /b 1
)

if not exist "data\graph\entity_index.json" (
    echo %RED%[ERROR]%RESET% Entity index missing: data\graph\entity_index.json
    pause
    exit /b 1
)

echo %GREEN%[OK]%RESET% Graph artifacts verified
echo.

REM ========================================================================
REM Summary
REM ========================================================================

echo %GREEN%========================================================================%RESET%
echo %GREEN%   DATA + DATABASE + GRAPH SETUP COMPLETE!%RESET%
echo %GREEN%========================================================================%RESET%
echo.
echo %GREEN%Pipeline Results:%RESET%
echo    1. Normalized: %NORMALIZED_FILE%
echo    2. Validated:  %VALIDATED_FILE%
echo    3. Chunked:    %CHUNKED_FILE%
echo    4. LanceDB:    data\lancedb\ (%LANCEDB_TABLE%)
echo    5. Graph:      data\graph\knowledge_graph.graphml
echo    6. Entities:   data\graph\entity_index.json
echo.
echo %BLUE%Next Steps:%RESET%
if /I "%RUN_MODE%"=="INCREMENTAL" (
    echo    1. Run %GREEN%python -m uvicorn src.main:app --host 0.0.0.0 --port 8000%RESET%
) else (
    echo    1. Run %GREEN%python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload%RESET%
)
echo    2. Open http://localhost:8000/docs to test the API
echo    3. Try querying your data!
echo.
pause
exit /b 0

:LightModeOnly
echo %YELLOW%========================================================================%RESET%
echo %YELLOW%   LIGHT MODE COMPLETE (SKIPPED HEAVY INGEST/GRAPH STEPS)%RESET%
echo %YELLOW%========================================================================%RESET%
echo.
echo %BLUE%[INFO]%RESET% LIGHT mode only ran steps 0-3 for low-disk environments.
echo %BLUE%[INFO]%RESET% On the high-capacity machine, run:
echo        %GREEN%setup_data.bat INCREMENTAL data\raw%RESET%
echo        or %GREEN%setup_data.bat FULL data\raw%RESET%
echo.
pause
exit /b 0

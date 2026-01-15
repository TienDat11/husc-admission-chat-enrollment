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

echo.
echo %BLUE%========================================================================%RESET%
echo %BLUE%   2025 RAG Lab - Data Ingestion Pipeline%RESET%
echo %BLUE%========================================================================%RESET%
echo.

REM ========================================================================
REM Step 0: Environment Setup
REM ========================================================================

echo %BLUE%[STEP 0]%RESET% Setting up environment...
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
echo.

REM Create data directories
echo %BLUE%[INFO]%RESET% Creating data directories...
if not exist "data" mkdir data
if not exist "data\raw" mkdir data\raw
if not exist "data\normalized" mkdir data\normalized
if not exist "data\validated" mkdir data\validated
if not exist "data\chunked" mkdir data\chunked
if not exist "index" mkdir index
echo %GREEN%[OK]%RESET% Directories ready
echo.

REM Check for input file
set "INPUT_FILE=..\2.jsonl"

if not exist "%INPUT_FILE%" (
    echo %YELLOW%[WARNING]%RESET% Input file not found: %INPUT_FILE%
    echo %YELLOW%[INFO]%RESET% Looking for alternative locations...

    REM Try data/raw/
    if exist "data\raw\2.jsonl" (
        set "INPUT_FILE=data\raw\2.jsonl"
        echo %GREEN%[OK]%RESET% Found: data\raw\2.jsonl
    ) else if exist "2.jsonl" (
        set "INPUT_FILE=2.jsonl"
        echo %GREEN%[OK]%RESET% Found: 2.jsonl
    ) else (
        echo %RED%[ERROR]%RESET% Cannot find input file 2.jsonl
        echo %RED%[ERROR]%RESET% Please ensure 2.jsonl exists in one of:
        echo               - ..\2.jsonl
        echo               - data\raw\2.jsonl
        echo               - 2.jsonl
        echo.
        pause
        exit /b 1
    )
)

echo %GREEN%[OK]%RESET% Input file found: %INPUT_FILE%
echo.

REM ========================================================================
REM Step 1: Normalize Raw Data
REM ========================================================================

echo %BLUE%========================================================================%RESET%
echo %BLUE%[STEP 1]%RESET% Normalizing raw data...
echo %BLUE%========================================================================%RESET%
echo.

set "NORMALIZED_FILE=data\normalized\normalized_2.jsonl"

echo Running: python src\normalize_data.py "%INPUT_FILE%" "%NORMALIZED_FILE%"
echo.

python src\normalize_data.py "%INPUT_FILE%" "%NORMALIZED_FILE%"

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

REM ========================================================================
REM Step 4: Build Vector Index
REM ========================================================================

echo %BLUE%========================================================================%RESET%
echo %BLUE%[STEP 4]%RESET% Building vector index...
echo %BLUE%========================================================================%RESET%
echo.

set "INDEX_FILE=index\vector_store.npz"

echo Running: python scripts\build_index.py "data\chunked" "%INDEX_FILE%"
echo.

python scripts\build_index.py "data\chunked" "%INDEX_FILE%"

if !errorlevel! neq 0 (
    echo.
    echo %RED%[ERROR]%RESET% Index build failed!
    echo %RED%[ERROR]%RESET% Please check the errors above
    pause
    exit /b 1
)

echo.
echo %GREEN%[OK]%RESET% Index build complete: %INDEX_FILE%
echo.

REM ========================================================================
REM Summary
REM ========================================================================

echo %GREEN%========================================================================%RESET%
echo %GREEN%   DATA INGESTION COMPLETE!%RESET%
echo %GREEN%========================================================================%RESET%
echo.
echo %GREEN%Pipeline Results:%RESET%
echo    1. Normalized: %NORMALIZED_FILE%
echo    2. Validated:  %VALIDATED_FILE%
echo    3. Chunked:    %CHUNKED_FILE%
echo    4. Indexed:    %INDEX_FILE%
echo.
echo %BLUE%Next Steps:%RESET%
echo    1. Run %GREEN%run_lab.bat%RESET% to start the API server
echo    2. Open http://localhost:8000/docs to test the API
echo    3. Try querying your data!
echo.
pause
exit /b 0

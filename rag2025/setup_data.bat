@echo off
REM ========================================================================
REM  2025 RAG Lab - Automated Data Ingestion Pipeline
REM ========================================================================
REM  This script automates the complete data ingestion workflow:
REM    1. Normalize raw JSONL data
REM    2. Validate JSONL data
REM    3. Chunk documents with adaptive profiles
REM    4. Build vector index with embeddings
REM
REM  Smart skip: each step is skipped if output already exists AND
REM  the output is NEWER than the input (no changes detected).
REM  Use --force flag to rebuild everything: setup_data.bat --force
REM ========================================================================

setlocal enabledelayedexpansion

REM Parse --force flag
set "FORCE_REBUILD=0"
if /i "%~1"=="--force" set "FORCE_REBUILD=1"

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
if "%FORCE_REBUILD%"=="1" (
    echo %YELLOW%   Mode: FORCE REBUILD - all steps will run%RESET%
) else (
    echo %YELLOW%   Mode: SMART SKIP - steps skipped if output is up-to-date%RESET%
    echo %YELLOW%   Tip: run with --force to rebuild everything%RESET%
)
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

REM Check and install tiktoken if missing
echo %BLUE%[INFO]%RESET% Checking required packages...
%PYTHON_CMD% -c "import tiktoken" >nul 2>&1
if !errorlevel! neq 0 (
    echo %YELLOW%[INFO]%RESET% tiktoken not found, installing...
    pip install tiktoken
    if !errorlevel! neq 0 (
        echo %RED%[ERROR]%RESET% Failed to install tiktoken
        echo %RED%[ERROR]%RESET% Please run: pip install tiktoken
        pause
        exit /b 1
    )
    echo %GREEN%[OK]%RESET% tiktoken installed successfully
) else (
    echo %GREEN%[OK]%RESET% tiktoken already installed
)
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

REM Smart skip check
set "SKIP_STEP1=0"
if "%FORCE_REBUILD%"=="0" (
    if exist "%NORMALIZED_FILE%" (
        REM Check if output is newer than input using forfiles
        forfiles /p "." /m "%INPUT_FILE%" /c "cmd /c exit 0" >nul 2>&1
        if exist "%NORMALIZED_FILE%" (
            REM Compare timestamps via xcopy dry-run trick
            xcopy /d /l /y "%INPUT_FILE%" "%NORMALIZED_FILE%" >nul 2>&1
            if !errorlevel! equ 0 (
                REM errorlevel 0 from xcopy /d means source is NEWER -> need rebuild
                set "SKIP_STEP1=0"
            ) else (
                REM errorlevel 1 means destination is up-to-date -> skip
                set "SKIP_STEP1=1"
            )
        )
    )
)

if "!SKIP_STEP1!"=="1" (
    echo %YELLOW%[SKIP]%RESET% Output already up-to-date: %NORMALIZED_FILE%
    echo %YELLOW%[SKIP]%RESET% Use --force to rebuild
    echo.
) else (
    echo Running: %PYTHON_CMD% src\normalize_data.py "%INPUT_FILE%" "%NORMALIZED_FILE%"
    echo.

    %PYTHON_CMD% src\normalize_data.py "%INPUT_FILE%" "%NORMALIZED_FILE%"

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
)

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

REM Smart skip check
set "SKIP_STEP2=0"
if "%FORCE_REBUILD%"=="0" (
    if exist "%VALIDATED_FILE%" (
        xcopy /d /l /y "%NORMALIZED_FILE%" "%VALIDATED_FILE%" >nul 2>&1
        if !errorlevel! equ 0 (
            set "SKIP_STEP2=0"
        ) else (
            set "SKIP_STEP2=1"
        )
    )
)

if "!SKIP_STEP2!"=="1" (
    echo %YELLOW%[SKIP]%RESET% Output already up-to-date: %VALIDATED_FILE%
    echo %YELLOW%[SKIP]%RESET% Use --force to rebuild
    echo.
) else (
    echo Running: %PYTHON_CMD% src\validate_jsonl.py "%NORMALIZED_FILE%" "%SCHEMA_FILE%" "%VALIDATED_FILE%"
    echo.

    %PYTHON_CMD% src\validate_jsonl.py "%NORMALIZED_FILE%" "%SCHEMA_FILE%" "%VALIDATED_FILE%"

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
)

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

REM Smart skip check
set "SKIP_STEP3=0"
if "%FORCE_REBUILD%"=="0" (
    if exist "%CHUNKED_FILE%" (
        xcopy /d /l /y "%VALIDATED_FILE%" "%CHUNKED_FILE%" >nul 2>&1
        if !errorlevel! equ 0 (
            set "SKIP_STEP3=0"
        ) else (
            set "SKIP_STEP3=1"
        )
    )
)

if "!SKIP_STEP3!"=="1" (
    echo %YELLOW%[SKIP]%RESET% Output already up-to-date: %CHUNKED_FILE%
    echo %YELLOW%[SKIP]%RESET% Use --force to rebuild
    echo.
) else (
    echo Running: %PYTHON_CMD% src\chunker.py "%VALIDATED_FILE%" "%CHUNKED_FILE%" "%CHUNK_PROFILES%" %CHUNK_PROFILE%
    echo.

    %PYTHON_CMD% src\chunker.py "%VALIDATED_FILE%" "%CHUNKED_FILE%" "%CHUNK_PROFILES%" %CHUNK_PROFILE%

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
)

REM ========================================================================
REM Step 4: Build Vector Index
REM ========================================================================

echo %BLUE%========================================================================%RESET%
echo %BLUE%[STEP 4]%RESET% Building vector index...
echo %BLUE%========================================================================%RESET%
echo.

set "INDEX_FILE=index\vector_store.npz"

REM Smart skip check
set "SKIP_STEP4=0"
if "%FORCE_REBUILD%"=="0" (
    if exist "%INDEX_FILE%" (
        xcopy /d /l /y "%CHUNKED_FILE%" "%INDEX_FILE%" >nul 2>&1
        if !errorlevel! equ 0 (
            set "SKIP_STEP4=0"
        ) else (
            set "SKIP_STEP4=1"
        )
    )
)

if "!SKIP_STEP4!"=="1" (
    echo %YELLOW%[SKIP]%RESET% Output already up-to-date: %INDEX_FILE%
    echo %YELLOW%[SKIP]%RESET% Use --force to rebuild
    echo.
) else (
    echo Running: %PYTHON_CMD% scripts\build_index.py "data\chunked" "%INDEX_FILE%"
    echo.

    %PYTHON_CMD% scripts\build_index.py "data\chunked" "%INDEX_FILE%"

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
)

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
echo    1. Run %GREEN%build_graph.bat%RESET% to build GraphRAG knowledge graph (optional)
echo    2. Run %GREEN%run_lab.bat%RESET% to start the API server
echo    3. Open http://localhost:8000/docs to test the API
echo    4. Try querying your data!
echo.
pause
exit /b 0

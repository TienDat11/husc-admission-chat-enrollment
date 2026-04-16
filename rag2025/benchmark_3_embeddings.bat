@echo off
REM ========================================================================
REM  Benchmark 3 Embedding Models (Qwen / Harrier / BGE)
REM ========================================================================
REM  Do: Query latency, indexing throughput, memory usage
REM  Ket qua luu vao: results\benchmark_3models.json
REM ========================================================================

setlocal enabledelayedexpansion

set "GREEN=[92m"
set "YELLOW=[93m"
set "RED=[91m"
set "BLUE=[94m"
set "CYAN=[96m"
set "RESET=[0m"

echo.
echo %CYAN%========================================================================%RESET%
echo %CYAN%   Benchmark 3 Embedding Models%RESET%
echo %CYAN%   Qwen3-Embedding-4B vs Harrier-OSS-v1-0.6B vs BGE-M3%RESET%
echo %CYAN%========================================================================%RESET%
echo.

REM Check directory
if not exist "src\main.py" (
    echo %RED%[ERROR]%RESET% Hay chay tu thu muc rag2025
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
echo %RED%[ERROR]%RESET% Khong tim thay Python!
pause
exit /b 1

:PythonFound
echo %GREEN%[OK]%RESET% Python: %PYTHON_CMD%

REM Activate venv
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo %GREEN%[OK]%RESET% Virtual environment activated
) else (
    echo %YELLOW%[WARNING]%RESET% Khong tim thay venv, chay bang Python he thong
)
echo.

REM Create results directory
if not exist "results" mkdir results

REM Set CPU optimization
set OMP_NUM_THREADS=8
set MKL_NUM_THREADS=8
echo %BLUE%[INFO]%RESET% CPU threads set to 8
echo.

echo %BLUE%========================================================================%RESET%
echo %BLUE%  Bat dau benchmark 3 models...%RESET%
echo %BLUE%  Moi model se tu dong tai tu HuggingFace lan dau (~1-8 GB)%RESET%
echo %BLUE%========================================================================%RESET%
echo.

%PYTHON_CMD% scripts\benchmark_embedding.py --compare --output results\benchmark_3models.json

if !errorlevel! neq 0 (
    echo.
    echo %RED%[ERROR]%RESET% Benchmark that bai!
    echo %RED%[ERROR]%RESET% Kiem tra log phia tren de xem loi
    pause
    exit /b 1
)

echo.
echo %GREEN%========================================================================%RESET%
echo %GREEN%   BENCHMARK HOAN TAT!%RESET%
echo %GREEN%========================================================================%RESET%
echo.
echo %GREEN%Ket qua:%RESET% results\benchmark_3models.json
echo.
echo %BLUE%Buoc tiep theo:%RESET%
echo   1. Chay %CYAN%compare_3_embeddings.bat%RESET% de so sanh retrieval quality
echo   2. Gui ket qua JSON cho Claude de phan tich
echo.
pause
exit /b 0

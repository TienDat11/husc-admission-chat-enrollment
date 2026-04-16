@echo off
REM ========================================================================
REM  So Sanh Retrieval Quality 3 Embedding Models
REM ========================================================================
REM  Do: Precision@K, NDCG@K, MRR tren tap du lieu thuc te
REM  Quy trinh: Voi moi model -> re-index LanceDB -> chay test queries
REM  Ket qua luu vao: results\retrieval_3models.json
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
echo %CYAN%   So Sanh Retrieval Quality - 3 Embedding Models%RESET%
echo %CYAN%   Qwen3-4B vs Harrier-0.6B vs BGE-M3%RESET%
echo %CYAN%========================================================================%RESET%
echo.
echo %YELLOW%LUU Y: Script nay se re-index LanceDB 3 lan (1 lan/model)%RESET%
echo %YELLOW%Thoi gian uoc tinh: 5-15 phut tuy theo CPU va du lieu%RESET%
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

REM Backup current .env
if exist ".env" (
    copy /Y ".env" ".env.benchmark_backup" >nul
    echo %GREEN%[OK]%RESET% Da backup .env -> .env.benchmark_backup
)
echo.

REM ========================================================================
REM  Model 1: Qwen3-Embedding-4B
REM ========================================================================

echo %BLUE%========================================================================%RESET%
echo %BLUE%  [1/3] Qwen3-Embedding-4B (dim=2560)%RESET%
echo %BLUE%========================================================================%RESET%
echo.

REM Write Qwen config to .env temporarily
call :WriteEnvEmbedding qwen "Qwen/Qwen3-Embedding-4B" 2560

echo %BLUE%[INFO]%RESET% Indexing voi Qwen3-4B...
%PYTHON_CMD% scripts\ingest_lancedb.py
if !errorlevel! neq 0 (
    echo %RED%[ERROR]%RESET% Qwen3 indexing that bai!
    goto :RestoreEnv
)

echo %BLUE%[INFO]%RESET% Evaluating retrieval quality...
%PYTHON_CMD% scripts\compare_retrieval.py --model "Qwen/Qwen3-Embedding-4B" --dim 2560 --provider qwen --k 5 --output results\retrieval_qwen3_4b.json
echo %GREEN%[OK]%RESET% Qwen3-4B done
echo.

REM ========================================================================
REM  Model 2: Harrier-OSS-v1-0.6B
REM ========================================================================

echo %BLUE%========================================================================%RESET%
echo %BLUE%  [2/3] Harrier-OSS-v1-0.6B (dim=1024)%RESET%
echo %BLUE%========================================================================%RESET%
echo.

call :WriteEnvEmbedding harrier "microsoft/harrier-oss-v1-0.6b" 1024

echo %BLUE%[INFO]%RESET% Indexing voi Harrier-0.6B...
%PYTHON_CMD% scripts\ingest_lancedb.py
if !errorlevel! neq 0 (
    echo %RED%[ERROR]%RESET% Harrier indexing that bai!
    goto :RestoreEnv
)

echo %BLUE%[INFO]%RESET% Evaluating retrieval quality...
%PYTHON_CMD% scripts\compare_retrieval.py --model "microsoft/harrier-oss-v1-0.6b" --dim 1024 --provider harrier --k 5 --output results\retrieval_harrier_0_6b.json
echo %GREEN%[OK]%RESET% Harrier-0.6B done
echo.

REM ========================================================================
REM  Model 3: BGE-M3
REM ========================================================================

echo %BLUE%========================================================================%RESET%
echo %BLUE%  [3/3] BGE-M3 (dim=1024)%RESET%
echo %BLUE%========================================================================%RESET%
echo.

call :WriteEnvEmbedding bge "BAAI/bge-m3" 1024

echo %BLUE%[INFO]%RESET% Indexing voi BGE-M3...
%PYTHON_CMD% scripts\ingest_lancedb.py
if !errorlevel! neq 0 (
    echo %RED%[ERROR]%RESET% BGE indexing that bai!
    goto :RestoreEnv
)

echo %BLUE%[INFO]%RESET% Evaluating retrieval quality...
%PYTHON_CMD% scripts\compare_retrieval.py --model "BAAI/bge-m3" --dim 1024 --provider bge --k 5 --output results\retrieval_bge_m3.json
echo %GREEN%[OK]%RESET% BGE-M3 done
echo.

REM ========================================================================
REM  Tong hop ket qua
REM ========================================================================

echo %BLUE%========================================================================%RESET%
echo %BLUE%  Tong hop 3 ket qua...%RESET%
echo %BLUE%========================================================================%RESET%
echo.

%PYTHON_CMD% -c "
import json
from pathlib import Path

results = {}
for name, path in [
    ('qwen3_4b', 'results/retrieval_qwen3_4b.json'),
    ('harrier_0_6b', 'results/retrieval_harrier_0_6b.json'),
    ('bge_m3', 'results/retrieval_bge_m3.json'),
]:
    p = Path(path)
    if p.exists():
        results[name] = json.loads(p.read_text(encoding='utf-8'))
    else:
        print(f'WARNING: {path} not found')

if results:
    output = {'results': results}
    Path('results/retrieval_3models.json').write_text(
        json.dumps(output, indent=2, ensure_ascii=False)
    )
    print('Saved: results/retrieval_3models.json')

    print()
    print('='*60)
    print('KET QUA SO SANH')
    print('='*60)
    for key, data in results.items():
        prec = data.get('precision_at_k', {}).get('mean', 0)
        ndcg = data.get('ndcg', {}).get('mean', 0)
        mrr = data.get('mrr', {}).get('mean', 0)
        print(f'  {key:20s} Precision={prec:.3f}  NDCG={ndcg:.3f}  MRR={mrr:.3f}')
"

:RestoreEnv
REM Restore original .env
if exist ".env.benchmark_backup" (
    copy /Y ".env.benchmark_backup" ".env" >nul
    del ".env.benchmark_backup"
    echo.
    echo %GREEN%[OK]%RESET% Da khoi phuc .env goc
)

echo.
echo %GREEN%========================================================================%RESET%
echo %GREEN%   SO SANH RETRIEVAL HOAN TAT!%RESET%
echo %GREEN%========================================================================%RESET%
echo.
echo %GREEN%Ket qua tung model:%RESET%
echo   results\retrieval_qwen3_4b.json
echo   results\retrieval_harrier_0_6b.json
echo   results\retrieval_bge_m3.json
echo.
echo %GREEN%Ket qua tong hop:%RESET%
echo   results\retrieval_3models.json
echo.
echo %BLUE%Gui cac file JSON nay cho Claude de chon model tot nhat!%RESET%
echo.
pause
exit /b 0

REM ========================================================================
REM  Subroutine: Ghi EMBEDDING config vao .env
REM ========================================================================
:WriteEnvEmbedding
REM %1 = provider, %2 = model, %3 = dim
set "EMB_PROVIDER=%~1"
set "EMB_MODEL=%~2"
set "EMB_DIM=%~3"

REM Read current .env and replace embedding lines
if exist ".env.benchmark_backup" (
    set "SOURCE=.env.benchmark_backup"
) else (
    set "SOURCE=.env"
)

REM Write new .env with updated embedding settings
(
    for /f "usebackq tokens=1,* delims==" %%A in ("%SOURCE%") do (
        set "LINE=%%A"
        if "%%A"=="EMBEDDING_PROVIDER" (
            echo EMBEDDING_PROVIDER=!EMB_PROVIDER!
        ) else if "%%A"=="EMBEDDING_MODEL" (
            echo EMBEDDING_MODEL=!EMB_MODEL!
        ) else if "%%A"=="EMBEDDING_DIM" (
            echo EMBEDDING_DIM=!EMB_DIM!
        ) else (
            if "%%B"=="" (
                echo %%A
            ) else (
                echo %%A=%%B
            )
        )
    )
) > ".env.tmp"

move /Y ".env.tmp" ".env" >nul
echo %GREEN%[OK]%RESET% .env updated: PROVIDER=%EMB_PROVIDER% MODEL=%EMB_MODEL% DIM=%EMB_DIM%

goto :eof

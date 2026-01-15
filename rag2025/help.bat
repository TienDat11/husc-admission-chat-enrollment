@echo off
REM ========================================================================
REM  2025 RAG Lab - Help & Commands Reference
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
echo %CYAN%   2025 RAG Lab - Available Commands%RESET%
echo %CYAN%========================================================================%RESET%
echo.

echo %BLUE%QUICK START (3 Steps):%RESET%
echo.
echo   %GREEN%1. apply_config.bat%RESET%    - Fix configuration (USE_QDRANT=false, DIM=384)
echo   %GREEN%2. setup_data.bat%RESET%      - Run data ingestion pipeline (validate → chunk → index)
echo   %GREEN%3. run_lab.bat%RESET%         - Start server with preflight checks
echo.

echo %BLUE%CONFIGURATION:%RESET%
echo.
echo   %CYAN%apply_config.bat%RESET%       - Apply corrected .env configuration
echo                            Backs up old .env to .env.backup
echo.
echo   %CYAN%fix_deps.bat%RESET%           - Install missing dependencies (jsonschema, etc.)
echo                            Use this if setup_data.bat fails with import errors
echo.
echo   %CYAN%type .env%RESET%              - View current configuration
echo   %CYAN%notepad .env%RESET%           - Edit configuration manually
echo.

echo %BLUE%DATA INGESTION:%RESET%
echo.
echo   %CYAN%setup_data.bat%RESET%         - Full pipeline (validate → chunk → index)
echo                            Input:  ../2.jsonl
echo                            Output: index/vector_store.npz
echo.
echo   %CYAN%Manual Pipeline:%RESET%
echo     1. python src\validate_jsonl.py input.jsonl data\validated\output.jsonl
echo     2. python src\chunker.py data\validated\output.jsonl data\chunked\output.jsonl
echo     3. python scripts\build_index.py data\chunked\output.jsonl index\vector_store.npz
echo.

echo %BLUE%SERVER:%RESET%
echo.
echo   %CYAN%run_lab.bat%RESET%            - Start server with preflight checks
echo                            URL: http://localhost:8000
echo                            Docs: http://localhost:8000/docs
echo.
echo   %CYAN%python scripts\preflight_check.py%RESET%
echo                            - Run health checks only (no server start)
echo.

echo %BLUE%TESTING:%RESET%
echo.
echo   %CYAN%python scripts\verify_api.py%RESET%
echo                            - Test API endpoints
echo.
echo   %CYAN%python scripts\demo_epic1.py%RESET%
echo                            - Demo chunking functionality
echo.

echo %BLUE%UTILITIES:%RESET%
echo.
echo   %CYAN%help.bat%RESET%               - Show this help message
echo.
echo   %CYAN%dir data\validated%RESET%     - List validated files
echo   %CYAN%dir data\chunked%RESET%       - List chunked files
echo   %CYAN%dir index%RESET%              - List index files
echo.

echo %BLUE%DOCUMENTATION:%RESET%
echo.
echo   %CYAN%SETUP_INSTRUCTIONS.md%RESET%  - Complete setup guide
echo   %CYAN%STARTUP_GUIDE.md%RESET%       - Preflight checks documentation
echo   %CYAN%CONFIGURATION_FIXES.md%RESET% - Configuration troubleshooting
echo   %CYAN%README.md%RESET%              - Project overview
echo.

echo %BLUE%COMMON WORKFLOWS:%RESET%
echo.
echo   %YELLOW%First-time setup:%RESET%
echo     apply_config.bat
echo     setup_data.bat
echo     run_lab.bat
echo.
echo   %YELLOW%Add new data:%RESET%
echo     python src\validate_jsonl.py new_data.jsonl data\validated\new_data.jsonl
echo     python src\chunker.py data\validated\new_data.jsonl data\chunked\new_data.jsonl
echo     python scripts\build_index.py data\chunked\new_data.jsonl index\vector_store.npz
echo     run_lab.bat
echo.
echo   %YELLOW%Troubleshooting:%RESET%
echo     python scripts\preflight_check.py  (check system health)
echo     type .env ^| findstr "USE_QDRANT EMBEDDING_DIM"  (verify config)
echo     dir index\vector_store.npz  (verify index exists)
echo.

echo %CYAN%========================================================================%RESET%
echo %CYAN%For detailed help, see: SETUP_INSTRUCTIONS.md%RESET%
echo %CYAN%========================================================================%RESET%
echo.
pause

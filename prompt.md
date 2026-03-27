PS D:\husc-admission-chat-enrollment> cd .\rag2025\
PS D:\husc-admission-chat-enrollment\rag2025> .\run_lab.bat

[94m========================================================================[0m
[94m   2025 RAG Lab - Startup Sequence[0m
[94m========================================================================[0m

[94m[STEP 1][0m Validating environment...

[92m[OK][0m Found src\main.py - correct directory

[94m[STEP 2][0m Setting up Python environment...

[92m[OK][0m Using Python: python
Python 3.13.0

[94m[INFO][0m Activating virtual environment...
[92m[OK][0m Virtual environment activated

[94m[INFO][0m Checking dependencies...
[92m[OK][0m Dependencies ready

[94m[STEP 3][0m Running preflight checks...


======================================================================
                       RAG Lab Preflight Checks
======================================================================

Running pre-startup validation...


======================================================================
                   CHECK 1: Configuration Integrity
======================================================================

[OK] Settings loaded successfully
  • Embedding Model: Qwen/Qwen3-Embedding-4B
  • Embedding Dimension: 2560
  • Index Directory: index
[OK] Configuration check passed


======================================================================
                    CHECK 2: LanceDB Connectivity
======================================================================

  • URI: ./data/lancedb
  • Table: rag2025
[OK] LanceDB connection successful
  • Rows: 180


======================================================================
                      CHECK 3: Data Availability
======================================================================

[OK] LanceDB table loaded: 180 rows


======================================================================
                          Preflight Summary
======================================================================

[OK] Configuration: PASSED
[OK] LanceDB: PASSED
[OK] Data: PASSED


[OK] All preflight checks passed! Starting server...

[92m========================================================================[0m
[92m   PREFLIGHT CHECKS PASSED[0m
[92m========================================================================[0m

[94m[STEP 4][0m Starting API server...

[92mServer URL:[0m http://localhost:8000
[92mSwagger UI:[0m http://localhost:8000/docs

[93mPress Ctrl+C to stop the server[0m

INFO:     Will watch for changes in these directories: ['D:\\husc-admission-chat-enrollment\\rag2025']
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [25920] using WatchFiles
D:\husc-admission-chat-enrollment\rag2025\src\services\query_enhancer.py:16: FutureWarning: 

All support for the `google.generativeai` package has ended. It will no longer be receiving
updates or bug fixes. Please switch to the `google.genai` package as soon as possible.
See README for more details:

https://github.com/google-gemini/deprecated-generative-ai-python/blob/main/README.md

  import google.generativeai as genai
2026-03-27 16:55:20.475 | INFO     | services.query_enhancer:__init__:36 - Loaded HYDE prompt from D:\husc-admission-chat-enrollment\rag2025\prompts\hyde_system_prompt.txt
2026-03-27 16:55:21.301 | INFO     | services.llm_generator:__init__:41 - Loaded generation prompt from D:\husc-admission-chat-enrollment\rag2025\prompts\generation_system_prompt.txt
2026-03-27 16:55:21.301 | WARNING  | services.llm_generator:__init__:56 - ZAI_API_KEY not set, GLM-4.5 disabled
2026-03-27 16:55:21.301 | WARNING  | services.llm_generator:__init__:65 - GROQ_API_KEY not set, Groq disabled
Process SpawnProcess-1:
Traceback (most recent call last):
  File "C:\Users\khoad\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\process.py", line 313, in _bootstrap
    self.run()
    ~~~~~~~~^^
  File "C:\Users\khoad\AppData\Local\Programs\Python\Python313\Lib\multiprocessing\process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\husc-admission-chat-enrollment\rag2025\venv\Lib\site-packages\uvicorn\_subprocess.py", line 80, in subprocess_started
    target(sockets=sockets)
    ~~~~~~^^^^^^^^^^^^^^^^^
  File "D:\husc-admission-chat-enrollment\rag2025\venv\Lib\site-packages\uvicorn\server.py", line 75, in run
    return asyncio_run(self.serve(sockets=sockets), loop_factory=self.config.get_loop_factory())
  File "C:\Users\khoad\AppData\Local\Programs\Python\Python313\Lib\asyncio\runners.py", line 194, in run
    return runner.run(main)
           ~~~~~~~~~~^^^^^^
  File "C:\Users\khoad\AppData\Local\Programs\Python\Python313\Lib\asyncio\runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "C:\Users\khoad\AppData\Local\Programs\Python\Python313\Lib\asyncio\base_events.py", line 721, in run_until_complete
    return future.result()
           ~~~~~~~~~~~~~^^
  File "D:\husc-admission-chat-enrollment\rag2025\venv\Lib\site-packages\uvicorn\server.py", line 79, in serve
    await self._serve(sockets)
  File "D:\husc-admission-chat-enrollment\rag2025\venv\Lib\site-packages\uvicorn\server.py", line 86, in _serve
    config.load()
    ~~~~~~~~~~~^^
  File "D:\husc-admission-chat-enrollment\rag2025\venv\Lib\site-packages\uvicorn\config.py", line 441, in load
    self.loaded_app = import_from_string(self.app)
                      ~~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "D:\husc-admission-chat-enrollment\rag2025\venv\Lib\site-packages\uvicorn\importer.py", line 19, in import_from_string
    module = importlib.import_module(module_str)
  File "C:\Users\khoad\AppData\Local\Programs\Python\Python313\Lib\importlib\__init__.py", line 88, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 1022, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "D:\husc-admission-chat-enrollment\rag2025\src\main.py", line 34, in <module>
    from services.llm_generator import llm_generator
  File "D:\husc-admission-chat-enrollment\rag2025\src\services\llm_generator.py", line 231, in <module>
    llm_generator = LLMGenerator()
  File "D:\husc-admission-chat-enrollment\rag2025\src\services\llm_generator.py", line 68, in __init__
    raise Exception("At least one LLM API key required (ZAI_API_KEY or GROQ_API_KEY)!")
Exception: At least one LLM API key required (ZAI_API_KEY or GROQ_API_KEY)!

Log ở trên là của khi chạy run_lab.bat


PS D:\husc-admission-chat-enrollment> cd .\rag2025\
PS D:\husc-admission-chat-enrollment\rag2025> .\setup_data.bat data/raw

[94m========================================================================[0m
[94m   2025 RAG Lab - Data Ingestion Pipeline[0m
[94m========================================================================[0m

[94m[STEP 0][0m Setting up environment...

[92m[OK][0m Using Python: python

[94m[INFO][0m Activating virtual environment...
[92m[OK][0m Virtual environment activated

[94m[INFO][0m Creating data directories...
[92m[OK][0m Directories ready

[93m[WARNING][0m Input file not found: ..\2.jsonl
[93m[INFO][0m Looking for alternative locations...
[92m[OK][0m Found: data\raw\2.jsonl
[92m[OK][0m Input file found: data\raw\2.jsonl

[94m========================================================================[0m
[94m[STEP 1][0m Normalizing raw data...
[94m========================================================================[0m

Running: python src\normalize_data.py "data\raw\2.jsonl" "data\normalized\normalized_2.jsonl"

2026-03-27 16:39:08.393 | INFO     | __main__:normalize_jsonl:305 - Normalizing data\raw\2.jsonl -> data\normalized\normalized_2.jsonl
2026-03-27 16:39:08.402 | INFO     | __main__:normalize_jsonl:329 - Normalization complete: 4 records

Normalization Report:
  Total: 4
  Normalized: 4
  Skipped: 0
  Errors: 0

[92m[OK][0m Normalization complete: data\normalized\normalized_2.jsonl

[94m========================================================================[0m
[94m[STEP 2][0m Validating JSONL data...
[94m========================================================================[0m

Running: python src\validate_jsonl.py "data\normalized\normalized_2.jsonl" "config\rag_chunk_schema.json" "data\validated\validated_2.jsonl"

2026-03-27 16:39:08.699 | INFO     | __main__:validate_jsonl:117 - Starting validation: data\normalized\normalized_2.jsonl
2026-03-27 16:39:08.706 | INFO     | __main__:load_schema:95 - Loaded schema from config\rag_chunk_schema.json
2026-03-27 16:39:08.713 | DEBUG    | __main__:detect_encoding:69 - Detected encoding: utf-8 (confidence: 99.00%) for normalized_2.jsonl
2026-03-27 16:39:08.783 | INFO     | __main__:validate_jsonl:195 - Validated records written to data\validated\validated_2.jsonl
2026-03-27 16:39:08.783 | INFO     | __main__:validate_jsonl:197 - Validation Report:
  Total: 4 | Valid: 4 | Invalid: 0 | Success: 100.0%
  Encoding: utf-8
  Errors: 0 | Warnings: 0

Validation Report:
  Total: 4 | Valid: 4 | Invalid: 0 | Success: 100.0%
  Encoding: utf-8
  Errors: 0 | Warnings: 0

Top 5 Errors:

[92m[OK][0m Validation complete: data\validated\validated_2.jsonl

[94m========================================================================[0m
[94m[STEP 3][0m Chunking documents...
[94m========================================================================[0m

Running: python src\chunker.py "data\validated\validated_2.jsonl" "data\chunked\chunked_2.jsonl" "config\chunk_profiles.yaml" auto

2026-03-27 16:39:09.050 | INFO     | __main__:__init__:48 - Loaded 3 chunk profiles from config\chunk_profiles.yaml
2026-03-27 16:39:09.148 | INFO     | __main__:__init__:96 - Chunker initialized with encoding: cl100k_base
2026-03-27 16:39:09.150 | DEBUG    | __main__:chunk_document:116 - Chunking doc all_Bo_GDDT_ma_phuong_thuc_xet_tuyen_2025 with profile: auto
2026-03-27 16:39:09.159 | INFO     | __main__:chunk_document:149 - Doc 2_17963d0e_0001 chunked into 1 chunks with profile auto
2026-03-27 16:39:09.160 | DEBUG    | __main__:chunk_document:116 - Chunking doc csdt_admin_Bo_GDDT_ma_to_hop_xet_tuyen_2025 with profile: auto
2026-03-27 16:39:09.160 | INFO     | __main__:chunk_document:149 - Doc 2_17963d0e_0002 chunked into 1 chunks with profile auto
2026-03-27 16:39:09.160 | DEBUG    | __main__:chunk_document:116 - Chunking doc thi_sinh_dai_hoc_hue_cac_phuong_thuc_tuyen_sinh_2025 with profile: auto        
2026-03-27 16:39:09.161 | INFO     | __main__:chunk_document:149 - Doc 2_17963d0e_0003 chunked into 1 chunks with profile auto
2026-03-27 16:39:09.161 | DEBUG    | __main__:chunk_document:116 - Chunking doc thi_sinh_dai_hoc_hue_danh_muc_to_hop_nganh_2025 with profile: auto
2026-03-27 16:39:09.161 | INFO     | __main__:chunk_document:149 - Doc 2_17963d0e_0004 chunked into 1 chunks with profile auto
2026-03-27 16:39:09.161 | INFO     | __main__:chunk_jsonl:471 - Chunked 4 total chunks to data\chunked\chunked_2.jsonl

Chunked 4 total chunks.

[92m[OK][0m Chunking complete: data\chunked\chunked_2.jsonl

[94m========================================================================[0m
[94m[STEP 4][0m Building vector index...
[94m========================================================================[0m

Running: python scripts\build_index.py "data\chunked" "index\vector_store.npz"

2026-03-27 16:39:22.510 | INFO     | __main__:build_index_from_chunked_folder:187 - ============================================================
2026-03-27 16:39:22.510 | INFO     | __main__:build_index_from_chunked_folder:188 - Building Vector Index from All Chunked Files
2026-03-27 16:39:22.510 | INFO     | __main__:build_index_from_chunked_folder:189 - ============================================================
2026-03-27 16:39:22.510 | INFO     | __main__:build_index_from_chunked_folder:190 - Input folder: data\chunked
2026-03-27 16:39:22.511 | INFO     | __main__:build_index_from_chunked_folder:191 - Output: index\vector_store.npz
2026-03-27 16:39:22.511 | INFO     | __main__:build_index_from_chunked_folder:199 - Found 13 chunked files:
2026-03-27 16:39:22.511 | INFO     | __main__:build_index_from_chunked_folder:201 -   - chunked_1.jsonl
2026-03-27 16:39:22.511 | INFO     | __main__:build_index_from_chunked_folder:201 -   - chunked_10.jsonl
2026-03-27 16:39:22.511 | INFO     | __main__:build_index_from_chunked_folder:201 -   - chunked_10_enhanced.jsonl
2026-03-27 16:39:22.512 | INFO     | __main__:build_index_from_chunked_folder:201 -   - chunked_11.jsonl
2026-03-27 16:39:22.512 | INFO     | __main__:build_index_from_chunked_folder:201 -   - chunked_2.jsonl
2026-03-27 16:39:22.512 | INFO     | __main__:build_index_from_chunked_folder:201 -   - chunked_3.jsonl
2026-03-27 16:39:22.512 | INFO     | __main__:build_index_from_chunked_folder:201 -   - chunked_4.jsonl
2026-03-27 16:39:22.512 | INFO     | __main__:build_index_from_chunked_folder:201 -   - chunked_5.jsonl
2026-03-27 16:39:22.512 | INFO     | __main__:build_index_from_chunked_folder:201 -   - chunked_6.jsonl
2026-03-27 16:39:22.512 | INFO     | __main__:build_index_from_chunked_folder:201 -   - chunked_7.jsonl
2026-03-27 16:39:22.513 | INFO     | __main__:build_index_from_chunked_folder:201 -   - chunked_8.jsonl
2026-03-27 16:39:22.513 | INFO     | __main__:build_index_from_chunked_folder:201 -   - chunked_9.jsonl
2026-03-27 16:39:22.513 | INFO     | __main__:build_index_from_chunked_folder:201 -   - chunked_test.jsonl
2026-03-27 16:39:22.513 | INFO     | __main__:load_chunks:44 - Loading chunks from: data\chunked\chunked_1.jsonl
2026-03-27 16:39:22.520 | INFO     | __main__:load_chunks:111 - Loaded 18 chunks from data\chunked\chunked_1.jsonl
2026-03-27 16:39:22.520 | INFO     | __main__:load_chunks:44 - Loading chunks from: data\chunked\chunked_10.jsonl
2026-03-27 16:39:22.528 | INFO     | __main__:load_chunks:111 - Loaded 32 chunks from data\chunked\chunked_10.jsonl
2026-03-27 16:39:22.528 | INFO     | __main__:load_chunks:44 - Loading chunks from: data\chunked\chunked_10_enhanced.jsonl
2026-03-27 16:39:22.535 | INFO     | __main__:load_chunks:111 - Loaded 35 chunks from data\chunked\chunked_10_enhanced.jsonl
2026-03-27 16:39:22.536 | INFO     | __main__:load_chunks:44 - Loading chunks from: data\chunked\chunked_11.jsonl
2026-03-27 16:39:22.543 | INFO     | __main__:load_chunks:111 - Loaded 12 chunks from data\chunked\chunked_11.jsonl
2026-03-27 16:39:22.543 | INFO     | __main__:load_chunks:44 - Loading chunks from: data\chunked\chunked_2.jsonl
2026-03-27 16:39:22.549 | INFO     | __main__:load_chunks:111 - Loaded 4 chunks from data\chunked\chunked_2.jsonl
2026-03-27 16:39:22.550 | INFO     | __main__:load_chunks:44 - Loading chunks from: data\chunked\chunked_3.jsonl
2026-03-27 16:39:22.576 | INFO     | __main__:load_chunks:111 - Loaded 33 chunks from data\chunked\chunked_3.jsonl
2026-03-27 16:39:22.577 | INFO     | __main__:load_chunks:44 - Loading chunks from: data\chunked\chunked_4.jsonl
2026-03-27 16:39:22.584 | INFO     | __main__:load_chunks:111 - Loaded 2 chunks from data\chunked\chunked_4.jsonl
2026-03-27 16:39:22.584 | INFO     | __main__:load_chunks:44 - Loading chunks from: data\chunked\chunked_5.jsonl
2026-03-27 16:39:22.591 | INFO     | __main__:load_chunks:111 - Loaded 13 chunks from data\chunked\chunked_5.jsonl
2026-03-27 16:39:22.592 | INFO     | __main__:load_chunks:44 - Loading chunks from: data\chunked\chunked_6.jsonl
2026-03-27 16:39:22.598 | INFO     | __main__:load_chunks:111 - Loaded 13 chunks from data\chunked\chunked_6.jsonl
2026-03-27 16:39:22.599 | INFO     | __main__:load_chunks:44 - Loading chunks from: data\chunked\chunked_7.jsonl
2026-03-27 16:39:22.599 | INFO     | __main__:load_chunks:111 - Loaded 0 chunks from data\chunked\chunked_7.jsonl
2026-03-27 16:39:22.599 | INFO     | __main__:load_chunks:44 - Loading chunks from: data\chunked\chunked_8.jsonl
2026-03-27 16:39:22.611 | INFO     | __main__:load_chunks:111 - Loaded 16 chunks from data\chunked\chunked_8.jsonl
2026-03-27 16:39:22.611 | INFO     | __main__:load_chunks:44 - Loading chunks from: data\chunked\chunked_9.jsonl
2026-03-27 16:39:22.617 | INFO     | __main__:load_chunks:111 - Loaded 5 chunks from data\chunked\chunked_9.jsonl
2026-03-27 16:39:22.617 | INFO     | __main__:load_chunks:44 - Loading chunks from: data\chunked\chunked_test.jsonl
2026-03-27 16:39:22.623 | INFO     | __main__:load_chunks:111 - Loaded 4 chunks from data\chunked\chunked_test.jsonl
2026-03-27 16:39:22.624 | INFO     | __main__:build_index_from_chunked_folder:214 - Total chunks from all files: 187
2026-03-27 16:39:22.624 | INFO     | __main__:build_index_from_chunked_folder:217 - Initializing embedding service...
2026-03-27 16:39:22.624 | INFO     | services.embedding:__init__:29 - Loading embedding model: Qwen/Qwen3-Embedding-4B
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 398/398 [00:00<00:00, 3095.04it/s]
2026-03-27 16:39:36.535 | INFO     | __main__:build_index_from_chunked_folder:220 - EmbeddingService initialized: model=Qwen/Qwen3-Embedding-4B, dim=2560
2026-03-27 16:39:36.544 | INFO     | __main__:build_index_from_chunked_folder:223 - Generating embeddings for 187 chunks...
Log ở trên là chạy setup_data.bat 
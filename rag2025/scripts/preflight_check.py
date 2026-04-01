"""
Preflight Health Check Script for RAG Lab

Performs strict validation before server startup:
1. Configuration integrity check (embedding dimensions)
2. LanceDB connectivity
3. Data availability
4. LLM provider readiness (full vs degraded mode)

Exit codes:
- 0: All checks passed
- 1: Critical failure (missing config, connection failed)
"""
import sys
from pathlib import Path

# Add project root and src to sys.path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from config.settings import RAGSettings
from services.vector_store import NumpyVectorStore

# ANSI color codes for Windows
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_header(text: str):
    """Print section header."""
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}{text.center(70)}{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{GREEN}[OK] {text}{RESET}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{YELLOW}[WARNING] {text}{RESET}")


def print_error(text: str):
    """Print error message."""
    print(f"{RED}[ERROR] {text}{RESET}")


def check_configuration() -> bool:
    """
    Check 1: Configuration Integrity
    
    Validates:
    - Settings can be loaded
    - Embedding dimension matches model expectations
    - Required paths exist
    
    Returns:
        True if configuration is valid, False otherwise
    """
    print_header("CHECK 1: Configuration Integrity")
    
    try:
        # Load settings
        config = RAGSettings()
        print_success(f"Settings loaded successfully")
        print(f"  • Embedding Provider: {config.EMBEDDING_PROVIDER}")
        print(f"  • Embedding Model: {config.EMBEDDING_MODEL}")
        print(f"  • Embedding Dimension: {config.EMBEDDING_DIM}")
        print(f"  • Index Directory: {config.INDEX_DIR}")

        # Validate known model-dimension pairs (soft warning)
        model_dim_expectations = {
            "qwen/qwen3-embedding-0.6b": 1024,
            "qwen/qwen3-embedding-4b": 2560,
            "qwen/qwen3-embedding-8b": 4096,
            "microsoft/harrier-oss-v1-270m": 640,
            "microsoft/harrier-oss-v1-0.6b": 1024,
            "microsoft/harrier-oss-v1-27b": 5376,
            "baai/bge-m3": 1024,
        }

        model_lower = config.EMBEDDING_MODEL.lower()
        expected_dim = model_dim_expectations.get(model_lower)
        if expected_dim is not None and config.EMBEDDING_DIM != expected_dim:
            print_warning(
                f"Dimension mismatch: {config.EMBEDDING_MODEL} expects {expected_dim} dims, "
                f"but config has {config.EMBEDDING_DIM}"
            )
            print_warning("Please update EMBEDDING_DIM in .env to match selected model")
        
        # Check if index directory exists
        if not config.INDEX_DIR.exists():
            print_warning(f"Index directory does not exist: {config.INDEX_DIR}")
            print_warning(f"Creating directory...")
            config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
        
        print_success(f"Configuration check passed\n")
        return True
        
    except Exception as e:
        print_error(f"Configuration error: {e}")
        return False


def check_lancedb_connectivity() -> bool:
    """
    Check 2: LanceDB Connectivity.

    Returns:
        True if LanceDB table exists and is accessible, False otherwise
    """
    print_header("CHECK 2: LanceDB Connectivity")

    try:
        config = RAGSettings()
        if not config.USE_LANCEDB:
            print_warning("LanceDB disabled (USE_LANCEDB=false)")
            return False

        from src.infrastructure.lancedb_adapter import LanceDBAdapter

        print(f"  • URI: {config.LANCEDB_URI}")
        print(f"  • Table: {config.LANCEDB_TABLE}")

        adapter = LanceDBAdapter(uri=config.LANCEDB_URI, table_name=config.LANCEDB_TABLE)
        if not adapter.table_exists():
            print_warning(f"Table '{config.LANCEDB_TABLE}' not found. Attempting bootstrap from legacy index...")
            import subprocess
            bootstrap_script = project_root / "scripts" / "bootstrap_lancedb.py"
            result = subprocess.run([sys.executable, str(bootstrap_script)], cwd=str(project_root), check=False)
            if result.returncode != 0:
                print_error(f"Table '{config.LANCEDB_TABLE}' not found")
                print_error("Run: python scripts/ingest_lancedb.py")
                return False
            adapter = LanceDBAdapter(uri=config.LANCEDB_URI, table_name=config.LANCEDB_TABLE)
            if not adapter.table_exists():
                print_error(f"Bootstrap did not create table '{config.LANCEDB_TABLE}'")
                return False

        print_success("LanceDB connection successful")
        print(f"  • Rows: {adapter.count()}\n")
        return True

    except Exception as e:
        print_error(f"LanceDB check error: {e}")
        return False


def check_data_availability() -> bool:
    """
    Check 3: Data Availability
    
    Verifies:
    - Vector store file exists
    - Can be loaded
    - Contains vectors
    
    Returns:
        True if data check passes (allows empty store), False on critical error
    """
    print_header("CHECK 3: Data Availability")
    
    try:
        config = RAGSettings()

        # Primary check: LanceDB table
        from src.infrastructure.lancedb_adapter import LanceDBAdapter
        adapter = LanceDBAdapter(uri=config.LANCEDB_URI, table_name=config.LANCEDB_TABLE)
        if adapter.table_exists():
            row_count = adapter.count()
            if row_count == 0:
                print_warning(f"LanceDB table is EMPTY (0 rows)")
            else:
                print_success(f"LanceDB table loaded: {row_count} rows")
            print("")
            return True

        # Legacy fallback check for migration compatibility
        vector_store_path = config.INDEX_DIR / "vector_store.npz"
        if not vector_store_path.exists():
            print_warning(f"LanceDB table not found and legacy vector store not found: {vector_store_path}")
            print_warning("Run: python scripts/ingest_lancedb.py")
            print("")
            return True

        print_success(f"Legacy vector store file found: {vector_store_path}")

        try:
            store = NumpyVectorStore(dim=config.EMBEDDING_DIM)
            store.load(vector_store_path)
            vector_count = store.count()
            if vector_count == 0:
                print_warning(f"Legacy vector store is EMPTY (0 vectors)")
            else:
                print_success(f"Legacy vector store loaded: {vector_count} vectors")
                if store.metadatas:
                    sample_meta = store.metadatas[0]
                    print(f"  • Sample doc_id: {sample_meta.get('doc_id', 'unknown')}")
                    print(f"  • Sample chunk text length: {len(sample_meta.get('text', ''))}")
            print("")
            return True

        except Exception as e:
            print_error(f"Failed to load legacy vector store: {e}")
            print_error("The file may be corrupted. Try rebuilding ingestion.")
            return False

    except Exception as e:
        print_error(f"Data availability check error: {e}")
        return False


def check_llm_readiness() -> bool:
    """
    Check 4: LLM Provider Readiness

    Verifies runtime generation provider availability without forcing startup crash.

    Returns:
        True. Missing providers are reported as degraded-mode warnings.
    """
    print_header("CHECK 4: LLM Provider Readiness")

    try:
        config = RAGSettings()

        providers = []
        if config.RAMCLOUDS_API_KEY or config.OPENAI_API_KEY:
            model = getattr(config, "RAMCLOUDS_MODEL", "") or "(default model)"
            providers.append(f"ramclouds/openai-compat [{model}]")
        if config.GROQ_API_KEY:
            providers.append("groq")
        if config.ZAI_API_KEY:
            providers.append("zai/glm")

        if providers:
            print_success(f"Generation providers configured: {', '.join(providers)}")
            print("")
            return True

        print_warning("No generation provider key found (RAMCLOUDS_API_KEY/OPENAI_API_KEY/GROQ_API_KEY/ZAI_API_KEY)")
        print_warning("API can still boot for /docs and health checks, but answer generation may run in degraded fallback mode.")
        print_warning("Set at least one provider key for full generation quality.")
        print("")
        return True

    except Exception as e:
        print_error(f"LLM readiness check error: {e}")
        return True


def main():
    """Run all preflight checks."""
    print_header("RAG Lab Preflight Checks")
    print(f"Running pre-startup validation...\n")
    
    # Track results
    checks = {
        "Configuration": check_configuration(),
        "LanceDB": check_lancedb_connectivity(),
        "Data": check_data_availability(),
        "LLM": check_llm_readiness(),
    }
    
    # Summary
    print_header("Preflight Summary")
    
    all_passed = True
    for check_name, result in checks.items():
        if result:
            print_success(f"{check_name}: PASSED")
        else:
            print_error(f"{check_name}: FAILED")
            all_passed = False
    
    print(f"\n")
    
    if all_passed:
        print_success(f"All preflight checks passed! Starting server...\n")
        sys.exit(0)
    else:
        print_error(f"Preflight checks FAILED. Server will NOT start.")
        print_error(f"Please fix the errors above and try again.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

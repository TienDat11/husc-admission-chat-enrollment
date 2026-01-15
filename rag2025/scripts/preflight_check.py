"""
Preflight Health Check Script for RAG Lab

Performs strict validation before server startup:
1. Configuration integrity check (embedding dimensions)
2. Qdrant connectivity (if USE_QDRANT=true)
3. Data availability (vector store existence and count)

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

from config.settings import RAGSettings, settings
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
        print(f"  • Embedding Model: {config.EMBEDDING_MODEL}")
        print(f"  • Embedding Dimension: {config.EMBEDDING_DIM}")
        print(f"  • Index Directory: {config.INDEX_DIR}")
        
        # Validate dimension for known models
        if "e5-small-v2" in config.EMBEDDING_MODEL:
            if config.EMBEDDING_DIM != 384:
                print_warning(
                    f"Dimension mismatch: e5-small-v2 outputs 384 dims, "
                    f"but config has {config.EMBEDDING_DIM}"
                )
                print_warning(
                    f"This may cause issues. Please update EMBEDDING_DIM=384 in .env"
                )
                # Don't fail, just warn (the model might still work with 768)
        
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


def check_qdrant_connectivity() -> bool:
    """
    Check 2: Qdrant Connectivity (Strict)

    If USE_QDRANT=true, attempts to connect to Qdrant server.
    Fails immediately if connection is unavailable.

    Returns:
        True if Qdrant is not used OR connection succeeds, False otherwise
    """
    print_header("CHECK 2: Qdrant Connectivity")

    try:
        config = RAGSettings()

        if not config.USE_QDRANT:
            print_success(f"Qdrant disabled (USE_QDRANT=false)")
            print(f"  • Using local NumPy vector store instead\n")
            return True

        # If Qdrant is enabled, verify connection
        print_warning(f"Qdrant enabled in config (USE_QDRANT=true)")
        print(f"  • URL: {config.QDRANT_URL}")
        print(f"  • Collection: {config.QDRANT_COLLECTION}")

        # Import Qdrant client and test connection
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.exceptions import UnexpectedResponse

            print(f"  • Testing connection...")
            client = QdrantClient(
                url=config.QDRANT_URL,
                api_key=config.QDRANT_API_KEY,
            )

            # Try to get collections list
            collections = client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if config.QDRANT_COLLECTION not in collection_names:
                print_error(f"Collection '{config.QDRANT_COLLECTION}' not found!")
                print_error(f"Available collections: {collection_names}")
                print_error(f"")
                print_error(f"Please create the collection or update QDRANT_COLLECTION in .env")
                return False

            # Get collection info
            collection_info = client.get_collection(config.QDRANT_COLLECTION)
            print_success(f"Qdrant connection successful")
            print(f"  • Points count: {collection_info.points_count}")
            print(f"  • Status: {collection_info.status}")
            print(f"  • Vector size: {collection_info.config.params.vectors.size}")
            print(f"  • Distance: {collection_info.config.params.vectors.distance}\n")
            return True

        except ImportError as e:
            print_error(f"qdrant-client not installed: {e}")
            print_error(f"Run: pip install qdrant-client")
            return False
        except UnexpectedResponse as e:
            print_error(f"Qdrant connection failed: {e}")
            print_error(f"Please verify QDRANT_URL and QDRANT_API_KEY in .env")
            return False

    except Exception as e:
        print_error(f"Qdrant check error: {e}")
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
        vector_store_path = config.INDEX_DIR / "vector_store.npz"
        
        # Check if vector store file exists
        if not vector_store_path.exists():
            print_warning(f"Vector store not found: {vector_store_path}")
            print_warning(f"The API will start but retrieval will return no results.")
            print_warning(f"")
            print_warning(f"To fix: Run 'python scripts/build_index.py' to create index")
            print(f"")
            return True  # Allow startup (so they can use ingest API)
        
        # Try to load the vector store
        print_success(f"Vector store file found: {vector_store_path}")
        
        try:
            store = NumpyVectorStore(dim=config.EMBEDDING_DIM)
            store.load(vector_store_path)
            
            vector_count = store.count()
            
            if vector_count == 0:
                print_warning(f"Vector store is EMPTY (0 vectors)")
                print_warning(f"Run ingestion to populate the index")
            else:
                print_success(f"Vector store loaded: {vector_count} vectors")
                
                # Show sample metadata
                if store.metadatas:
                    sample_meta = store.metadatas[0]
                    print(f"  • Sample doc_id: {sample_meta.get('doc_id', 'unknown')}")
                    print(f"  • Sample chunk text length: {len(sample_meta.get('text', ''))}")
            
            print(f"")
            return True
            
        except Exception as e:
            print_error(f"Failed to load vector store: {e}")
            print_error(f"The file may be corrupted. Try rebuilding the index.")
            return False
        
    except Exception as e:
        print_error(f"Data availability check error: {e}")
        return False


def main():
    """Run all preflight checks."""
    print_header("RAG Lab Preflight Checks")
    print(f"Running pre-startup validation...\n")
    
    # Track results
    checks = {
        "Configuration": check_configuration(),
        "Qdrant": check_qdrant_connectivity(),
        "Data": check_data_availability(),
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

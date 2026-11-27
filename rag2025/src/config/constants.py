"""Named tunables for the rag2025 services.

Values equal the literals they replace. Env overrides exist only where a
setting is operator-tunable.
"""
import os

# --- LLM timeouts (seconds) ---
#: Read timeout for the generator call (was a literal in llm_generator.py).
LLM_READ_TIMEOUT_S = 30.0
#: Timeout for the query router's LLM calls.
QUERY_ROUTER_LLM_TIMEOUT_S = 8.0

# --- Student profile rate limit (per client IP) ---
STUDENT_PROFILE_RATE_LIMIT_PER_MINUTE = 30
STUDENT_PROFILE_RATE_LIMIT_WINDOW_S = 60


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


# --- Reverse proxy trust ---
#: Number of proxy hops in front of the app whose X-Forwarded-For entries are
#: trusted. Caddy is one hop; 0 ignores the header entirely.
TRUSTED_PROXY_COUNT = _env_int("TRUSTED_PROXY_COUNT", 1)

# --- Chunk preview (POST /debug/preview-chunks) input cap ---
#: Max characters accepted for a preview request's text.
CHUNK_PREVIEW_MAX_TEXT_LENGTH = 20000

# --- Input limits and graph-update tunables (env-overridable; defaults unchanged) ---
MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "1000"))
MAX_GRAPH_UPDATE_CHUNKS = int(os.getenv("MAX_GRAPH_UPDATE_CHUNKS", "100"))
MAX_GRAPH_CHUNK_TEXT_LENGTH = int(os.getenv("MAX_GRAPH_CHUNK_TEXT_LENGTH", "5000"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
GROUNDING_THRESHOLD = float(os.getenv("GROUNDING_THRESHOLD", "0.18"))

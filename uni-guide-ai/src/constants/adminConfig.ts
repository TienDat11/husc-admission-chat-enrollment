export interface AdminConfigEntry {
  key: string;
  label: string;
  value: string;
  source: string;
  howToChange: string;
}

export const adminConfig: AdminConfigEntry[] = [
  {
    key: "CURRENT_ADMISSION_YEAR",
    label: "Admission Year",
    value: "2026",
    source: "Loaded from .env (rag2025/.env)",
    howToChange: "Edit CURRENT_ADMISSION_YEAR in rag2025/.env and restart the server.",
  },
  {
    key: "LANCEDB_TABLE",
    label: "LanceDB Table",
    value: "husc_chunks",
    source: "Loaded from .env (rag2025/.env)",
    howToChange: "Edit LANCEDB_TABLE in rag2025/.env and re-ingest data into the new table.",
  },
  {
    key: "EMBEDDING_MODEL",
    label: "Embedding Model",
    value: "Qwen/Qwen3-Embedding-0.6B",
    source: "Loaded from .env (rag2025/.env)",
    howToChange: "Edit EMBEDDING_MODEL in rag2025/.env and re-run the embedding pipeline.",
  },
  {
    key: "NS24_BASE_URL",
    label: "ns24 Gateway Base URL",
    value: "https://api.ns24.io/v1",
    source: "config/settings.py default; overridable by NS24_BASE_URL in rag2025/.env",
    howToChange: "Edit NS24_BASE_URL in rag2025/.env and restart the server.",
  },
  {
    key: "NS24_MODEL",
    label: "Default Role Model",
    value: "gemini-3.5-flash-lite",
    source: "config/ns24_models.py ROLE_MODELS['default'] (fallback deepseek-v4.1-flash)",
    howToChange: "Edit ROLE_MODELS in rag2025/config/ns24_models.py and restart the server.",
  },
  {
    key: "NS24_ROUTER_MODEL",
    label: "Router, NER, HyDE, Guardrail Model",
    value: "gemini-3.5-flash-lite",
    source: "config/ns24_models.py ROLE_MODELS (fallback deepseek-v4.1-flash)",
    howToChange: "Edit ROLE_MODELS in rag2025/config/ns24_models.py and restart the server.",
  },
  {
    key: "NS24_GEN_MODEL",
    label: "Answer Generation Model",
    value: "gpt-6-luna",
    source: "config/ns24_models.py ROLE_MODELS['gen'] (fallback claude-haiku-5-5)",
    howToChange: "Edit ROLE_MODELS in rag2025/config/ns24_models.py and restart the server.",
  },
  {
    key: "RERANKER_MODEL",
    label: "Reranker Model",
    value: "BAAI/bge-reranker-v2-m3",
    source: "Loaded from .env (rag2025/.env)",
    howToChange: "Edit RERANKER_MODEL in rag2025/.env and restart the server.",
  },
];

export const NS24_API_KEY = "[REDACTED]";
export const NS24_API_KEY_NOTE = "Set in /srv/atlas-lab/shared/atlas.env on the server; never displayed.";

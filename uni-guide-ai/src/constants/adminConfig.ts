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
    key: "RAMCLOUDS_MODEL",
    label: "RAM Clouds Model",
    value: "gemini-3.6-flash-high",
    source: "Loaded from .env (rag2025/.env)",
    howToChange: "Edit RAMCLOUDS_MODEL in rag2025/.env and restart the server.",
  },
  {
    key: "RAMCLOUDS_BASE_URL",
    label: "RAM Clouds Base URL",
    value: "https://ramclouds.me/v1",
    source: "Loaded from .env (rag2025/.env)",
    howToChange: "Edit RAMCLOUDS_BASE_URL in rag2025/.env to point to your RAM Clouds deployment.",
  },
  {
    key: "RERANKER_MODEL",
    label: "Reranker Model",
    value: "BAAI/bge-reranker-v2-m3",
    source: "Loaded from .env (rag2025/.env)",
    howToChange: "Edit RERANKER_MODEL in rag2025/.env and restart the server.",
  },
];

export const RAMCLOUDS_API_KEY = "[REDACTED]";
export const RAMCLOUDS_API_KEY_NOTE =
  "Set in .env, never display full value.";

# @spec(S13.2) ChunkMetadataV3 schema with from_legacy() dual-read adapter
"""Pydantic v2 model for canonical chunk metadata + legacy migration adapter.

Strict mode (production reingest): every field required, validated.
from_legacy(): best-effort conversion from chunked_*.jsonl legacy schema for
transition window. Defaults are explicit and traceable.
"""
from __future__ import annotations
from datetime import date, datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


ChunkMethod = Literal["rule_v2", "haiku_v1", "claude_v1", "ensemble"]


class ChunkMetadataV3(BaseModel):
    """Canonical chunk metadata schema v3 — source-immutable, year-namespaced."""

    source_url: HttpUrl
    notification_id: int | None = None
    crawl_date: datetime
    data_year: int = Field(..., ge=2024, le=2030)
    chunk_method: ChunkMethod
    chunk_version_hash: str = Field(..., min_length=64, max_length=64)
    is_superseded: bool = False
    supersedes_id: str | None = None
    valid_from: date | None = None
    valid_to: date | None = None
    info_type: str
    audience: str = "thi_sinh"
    school: str = "HUSC"
    issuer: str | None = None
    effective_date: date | None = None
    expired: bool = False

    # Allow extension fields (passthrough for legacy keys we haven't promoted yet).
    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_legacy(cls, legacy: dict[str, Any], defaults: dict[str, Any] | None = None) -> "ChunkMetadataV3":
        """Build a v3 metadata object from a legacy chunk dict.

        Used during the transition window when legacy chunked_*.jsonl files still
        exist. Production reingest uses the strict constructor (no defaults). Any
        missing field is filled from `defaults` first, then from internal fallbacks.

        The fallback chain documents what was missing in legacy data:
          source_url:    legacy.source_url > defaults.source_url > "https://tuyensinh.husc.edu.vn/unknown"
          data_year:     legacy.data_year > legacy.year > defaults.data_year > 2025
          crawl_date:    legacy.crawl_date > defaults.crawl_date > datetime(year, 6, 1, tzinfo=UTC)
          chunk_method:  legacy.chunk_method > "rule_v2"
          chunk_version_hash: legacy.chunk_version_hash > "0" * 64
          info_type:     legacy.info_type > "unknown"
        """
        d = defaults or {}

        source_url = (
            legacy.get("source_url")
            or d.get("source_url")
            or "https://tuyensinh.husc.edu.vn/unknown"
        )
        # HIGH-3: fallback year from env (CURRENT_ADMISSION_YEAR), defaults 2026.
        # Previously hardcoded 2025 → silently mislabeled 2026 reingest content.
        import os
        env_year = int(os.getenv("CURRENT_ADMISSION_YEAR", "2026"))
        year_raw = (
            legacy.get("data_year")
            or legacy.get("year")
            or d.get("data_year")
            or env_year
        )
        year = int(year_raw)

        crawl_date = legacy.get("crawl_date") or d.get("crawl_date")
        if crawl_date is None:
            crawl_date = datetime(year, 6, 1, tzinfo=timezone.utc)
        elif isinstance(crawl_date, str):
            crawl_date = datetime.fromisoformat(crawl_date.replace("Z", "+00:00"))

        reserved = {
            "source_url", "notification_id", "crawl_date", "data_year", "year",
            "chunk_method", "chunk_version_hash", "is_superseded", "supersedes_id",
            "valid_from", "valid_to", "info_type", "audience", "school", "issuer",
            "effective_date", "expired",
        }
        passthrough = {k: v for k, v in legacy.items() if k not in reserved}

        return cls(
            source_url=source_url,
            notification_id=legacy.get("notification_id"),
            crawl_date=crawl_date,
            data_year=year,
            chunk_method=legacy.get("chunk_method", "rule_v2"),
            chunk_version_hash=legacy.get("chunk_version_hash") or ("0" * 64),
            is_superseded=bool(legacy.get("is_superseded", False)),
            supersedes_id=legacy.get("supersedes_id"),
            valid_from=legacy.get("valid_from"),
            valid_to=legacy.get("valid_to"),
            info_type=legacy.get("info_type", "unknown"),
            audience=legacy.get("audience", "thi_sinh"),
            school=legacy.get("school", "HUSC"),
            issuer=legacy.get("issuer"),
            effective_date=legacy.get("effective_date"),
            expired=bool(legacy.get("expired", False)),
            **passthrough,
        )

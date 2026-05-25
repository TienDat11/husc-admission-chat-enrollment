# @spec(S13.2) chunk metadata helpers — dual-read adapter for source label/url
"""Helpers to extract source identity from chunks during the schema v3 transition.

Reads from v3 fields (source_url, notification_id) when present; falls back to
legacy `source` field. All raw `metadata.get("source")` / `metadata["source"]`
access in the codebase MUST go through these helpers (enforced by the
check_no_raw_metadata_source.py grep gate).
"""
from __future__ import annotations
from typing import Any


_FALLBACK_LABEL = "Không rõ nguồn"


def _get_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    md = chunk.get("metadata") if isinstance(chunk, dict) else None
    return md if isinstance(md, dict) else {}


def get_source_url(chunk: dict[str, Any]) -> str | None:
    """Return canonical source_url from chunk metadata, or None when missing.

    Resolution order: metadata.source_url > top-level source_url > None.
    """
    md = _get_metadata(chunk)
    url = md.get("source_url") or chunk.get("source_url")
    return url if isinstance(url, str) and url else None


def get_notification_id(chunk: dict[str, Any]) -> int | None:
    """Return notification_id when present and integer-like; None otherwise."""
    md = _get_metadata(chunk)
    nid = md.get("notification_id")
    if nid is None:
        return None
    try:
        return int(nid)
    except (TypeError, ValueError):
        return None


def get_legacy_source(chunk: dict[str, Any]) -> str | None:
    """Return legacy `source` field (string) when present; None otherwise.

    This is the ONLY function permitted to read the legacy `source` key.
    All other call sites MUST go through `get_source_label()` instead.
    """
    md = _get_metadata(chunk)
    legacy = md.get("source")
    if isinstance(legacy, str) and legacy:
        return legacy
    legacy_top = chunk.get("source") if isinstance(chunk, dict) else None
    if isinstance(legacy_top, str) and legacy_top:
        return legacy_top
    return None


def get_source_label(chunk: dict[str, Any]) -> str:
    """Return a human-readable source label for display/citation.

    Resolution chain (v3-first, legacy-fallback):
      source_url > notification_id (formatted as "TB{id}") > legacy source > _FALLBACK_LABEL
    """
    url = get_source_url(chunk)
    if url:
        return url
    nid = get_notification_id(chunk)
    if nid is not None:
        return f"TB{nid}"
    legacy = get_legacy_source(chunk)
    if legacy:
        return legacy
    return _FALLBACK_LABEL


def get_source_breadcrumb(chunk: dict[str, Any]) -> str:
    """Return label suitable for breadcrumb display (legacy-compatible string).

    Used by chunker normalize paths that previously inlined `metadata["source"]`.
    """
    return get_source_label(chunk)

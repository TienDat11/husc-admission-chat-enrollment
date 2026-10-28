"""Canonical admissions record schema — versioned facts with provenance.

Public surface for the canonical domain layer. Importable standalone:
``python -c "import src.domain.canonical"`` from ``rag2025/``.
"""
from src.domain.canonical.record import (
    SCHEMA_VERSION,
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
    from_canonical_json,
    to_canonical_json,
)

__all__ = [
    "SCHEMA_VERSION",
    "AuthorityLevel",
    "CanonicalRecord",
    "RecordStatus",
    "RecordType",
    "from_canonical_json",
    "to_canonical_json",
]

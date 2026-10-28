"""Version-controlled list of high-impact canonical fields - ticket 25.

A field is *high-impact* when a change to it changes what a student is
served: the answer text, the numbers inside it, the window in which the
fact is live, or the provenance a citation points at. The reviewer's diff
must flag these fields no matter which client renders it, so the list lives
in code and is versioned, not in runtime config:

* an API consumer and the release notes replay the same classification;
* bumping ``HIGH_IMPACT_VERSION`` is a deliberate code change, so two diffs
  stamped with the same version are comparable and one stamped with a
  different version is not silently mixed with it.

See ``.scratch/husc-rag-admin-upgrade/issues/25-admin-review-diff-preview.md``.
"""
from __future__ import annotations

# Version of *this list*, not of the records it classifies. Any change to the
# membership of ``HIGH_IMPACT_FIELDS`` MUST bump this string.
HIGH_IMPACT_VERSION = "1"

# Top-level record fields whose change reaches the student-facing answer.
# ``text`` is the chunk content itself; ``values`` is the structured
# score/quota/tuition payload the spec names as high-impact facts (it is an
# allowed extra on ``CanonicalRecord``). Provenance fields are included
# because a changed source is a changed citation, and a wrong citation is a
# wrong answer even when the prose matches.
HIGH_IMPACT_FIELDS: frozenset[str] = frozenset(
    {
        "text",
        "values",
        "record_type",
        "admission_year",
        "effective_from",
        "effective_to",
        "authority_level",
        "locale",
        "source_document_id",
        "source_url",
        "source_page",
    }
)

# Fields that are lifecycle bookkeeping or version identity: they change on
# almost every edit and never change the served answer, so flagging them
# would train reviewers to ignore the flag.
NON_HIGH_IMPACT_FIELDS: frozenset[str] = frozenset(
    {
        "schema_version",
        "record_id",
        "version",
        "status",
        "published_at",
        "authored_by",
        "supersedes",
    }
)

# The two lists are meant to partition the known fields; a field in both
# would make the classification depend on which set the code checks first —
# exactly the silent drift a versioned list exists to prevent. Fail at import
# instead: HIGH_IMPACT_FIELDS is load-bearing, not decorative.
_CONFLICT = HIGH_IMPACT_FIELDS & NON_HIGH_IMPACT_FIELDS
if _CONFLICT:  # pragma: no cover - guards against future edits
    raise ValueError(
        "field(s) classified both high- and non-high-impact: "
        + ", ".join(sorted(_CONFLICT))
    )


def is_high_impact(field: str) -> bool:
    """Whether a change to ``field`` must be flagged high-impact in a diff.

    The versioned list answers for known fields; an unknown field defaults to
    high-impact: the list is versioned precisely so that a field nobody has
    classified yet is *shown* to the reviewer rather than silently downgraded.
    """
    if field in HIGH_IMPACT_FIELDS:
        return True
    if field in NON_HIGH_IMPACT_FIELDS:
        return False
    return True


def high_impact_version() -> str:
    """The version stamp the diff classification carries."""
    return HIGH_IMPACT_VERSION

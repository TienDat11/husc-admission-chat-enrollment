"""The knowledge graph as a release artifact — pin it at publish, verify it at restore.

The graph is one of the two artifacts a release *serves* (the LanceDB table is
the other). This module owns the one layout convention for where a release's
graph lives and the one pair of operations on it:

* :func:`pin_graph_artifact` — publish side. Copy the built graph into the
  release directory and report the pin (path + sha256 + schema version) the
  manifest records.
* :func:`load_pinned_graph` — restore side. Resolve the pin, verify the bytes
  against it, and only then parse. A rollback serves the artifact a release
  pinned, never "whatever graph happens to be on disk".

Why the bytes are verified rather than trusted: a rollback that loaded an
unverified graph could serve a *third* release's graph (rebuilt in place after
the release was published, say) while reporting the release it was asked for.
The digest is the only thing that makes "restored R1's graph" a claim about
bytes rather than about a file name.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from src.domain.graph import KnowledgeGraph
from src.domain.release import ReleaseManifest, compute_file_sha256
from src.infrastructure.canonical_store import read_manifest, write_manifest

#: Where a release's graph artifact lives, relative to its release directory.
#: One convention, because the pin stored in the manifest is resolved against
#: it and a release restored under a different layout would not be found.
GRAPH_ARTIFACT_RELATIVE_PATH = "graph/knowledge_graph.graphml"

__all__ = [
    "GRAPH_ARTIFACT_RELATIVE_PATH",
    "GraphArtifactError",
    "GraphArtifactMismatchError",
    "GraphArtifactMissingError",
    "GraphArtifactNotPinnedError",
    "GraphArtifactPin",
    "load_pinned_graph",
    "pin_graph_artifact",
    "publish_release_graph",
    "verify_pinned_graph",
]

class GraphArtifactError(Exception):
    """Base class for graph-artifact failures."""

class GraphArtifactNotPinnedError(GraphArtifactError):
    """The release pins no graph artifact."""

class GraphArtifactMissingError(GraphArtifactError):
    """The release pins a graph artifact, and it is not there."""

class GraphArtifactMismatchError(GraphArtifactError):
    """The artifact is there and is not the artifact the release pinned."""

@dataclass(frozen=True)
class GraphArtifactPin:
    """What a manifest records about a release's graph artifact."""

    path: str
    sha256: str
    schema_version: str

def _artifact_destination(release_dir: str | Path) -> Path:
    return Path(release_dir) / GRAPH_ARTIFACT_RELATIVE_PATH

def pin_graph_artifact(
    release_dir: str | Path, source: str | Path
) -> GraphArtifactPin:
    """Publish ``source`` as ``release_dir``'s graph artifact; return its pin.

    The artifact is copied, not moved: the caller keeps the build it just made
    (``data/graph/knowledge_graph.graphml``, the file the running process boots
    from), and the release keeps its own byte-identical copy. The returned
    digest is measured on the **copy**, and compared against the source, so a
    pin is always a claim about the bytes that are actually in the release.

    Call this **before** the release directory is marked immutable
    (:func:`~src.infrastructure.canonical_store.mark_release_immutable`): an
    immutable release cannot take its graph afterwards, which is correct — the
    artifact is part of what the release is.

    Raises:
        GraphArtifactError: the source is missing, is not a loadable GraphML
            graph, or the copy does not match it.
    """
    src = Path(source)
    if not src.is_file():
        raise GraphArtifactError(
            f"no graph artifact to pin at {src} — build the graph first "
            "(scripts/build_graph.py)"
        )
    try:
        graph = KnowledgeGraph.load(src)
    except Exception as exc:
        raise GraphArtifactError(
            f"graph artifact {src} is not a loadable GraphML knowledge graph: {exc!r}"
        ) from exc
    schema_version = graph.schema_version
    if not schema_version or schema_version == "unknown":
        raise GraphArtifactError(
            f"graph artifact {src} carries no schema_version — a release cannot pin "
            "a graph whose schema nobody can name"
        )

    dest = _artifact_destination(release_dir)
    if dest.exists() and not os.access(dest, os.W_OK):
        raise GraphArtifactError(
            f"release {Path(release_dir).name} already pins a read-only graph "
            f"artifact at {dest} — a release's graph is part of what the release is, "
            "so it is pinned once and then immutable"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dest)

    digest = compute_file_sha256(dest)
    source_digest = compute_file_sha256(src)
    if digest != source_digest:
        raise GraphArtifactError(
            f"graph artifact copy at {dest} does not match its source {src} "
            f"({digest} != {source_digest}) — refusing to pin bytes that are not the "
            "ones that were built"
        )
    pin = GraphArtifactPin(
        path=GRAPH_ARTIFACT_RELATIVE_PATH,
        sha256=digest,
        schema_version=schema_version,
    )
    logger.info(
        f"Graph artifact pinned: {dest} (sha256={digest[:16]}…, "
        f"schema_version={schema_version}, nodes={graph.node_count}, "
        f"edges={graph.edge_count})"
    )
    return pin

def verify_pinned_graph(manifest: ReleaseManifest, release_dir: str | Path) -> Path:
    """The pinned artifact's path, after proving it is the pinned bytes.

    The cheap half of :func:`load_pinned_graph`: resolve, exist, digest. Used by
    a rollback's validate step, which must prove the target restorable **before**
    anything moves — and must not pay a GraphML parse to do it. The digest is
    the check that matters there: the pin was written by a publish that had
    already parsed the artifact, so bytes that hash to the pin are bytes that
    parsed.

    Raises:
        GraphArtifactNotPinnedError: the release pins nothing.
        GraphArtifactMissingError: the pinned artifact is not on disk.
        GraphArtifactMismatchError: the artifact is not the pinned bytes.
        GraphArtifactError: the pin escapes its release directory.
    """
    if not manifest.pins_graph:
        raise GraphArtifactNotPinnedError(
            f"release {manifest.release_id} pins no graph artifact"
        )
    try:
        artifact = manifest.pinned_graph_path(release_dir)
    except ValueError as exc:
        raise GraphArtifactError(str(exc)) from exc
    if artifact is None:  # pragma: no cover — pins_graph implies a path
        raise GraphArtifactNotPinnedError(
            f"release {manifest.release_id} pins a graph digest but no path"
        )
    if not artifact.is_file():
        raise GraphArtifactMissingError(
            f"release {manifest.release_id} pins its graph at "
            f"{manifest.graph_path} and there is no file there ({artifact})"
        )
    digest = compute_file_sha256(artifact)
    if digest != manifest.graph_sha256:
        raise GraphArtifactMismatchError(
            f"release {manifest.release_id}'s graph artifact {artifact} is not the "
            f"artifact the release pinned (sha256 {digest}, pinned "
            f"{manifest.graph_sha256})"
        )
    return artifact

def publish_release_graph(release_dir: str | Path, source: str | Path) -> ReleaseManifest:
    """Pin ``source`` as ``release_dir``'s graph and record it in the manifest.

    The one call a publish path makes for the graph side of a release: copy the
    built artifact in (:func:`pin_graph_artifact`), read the release's manifest,
    write it back with the pin. It returns the updated manifest so the caller
    can log or assert what the release now pins.

    Reading and writing the manifest here rather than at the call site is
    deliberate — a publish that copied the graph but forgot to record the pin
    would produce a release that serves a graph it cannot restore, which is the
    exact state a rollback then has to refuse.

    Raises:
        GraphArtifactError: as :func:`pin_graph_artifact`.
        FileNotFoundError: the release has no manifest yet — a release directory
            without one is not a release, so this refuses rather than creating
            a half-release.
    """
    pin = pin_graph_artifact(release_dir, source)
    manifest = read_manifest(release_dir)
    updated = manifest.with_graph_pin(
        path=pin.path, sha256=pin.sha256, schema_version=pin.schema_version
    )
    write_manifest(release_dir, updated)
    logger.info(
        f"Release {manifest.release_id} pins its graph: {pin.path} "
        f"(sha256={pin.sha256[:16]}…, schema_version={pin.schema_version})"
    )
    return updated

def load_pinned_graph(
    manifest: ReleaseManifest, release_dir: str | Path
) -> KnowledgeGraph:
    """The graph ``manifest`` pins, verified against the pin, loaded.

    Raises everything :func:`verify_pinned_graph` raises, plus:
        GraphArtifactError: the verified bytes are not a loadable GraphML graph.
        GraphArtifactMismatchError: the artifact declares a schema_version other
            than the one the release pinned.
    """
    artifact = verify_pinned_graph(manifest, release_dir)
    try:
        graph = KnowledgeGraph.load(artifact)
    except Exception as exc:
        raise GraphArtifactError(
            f"release {manifest.release_id}'s pinned graph artifact {artifact} does "
            f"not load: {exc!r}"
        ) from exc
    if graph.schema_version != manifest.graph_schema_version:
        raise GraphArtifactMismatchError(
            f"release {manifest.release_id}'s graph artifact declares schema_version "
            f"{graph.schema_version!r} but the release pinned "
            f"{manifest.graph_schema_version!r}"
        )
    return graph

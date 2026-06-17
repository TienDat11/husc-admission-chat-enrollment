"""
Auto-Crawl and Incremental Update Pipeline (Phase E)

7-step ingestion pipeline: crawl → hash → chunk → metadata → supersession → embed → graph
Supports: incremental crawl, hash dedup, semantic dedup, checkpoint/resume, snapshot/rollback.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

from src.infrastructure.win32_utils import (
    check_disk_space,
    cleanup_old_snapshots,
    ensure_extended_path,
    safe_copytree,
)

from src.services.lancedb_retrieval import _detect_current_admission_year

# --- Configuration ---

HASH_LOG_PATH = "rag2025/data/crawl_hash_log.json"
CRAWLER_STATE_PATH = "rag2025/crawler_state.json"
CHECKPOINT_PATH = "rag2025/checkpoint_incremental.json"
SNAPSHOT_BASE = "rag2025/data/snapshots"
DATA_DIRS = [
    "rag2025/data/chunked",
    "rag2025/data/lancedb",
    "rag2025/data/graph",
]

CURRENT_YEAR = _detect_current_admission_year()
MIN_FREE_GB = 2.0
MAX_RETRIES = {"crawl": 3, "chunk": 2, "embed": 3, "graph": 3}


# --- Data Models ---


@dataclass
class CrawlerState:
    """Tracks incremental crawl progress."""

    max_id: int = 0
    last_crawl: str = ""
    checkpoint: str = ""
    year_partition: str = str(CURRENT_YEAR)
    retry_count: int = 0
    hash_log: Dict[str, str] = field(default_factory=dict)

    def save(self, path: str = CRAWLER_STATE_PATH) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "max_id": self.max_id,
                    "last_crawl": self.last_crawl,
                    "checkpoint": self.checkpoint,
                    "year_partition": self.year_partition,
                    "retry_count": self.retry_count,
                    "hash_log": self.hash_log,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str = CRAWLER_STATE_PATH) -> "CrawlerState":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return cls(
                max_id=data.get("max_id", 0),
                last_crawl=data.get("last_crawl", ""),
                checkpoint=data.get("checkpoint", ""),
                year_partition=data.get("year_partition", str(CURRENT_YEAR)),
                retry_count=data.get("retry_count", 0),
                hash_log=data.get("hash_log", {}),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Corrupted crawler_state.json, starting fresh: {e}")
            return cls()


@dataclass
class Checkpoint:
    """Tracks pipeline step completion for resume."""

    run_id: str = ""
    last_success_step: int = 0
    last_success_chunk_id: str = ""
    failed_step: int = 0
    error_message: str = ""
    timestamp: str = ""
    recovery_mode: str = "incremental"

    def save(self, path: str = CHECKPOINT_PATH) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "last_success_step": self.last_success_step,
                    "last_success_chunk_id": self.last_success_chunk_id,
                    "failed_step": self.failed_step,
                    "error_message": self.error_message,
                    "timestamp": self.timestamp,
                    "recovery_mode": self.recovery_mode,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str = CHECKPOINT_PATH) -> Optional["Checkpoint"]:
        p = Path(path)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return cls(
                run_id=data.get("run_id", ""),
                last_success_step=data.get("last_success_step", 0),
                last_success_chunk_id=data.get("last_success_chunk_id", ""),
                failed_step=data.get("failed_step", 0),
                error_message=data.get("error_message", ""),
                timestamp=data.get("timestamp", ""),
                recovery_mode=data.get("recovery_mode", "incremental"),
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def clear(self, path: str = CHECKPOINT_PATH) -> None:
        p = Path(path)
        if p.exists():
            p.unlink()


# --- Hash Dedup ---


def compute_hash(content: str) -> str:
    """SHA256 hash of UTF-8 content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def hash_dedup(content: str, url: str, hash_log: Dict[str, str]) -> bool:
    """Check if content is new (returns True) or duplicate (returns False).

    Updates hash_log in-place with new content hashes.
    """
    h = compute_hash(content)
    if h in hash_log:
        logger.debug(f"Hash dedup skip: {url} (hash={h[:12]}...)")
        return False
    hash_log[h] = url
    return True


def load_hash_log(path: str = HASH_LOG_PATH) -> Dict[str, str]:
    """Load hash dedup log from disk."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, KeyError):
        return {}


def save_hash_log(hash_log: Dict[str, str], path: str = HASH_LOG_PATH) -> None:
    """Persist hash dedup log to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(hash_log, ensure_ascii=False, indent=2), encoding="utf-8")


# --- Supersession ---


def check_supersession(
    new_chunk_metadata: Dict[str, Any],
    existing_chunks_metadata: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Mark old chunks as superseded when new version detected.

    Matches by section + data_year. Returns updated existing chunks.
    """
    new_section = new_chunk_metadata.get("section", "")
    new_year = new_chunk_metadata.get("data_year")

    for old in existing_chunks_metadata:
        old_section = old.get("section", "")
        old_year = old.get("data_year")

        if old_section == new_section and old_year == new_year:
            old["is_superseded"] = True
            old["superseded_by"] = new_chunk_metadata.get("chunk_id", "")

    # Set version on new chunk
    versions = [c.get("version", 1) for c in existing_chunks_metadata]
    new_chunk_metadata["version"] = max(versions or [0]) + 1

    return existing_chunks_metadata


# --- Snapshot / Rollback ---


def snapshot_before_ingest(run_id: str) -> str:
    """Create timestamped snapshots of all mutable state before ingest.

    Returns snapshot directory path.
    """
    check_disk_space("D:/")
    cleanup_old_snapshots(SNAPSHOT_BASE, keep=3)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snap_dir = Path(SNAPSHOT_BASE) / f"{run_id}_{ts}"
    snap_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "timestamp": ts,
        "targets": [],
    }

    for target in DATA_DIRS:
        src = Path(target)
        if not src.exists():
            continue
        dest = snap_dir / src.name
        safe_copytree(str(src), str(dest))
        manifest["targets"].append(str(target))

    # Snapshot crawler state if exists
    cs_path = Path(CRAWLER_STATE_PATH)
    if cs_path.exists():
        shutil.copy2(str(cs_path), str(snap_dir / "crawler_state.json"))

    manifest_path = snap_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info(f"Snapshot created: {snap_dir}")
    return str(snap_dir)


def rollback_on_failure(snapshot_path: str) -> None:
    """Restore all mutable state from a snapshot directory."""
    snap_dir = Path(snapshot_path)
    if not snap_dir.exists():
        raise FileNotFoundError(f"Snapshot not found: {snapshot_path}")

    manifest_path = snap_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found in {snapshot_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for target in manifest.get("targets", []):
        src_name = Path(target).name
        dest = Path(target)
        src = snap_dir / src_name

        if not src.exists():
            logger.warning(f"Snapshot missing for {target}, skipping")
            continue

        if dest.exists():
            shutil.rmtree(str(dest), ignore_errors=True)
        safe_copytree(str(src), str(dest))

    # Restore crawler state
    cs_snap = snap_dir / "crawler_state.json"
    if cs_snap.exists():
        shutil.copy2(str(cs_snap), CRAWLER_STATE_PATH)

    logger.info(f"Rollback complete from {snapshot_path}")


# --- Pipeline Orchestrator ---


class IngestPipeline:
    """7-step incremental ingestion pipeline.

    Step 1 — CRAWL: Fetch raw content from discovered URLs
    Step 2 — HASH: Compute SHA256 content hash → dedup log
    Step 3 — CHUNK: Apply hybrid chunking (profile=auto/tuyensinh)
    Step 4 — METADATA: Assign temporal schema (data_year, crawl_date, version)
    Step 5 — SUPERSESSION: Check same-topic chunks → mark superseded
    Step 6 — EMBED: Embedding + LanceDB upsert
    Step 7 — GRAPH: Batch NER → graph delta update
    """

    def __init__(self, state: Optional[CrawlerState] = None) -> None:
        self._state = state or CrawlerState.load()
        self._hash_log = load_hash_log()
        self._checkpoint = Checkpoint()

    @property
    def state(self) -> CrawlerState:
        return self._state

    def run(
        self,
        crawl_fn=None,
        chunk_fn=None,
        embed_fn=None,
        graph_fn=None,
        start_step: int = 1,
    ) -> Dict[str, Any]:
        """Execute the 7-step pipeline with retry and checkpoint support.

        Args:
            crawl_fn: Callable returning list of (url, content) tuples.
            chunk_fn: Callable(text, metadata) -> list of chunk dicts.
            embed_fn: Callable(chunks) -> int (count embedded).
            graph_fn: Callable(chunks) -> int (count graphed).
            start_step: Resume from this step (1-indexed).

        Returns:
            Pipeline result dict with step statuses.
        """
        run_id = f"incr_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        self._checkpoint = Checkpoint(run_id=run_id, timestamp=datetime.now(timezone.utc).isoformat())

        snapshot_path = ""
        result: Dict[str, Any] = {
            "run_id": run_id,
            "steps": {},
            "chunks_produced": 0,
            "skipped_duplicates": 0,
        }

        try:
            # Step 1: CRAWL
            if start_step <= 1:
                raw_pages = self._step_crawl(crawl_fn)
                result["steps"]["crawl"] = f"{len(raw_pages)} pages"
                self._checkpoint.last_success_step = 1
            else:
                raw_pages = []

            # Step 2: HASH dedup
            if start_step <= 2:
                unique_pages = self._step_hash(raw_pages)
                result["skipped_duplicates"] = len(raw_pages) - len(unique_pages)
                result["steps"]["hash"] = f"{len(unique_pages)} unique"
                self._checkpoint.last_success_step = 2
            else:
                unique_pages = []

            # Step 3: CHUNK
            if start_step <= 3:
                all_chunks = self._step_chunk(unique_pages, chunk_fn)
                result["chunks_produced"] = len(all_chunks)
                result["steps"]["chunk"] = f"{len(all_chunks)} chunks"
                self._checkpoint.last_success_step = 3
                self._checkpoint.last_success_chunk_id = all_chunks[-1].get("id", "") if all_chunks else ""
            else:
                all_chunks = []

            # Step 4: METADATA
            if start_step <= 4:
                all_chunks = self._step_metadata(all_chunks)
                result["steps"]["metadata"] = "assigned"
                self._checkpoint.last_success_step = 4

            # Step 5: SUPERSESSION
            if start_step <= 5:
                all_chunks = self._step_supersession(all_chunks)
                result["steps"]["supersession"] = "checked"
                self._checkpoint.last_success_step = 5

            # Step 6: EMBED
            if start_step <= 6:
                embedded = self._step_embed(all_chunks, embed_fn)
                result["steps"]["embed"] = f"{embedded} embedded"
                self._checkpoint.last_success_step = 6

            # Step 7: GRAPH
            if start_step <= 7:
                graphed = self._step_graph(all_chunks, graph_fn)
                result["steps"]["graph"] = f"{graphed} graphed"
                self._checkpoint.last_success_step = 7

            # Finalize
            self._state.last_crawl = datetime.now(timezone.utc).isoformat()
            self._state.save()
            save_hash_log(self._hash_log)
            self._checkpoint.clear()

            result["status"] = "success"

        except PipelineStepError as e:
            self._checkpoint.failed_step = e.step
            self._checkpoint.error_message = str(e)
            self._checkpoint.save()
            result["status"] = f"failed at step {e.step}"
            result["error"] = str(e)
            logger.error(f"Pipeline failed at step {e.step}: {e}")

        return result

    # --- Step implementations ---

    def _step_crawl(self, crawl_fn) -> List[Tuple[str, str]]:
        """Step 1: Fetch raw content. Returns [(url, content)]."""
        if crawl_fn is None:
            logger.info("No crawl_fn provided, skipping crawl step")
            return []

        for attempt in range(MAX_RETRIES["crawl"]):
            try:
                pages = crawl_fn()
                return pages
            except Exception as e:
                wait = 30 * (attempt + 1)
                logger.warning(f"Crawl retry {attempt+1}/{MAX_RETRIES['crawl']}: {e}")
                if attempt < MAX_RETRIES["crawl"] - 1:
                    time.sleep(wait)
        raise PipelineStepError(1, "Crawl failed after all retries")

    def _step_hash(self, pages: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Step 2: Hash dedup at document level."""
        unique = []
        for url, content in pages:
            if hash_dedup(content, url, self._hash_log):
                unique.append((url, content))
        return unique

    def _step_chunk(self, pages: List[Tuple[str, str]], chunk_fn) -> List[Dict]:
        """Step 3: Hybrid chunking."""
        if chunk_fn is None:
            logger.info("No chunk_fn provided, skipping chunk step")
            return []

        all_chunks = []
        for url, content in pages:
            for attempt in range(MAX_RETRIES["chunk"]):
                try:
                    chunks = chunk_fn(content, {"source_url": url})
                    all_chunks.extend(chunks)
                    break
                except Exception as e:
                    logger.warning(f"Chunk retry {attempt+1}: {e}")
                    if attempt == MAX_RETRIES["chunk"] - 1:
                        logger.error(f"Chunking failed for {url}, skipping")
        return all_chunks

    def _step_metadata(self, chunks: List[Dict]) -> List[Dict]:
        """Step 4: Assign temporal metadata."""
        now = datetime.now(timezone.utc).isoformat()
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            meta.setdefault("data_year", CURRENT_YEAR)
            meta.setdefault("crawl_date", now)
            meta.setdefault("school", "HUSC")
            meta.setdefault("version", 1)
            meta.setdefault("is_superseded", False)
            meta.setdefault("freshness_score", 1.0)
            chunk["metadata"] = meta
        return chunks

    def _step_supersession(self, chunks: List[Dict]) -> List[Dict]:
        """Step 5: Check supersession against existing chunks."""
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            existing = self._load_existing_chunks(meta.get("section", ""))
            if existing:
                check_supersession(meta, existing)
        return chunks

    def _step_embed(self, chunks: List[Dict], embed_fn) -> int:
        """Step 6: Embed and upsert to LanceDB."""
        if embed_fn is None:
            logger.info("No embed_fn provided, skipping embed step")
            return 0
        if not chunks:
            return 0

        for attempt in range(MAX_RETRIES["embed"]):
            try:
                count = embed_fn(chunks)
                return count
            except Exception as e:
                wait = 2**attempt
                logger.warning(f"Embed retry {attempt+1}: {e}, waiting {wait}s")
                if attempt < MAX_RETRIES["embed"] - 1:
                    time.sleep(wait)
        raise PipelineStepError(6, "Embed failed after all retries")

    def _step_graph(self, chunks: List[Dict], graph_fn) -> int:
        """Step 7: Batch NER and graph delta update."""
        if graph_fn is None:
            logger.info("No graph_fn provided, skipping graph step")
            return 0
        if not chunks:
            return 0

        for attempt in range(MAX_RETRIES["graph"]):
            try:
                count = graph_fn(chunks)
                return count
            except Exception as e:
                logger.warning(f"Graph retry {attempt+1}: {e}")
                if attempt < MAX_RETRIES["graph"] - 1:
                    time.sleep(2**attempt)
        logger.error("Graph update failed after all retries — manual rebuild needed")
        return 0

    def _load_existing_chunks(self, section: str) -> List[Dict]:
        """Load existing chunks for supersession check (simplified)."""
        chunked_dir = Path("rag2025/data/chunked")
        if not chunked_dir.exists():
            return []

        existing = []
        for jsonl_path in chunked_dir.glob("*.jsonl"):
            try:
                for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    meta = data.get("metadata", {})
                    if meta.get("section") == section:
                        existing.append(meta)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
        return existing


class PipelineStepError(Exception):
    """Raised when a pipeline step fails after all retries."""

    def __init__(self, step: int, message: str) -> None:
        self.step = step
        super().__init__(f"Step {step} failed: {message}")


# --- Resume ---


def resume_pipeline() -> Optional[Dict[str, Any]]:
    """Resume a failed pipeline from the last checkpoint.

    Returns pipeline result if checkpoint exists, None otherwise.
    """
    ckpt = Checkpoint.load()
    if ckpt is None:
        logger.info("No checkpoint found, nothing to resume")
        return None

    logger.info(
        f"Resuming pipeline from step {ckpt.failed_step + 1} "
        f"(run_id={ckpt.run_id}, failed at step {ckpt.failed_step}: {ckpt.error_message})"
    )

    pipeline = IngestPipeline()
    return pipeline.run(start_step=ckpt.failed_step + 1)


# --- Scheduler Helper ---


def create_scheduled_task(task_name: str = "HUSC-Ingest") -> bool:
    """Create Windows Task Scheduler entry for daily crawl at 06:00.

    Returns True if task was created successfully.
    """
    import subprocess
    import sys

    python_exe = sys.executable
    script = "-m rag2025.scripts.crawl_incremental"

    cmd = [
        "schtasks", "/create",
        "/sc", "DAILY",
        "/st", "06:00",
        "/tr", f'"{python_exe}" {script}',
        "/tn", task_name,
        "/f",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info(f"Scheduled task '{task_name}' created for daily 06:00 crawl")
            return True
        logger.error(f"schtasks failed: {result.stderr}")
        return False
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.error(f"Failed to create scheduled task: {e}")
        return False

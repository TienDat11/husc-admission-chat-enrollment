# @spec(S13.1) audit pre-reingest LanceDB year distribution

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lancedb
from loguru import logger


def audit_table(uri: str, table_name: str) -> dict[str, Any]:
    """Audit a LanceDB table and return year/method/info_type distributions."""
    db = lancedb.connect(uri)
    tbl = db.open_table(table_name)
    rows = list(tbl.to_arrow().to_pylist())

    year_counter: Counter = Counter()
    method_counter: Counter = Counter()
    info_type_counter: Counter = Counter()
    missing_data_year = 0
    missing_source_url = 0

    for row in rows:
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        data_year = metadata.get("data_year")
        chunk_method = metadata.get("chunk_method")
        info_type = metadata.get("info_type")
        source_url = metadata.get("source_url")

        year_key = str(data_year) if data_year is not None else "null"
        method_key = str(chunk_method) if chunk_method is not None else "null"
        info_key = str(info_type) if info_type is not None else "null"

        year_counter[year_key] += 1
        method_counter[method_key] += 1
        info_type_counter[info_key] += 1

        if data_year is None:
            missing_data_year += 1
        if source_url is None:
            missing_source_url += 1

    return {
        "audit_date": datetime.now(timezone.utc).isoformat(),
        "table": table_name,
        "total_rows": len(rows),
        "year_distribution": dict(year_counter),
        "method_distribution": dict(method_counter),
        "info_type_distribution": dict(info_type_counter),
        "missing_data_year_count": missing_data_year,
        "missing_source_url_count": missing_source_url,
    }


def save_audit(audit: dict, output_path: Path) -> None:
    """Write audit dict to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit LanceDB table year distribution"
    )
    parser.add_argument(
        "--lancedb-uri",
        default=os.getenv("LANCEDB_URI", "rag2025/data/lancedb"),
        help="LanceDB URI (default: env LANCEDB_URI or rag2025/data/lancedb)",
    )
    parser.add_argument(
        "--table",
        default=os.getenv("LANCEDB_TABLE", "husc"),
        help="Table name (default: env LANCEDB_TABLE or husc)",
    )
    parser.add_argument(
        "--output",
        default="rag2025/data/audit_pre_reingest.json",
        help="Output JSON path (default: rag2025/data/audit_pre_reingest.json)",
    )
    args = parser.parse_args()

    try:
        logger.info("Auditing table {!r} at {!r}", args.table, args.lancedb_uri)
        audit = audit_table(args.lancedb_uri, args.table)
        save_audit(audit, Path(args.output))
        logger.info("Audit saved to {!r}", args.output)
        print(json.dumps(audit, indent=2, ensure_ascii=False))
        return 0
    except Exception:
        logger.exception("Audit failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""
Data Normalization Script

Converts raw JSONL data to format expected by RAGChunk schema.
Generates missing required fields (doc_id, id, etc.).
"""
import json
import sys
import uuid
from pathlib import Path

from loguru import logger


def cast_to_int(value):
    """Cast string to int if it represents a valid integer."""
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def normalize_record(record: dict, source_file: str, line_num: int) -> dict | None:
    """
    Normalize a single record to match RAGChunk schema.
    
    Args:
        record: Raw record dictionary
        source_file: Source filename for doc_id generation
        line_num: Line number for ID generation
        
    Returns:
        Normalized record or None if invalid
    """
    # Generate doc_id if missing (use source filename)
    if "doc_id" not in record:
        base_name = Path(source_file).stem
        record["doc_id"] = f"{base_name}_doc_{line_num:04d}"
    
    # Generate unique id if missing
    if "id" not in record:
        record["id"] = f"{record['doc_id']}_chunk_0"
    
    # Ensure chunk_id exists and is integer
    if "chunk_id" not in record:
        record["chunk_id"] = 0
    else:
        record["chunk_id"] = cast_to_int(record["chunk_id"])
    
    # Validate text field exists
    if "text" not in record or not record["text"]:
        logger.warning(f"Line {line_num}: Missing or empty 'text' field, skipping")
        return None
    
    # Ensure metadata exists
    if "metadata" not in record:
        record["metadata"] = {}
    
    # Ensure metadata has required 'source' field
    if "source" not in record["metadata"]:
        record["metadata"]["source"] = source_file
    
    # Cast numeric fields in metadata
    if "year" in record["metadata"]:
        record["metadata"]["year"] = cast_to_int(record["metadata"]["year"])
    
    return record


def normalize_jsonl(input_path: Path, output_path: Path) -> dict:
    """
    Normalize JSONL file.
    
    Args:
        input_path: Input raw JSONL file
        output_path: Output normalized JSONL file
        
    Returns:
        Statistics dict
    """
    stats = {
        "total": 0,
        "normalized": 0,
        "skipped": 0,
        "errors": []
    }
    
    logger.info(f"Normalizing {input_path} -> {output_path}")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        
        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            
            stats["total"] += 1
            
            try:
                record = json.loads(line)
                normalized = normalize_record(record, input_path.name, line_num)
                
                if normalized:
                    fout.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                    stats["normalized"] += 1
                else:
                    stats["skipped"] += 1
                    
            except json.JSONDecodeError as e:
                logger.error(f"Line {line_num}: JSON decode error - {e}")
                stats["errors"].append({"line": line_num, "error": str(e)})
                stats["skipped"] += 1
            except Exception as e:
                logger.error(f"Line {line_num}: Error - {e}")
                stats["errors"].append({"line": line_num, "error": str(e)})
                stats["skipped"] += 1
    
    logger.info(f"Normalization complete: {stats['normalized']}/{stats['total']} records")
    return stats


def main():
    """CLI entry point."""
    if len(sys.argv) < 3:
        print("Usage: python normalize_data.py <input.jsonl> <output.jsonl>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    stats = normalize_jsonl(input_path, output_path)
    
    print(f"\nNormalization Report:")
    print(f"  Total: {stats['total']}")
    print(f"  Normalized: {stats['normalized']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Errors: {len(stats['errors'])}")
    
    if stats["errors"]:
        print(f"\nTop 5 Errors:")
        for err in stats["errors"][:5]:
            print(f"  Line {err['line']}: {err['error']}")
    
    sys.exit(0 if stats["skipped"] == 0 else 1)


if __name__ == "__main__":
    main()

"""
JSONL Schema Validation Module (Epic 1.1)

Validates .jsonl files against RAGChunk JSON Schema.
Detects encoding issues, malformed JSON, and schema violations.
"""
import json
import re
from pathlib import Path
from typing import Any

import chardet
from jsonschema import ValidationError, validate
from loguru import logger
from pydantic import BaseModel, Field


class ValidationReport(BaseModel):
    """Validation report model."""

    total_lines: int = Field(description="Total lines processed")
    valid_lines: int = Field(description="Number of valid records")
    invalid_lines: int = Field(description="Number of invalid records")
    errors: list[dict[str, Any]] = Field(
        default_factory=list, description="List of validation errors"
    )
    warnings: list[str] = Field(default_factory=list, description="List of warnings")
    encoding: str = Field(description="Detected file encoding")
    validated_file: Path | None = Field(
        default=None, description="Path to validated output file"
    )

    @property
    def success_rate(self) -> float:
        """Calculate validation success rate."""
        if self.total_lines == 0:
            return 0.0
        return self.valid_lines / self.total_lines

    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"Validation Report:\n"
            f"  Total: {self.total_lines} | Valid: {self.valid_lines} | "
            f"Invalid: {self.invalid_lines} | Success: {self.success_rate:.1%}\n"
            f"  Encoding: {self.encoding}\n"
            f"  Errors: {len(self.errors)} | Warnings: {len(self.warnings)}"
        )


def detect_encoding(file_path: Path, sample_size: int = 10000) -> str:
    """
    Detect file encoding using chardet.

    Args:
        file_path: Path to file
        sample_size: Number of bytes to sample

    Returns:
        Detected encoding string (e.g., 'utf-8')
    """
    with open(file_path, "rb") as f:
        raw_data = f.read(sample_size)

    result = chardet.detect(raw_data)
    encoding = result["encoding"] or "utf-8"
    confidence = result.get("confidence", 0.0)

    logger.debug(
        f"Detected encoding: {encoding} (confidence: {confidence:.2%}) for {file_path.name}"
    )

    # Fallback to utf-8 for low confidence
    if confidence < 0.7:
        logger.warning(
            f"Low confidence ({confidence:.2%}) for {encoding}, falling back to utf-8"
        )
        return "utf-8"

    return encoding


def load_schema(schema_path: Path) -> dict[str, Any]:
    """
    Load JSON Schema from file.

    Args:
        schema_path: Path to JSON Schema file

    Returns:
        Schema dictionary
    """
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    logger.info(f"Loaded schema from {schema_path}")
    return schema


def validate_jsonl(
    input_path: Path,
    schema_path: Path,
    output_path: Path | None = None,
    strict: bool = True,
) -> ValidationReport:
    """
    Validate .jsonl file against JSON Schema.

    Args:
        input_path: Path to input .jsonl file
        schema_path: Path to JSON Schema file
        output_path: Optional path to write validated records
        strict: If True, skip invalid records; if False, include with warnings

    Returns:
        ValidationReport with detailed results
    """
    logger.info(f"Starting validation: {input_path}")

    # Load schema
    schema = load_schema(schema_path)

    # Detect encoding
    encoding = detect_encoding(input_path)

    # Initialize report
    report = ValidationReport(total_lines=0, valid_lines=0, invalid_lines=0, encoding=encoding)

    # Open output file if specified
    output_file = None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = open(output_path, "w", encoding="utf-8")
        report.validated_file = output_path

    try:
        with open(input_path, "r", encoding=encoding, errors="replace") as f:
            for line_num, line in enumerate(f, start=1):
                report.total_lines += 1
                line = line.strip()

                # Skip empty lines
                if not line:
                    report.warnings.append(f"Line {line_num}: Empty line skipped")
                    continue

                try:
                    # Parse JSON
                    record = json.loads(line)

                    # Validate against schema
                    validate(instance=record, schema=schema)

                    # Additional custom validations
                    _validate_custom_rules(record, line_num, report)

                    # Record is valid
                    report.valid_lines += 1
                    if output_file:
                        output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

                except json.JSONDecodeError as e:
                    report.invalid_lines += 1
                    report.errors.append(
                        {
                            "line": line_num,
                            "error": "JSON parse error",
                            "message": str(e),
                            "snippet": line[:100],
                        }
                    )
                    logger.warning(f"Line {line_num}: JSON parse error - {e}")

                except ValidationError as e:
                    report.invalid_lines += 1
                    report.errors.append(
                        {
                            "line": line_num,
                            "error": "Schema validation failed",
                            "field": ".".join(str(p) for p in e.path),
                            "message": e.message,
                        }
                    )
                    logger.warning(f"Line {line_num}: Schema validation failed - {e.message}")

                except Exception as e:
                    report.invalid_lines += 1
                    report.errors.append(
                        {"line": line_num, "error": "Unexpected error", "message": str(e)}
                    )
                    logger.error(f"Line {line_num}: Unexpected error - {e}")

    finally:
        if output_file:
            output_file.close()
            logger.info(f"Validated records written to {output_path}")

    logger.info(report.summary())
    return report


def _validate_custom_rules(record: dict[str, Any], line_num: int, report: ValidationReport) -> None:
    """
    Apply custom validation rules beyond JSON Schema.

    Args:
        record: Parsed JSON record
        line_num: Line number
        report: ValidationReport to append warnings
    """
    # Rule 1: Check text length (min 10 chars)
    text = record.get("text", "")
    if len(text.strip()) < 10:
        report.warnings.append(f"Line {line_num}: Text too short (<10 chars)")

    # Rule 2: Check metadata.source is non-empty
    metadata = record.get("metadata", {})
    if not metadata.get("source"):
        report.warnings.append(f"Line {line_num}: Missing metadata.source")

    # Rule 3: Check vector dimension if present
    vector = record.get("vector")
    if vector is not None and len(vector) != 768:
        report.warnings.append(
            f"Line {line_num}: Vector dimension is {len(vector)}, expected 768"
        )

    # Rule 4: Validate date formats
    effective_date = metadata.get("effective_date")
    if effective_date and not _is_valid_date(effective_date):
        report.warnings.append(
            f"Line {line_num}: Invalid effective_date format: {effective_date}"
        )


def _is_valid_date(date_str: str) -> bool:
    """Check if date string matches ISO 8601 format (YYYY-MM-DD)."""
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", date_str))


def main():
    """CLI entry point for validation script."""
    import sys

    if len(sys.argv) < 3:
        print("Usage: python validate_jsonl.py <input.jsonl> <schema.json> [output.jsonl]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    schema_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None

    # Run validation
    report = validate_jsonl(input_path, schema_path, output_path, strict=True)

    # Print summary
    print("\n" + report.summary())
    print(f"\nTop 5 Errors:")
    for err in report.errors[:5]:
        print(f"  Line {err['line']}: {err['error']} - {err.get('message', '')}")

    # Exit code
    sys.exit(0 if report.invalid_lines == 0 else 1)


if __name__ == "__main__":
    main()

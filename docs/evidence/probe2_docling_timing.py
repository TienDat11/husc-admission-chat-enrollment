import time, sys, json
from pathlib import Path
from docling.document_converter import DocumentConverter

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("D:/chunking/husc-admission-chat-enrollment/new_data/THONG_TIN_TUYEN_SINH_202611.pdf")
out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("D:/tmp/docling_probe_result.json")

t0 = time.time()
res = DocumentConverter().convert(str(p))
md = res.document.export_to_markdown()
dt = time.time() - t0

result = {
    "file": p.name,
    "seconds": round(dt, 1),
    "md_chars": len(md),
    "pages": len(res.document.pages) if hasattr(res.document, "pages") else "n/a",
    "has_pipe_table": "|" in md,
    "sample_head": md[:600],
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(json.dumps(result, ensure_ascii=False, indent=2))

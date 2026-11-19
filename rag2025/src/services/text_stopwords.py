"""Vietnamese stopwords for the retrieval diff sandbox (U8 PL-6.3).

Extracted from the inline list formerly living in guardrail.py so the sandbox
and the guardrail share one canonical list. Tokens are produced by NFC ->
lowercase -> split \\w+ -> drop these stopwords.
"""

from __future__ import annotations

STOPWORDS_VI: frozenset[str] = frozenset({
    "a", "à", "ã", "á", "ạ", "ả", "â", "ă", "b", "bằng", "bị", "bởi",
    "c", "cả", "các", "cái", "cần", "còn", "của", "cụ", "cũng", "d", "đã",
    "đây", "đó", "để", "đều", "được", "gì", "hả", "hãy", "họ", "khi",
    "không", "là", "lại", "liên", "mà", "mãi", "mọi", "một", "này",
    "nhưng", "nơi", "nữa", "qua", "ra", "rất", "rồi", "sau", "sẽ", "so",
    "sự", "tại", "theo", "thì", "trên", "trước", "từng", "và", "vậy", "vì",
    "vào", "về", "với", "đang", "đến", "đặt",
})

__all__ = ["STOPWORDS_VI"]

"""Diagnostic repro for guardrail false-positive on 'Đăng ký vào trường như lào?'."""
from __future__ import annotations
import asyncio
import os
import sys
from pathlib import Path

# Make the rag2025 package importable
HERE = Path(__file__).resolve().parent
RAG = HERE.parent
sys.path.insert(0, str(RAG))

from dotenv import load_dotenv  # type: ignore
load_dotenv(RAG / ".env")

from config.settings import RAGSettings  # type: ignore
from src.services.guardrail import GuardrailService  # type: ignore


def banner(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def diag_query(svc: GuardrailService, q: str) -> None:
    banner(f"QUERY: {q!r}")
    folded = svc._fold_diacritics(q)
    print(f"_fold_diacritics           -> {folded!r}")
    print(f"_contains_sensitive_pii    -> {svc._contains_sensitive_pii(q)}")
    print(f"_looks_admission_related   -> {svc._looks_admission_related(q)}")
    print(f"_mentions_husc             -> {svc._mentions_husc(q)}")
    print(f"_mentions_other_school     -> {svc._mentions_other_school(q)}")
    print(f"_extract_major_phrase      -> {svc._extract_major_phrase(q)!r}")
    phrase = svc._extract_major_phrase(q)
    folded_phrase = svc._fold_diacritics(phrase) if phrase else None
    print(f"  folded_phrase             -> {folded_phrase!r}")
    if folded_phrase:
        print(f"  _major_in_denylist(folded_phrase) -> {svc._major_in_denylist(folded_phrase)}")
        allowed = svc._load_major_scope()
        print(f"  _major_in_allowlist(...)         -> {svc._major_in_allowlist(folded_phrase, allowed)}")
    print(f"_major_in_denylist(folded_q)  -> {svc._major_in_denylist(folded)}")
    print(f"_major_out_of_scope        -> {svc._major_out_of_scope(q)}")


async def main() -> None:
    settings = RAGSettings()
    svc = GuardrailService(settings)

    queries = [
        "Đăng ký vào trường như lào?",
        "Làm sao để đăng ký xét tuyển?",
        "Cách nộp hồ sơ vào trường?",
        "Đăng ký vào trường thế nào?",
        "Thủ tục nhập học ra sao?",
    ]
    for q in queries:
        diag_query(svc, q)

    banner("FULL precheck() per query (LLM-backed when needed)")
    for q in queries:
        d = await svc.precheck(q)
        print(f"\nQ: {q!r}")
        print(f"  is_in_scope    -> {d.is_in_scope}")
        print(f"  internal_code  -> {d.internal_code}")
        print(f"  reason         -> {d.reason}")
        print(f"  short_answer   -> {d.short_answer[:160]!r}{'...' if len(d.short_answer) > 160 else ''}")


if __name__ == "__main__":
    asyncio.run(main())

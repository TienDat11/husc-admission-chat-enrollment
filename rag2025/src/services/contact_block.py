# @spec(T10) contact_block — year-agnostic admissions-office contact block.
"""Move out of ``query_router`` so ``abstention_policy`` (and any lightweight
seam) can import the contact block without dragging in the router's heavy
import chain (numpy / loguru / llm_client / ner_service).

The hotline is sourced in priority order from:

1. ``HUSC_HOTLINE`` env var (operator override, e.g. CI / staged deploy)
2. the /api/meta payload (``MetaResponse.contact_hotline``) when available
3. ``_HOTLINE_FALLBACK`` below (the previous literal — preserves prod
   behavior when the data-driven paths return nothing).

``build_contact_block(meta=...)`` renders; ``CONTACT_BLOCK`` is the
module-level pre-rendered fallback for callers that can't supply a meta
payload (auto-answer path). Strings are byte-identical to the former
``query_router`` definitions.
"""
from __future__ import annotations

import os
from typing import Optional


_HOTLINE_FALLBACK = "0234.3823290"  # Phòng Đào tạo (HUSC)


def _resolve_hotline(meta: Optional[dict] = None) -> str:
    """Pick the best available hotline (env > meta > fallback).

    Args:
        meta: optional dict (e.g. /api/meta response). If it contains a
            non-empty `contact_hotline` key, that value wins over the
            fallback but is overridden by HUSC_HOTLINE env.

    Returns:
        The hotline string to embed in CONTACT_BLOCK.
    """
    env = os.getenv("HUSC_HOTLINE")
    if env and env.strip():
        return env.strip()
    if meta:
        v = meta.get("contact_hotline")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return _HOTLINE_FALLBACK


def build_contact_block(meta: Optional[dict] = None) -> str:
    """Render the CONTACT_BLOCK with the resolved hotline.

    `meta` is the /api/meta response (or a subset dict containing
    `contact_hotline`). When None or empty, the env-var or fallback
    literal is used.
    """
    hotline = _resolve_hotline(meta)
    return (
        "Bạn có thể tra cứu thông tin tuyển sinh chính thức của Trường Đại học Khoa học - "
        "Đại học Huế (HUSC) tại:\n\n"
        "🌐 Cổng tuyển sinh: https://tuyensinh.husc.edu.vn\n"
        "🌐 Trang chủ: https://husc.edu.vn\n"
        f"📞 Hotline: {hotline} (Phòng Đào tạo)\n\n"
        "Bạn muốn biết cụ thể về ngành nào, điểm chuẩn, học phí, tổ hợp xét tuyển hay phương thức xét tuyển? Mình sẽ tra cứu chi tiết giúp bạn."
    )


# Pre-rendered default (env not set, no meta) — matches the pre-fix
# hardcoded string so legacy callers see no behavior change.
CONTACT_BLOCK = build_contact_block()

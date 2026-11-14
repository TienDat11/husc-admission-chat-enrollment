"""CORS origin policy and the ENV allow-list that governs it.

Pure: takes the raw ``ENV`` and ``ALLOWED_ORIGINS`` strings and returns the
origins and localhost regex, or raises. ``main.py`` calls this once at import.
"""

from __future__ import annotations

DEV_ENVS = frozenset({"", "dev", "development", "local", "test"})
STRICT_ENVS = frozenset({"prod", "production", "staging"})

DEV_DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]
# Any localhost/127.0.0.1 port, so the Vite dev server works on whatever port it gets.
LOCALHOST_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"


def resolve_cors_settings(env: str, raw_origins: str) -> tuple[list[str], str | None]:
    """Return ``(allow_origins, allow_origin_regex)`` or raise ``RuntimeError``.

    ``env`` is an allow-list: an unknown value refuses to start rather than being
    treated as development, so a typo such as ``prdo`` cannot open CORS wide.
    """
    env_name = env.strip().lower()
    if env_name not in DEV_ENVS and env_name not in STRICT_ENVS:
        raise RuntimeError(
            f"unknown ENV={env!r}; expected one of dev, development, local, test, prod, production, staging"
        )

    origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    if "*" in origins:
        raise RuntimeError("ALLOWED_ORIGINS must not contain '*' (credentials are enabled)")

    if env_name in STRICT_ENVS and not origins:
        raise RuntimeError(
            f"ENV={env_name} requires ALLOWED_ORIGINS to be set; refusing to start"
        )

    if origins:
        return origins, None
    return list(DEV_DEFAULT_ORIGINS), LOCALHOST_ORIGIN_REGEX

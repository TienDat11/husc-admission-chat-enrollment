"""Request-id contextvar shared by every LLM call in one user request.

Lives under ``config`` (imported by one canonical name everywhere) so that the
two import spellings of ``services.llm_client`` / ``src.services.llm_client``
read and write the SAME variable. A variable defined inside llm_client would
be duplicated per import spelling and attribution would silently drop.
"""
from __future__ import annotations

import contextvars
from typing import Optional

_request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "llm_request_id", default=None
)


def set_request_id(request_id: Optional[str]) -> contextvars.Token:
    """Bind ``request_id`` for every LLM call made in this context."""
    return _request_id_var.set(request_id)


def reset_request_id(token: contextvars.Token) -> None:
    """Restore the previous request id (pair with :func:`set_request_id`)."""
    _request_id_var.reset(token)


def get_request_id() -> Optional[str]:
    return _request_id_var.get()

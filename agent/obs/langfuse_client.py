"""Langfuse client setup. Lazy: built on first use, returns None if env unset."""

from __future__ import annotations

import os
from functools import lru_cache

from langfuse import Langfuse


@lru_cache(maxsize=1)
def get_langfuse() -> Langfuse | None:
    """Return a process-wide Langfuse client, or None if env vars are not configured."""
    public = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST")
    if not (public and secret and host):
        return None
    return Langfuse(public_key=public, secret_key=secret, host=host)


def langfuse_callback_handler() -> object | None:
    """Return a LangChain callback handler bound to our Langfuse client.

    Used in LangGraph by passing as a callback to .ainvoke():
      `await graph.ainvoke(state, config={"callbacks": [langfuse_callback_handler()]})`
    Auto-traces every LangChain LLM call as a span on the active trace.

    NOTE: returns None for now. The langfuse LangChain integration import path moved
    between versions (`langfuse.callback` → `langfuse.langchain`) and the new path
    requires `langchain` (not just `langchain-core`) to be installed. The handler will
    be wired up in Phase 7 when Langfuse tracing is actually used end-to-end.
    """
    return None

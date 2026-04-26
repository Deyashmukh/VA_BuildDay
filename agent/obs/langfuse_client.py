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



"""Observability primitives — JSON logger and Langfuse client."""

from agent.obs.langfuse_client import get_langfuse
from agent.obs.log import build_logger

__all__ = ["build_logger", "get_langfuse"]

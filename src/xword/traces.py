"""LangSmith tracing (Blueprint observability). No-ops without a key."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TypeVar

from dotenv import load_dotenv

from .paths import ROOT

F = TypeVar("F", bound=Callable)


def enable_tracing() -> bool:
    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT / ".env.local")
    key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not key:
        return False
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGCHAIN_PROJECT") or "xword")
    if not os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = key
    return True


def maybe_traceable(name: str) -> Callable[[F], F]:
    if not enable_tracing():
        return lambda fn: fn
    try:
        from langsmith import traceable

        return traceable(name=name)
    except ImportError:
        return lambda fn: fn

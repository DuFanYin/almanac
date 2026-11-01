"""Each model call as it happens (LLMCall), and the collector a run keeps them in.

One collector per run (a research question, a board's day, a command line call): LLMClient writes events here,
the caller reads them (its calls, their tokens and cost). No global state: concurrent runs are isolated.
"""

from __future__ import annotations

import threading
from typing import Literal, TypeAlias

from pydantic import BaseModel

LLMCallStatus: TypeAlias = Literal["calling", "success", "retry", "failed"]


class LLMCall(BaseModel):
    id: str
    node: str
    model: str
    attempt: int
    status: LLMCallStatus
    latency_ms: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None


class LLMCallCollector:
    def __init__(self) -> None:
        self._calls: list[LLMCall] = []
        self._lock = threading.Lock()
        self._seq = 0

    def next_id(self) -> str:
        with self._lock:
            self._seq += 1
            return f"llm_{self._seq:03d}"

    def record(self, call: LLMCall) -> None:
        with self._lock:
            self._calls.append(call)

    def all(self) -> list[LLMCall]:
        """Every call event so far."""
        with self._lock:
            return list(self._calls)

    def totals(self) -> tuple[float, int, int]:
        """Return (total_cost_usd, total_prompt_tokens, total_completion_tokens)."""
        with self._lock:
            cost = sum(c.cost_usd or 0.0 for c in self._calls)
            prompt = sum(c.prompt_tokens or 0 for c in self._calls)
            completion = sum(c.completion_tokens or 0 for c in self._calls)
            return round(cost, 6), prompt, completion

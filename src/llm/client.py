"""The model provider's client: one OpenAI-compatible JSON call.

- the provider (deepseek, openai or openrouter) and its key from src/llm/config.py;
- a chain of models tried in order: a model is retried on a timeout, a 429 or 5xx, an unexpected error or an
  answer that is not JSON (with backoff); a fatal answer (another 4xx, a truncated or empty answer) skips to the next;
- every attempt is recorded in the caller's LLMCallCollector (calls.py), with its tokens and cost.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import httpx
from src.llm.calls import LLMCall
from src.llm.config import LLM_API_KEY, LLM_APP_TITLE, LLM_BASE_URL, LLM_HTTP_REFERER, LLM_PROVIDER, provider_settings

if TYPE_CHECKING:
    from src.llm.calls import LLMCallCollector

logger = logging.getLogger(__name__)

_DEFAULT_MODELS = {
    "deepseek": ["deepseek-v4-flash", "deepseek-v4-pro"],
    "openai": ["gpt-4.1", "gpt-4.1-mini"],
    "openrouter": ["openai/gpt-oss-120b:free", "meta-llama/llama-3.3-70b-instruct:free"],
}

# Output ceilings, each under the provider's hard limit (above it the call fails with a 400); a cap, not a target.
_MAX_TOKENS_BY_PROVIDER = {"deepseek": 65536, "openai": 32768, "openrouter": 8192}

# Cost per 1M tokens (input, output) in USD, public pricing; DeepSeek's input at the cache-miss rate (worst case).
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "deepseek-v4-flash": (0.14, 0.28),
    "deepseek-v4-pro": (0.435, 0.87),
}

_RETRYABLE_CODES = {429, 500, 502, 503, 504}


def _billed(model: str, usage: dict | None) -> dict:
    """A call's tokens and cost from the provider's usage; empty without it."""
    prompt, completion = (usage or {}).get("prompt_tokens"), (usage or {}).get("completion_tokens")
    if prompt is None and completion is None:
        return {}
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cost_usd": _compute_cost(model, prompt or 0, completion or 0),
    }


def _compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    for key, (input_cost, output_cost) in _PRICING.items():
        if model.startswith(key):
            return round((prompt_tokens * input_cost + completion_tokens * output_cost) / 1_000_000, 8)
    return None


class _RetryableError(Exception):
    """Retry this model. usage: the provider's token counts when it answered (and billed) before the answer failed."""

    def __init__(self, msg, usage=None):
        super().__init__(msg)
        self.usage = usage


class _FatalError(_RetryableError):
    """Skip to the next model (usage as for _RetryableError)."""


class LLMClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        models: list[str] | None = None,
        base_url: str = LLM_BASE_URL,
        timeout_seconds: float = 60.0,
        max_retries: int = 2,
        retry_backoff: float = 2.0,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        collector: LLMCallCollector | None = None,
        provider: str | None = None,
    ) -> None:
        """provider: another than the main one (src/llm/config.py provider_settings), with its own base URL and key."""
        other = bool(provider) and provider != LLM_PROVIDER
        if other:  # its own URL and key, never the main provider's key
            base_url, own_key = provider_settings(provider)
            api_key = api_key or own_key
        p = provider or LLM_PROVIDER
        self.provider = p if p in _DEFAULT_MODELS else "openrouter"
        self.api_key = api_key if other else api_key or LLM_API_KEY
        # one model, or a chain tried in order (`models`), or the provider's default chain
        self._models = list(models) if models else [model] if model else list(_DEFAULT_MODELS[self.provider])
        self.base_url = (base_url or LLM_BASE_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.max_tokens = max_tokens or _MAX_TOKENS_BY_PROVIDER[self.provider]
        self.temperature = temperature
        self._collector = collector

    async def complete(self, prompt: str, *, system: str | None = None, node: str = "unknown") -> str:
        """The answer, a JSON object's text; RuntimeError once every model of the chain has failed."""
        if not self.api_key:
            raise RuntimeError("LLM API key is not set. Set LLM_API_KEY in your environment.")
        last_error: Exception | None = None
        for model_id in self._models:
            for attempt in range(1, self.max_retries + 2):
                call_id = self._collector.next_id() if self._collector else "llm_notrace"
                started_at = datetime.now(UTC).isoformat()
                base = {"id": call_id, "node": node, "model": model_id, "attempt": attempt, "started_at": started_at}
                self._emit(LLMCall(**base, status="calling"))
                try:
                    content, usage = await self._call(model_id, prompt, system=system)
                except (_RetryableError, _FatalError, Exception) as exc:
                    fatal = isinstance(exc, _FatalError)
                    last_error = exc
                    self._emit(
                        LLMCall(
                            **base,
                            **_done(started_at),
                            status="failed" if fatal else "retry",
                            error=str(exc)[:200],
                            **_billed(model_id, getattr(exc, "usage", None)),  # an answer that failed is still paid
                        )
                    )
                    if fatal:
                        logger.warning("model %s fatal error: %s; skipping", model_id, exc)
                        break
                    if attempt <= self.max_retries:
                        wait = self.retry_backoff**attempt
                        logger.warning(
                            "model %s attempt %d/%d failed (%s); retrying in %.1fs",
                            model_id,
                            attempt,
                            self.max_retries + 1,
                            exc,
                            wait,
                        )
                        await asyncio.sleep(wait)
                    continue
                self._emit(LLMCall(**base, **_done(started_at), status="success", **_billed(model_id, usage)))
                return content
        raise RuntimeError(f"[llm] All models exhausted. Last error: {last_error}") from last_error

    def _headers(self) -> dict:
        h = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if self.provider == "openrouter":
            if LLM_HTTP_REFERER:
                h["HTTP-Referer"] = LLM_HTTP_REFERER
            if LLM_APP_TITLE:
                h["X-Title"] = LLM_APP_TITLE
        return h

    def _emit(self, call: LLMCall) -> None:
        if self._collector is not None:
            self._collector.record(call)

    async def _call(self, model_id: str, prompt: str, *, system: str | None) -> tuple[str, dict | None]:
        payload: dict = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system or "Return only valid JSON, no markdown or extra text."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.provider == "deepseek":
            # V4 thinks by default: slow, and its reasoning is not used here
            payload["thinking"] = {"type": "disabled"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=payload)
        except httpx.TimeoutException as exc:
            raise _RetryableError(f"timeout: {exc}") from exc
        except httpx.NetworkError as exc:
            raise _RetryableError(f"network error: {exc}") from exc

        if response.status_code in _RETRYABLE_CODES:
            raise _RetryableError(f"HTTP {response.status_code}")
        if response.status_code >= 400:
            raise _FatalError(f"HTTP {response.status_code}: {response.text[:200]}")

        try:
            data = response.json()
        except ValueError as exc:
            raise _RetryableError(f"invalid JSON response body: {exc}") from exc
        if "error" in data and "choices" not in data:
            code, msg = data["error"].get("code", 0), data["error"].get("message", "unknown")
            if code in _RETRYABLE_CODES or code == 0:
                raise _RetryableError(f"API error {code}: {msg}")
            raise _FatalError(f"API error {code}: {msg}")

        usage = data.get("usage") if isinstance(data, dict) else None
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise _FatalError("response missing message content", usage) from exc
        # cut off at max_tokens: the same prompt would be cut off again, so the next model rather than a retry
        if choice.get("finish_reason") == "length":
            raise _FatalError(f"response truncated at max_tokens={self.max_tokens} (finish_reason=length)", usage)
        if isinstance(content, list):
            content = "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
        if not isinstance(content, str) or not content.strip():
            raise _FatalError("model returned empty content", usage)

        content = content.strip()
        match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", content, re.DOTALL)  # a fence some models add
        content = match.group(1).strip() if match else content
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise _RetryableError(f"invalid JSON from model: {content[:120]}", usage) from exc
        if not isinstance(parsed, dict):  # every caller reads an object
            raise _RetryableError(f"the answer is not a JSON object: {content[:120]}", usage)
        return content, usage


def _done(started_at: str) -> dict:
    finished_at = datetime.now(UTC).isoformat()
    ms = (datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)).total_seconds() * 1000
    return {"finished_at": finished_at, "latency_ms": max(0, int(ms))}

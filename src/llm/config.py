"""The model provider's settings, from the environment; the repository's .env fills in what the environment does
not set (mode 600, out of git), so the server, the board and the command line read the same key."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_env(path: Path = ROOT / ".env") -> None:
    """KEY=value lines into os.environ, never overriding what is already set."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return
    for line in lines:
        key, sep, value = line.strip().partition("=")
        if sep and key and not key.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_env()

LLM_PROVIDER: str = (os.getenv("LLM_PROVIDER") or "openrouter").strip().lower()
LLM_API_KEY: str | None = os.getenv("LLM_API_KEY") or None
# All providers expose an OpenAI-compatible /chat/completions endpoint.
_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
LLM_BASE_URL: str = _BASE_URLS.get(LLM_PROVIDER, _BASE_URLS["openrouter"])


def provider_settings(provider: str) -> tuple[str, str | None]:
    """(base URL, key) for a provider: the main one's (LLM_PROVIDER, LLM_API_KEY), or another's from its own
    <PROVIDER>_API_KEY (as OPENROUTER_API_KEY), None when it has none set."""
    p = provider.strip().lower()
    key = LLM_API_KEY if p == LLM_PROVIDER else os.getenv(f"{p.upper()}_API_KEY") or None
    return _BASE_URLS.get(p, _BASE_URLS["openrouter"]), key


# OpenRouter optional headers — set these in code if needed, not via env.
LLM_HTTP_REFERER: str | None = None
LLM_APP_TITLE: str | None = None

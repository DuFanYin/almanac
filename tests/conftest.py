"""Tests never call a model: a stand-in key, set before src/llm/config.py reads the environment, so they pass the
same on a fresh clone without .env as here."""

import os

os.environ["LLM_API_KEY"] = "test-key"
os.environ.setdefault("LLM_PROVIDER", "deepseek")

"""The version of a method, for a caller to keep with each answer: a hash of its prompts and of the code that
asks and summarises (the method's module and src/llm/prompts.py)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.llm.methods.prompts import METHOD_PROMPTS

RENDER = Path(__file__).resolve().parents[1] / "prompts.py"


def method_hash(section: str, module_file: str) -> str:
    h = hashlib.sha1(json.dumps(METHOD_PROMPTS[section], sort_keys=True).encode())
    for f in (Path(module_file), RENDER):
        h.update(f.read_bytes())
    return h.hexdigest()[:10]

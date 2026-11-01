"""The methods as a command line, for programs in other languages or environments: JSON in on stdin, JSON out
on stdout; nothing is stored here. The provider and its key come from the environment (LLM_PROVIDER,
LLM_API_KEY), as for the board.

    python -m src.llm.methods forecast    {"today", "questions": [{"id", "title", "detail"?, "outcomes": [...]}],
                                           "data": [{"id", "name", "value", "day"?}], "models": [...], "samples"?,
                                           "temperature"?}   (an outcome: {"id", "label"})
                                          -> {"method", "temperature", "answers", "summary": {question id:
                                              {"p", "iqr", "n", "failed", "incoherence"}}}
    python -m src.llm.methods write       {"role", "instructions", "kind": "text" | "points", "data": [...], "model"?}
                                          -> {"method", "output", "valid", "attempts"}
    python -m src.llm.methods indicators  {"question", "detail"?, "outcomes": [...], "model"?}
                                          -> {"method", "output": {"drivers": [...]}}

`method` is the hash of the method's prompts and code: a caller keeps it with the answer. `usage` is what the
call cost: model calls, tokens, USD, and the models that answered.
"""

from __future__ import annotations

import asyncio
import json
import sys

from src.llm.calls import LLMCallCollector
from src.llm.client import LLMClient
from src.llm.methods import forecast, grounded, indicators

COLLECTOR = LLMCallCollector()


def usage() -> dict:
    cost, prompt, completion = COLLECTOR.totals()
    done = [c for c in COLLECTOR.all() if c.status == "success"]
    return {
        "calls": len(done),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cost_usd": cost,
        "models": sorted({c.model for c in done}),
    }


async def run_forecast(req: dict) -> dict:
    return await forecast.forecast(
        req["questions"],
        req["data"],
        today=req["today"],
        models=req["models"],
        samples=req.get("samples", 3),
        temperature=req.get("temperature", 0.7),
        client_for=lambda m: LLMClient(
            model=m, temperature=req.get("temperature", 0.7), max_tokens=800, collector=COLLECTOR
        ),
    )


async def run_write(req: dict) -> dict:
    return await grounded.write(
        req["role"],
        req["instructions"],
        req["data"],
        kind=req.get("kind", "text"),
        client=LLMClient(model=req.get("model"), max_tokens=2000, collector=COLLECTOR),
    )


async def run_indicators(req: dict) -> dict:
    return await indicators.propose(
        req["question"],
        req["outcomes"],
        detail=req.get("detail", ""),
        client=LLMClient(model=req.get("model"), max_tokens=4000, collector=COLLECTOR),
    )


COMMANDS = {"forecast": run_forecast, "write": run_write, "indicators": run_indicators}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in COMMANDS:
        print(__doc__, file=sys.stderr)
        return 2
    out = asyncio.run(COMMANDS[sys.argv[1]](json.load(sys.stdin)))
    json.dump({**out, "usage": usage()}, sys.stdout, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

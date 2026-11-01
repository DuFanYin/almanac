"""Research (src/research) on one of the board's questions: the news searched and read into checked notes (which the
scenarios' writer reads), then the forecast method on the board's data and the notes (VIEWS, the share of the data
left out, data/llm.py's temperature), with research's own model (`research_model` in local/settings.toml, else
MODEL) and fewer samples: each answer has a long prompt. Its forecast is kept and scored beside the world's."""

import asyncio

from src.llm.calls import LLMCallCollector
from src.llm.client import LLMClient
from src.research import research

from ..core import store
from . import llm

MODEL = "deepseek-v4-pro"  # few questions, every four weeks: worth the better model
SAMPLES = 3  # each view once
VIEWS = [  # the forecast's perspectives, one sample each
    "a macroeconomist",
    "a market historian who asks how often such things happened before",
    "a risk manager looking for what everyone else is missing",
]
DROP = 0.2  # the share of the data each sample leaves out


def model():
    return str(store.settings().get("research_model") or MODEL)


def method():
    """Research's version: its own and its methods' prompts and code, and the model."""
    return f"{research.method_hash()}-{model()}"


def run(today, question, data, samples=SAMPLES):
    """(research's result, its meta) for one question {"id", "title", "detail", "buckets"} on the numbered data."""
    llm.ready()
    calls, m = LLMCallCollector(), model()
    out = asyncio.run(
        research.research(
            {
                "id": question["id"],
                "title": question["title"],
                "detail": question.get("detail", ""),
                "outcomes": question["buckets"],
            },
            data,
            today=today,
            client=LLMClient(model=m, temperature=0.3, max_tokens=3000, collector=calls),
            forecast_kw={
                "models": [m],
                "samples": samples,
                "temperature": llm.TEMPERATURE,
                "views": VIEWS,
                "drop": DROP,
                "seed": today,
                "client_for": lambda x: LLMClient(
                    model=x, temperature=llm.TEMPERATURE, max_tokens=800, collector=calls
                ),
            },
        )
    )
    return out, {"prompt_hash": out["method"], "models": [m], "samples": samples, "usage": llm.usage(calls)}

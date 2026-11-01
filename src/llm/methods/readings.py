"""A text read into a number on a stated scale (a central bank's statement, how hawkish; a set of headlines, what
mood): asked `samples` times, each answer giving whether the text bears on the question, a number on the scale, and
one to three phrases quoted from the text that carry it. An answer whose quotes are not in the text word for word, or
whose number is off the scale, is dropped (as research's notes are); the number is the mean of the answers that kept
the text relevant, with their spread. The caller decides what to read and keeps the readings as a series."""

from __future__ import annotations

import asyncio
import json
import re
import statistics

from src.llm.methods.method import method_hash as _method_hash
from src.llm.methods.prompts import METHOD_PROMPTS
from src.llm.prompts import render

LOW, HIGH = -2.0, 2.0
MAX_CHARS = 15000  # of the text, at most
MIN_QUOTE = 12  # characters a quote has at least: a word or two would be found in any text


def method_hash() -> str:
    return _method_hash("readings", __file__)


def plain(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def checked(answer: dict, text: str) -> tuple[dict | None, str | None]:
    """(the answer as kept, None) or (None, why not)."""
    try:
        r = answer["relevant"]
        if isinstance(r, str) and r.strip().lower() in ("true", "false"):
            r = r.strip().lower() == "true"
        if not isinstance(r, bool):
            raise TypeError(f"relevant is {r!r}, not true or false")
        relevant = r
        quotes = [str(q) for q in answer.get("quotes") or []][:3]
        value = float(answer["value"]) if relevant else None
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        return None, f"malformed: {type(e).__name__} {e}"[:200]
    if relevant and not LOW <= value <= HIGH:
        return None, f"{value} is off the scale"
    body = plain(text)
    if relevant and (not quotes or any(len(plain(q)) < MIN_QUOTE or plain(q) not in body for q in quotes)):
        return None, f"a quote is shorter than {MIN_QUOTE} characters or not in the text"
    return {"relevant": relevant, "value": value, "quotes": quotes, "why": str(answer.get("why", ""))[:300]}, None


async def read(text: str, *, what: str, scale: str, title: str, day: str, samples: int, client_for) -> dict:
    """{"method", "relevant", "value" (None when not relevant or no answer kept), "spread", "answers", "dropped"}.
    what: what to read the text for; scale: the scale's ends and middle, in words."""
    text = text[:MAX_CHARS]
    system, user = render(METHOD_PROMPTS["readings"]["read"], what=what, scale=scale, title=title, day=day, text=text)

    async def one(i):
        try:
            return json.loads(await client_for(i).complete(user, system=system, node="readings.read"))
        except Exception as e:  # a failed call is a dropped answer
            return {"error": f"{type(e).__name__}: {e}"[:200]}

    kept, dropped = [], []
    for raw in await asyncio.gather(*(one(i) for i in range(samples))):
        if not isinstance(raw, dict):
            a, why = None, "not a JSON object"
        else:
            a, why = checked(raw, text) if "error" not in raw else (None, raw["error"])
        (kept.append(a) if a else dropped.append({"answer": raw, "why": why}))
    rel = [a for a in kept if a["relevant"]]
    relevant = bool(kept) and len(rel) * 2 > len(kept)
    values = [a["value"] for a in rel]
    return {
        "method": method_hash(),
        "relevant": relevant,
        "value": statistics.fmean(values) if relevant else None,
        "spread": statistics.pstdev(values) if relevant and len(values) > 1 else None,
        "answers": kept,
        "dropped": dropped,
    }

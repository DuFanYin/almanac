"""Probabilities for fixed, mutually exclusive outcomes, asked several ways so the spread shows.

Every question is asked by every model, in every wording, several times; each sample can take a different view
(a perspective named in the system message, by turns) and leave out a random part of the data (drop), so the
spread shows how much the answer rests on one way of looking or on a few numbers: the outcomes listed in order, the same
reversed, and each outcome on its own (the raw sum of those single answers shows how coherent the model is;
then normalised). The summary per question is each outcome's median over the answers (renormalised), its
interquartile range, how many answers, and how far the single-wording sums stray from 1 (incoherence). Numbers
come only from the caller's data; the caller owns the questions and what the probabilities are used for.
"""

from __future__ import annotations

import asyncio
import json
import random
import statistics
from collections.abc import Callable

from src.llm.client import LLMClient
from src.llm.methods.method import method_hash as _method_hash
from src.llm.methods.prompts import METHOD_PROMPTS
from src.llm.prompts import numbered_data, render

WORDINGS = ("listed", "reversed", "single")


def method_hash() -> str:
    return _method_hash("forecast", __file__)


def outcome_lines(outcomes: list[dict]) -> str:
    return "\n".join(f"- {o['id']}: {o['label']}" for o in outcomes)


def _as(system: str, view: str | None) -> str:
    return f"{system} Think as {view} would." if view else system


async def ask(client: LLMClient, question: dict, wording: str, today: str, data: str, view: str | None = None) -> dict:
    """One answer: {"probabilities" (normalised, None if all 0), "raw_sum", "raw"}."""
    outcomes = question["outcomes"]
    if wording == "single":
        raw, probs = {}, {}
        for o in outcomes:
            system, user = render(
                METHOD_PROMPTS["forecast"]["single"],
                today=today,
                question=question["title"],
                detail=question.get("detail", ""),
                outcome=o["label"],
                data=data,
            )
            out = json.loads(await client.complete(user, system=_as(system, view), node="forecast.single"))
            raw[o["id"]] = out
            probs[o["id"]] = max(0.0, float(out.get("probability", 0)))
    else:
        order = outcomes if wording == "listed" else list(reversed(outcomes))
        system, user = render(
            METHOD_PROMPTS["forecast"]["listed"],
            today=today,
            question=question["title"],
            detail=question.get("detail", ""),
            outcomes=outcome_lines(order),
            data=data,
        )
        raw = json.loads(await client.complete(user, system=_as(system, view), node="forecast.listed"))
        given = raw.get("probabilities", {}) if isinstance(raw.get("probabilities"), dict) else {}
        probs = {o["id"]: max(0.0, float(given.get(o["id"], 0))) for o in outcomes}
    total = sum(probs.values())
    return {"probabilities": {k: v / total for k, v in probs.items()} if total else None, "raw_sum": total, "raw": raw}


def summarise(question: dict, answers: list[dict]) -> dict | None:
    """The question's summary over its usable answers; None without any. "views": each view's medians."""
    good = [a for a in answers if a.get("probabilities")]
    if not good:
        return None
    ids = [o["id"] for o in question["outcomes"]]
    med = {k: statistics.median(a["probabilities"][k] for a in good) for k in ids}
    total = sum(med.values()) or 1.0
    iqr = {}
    for k in ids:
        xs = sorted(a["probabilities"][k] for a in good)
        iqr[k] = xs[(3 * len(xs)) // 4] - xs[len(xs) // 4] if len(xs) >= 4 else None
    singles = [abs(a["raw_sum"] - 1) for a in good if a["wording"] == "single"]
    by_view = {}
    for a in good:
        by_view.setdefault(a.get("view"), []).append(a)
    views = {}
    if len(by_view) > 1:
        for v, xs in by_view.items():
            m = {k: statistics.median(a["probabilities"][k] for a in xs) for k in ids}
            t = sum(m.values()) or 1.0
            views[v] = {k: x / t for k, x in m.items()}
    return {
        "p": {k: v / total for k, v in med.items()},
        "iqr": iqr,
        "n": len(good),
        "failed": len(answers) - len(good),
        "incoherence": statistics.median(singles) if singles else None,
        "views": views,  # each view's medians (renormalised), when there is more than one
    }


async def forecast(
    questions: list[dict],
    data: list[dict],
    *,
    today: str,
    models: list[str],
    samples: int = 3,
    temperature: float = 0.7,
    concurrency: int = 4,
    client_for: Callable[[str], LLMClient] | None = None,
    views: list[str] | None = None,
    drop: float = 0.0,
    seed: str = "",
) -> dict:
    """{"method", "temperature", "answers": [...], "summary": {question id: summary}} for questions
    [{"id", "title", "detail"?, "outcomes": [{"id", "label"}]}] and data [{"id", "name", "value", "day"?}].
    views: perspectives taken by turns, sample by sample; drop: each data item left out of a sample's prompt with
    this chance (at least one kept), seeded by `seed`, the question, the wording and the sample, so a run can be
    repeated. One failed answer is left out of its question's summary, not fatal."""
    make = client_for or (lambda m: LLMClient(model=m, temperature=temperature, max_tokens=800))
    sem = asyncio.Semaphore(concurrency)

    async def one(q, model, wording, sample):
        view = views[sample % len(views)] if views else None
        kept = data
        if drop > 0 and len(data) > 1:
            rng = random.Random(f"{seed}:{q['id']}:{model}:{wording}:{sample}")
            kept = [d for d in data if rng.random() >= drop] or [rng.choice(data)]
        base = {
            "question": q["id"],
            "model": model,
            "wording": wording,
            "sample": sample,
            "view": view,
            "dropped": sorted({d["id"] for d in data} - {d["id"] for d in kept}),
        }
        try:
            async with sem:
                return {**base, **await ask(make(model), q, wording, today, numbered_data(kept), view)}
        except Exception as e:
            return {**base, "error": f"{type(e).__name__}: {e}"[:300]}

    jobs = [one(q, m, w, s) for q in questions for m in models for w in WORDINGS for s in range(samples)]
    answers = await asyncio.gather(*jobs)
    summary = {}
    for q in questions:
        sm = summarise(q, [a for a in answers if a["question"] == q["id"]])
        if sm:
            summary[q["id"]] = sm
    return {"method": method_hash(), "temperature": temperature, "answers": answers, "summary": summary}

"""Research for one forecasting question: the evidence the caller's data does not hold (the news), read into notes
that are checked, then the forecast method on the data and the notes.

1. plan: what would move the answer (drivers) and the news searches that show where those stand;
2. search (sources.py), each item new by its link or title;
3. notes: grounded writing (src/llm/methods/grounded.py) over the data and the items, each point citing the items
   it rests on; a point with a number that is not the cited item's, or citing an item that does not exist, is
   dropped, so the forecast sees only checked points;
4. judge: are the notes enough to judge each driver today? If not, its searches, and steps 2-4 again (ROUNDS);
5. forecast (src/llm/methods/forecast.py) on the caller's data and the notes, asked as the caller asks it.

The caller owns the question, the data and how the forecast is asked (models, samples, views); this returns what
was found, the notes, every round's judgment and the forecast's summary and answers, so each can be kept and
checked. A failed search gives nothing; a failed plan, notes or judge call raises."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path

from src.llm.client import LLMClient
from src.llm.methods import forecast, grounded
from src.llm.prompts import numbered_data, quoted, render
from src.research import sources
from src.research.prompts import PROMPTS

ROUNDS = 2  # searches at most twice: the plan's, then the judge's
MAX_QUERIES = 5
MAX_ITEMS = 40  # items a round may add
CITE = re.compile(r"\[(D\d+)\]")


def method_hash() -> str:
    """Research's version: its prompts and code, and those of the methods it asks."""
    h = hashlib.sha1(json.dumps(PROMPTS, sort_keys=True).encode())
    for f in (Path(__file__), Path(sources.__file__)):
        h.update(f.read_bytes())
    h.update((forecast.method_hash() + grounded.method_hash()).encode())
    return h.hexdigest()[:10]


def outcome_lines(question: dict) -> str:
    return forecast.outcome_lines(question["outcomes"])


def as_items(found: list[dict], first_id: int) -> list[dict]:
    """Search results as data items the model cites ([D61]), each one's text fenced as a quote."""
    return [
        {
            "id": f"D{first_id + i}",
            "name": ", ".join(x for x in (x["kind"], x["source"], x["day"]) if x),
            "value": quoted(f"{x['title']}. {x['text']}".rstrip(". "), 600),
            "url": x["url"],
        }
        for i, x in enumerate(found)
    ]


def checked(points: list[str], items: list[dict]) -> tuple[list[str], list[dict]]:
    """(the points that pass, [{"point", "unknown_numbers", "unknown_ids"}] for those that do not)."""
    ids = {d["id"] for d in items}
    good, bad = [], []
    for p in points:
        nums, cites = grounded.unknown_numbers(p, items), [c for c in CITE.findall(p) if c not in ids]
        if nums or cites:
            bad.append({"point": p, "unknown_numbers": nums, "unknown_ids": cites})
        else:
            good.append(p)
    return good, bad


async def ask_json(client: LLMClient, prompt: str, node: str, **kw) -> dict:
    system, user = render(PROMPTS[prompt], **kw)
    out = json.loads(await client.complete(user, system=system, node=node))
    if not isinstance(out, dict):
        raise ValueError(f"{node}: expected a JSON object")
    return out


async def research(
    question: dict,
    data: list[dict],
    *,
    today: str,
    client: LLMClient,
    forecast_kw: dict,
    search: Callable[[str], list[dict]] = sources.search,
    rounds: int = ROUNDS,
) -> dict:
    """{"method", "drivers", "rounds": [{"queries", "found", "kept", "dropped", "judge"}], "evidence", "notes",
    "summary" (the forecast's, None without a usable answer), "answers"} for question {"id", "title", "detail"?,
    "outcomes": [{"id", "label"}]} and the caller's data [{"id": "D1", "name", "value"}]. forecast_kw: how the
    forecast is asked (models, samples, temperature, views, drop, seed, client_for)."""
    ask = {"today": today, "question": question["title"], "detail": question.get("detail", "")}
    plan = await ask_json(
        client, "plan", "research.plan", **ask, outcomes=outcome_lines(question), data=numbered_data(data)
    )
    drivers = [str(x) for x in plan.get("drivers") or []][:5]
    queries = [str(x) for x in plan.get("queries") or []][:MAX_QUERIES]
    evidence, seen, asked, log, notes = [], set(), set(), [], []
    for n in range(1, rounds + 1):
        asked |= set(queries)
        found = []
        for batch in await asyncio.gather(*(asyncio.to_thread(search, q) for q in queries)):
            for x in batch:
                key = x.get("url") or x["title"]
                if key not in seen and len(found) < MAX_ITEMS:
                    seen.add(key)
                    found.append(x)
        evidence += as_items(found, len(data) + len(evidence) + 1)
        items = data + evidence
        spec = PROMPTS["notes"]
        out = await grounded.write(
            spec["role"],
            spec["instructions"].format(**ask, outcomes=outcome_lines(question), drivers="; ".join(drivers)),
            items,
            kind="points",
            client=client,
        )
        notes, dropped = checked([str(p) for p in out["output"].get("points") or []], items)
        verdict = await ask_json(
            client,
            "judge",
            "research.judge",
            today=today,
            question=question["title"],
            drivers="; ".join(drivers),
            notes="\n".join(f"- {p}" for p in notes) or "none",
            found=len(evidence),
            round=n,
            rounds=rounds,
        )
        log.append({"queries": queries, "found": len(found), "kept": len(notes), "dropped": dropped, "judge": verdict})
        queries = [str(q) for q in verdict.get("queries") or [] if str(q) not in asked][:MAX_QUERIES]
        if verdict.get("enough") or not queries:
            break
    noted = [
        {"id": f"D{len(data) + len(evidence) + i + 1}", "name": "Research note", "value": p}
        for i, p in enumerate(notes)
    ]
    out = await forecast.forecast([question], data + noted, today=today, **forecast_kw)
    return {
        "method": method_hash(),
        "drivers": drivers,
        "rounds": log,
        "evidence": evidence,
        "notes": noted,
        "summary": out["summary"].get(question["id"]),
        "answers": out["answers"],
    }

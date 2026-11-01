"""Indicators, drafted by a model. propose: a first draft of the drivers of a forecasting question and the observable
indicators that would tell which outcome is becoming more likely, each with thresholds and likelihood ratios, for a
person to check. for_board: new indicators for a board, in its own form (an expression over the series it keeps,
a red and a yellow line, the forecasts to try it on), shape-checked only: the board checks each on its data and
judges it on history and on what comes after (nothing here fits them to data). expect: what an indicator says about
the factors ahead (a factor, a way, a span of days, a mechanism), for one too new to be measured."""

from __future__ import annotations

import json
import re

from src.llm.client import LLMClient
from src.llm.methods.forecast import outcome_lines
from src.llm.methods.method import method_hash as _method_hash
from src.llm.methods.prompts import METHOD_PROMPTS
from src.llm.prompts import numbered_data, render

ID = re.compile(r"^[a-z][a-z0-9_]{2,30}$")
LINE = re.compile(r"^(<|<=|>|>=) -?\d+(\.\d+)?$")


def method_hash() -> str:
    return _method_hash("indicators", __file__)


async def propose(question: str, outcomes: list[dict], *, detail: str = "", client: LLMClient | None = None) -> dict:
    """{"method", "output": {"drivers": [{"name", "why", "indicators": [...]}]}}."""
    client = client or LLMClient(max_tokens=4000)
    system, user = render(
        METHOD_PROMPTS["indicators"]["propose"], question=question, detail=detail, outcomes=outcome_lines(outcomes)
    )
    return {
        "method": method_hash(),
        "output": json.loads(await client.complete(user, system=system, node="indicators.propose")),
    }


def shaped(x: dict, targets: set[str], taken: set[str]) -> tuple[dict | None, str | None]:
    """(the draft in the board's form, None) or (None, why not). targets: the forecasts it may be tried on; taken:
    the ids in use."""
    try:
        sp = {
            "id": str(x["id"]).strip().lower(),
            "name": str(x["name"])[:80],
            "driver": str(x.get("driver", ""))[:40],
            "value": str(x["value"]).strip(),
            "red": str(x["red"]).strip(),
            "try_on": [str(t) for t in x["try_on"]],
            "why": str(x.get("why", ""))[:400],
        }
        if x.get("yellow"):
            sp["yellow"] = str(x["yellow"]).strip()
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        return None, f"malformed: {type(e).__name__} {e}"[:200]
    if not ID.match(sp["id"]) or sp["id"] in taken:
        return None, "the id must be new, lower case, letters, digits and _"
    if not LINE.match(sp["red"]) or ("yellow" in sp and not LINE.match(sp["yellow"])):
        return None, 'a line is "op number", as "> 0.2"'
    if not sp["try_on"] or not set(sp["try_on"]) <= targets:
        return None, f"try_on must name some of {sorted(targets)}"
    return sp, None


async def for_board(
    *,
    today: str,
    targets: str,
    series: list[str],
    language: str,
    existing: list[str],
    data: list[dict],
    n: int,
    client: LLMClient,
) -> dict:
    """{"method", "signposts": [shaped], "rejected": [{"draft", "why"}]}. targets: the forecasts, one a line ("- id:
    ..."); series: "id: name, how often, since"; existing: "id: name = expression, red line"."""
    system, user = render(
        METHOD_PROMPTS["indicators"]["board"],
        today=today,
        targets=targets,
        series="\n".join(f"- {s}" for s in series),
        language=language,
        existing="\n".join(f"- {e}" for e in existing) or "(none)",
        data=numbered_data(data),
        n=n,
    )
    raw = json.loads(await client.complete(user, system=system, node="indicators.board"))
    known = set(re.findall(r"^- ([a-z0-9_]+):", targets, re.M))
    taken = {e.split(":")[0] for e in existing}
    kept, rejected = [], []
    for x in (raw.get("signposts") or [])[: n + 2] if isinstance(raw.get("signposts"), list) else []:
        sp, why = shaped(x, known, taken) if isinstance(x, dict) else (None, "not an object")
        if sp:
            kept.append(sp)
            taken.add(sp["id"])
        else:
            rejected.append({"draft": x, "why": why})
    return {"method": method_hash(), "signposts": kept[:n], "rejected": rejected}


WAYS = {"up", "down", "wider", "narrower"}


def expectation(x: dict, factors: set[str]) -> tuple[dict | None, str | None]:
    """(an expectation in the board's form, None) or (None, why not)."""
    try:
        e = {
            "factor": str(x["factor"]),
            "way": str(x["way"]),
            "days": [int(x["days"][0]), int(x["days"][1])],
            "why": str(x.get("why", ""))[:300],
        }
    except (KeyError, TypeError, ValueError, IndexError) as err:
        return None, f"malformed: {type(err).__name__} {err}"[:200]
    if e["factor"] not in factors:
        return None, f"factor must be one of {sorted(factors)}"
    if e["way"] not in WAYS:
        return None, f"way must be one of {sorted(WAYS)}"
    if not 0 <= e["days"][0] < e["days"][1] <= 1095:
        return None, "days: from before to, within 0 to 1095"
    return e, None


async def expect(indicator: dict, factors: dict[str, str], *, client: LLMClient) -> dict:
    """{"method", "expect": [the shaped expectations], "rejected": [{"draft", "why"}]} for an indicator ({"name",
    "what", "scale", "value", "red", "why"}) over the caller's factors ({id: what and how measured})."""
    system, user = render(
        METHOD_PROMPTS["indicators"]["expect"],
        **{k: indicator.get(k, "") for k in ("name", "what", "scale", "value", "red", "why")},
        factors="\n".join(f"- {k}: {v}" for k, v in factors.items()),
    )
    raw = json.loads(await client.complete(user, system=system, node="indicators.expect"))
    out, rejected = [], []
    for x in (raw.get("expect") or [])[:3] if isinstance(raw.get("expect"), list) else []:
        e, why = expectation(x, set(factors)) if isinstance(x, dict) else (None, "not an object")
        if e:
            out.append(e)
        else:
            rejected.append({"draft": x, "why": why})
    return {"method": method_hash(), "expect": out, "rejected": rejected}

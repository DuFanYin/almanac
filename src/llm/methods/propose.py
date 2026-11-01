"""Questions a board might add, drafted by a model for a person to approve: each splits the future of one series
the board keeps into outcomes that cannot both happen and cover everything, so it resolves from that series alone.
And, the same way, factors it might add (factors); and, for a draft that passed the caller's checks, an independent
reading of whether to adopt it (judge).

Only the shape is checked here (the fields, the kinds, conditions written "name op number", a last bucket that
takes the rest); the caller checks each draft against its data (history's base rate, overlap with what it asks,
what it would change) and a person adopts it or not. Nothing here registers anything."""

from __future__ import annotations

import json
import re

from src.llm.client import LLMClient
from src.llm.methods.method import method_hash as _method_hash
from src.llm.methods.prompts import METHOD_PROMPTS
from src.llm.prompts import numbered_data, render

ID = re.compile(r"^[a-z][a-z0-9_]{2,30}$")
COND = re.compile(r"^[a-z_]+ (<|<=|>|>=) -?\d+(\.\d+)?$")
NAMES = {"path": {"ret", "dd", "hi", "lo"}, "ret": {"ret"}, "end": {"end"}, "yoy_change": {"chg", "yoy"}}


def method_hash() -> str:
    return _method_hash("propose", __file__)


def shaped(x: dict, series: set[str]) -> tuple[dict | None, str | None]:
    """(the draft as the board would register it, None) or (None, why not)."""
    try:
        kind, sid = str(x["kind"]), str(x["series"])
        q = {
            "id": str(x["id"]).strip().lower(),
            "title": str(x["title"])[:120],
            "purpose": "score" if x.get("purpose") == "score" else "decision",
            "kind": kind,
            "series": sid,
            "detail": str(x.get("detail", ""))[:300],
            "why": str(x.get("why", ""))[:600],
        }
        buckets = [
            {
                "id": str(b["id"]),
                "label": str(b["label"])[:80],
                "when": [str(c).strip() for c in b.get("when") or []],
                "tone": max(-2.0, min(2.0, float(b.get("tone", 0)))),
            }
            for b in x["buckets"]
        ]
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        return None, f"malformed: {type(e).__name__} {e}"[:200]
    if not ID.match(q["id"]):
        return None, "id must be snake_case, 3 to 31 characters"
    if not all(re.fullmatch(r"[a-z0-9_]{1,30}", b["id"]) for b in buckets) or len({b["id"] for b in buckets}) < len(
        buckets
    ):
        return None, "each outcome's id must be distinct snake_case, at most 30 characters"
    if sid not in series:
        return None, f"no series {sid}"
    if kind == "path":
        try:
            q["days"] = int(x["days"])
        except (KeyError, TypeError, ValueError):
            return None, "a path question needs days"
        if not 28 <= q["days"] <= 1100:
            return None, "days must be 28 to 1100"
        names = NAMES["path"]
    elif kind == "period":
        q["every"], q["measure"] = str(x.get("every")), str(x.get("measure"))
        if q["every"] not in ("month", "quarter") or q["measure"] not in ("ret", "end", "yoy_change"):
            return None, "a period question needs every (month, quarter) and measure (ret, end, yoy_change)"
        names = NAMES[q["measure"]]
    else:
        return None, f"kind must be path or period, not {kind}"
    if not 2 <= len(buckets) <= 6 or len({b["id"] for b in buckets}) < len(buckets):
        return None, "2 to 6 buckets with distinct ids"
    if buckets[-1]["when"] or any(not b["when"] for b in buckets[:-1]):
        return None, "every bucket but the last needs a condition; the last takes the rest"
    for b in buckets:
        for c in b["when"]:
            if not COND.match(c) or c.split()[0] not in names:
                return None, f"condition {c!r}: write it 'name op number' with name one of {sorted(names)}"
    q["buckets"] = buckets
    return q, None


async def propose(
    questions: list[dict],
    series: list[dict],
    data: list[dict],
    *,
    today: str,
    context: str,
    client: LLMClient,
    n: int = 3,
) -> dict:
    """{"method", "proposals": [the shaped drafts], "rejected": [{"draft", "why"}]} for the questions the caller
    asks ([{"id", "title", "outcomes": [labels]}]), its series ([{"id", "name", "every", "since", "last"}]), its
    numbered data and what it sees (context, text)."""
    system, user = render(
        METHOD_PROMPTS["propose"]["questions"],
        today=today,
        questions="\n".join(f"- {q['id']}: {q['title']} ({'; '.join(q['outcomes'])})" for q in questions),
        series="\n".join(
            f"- {s['id']}: {s['name']}, {s['every']}, since {s['since']}, last {s['last']}" for s in series
        ),
        context=context,
        data=numbered_data(data),
        n=n,
    )
    raw = json.loads(await client.complete(user, system=system, node="propose.questions"))
    ids = {s["id"] for s in series}
    asked = {q["id"] for q in questions}
    proposals, rejected = [], []
    for x in (raw.get("questions") or [])[:n] if isinstance(raw.get("questions"), list) else []:
        q, why = shaped(x, ids) if isinstance(x, dict) else (None, "not an object")
        if q and q["id"] in asked:
            q, why = None, f"{q['id']} is already asked"
        if q:
            proposals.append(q)
        else:
            rejected.append({"draft": x, "why": why})
    return {"method": method_hash(), "proposals": proposals, "rejected": rejected}


MEASURES = {"ret", "diff", "ann"}


def shaped_factor(x: dict, series: set[str], taken: set[str]) -> tuple[dict | None, str | None]:
    """(the factor as the board would define it, None) or (None, why not)."""
    try:
        f = {
            "id": str(x["id"]).strip().lower(),
            "label": str(x["label"])[:60],
            "series": str(x["series"]),
            "measure": str(x["measure"]),
            "up": int(x.get("up", 0)),
            "why": str(x.get("why", ""))[:500],
        }
        if x.get("unit"):
            f["unit"] = str(x["unit"])[:10]
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        return None, f"malformed: {type(e).__name__} {e}"[:200]
    if not ID.match(f["id"]) or f["id"] in taken:
        return None, "the id must be new, snake_case, 3 to 31 characters"
    if f["series"] not in series:
        return None, f"no series {f['series']}"
    if f["measure"] not in MEASURES or f["up"] not in (-1, 0, 1):
        return None, f"measure one of {sorted(MEASURES)}, up -1, 0 or 1"
    return f, None


async def factors(
    *, today: str, existing: list[str], series: list[dict], data: list[dict], n: int, client: LLMClient
) -> dict:
    """{"method", "factors": [shaped], "rejected"}: new factors for the board's distribution. existing: "id: label,
    series, measure"; series: [{"id", "name", "every", "since", "last"}]."""
    system, user = render(
        METHOD_PROMPTS["propose"]["factors"],
        today=today,
        existing="\n".join(f"- {e}" for e in existing),
        series="\n".join(f"- {s['id']}: {s['name']}, {s['every']}, since {s['since']}" for s in series),
        data=numbered_data(data),
        n=n,
    )
    raw = json.loads(await client.complete(user, system=system, node="propose.factors"))
    ids, taken = {s["id"] for s in series}, {e.split(":")[0] for e in existing}
    kept, rejected = [], []
    for x in (raw.get("factors") or [])[:n] if isinstance(raw.get("factors"), list) else []:
        f, why = shaped_factor(x, ids, taken) if isinstance(x, dict) else (None, "not an object")
        (kept.append(f) if f else rejected.append({"draft": x, "why": why}))
    return {"method": method_hash(), "factors": kept, "rejected": rejected}


async def judge(
    *, today: str, kind: str, proposal: dict, checks: list[dict], has: list[str], client: LLMClient
) -> dict:
    """{"method", "adopt": bool, "why"}: one independent reading of whether a draft that passed the caller's checks
    should be adopted. kind: "question" or "factor"; checks: [{"name", "ok", "say"}]; has: what the board already
    asks (or its factors), one a line."""
    system, user = render(
        METHOD_PROMPTS["propose"]["judge"],
        today=today,
        kind=kind,
        proposal=json.dumps(proposal, ensure_ascii=False, indent=1)[:4000],
        checks="\n".join(f"- {c.get('name')}: {'ok' if c.get('ok') else 'failed'}, {c.get('say', '')}" for c in checks),
        has="\n".join(f"- {h}" for h in has),
    )
    raw = json.loads(await client.complete(user, system=system, node="propose.judge"))
    return {"method": method_hash(), "adopt": raw.get("adopt") is True, "why": str(raw.get("why") or "")[:500]}

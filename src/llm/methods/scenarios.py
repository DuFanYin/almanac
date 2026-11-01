"""Scenarios for a board's world, by several LLMs. The board simulates what can
happen over each length from history (every window weighed by today's signposts); the LLMs write, from today's data
and that simulation, the scenarios worth telling apart, each in a direction of its own (a story with the ranges of
one or two factors, any of them), and say of each one, on its own, how much likelier or less likely it is than the
simulation's share for what the simulation does not include: a factor. The board takes the simulation's share of
each (the paths in its ranges; scenarios may overlap) times its factor as the probability it should have, reweighs
the paths to give every scenario its own while moving the simulation least, and computes what each scenario means
for someone's holdings (better or worse than usual is a result, not a judgment). The roles:

write    one model writes the scenarios of every length, revising the standing ones: no probabilities, no sides
read     each reader (a perspective on a model) reads one scenario at a time, never the others, so there is no total
         to share out: the factor, what the simulation misses, why
review   another model checks the readers' reasons, not their numbers (is what a reason cites outside the simulation,
         is it evidence or a habit, is it true); it may ask the board's own data a few questions (tools, read-only)
         first; the readers of a scenario it found fault with read it again, told what it said
coach    now and then, what the scores of the scenarios that are over say about the factors given, for every role

Only the shape is checked here (known lengths and factors, ranges low below high or open at one end, one to CUT of
them, at most the length's count, a factor above 0)."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import statistics
from collections.abc import Callable

from src.llm.client import LLMClient
from src.llm.methods.method import method_hash as _method_hash
from src.llm.methods.prompts import METHOD_PROMPTS
from src.llm.prompts import numbered_data, render

CUT = 3  # a scenario's ranges: on at most this many factors (the writer is told why one or two)
ASKS = 5  # the reviewer's questions to the board's data, at most
SLUG = re.compile(r"[^a-z0-9]+")
log = logging.getLogger("llm")


def method_hash() -> str:
    return _method_hash("scenarios", __file__)


def slug(name: str) -> str:
    return SLUG.sub("-", name.lower()).strip("-")[:40]


def span(lo, hi) -> str:
    """A range in words: "-0.1 to 0.05", "below -0.1", "above 0.05"."""
    if lo is None:
        return f"below {hi:+.4g}"
    if hi is None:
        return f"above {lo:+.4g}"
    return f"{lo:+.4g} to {hi:+.4g}"


def shaped(x: dict, *, factors: set, rates=()) -> tuple[dict | None, str | None]:
    """(the scenario as the board keeps it: {"id", "name", "story", "ranges" {factor: [low, high], either None where
    open}}, None) or (None, why not). rates: the factors given as a decimal (a return, a yearly rate), whose
    ranges are read as percents when written past 1.5 (-3.5 for -3.5%)."""
    try:
        ranges = {}
        for k, v in (x.get("ranges") or {}).items():
            if k not in factors:
                return None, f"unknown factor {k}"
            lo, hi = (None if v[0] is None else float(v[0])), (None if v[1] is None else float(v[1]))
            given = [y for y in (lo, hi) if y is not None]
            if not given:
                return None, f"{k}: a range open at both ends says nothing"
            if k in rates and max(abs(y) for y in given) > 1.5:  # written in percent
                lo, hi = (None if lo is None else lo / 100), (None if hi is None else hi / 100)
            if lo is not None and hi is not None and not lo < hi:
                return None, f"{k}: low must be below high"
            ranges[k] = [lo, hi]
        if not 1 <= len(ranges) <= CUT:
            return None, f"give ranges on one to {CUT} factors"
        name = str(x["name"]).strip()[:60]
        if not slug(name):
            return None, "a name"
        story = str(x.get("story", ""))[:700]
        return {"id": slug(name), "name": name, "story": story, "ranges": ranges}, None
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as e:
        return None, f"malformed: {type(e).__name__} {e}"[:200]


def known(factors: str) -> dict:
    """The factor ids, and those that are rates, from the factors' text the writer is given."""
    return dict(
        factors=set(re.findall(r"^- ([a-z0-9_]+):", factors, re.M)),
        rates=set(re.findall(r"^- ([a-z0-9_]+):[^\n]*\((?:its return|its yearly rate)[^)]*decimal", factors, re.M)),
    )


def lengths_text(lengths: dict) -> str:
    """lengths: {length: (label, days, most)}."""
    return "\n".join(
        f"- {h}: {label} ({days} days), at most {most} scenarios" for h, (label, days, most) in lengths.items()
    )


def sets(raw: dict, lengths: dict, factors: str) -> tuple[dict, list]:
    """({length: [scenario]}, rejected [{"length", "draft", "why"}]) from the writer's answer: each length's scenarios
    shaped, one name once, at most its count."""
    ok = known(factors)
    out, rejected = {}, []
    got = raw.get("lengths") if isinstance(raw.get("lengths"), dict) else {}
    for h, (_, _, most) in lengths.items():
        kept, seen = [], set()
        drafts = got.get(h) if isinstance(got.get(h), list) else []
        for x in drafts:
            s, why = shaped(x, **ok) if isinstance(x, dict) else (None, "not an object")
            if s and s["id"] in seen:
                s, why = None, "a name used twice"
            if s:
                kept.append(s)
                seen.add(s["id"])
            else:
                rejected.append({"length": h, "draft": x, "why": why})
        if kept:
            out[h] = kept[:most]
    return out, rejected


def told(s: dict) -> str:
    """A scenario in words: name, story, the factors' ranges over the length."""
    r = "; ".join(f"{k} {span(lo, hi)}" for k, (lo, hi) in s["ranges"].items())
    return f"{s['name']}. {s['story']} Ranges: {r}."


async def write(
    *,
    today: str,
    lengths: dict,
    simulation: str,
    factors: str,
    previous: str,
    record: str,
    coach: str,
    why: str,
    notes: str,
    data: list[dict],
    client: LLMClient,
) -> dict:
    """{"method", "sets": {length: [scenario]}, "rejected"}. lengths: {length: (label, days, most)}; simulation: what
    the simulation is built from (the signposts today), in words; previous: the scenarios standing, in words; record:
    how earlier ones turned out; coach: the coach's note; why: why now; notes: research's notes on the week's key
    questions."""
    system, user = render(
        METHOD_PROMPTS["scenarios"]["write"],
        today=today,
        lengths=lengths_text(lengths),
        simulation=simulation,
        factors=factors,
        previous=previous or "(none: this is the first writing)",
        record=record or "(none scored yet)",
        coach=coach or "(none yet)",
        why=why or "The weekly writing.",
        notes=notes or "(none)",
        data=numbered_data(data),
    )
    raw = json.loads(await client.complete(user, system=system, node="scenarios.write"))
    got, rejected = sets(raw, lengths, factors)
    return {"method": method_hash(), "sets": got, "rejected": rejected}


def answer(raw: dict) -> dict | None:
    """A reader's answer kept in shape: {"times" (above 0), "missing", "why"}, or None."""
    try:
        times = float(raw.get("times"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(times) or times <= 0:
        return None
    return {"times": times, "missing": str(raw.get("missing") or "")[:400], "why": str(raw.get("why") or "")[:800]}


async def read(
    *,
    today: str,
    length: str,
    scenario: dict,
    share: str,
    simulation: str,
    factors: str,
    view: str,
    market: str,
    record: str,
    coach: str,
    notes: str,
    data: list[dict],
    client: LLMClient,
    first: dict | None = None,
    review: str = "",
) -> dict | None:
    """One reader's answer() on one scenario, the others unseen. length: its length in words; share: its share of the
    simulation and of history, in words; simulation: the signposts today and the factors at its length, in words;
    view: the reader's perspective. With first (its answer before) and review (what the reviewer said of the
    readers' reasons on it): its answer again."""
    spec = METHOD_PROMPTS["scenarios"]["read_again" if first else "read"]
    spec = {**spec, "system": spec["system"].format(view=view)}  # the reader's perspective is in its role
    system, user = render(
        spec,
        today=today,
        length=length,
        scenario=told(scenario),
        share=share,
        simulation=simulation,
        factors=factors,
        market=market,
        record=record or "(none scored yet)",
        coach=coach or "(none yet)",
        notes=notes or "(none)",
        data=numbered_data(data),
        first=f"x{first['times']:.3g}; missing: {first['missing']}; why: {first['why']}" if first else "",
        review=review,
    )
    return answer(json.loads(await client.complete(user, system=system, node="scenarios.read")))


def points(raw: dict, sets_: dict) -> list[dict]:
    """The reviewer's points kept in shape: [{"length", "scenario" (an id of that length), "says", "why"}]."""
    out = []
    for p in raw.get("points") or [] if isinstance(raw.get("points"), list) else []:
        if not isinstance(p, dict) or p.get("length") not in sets_ or not str(p.get("why") or "").strip():
            continue
        if p.get("scenario") not in {s["id"] for s in sets_[p["length"]]}:
            continue
        out.append(
            {
                "length": p["length"],
                "scenario": p["scenario"],
                "says": str(p.get("says", ""))[:120],
                "why": str(p["why"])[:600],
            }
        )
    return out


async def review(
    *,
    today: str,
    lengths: dict,
    simulation: str,
    factors: str,
    readings: str,
    sets_: dict,
    market: str,
    record: str,
    tools: dict[str, tuple[str, Callable[[dict], str]]],
    client: LLMClient,
    asks: int = ASKS,
) -> dict:
    """{"method", "points": points(), "asked": [{"tool", "args", "answer"}]}: the reviewer's check of the readers'
    reasons. readings: each scenario with every reader's factor, what it says the simulation misses and why, in
    words. tools: {name: (what it answers and its arguments, in words; a function of the arguments returning the
    answer in words)}, the board's own data, read-only; asked at most `asks` times, each answer added to what it is
    told before it is asked again; then it must answer."""
    asked: list[dict] = []
    menu = "\n".join(f"- {k}: {what}" for k, (what, _) in tools.items())
    for turn in range(asks + 1):
        last = turn == asks
        so_far = "\n".join(f"- {a['tool']}({json.dumps(a['args'])}): {a['answer']}" for a in asked) or "(none yet)"
        system, user = render(
            METHOD_PROMPTS["scenarios"]["review"],
            today=today,
            lengths=lengths_text(lengths),
            simulation=simulation,
            factors=factors,
            readings=readings,
            market=market,
            record=record or "(none scored yet)",
            tools=menu,
            asked=so_far,
            left=0 if last else asks - len(asked),
        )
        raw = json.loads(await client.complete(user, system=system, node="scenarios.review"))
        ask = raw.get("ask") if isinstance(raw.get("ask"), dict) else None
        if ask and not last and ask.get("tool") in tools:
            args = ask.get("args") if isinstance(ask.get("args"), dict) else {}
            try:
                got = tools[ask["tool"]][1](args)
            except Exception as e:  # a question the data cannot answer: said so, and it may ask another
                got = f"cannot answer: {type(e).__name__} {e}"[:300]
            asked.append({"tool": ask["tool"], "args": args, "answer": str(got)[:1500]})
            continue
        return {"method": method_hash(), "points": points(raw, sets_), "asked": asked}
    return {"method": method_hash(), "points": [], "asked": asked}


async def coach(*, today: str, record: str, client: LLMClient) -> dict:
    """{"method", "note"}: what the scenarios that are over say about the factors given, for every role."""
    system, user = render(METHOD_PROMPTS["scenarios"]["coach"], today=today, record=record or "(none scored yet)")
    raw = json.loads(await client.complete(user, system=system, node="scenarios.coach"))
    return {"method": method_hash(), "note": str(raw.get("note") or "")[:1500]}


# ---- the roles together: write, write again over what no scenario covers, read, review, read again

READERS = (  # who reads each scenario: each reader's perspective, on the roster's models in turn
    "a macroeconomist who reads where the cycle is",
    "a market historian who asks how often such things happened before",
    "an optimist who looks for what is going right",
    "a pessimist who looks for what can go wrong",
)
READ_TEMPERATURE = 0.5  # the readers': their views differ by perspective, not by chance
AT_ONCE = 8  # the readers' calls in flight at once
MORE = 4  # scenarios a length may add when its first ones leave much of the simulation out


async def run(
    *,
    roster: list[str],
    client: Callable[[str, float, int], LLMClient],
    lengths: dict,
    today: str,
    simulation: str,
    factors: dict,
    previous: str,
    record: str,
    coach: str,
    why: str,
    notes: str,
    data: list[dict],
    shares: Callable[[dict], dict],
    market: str,
    tools: dict,
    uncovered: Callable[[dict], dict] = lambda sets_: {},
) -> dict:
    """The world's scenarios by several LLMs: written by the first of the roster; when they leave much of the
    simulation out (uncovered), the writer told where and writing again once, adding scenarios; each scenario read on
    its own by every reader (READERS, on the roster's models in turn); the readers' reasons checked by another (the
    roster's second, when it has one), which may ask the caller's data (tools); the readers of a scenario it found
    fault with reading it again, told what it said. roster: the models ("provider:model"); client(model, temperature,
    max_tokens): a client for one; lengths: {length: (label, days, most)}; simulation: the signposts today, in words;
    factors: {None: every length's factors, length: that length's}, in words; shares: a function of the sets giving
    {length: {scenario: its share of the simulation and of history, in words}}; uncovered: a function of the sets
    giving {length: the paths in no scenario, in words} for the lengths where too many are. {"method", "sets",
    "rejected", "covered" (what the first writing left out, when it was told), "readers" [{"view", "model"}],
    "readings" {length: {scenario: [each reader's {"first", "final"}: answer() or None]}}, "review" {"points",
    "asked", "model"}}."""
    readers = [{"view": v, "model": roster[i % len(roster)]} for i, v in enumerate(READERS)]
    reviewer = roster[1 % len(roster)]
    gate = asyncio.Semaphore(AT_ONCE)
    out = await write(
        today=today,
        lengths=lengths,
        simulation=simulation,
        factors=factors[None],
        previous=previous,
        record=record,
        coach=coach,
        why=why,
        notes=notes,
        data=data,
        client=client(roster[0], 0.9, 8000),
    )
    gaps = uncovered(out["sets"]) if out["sets"] else {}
    if gaps:  # much of the simulation in no scenario: the writer told where, and adds scenarios that tell it
        again = await write(
            today=today,
            lengths={h: (*lengths[h][:2], lengths[h][2] + MORE) for h in gaps},
            simulation=simulation,
            factors=factors[None],
            previous=written_text({h: out["sets"][h] for h in gaps}, lengths),
            record=record,
            coach=coach,
            why=why
            + " The scenarios you just wrote (standing, above) leave much of the simulation in no scenario: "
            + "; ".join(f"{lengths[h][0]}: {w}" for h, w in gaps.items())
            + ". Keep the scenarios that hold and add scenarios that tell those paths, each in a direction of "
            "its own, within each length's count (now higher). A scenario need not be narrow: one factor and a "
            "wide range hold many paths, and an ordinary stretch with nothing much happening is a scenario too.",
            notes=notes,
            data=data,
            client=client(roster[0], 0.9, 8000),
        )
        out = {
            **out,
            "sets": {**out["sets"], **again["sets"]},
            "rejected": out["rejected"] + again["rejected"],
            "covered": gaps,
        }
    if not out["sets"]:
        return {**out, "readers": readers, "readings": {}, "review": None}
    sets_ = out["sets"]
    share = shares(sets_)

    async def read_one(h, s, r, first=None, review=""):
        async with gate:
            try:
                return await read(
                    today=today,
                    length=f"{lengths[h][0]} ({lengths[h][1]} days)",
                    scenario=s,
                    share=share[h][s["id"]],
                    simulation=simulation,
                    factors=factors[h],
                    view=r["view"],
                    market=market,
                    record=record,
                    coach=coach,
                    notes=notes,
                    data=data,
                    first=first,
                    review=review,
                    client=client(r["model"], READ_TEMPERATURE, 1500),
                )
            except Exception:  # a reader that fails says nothing on this scenario
                return None

    each = [(h, s, r) for h, scs in sets_.items() for s in scs for r in readers]
    got = await asyncio.gather(*(read_one(h, s, r) for h, s, r in each))
    readings = {h: {s["id"]: [] for s in scs} for h, scs in sets_.items()}
    for (h, s, _), a in zip(each, got):
        readings[h][s["id"]].append({"first": a, "final": a})
    try:
        rev = await review(
            today=today,
            lengths=lengths,
            simulation=simulation,
            factors=factors[None],
            readings=readings_text(sets_, readings, readers, share),
            sets_=sets_,
            market=market,
            record=record,
            tools=tools,
            client=client(reviewer, 0.3, 8000),
        )
    except Exception as e:  # no review: the readers' first answers stand, said so on the page
        log.warning("scenarios review: %s: %s", type(e).__name__, e)
        rev = {"method": method_hash(), "points": [], "asked": [], "failed": f"{type(e).__name__}"}
    said = {}
    for p in rev["points"]:
        said.setdefault((p["length"], p["scenario"]), []).append(f"- {p['says']}: {p['why']}")
    again = [
        (h, s, i, r)
        for h, scs in sets_.items()
        for s in scs
        if (h, s["id"]) in said
        for i, r in enumerate(readers)
        if readings[h][s["id"]][i]["first"]
    ]
    got = await asyncio.gather(
        *(read_one(h, s, r, readings[h][s["id"]][i]["first"], "\n".join(said[h, s["id"]])) for h, s, i, r in again)
    )
    for (h, s, i, _), a in zip(again, got):
        if a:
            readings[h][s["id"]][i]["final"] = a
    return {**out, "readers": readers, "readings": readings, "review": {**rev, "model": reviewer}}


def written_text(sets_, lengths):
    """Scenarios just written, in words, for the writer to add to."""
    return "\n".join(f"{h} ({lengths[h][0]}):\n" + "\n".join(f"  - {told(s)}" for s in scs) for h, scs in sets_.items())


def times(readings):
    """{length: {scenario: factor}}: each scenario's median of the readers' final factors, taken on their logarithms
    (a factor is a ratio: x2 and x0.5 are as far from 1), for the scenarios some reader answered."""
    out = {}
    for h, by in readings.items():
        for sid, rs in by.items():
            got = [math.log(r["final"]["times"]) for r in rs if r.get("final")]
            if got:
                out.setdefault(h, {})[sid] = math.exp(statistics.median(got))
    return out


def readings_text(sets_, readings, readers, share):
    """Each scenario with its share and every reader's answer, in words, for the reviewer."""
    lines = []
    for h, scs in sets_.items():
        lines.append(f"{h}:")
        for s in scs:
            lines.append(f"  - {s['id']}: {told(s)} Its share: {share[h][s['id']]}")
            for r, x in zip(readers, readings[h][s["id"]]):
                a = x["first"]
                if a:
                    lines.append(f"    - {r['view']}: x{a['times']:.3g}; missing: {a['missing']}; why: {a['why']}")
    return "\n".join(lines)

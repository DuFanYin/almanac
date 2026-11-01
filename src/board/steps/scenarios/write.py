"""The scenarios written: weekly, every length in one go, revising the standing ones; again
before the week is up when something happens the simulation does not know of yet (a factor's unusual move), at most
once in EVENT_GAP days; the coach's note every COACH_EVERY days once a set is scored; and what changed, for the
day's news."""

import json
import logging
from datetime import date, timedelta

from ...core import lineage, store
from ...data import llm
from ...model import factors as fx
from ...model import scenarios as sm
from .context import (
    factors_text,
    market_text,
    record_text,
    share_words,
    simulation_text,
    standing_text,
    tools,
    uncovered,
)
from .sets import last_written, lengths, live

log = logging.getLogger("board")


# three years has some 18 windows that do not overlap
EVERY = 7  # days between the weekly writings


EVENT_GAP = 2  # days: written again on an event at most once in this many


COACH_EVERY = 28  # days between the coach's notes


def events(fcfg, series, today):
    """What happened today that calls for writing again: a daily factor's 5-day move beyond history's 1st or 99th
    percentile, the day it gets there. [in words]."""
    out = []
    for f in fcfg["factor"].values():
        first = f.get("history_from") or fcfg["history_from"]
        m = fx.unusual_move(f, series, today, first)
        if m and not fx.unusual_move(f, series, today - timedelta(days=1), first):
            out.append(
                f"{f['label']} moved {fx.fmt(f, m[0])} in {fx.MOVE_DAYS} days, beyond 98% of such moves since "
                f"{fcfg['history_from'][:4]} ({fx.fmt(f, m[1])} to {fx.fmt(f, m[2])})"
            )
    return out


def coach_note(db, labels, today, roster, llm_on=True):
    """The coach's latest note: written again every COACH_EVERY days once some set is scored (texts, kind
    scenarios_coach). "" before any."""
    row = db.execute("SELECT day, body FROM texts WHERE kind = 'scenarios_coach' ORDER BY day DESC LIMIT 1").fetchone()
    record = record_text(db, labels, n=40)
    due = record and (not row or (today - date.fromisoformat(row[0])).days >= COACH_EVERY)
    if due and llm_on:
        got = llm.scenarios_coach(today.isoformat(), record, roster)
        if got["note"]:
            db.execute(
                "INSERT OR REPLACE INTO texts VALUES (?,?,?,?)",
                (today.isoformat(), "scenarios_coach", got["note"], json.dumps({"method": got["method"]})),
            )
            return got["note"]
    return row[1] if row else ""


def write(db, cfg, world, data, notes, today, series=None, force=False, happened=()):
    """The scenarios of every length written when the week is up (or force), or on what happened (happened:
    events()) at most once in EVENT_GAP days: written by one LLM, each scenario read on its own by several, the reasons
    checked, read again where they were found at fault (data/llm.py scenarios). Each scenario's factor the median of
    the readers' last (1 when none answered); each length's set put on today's simulation and kept as the live one
    (the one before replaced). ({length: the set before, for the news}, [news])."""
    fcfg = cfg["factors"]
    sets = live(db)
    last, weekly = last_written(db), last_written(db, weekly=True)
    is_weekly = force or not weekly or (today - weekly).days >= EVERY
    if not is_weekly and not (happened and (not last or (today - last).days >= EVENT_GAP)):
        return {}, []
    L = lengths(fcfg)
    labels = {h: x[0] for h, x in L.items()}
    why = "The weekly writing." if is_weekly else "Since the last writing: " + "; ".join(happened) + "."
    if is_weekly and happened:
        why += " Also: " + "; ".join(happened) + "."
    roster = store.settings().get("scenario_models") or llm.ROSTER
    coach = coach_note(db, labels, today, roster)
    out = llm.scenarios(
        roster=roster,
        lengths=L,
        today=today.isoformat(),
        simulation=simulation_text(world, cfg.get("signposts")),
        factors={None: factors_text(fcfg, world), **{h: factors_text(fcfg, world, x[1]) for h, x in L.items()}},
        previous=standing_text(sets, labels),
        record=record_text(db, labels),
        coach=coach,
        why=why,
        notes="\n".join(f"- {n['name']}: {n['value']}" for n in notes),
        data=data,
        shares=lambda sets_: share_words(world, L, sets_),
        uncovered=lambda sets_: uncovered(world, L, sets_),
        market=market_text(series or {}, today),
        tools=tools(world, L),
    )
    data_node = lineage.latest(db, "snapshot")
    factor = llm.times(out["readings"])
    news, before = [], {}
    for h, scs in out["sets"].items():
        days = L[h][1]
        a = world.weights(days)
        for x in scs:  # a scenario no reader answered: no evidence against the simulation, so its factor is 1
            x["times"] = round(factor.get(h, {}).get(x["id"], 1.0), 4)
        got = sm.at_length(a, scs)
        if not got:
            log.warning("scenarios %s: none read, the set before stays", h)
            continue
        for x, e in zip(scs, got["each"]):
            rs = out["readings"][h][x["id"]]
            x.update(
                readings=[
                    {"view": r["view"], "model": r["model"], **{k: v for k, v in got_r.items()}}
                    for r, got_r in zip(out["readers"], rs)
                ],
                share=round(e["share"], 6),
                base=round(e["base"], 6),
                history=round(e["history"], 6),
                episodes=e["episodes"],
                made=e["made"],
                target=round(e["target"], 6),
                weight=round(e["weight"], 6),
            )
        used = sorted({k for x in scs for k in x["ranges"]})
        sd = sm.scale(a["X"])
        review = out.get("review") or {}
        spec = {
            "scenarios": scs,
            "periods": got["periods"],
            "trigger": "weekly" if is_weekly else "event",
            "why": why,
            "rejected": [r for r in out["rejected"] if r["length"] == h],
            "covered": (out.get("covered") or {}).get(h),  # what the first writing left out, told to the writer
            "days": days,
            "sd": {k: float(sd[a["names"].index(k)]) for k in used if k in a["names"]},
            "factor_defs": {k: {x: fcfg["factor"][k][x] for x in ("series", "measure")} for k in used},
            "readers": out["readers"],
            "review": {
                "model": review.get("model"),
                "points": [p for p in review.get("points", []) if p["length"] == h],
                "asked": review.get("asked", []),
            },
            "coach": coach,
            "method": out["method"],
        }
        node = lineage.record(
            db,
            "scenario_set",
            f"scenarios:{h}",
            {x["id"]: round(x["times"], 4) for x in scs},
            {"stories": {x["id"]: x["story"] for x in scs}, "method": out["method"], "usage": out["usage"]},
            [data_node],
            step="scenarios",
        )
        before[h] = sets.get(h)
        db.execute("UPDATE scenario_sets SET status = 'replaced' WHERE horizon = ? AND status = 'live'", (h,))
        db.execute(
            "INSERT OR REPLACE INTO scenario_sets VALUES (?,?,?,?,?,?,?,?)",
            (
                f"{h}-{today:%Y%m%d}",
                h,
                today.isoformat(),
                (today + timedelta(days=days)).isoformat(),
                json.dumps(spec),
                "live",
                node,
                None,
            ),
        )
        news += changes(before[h], scs, labels[h])
    db.commit()
    log.info("scenarios: %d lengths written, $%s", len(before), out["usage"].get("cost_usd"))
    return before, news


NEWS_MOVE = 0.10  # a scenario whose probability moved this much is news


def changes(before, now, label):
    """A length's scenarios whose probability when written moved NEWS_MOVE or more since the set before, in words. A
    scenario new or gone is not news: every writing revises some, and the page shows them all."""
    was = {s["id"]: s for s in (before or {}).get("scenarios", [])}
    return [
        f"Next {label}: {s['name']} {was[s['id']]['weight']:.0%} → {s['weight']:.0%}"
        for s in now
        if s["id"] in was and abs(s["weight"] - was[s["id"]]["weight"]) >= NEWS_MOVE
    ]

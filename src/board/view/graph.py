"""The board as a graph for the page's map."""

import math
from datetime import date

import pandas as pd

from ..core import expr, lineage
from ..model.decide import ACTS
from ..model.rules import rule_inputs
from .fmt import r

SOURCES = {"kalshi": "Kalshi markets", "own": "Your view"}


def reach(part, segs, lo, hi):
    """How much a signpost pushes the world between `lo` and `hi` days from today (where its news lands): of its parts
    (on the move and the spread, by factor), the largest, each stretch's counted for its share inside that span."""
    out = 0.0
    for by in part.values():
        for v in by.values():
            out = max(out, sum(abs(x) * max(0, min(b, hi) - max(a, lo)) / (b - a) for x, (a, b) in zip(v, segs)))
    return out


def nearest(days, horizons):
    """The length the world is read at nearest a number of days (in proportion)."""
    return min(horizons, key=lambda h: abs(math.log(max(days, 1) / horizons[h]["days"])))


def graph(db, cfg, df, questions, world, horizons, actions, you, today):
    """The board as a graph in its layers: the world, left to right (evidence: the data and the other sources;
    readings: the signposts; the world at each length it is read at, with the LLM's scenarios; the forecasts read off
    it), you on the far right (your style, your targets, your rules, your holdings), and the actions between, where
    the two meet (a:<id>, edges in from both sides: a lean from every length the decision step acts at, a guardrail's
    move from the length past its limit, a rule from what it reads). A signpost leads to each length its news lands in
    today (it pushes the stretches between the length before and that one; "push": how much); a forecast is read off
    the length nearest the days it has left."""
    edges, used, push = [], set(), {}
    parts = world["data"].parts if world else {}
    segs = world["data"].segs if world else []
    for sp in cfg["signposts"]["signpost"]:
        for s in expr.names(sp["value"]):
            edges.append((f"d:{s}", f"s:{sp['id']}"))
            used.add(s)
        before = 0
        for hk, h in sorted((horizons or {}).items(), key=lambda kv: kv[1]["days"]):
            x = reach(parts.get(sp["id"]) or {}, segs, before, h["days"])
            before = h["days"]
            if x >= 0.02:
                edges.append((f"s:{sp['id']}", f"h:{hk}"))
                push[f"s:{sp['id']}>h:{hk}"] = r(x, 3)
    for q in questions:
        if horizons:
            left = (date.fromisoformat(q["closes"]) - today).days if q.get("closes") else 365
            edges.append((f"h:{nearest(left, horizons)}", f"q:{q['id']}"))
        if q["kind"] == "kalshi":
            edges.append(("d:kalshi", f"q:{q['id']}"))
        if (q.get("sources") or {}).get("own"):
            edges.append(("d:own", f"q:{q['id']}"))
    held = (you or {}).get("mode") == "holdings"
    for a in actions or []:
        aid = f"a:{a['id']}"
        if a["kind"] == "rule":
            rule = next(r for r in cfg["rules"]["rule"] if r["id"] == a["rule"])
            series, sps, holdings = rule_inputs(cfg["rules"], rule)
            edges += [(f"d:{x}", aid) for x in series]
            used |= series
            edges += [(f"s:{x}", aid) for x in sps]
            edges.append(("y:rules", aid))
            if holdings and held:
                edges.append(("y:holdings", aid))
        elif a.get("horizon"):  # a guardrail's move: on the futures of the length past its limit
            edges.append((f"h:{a['horizon']}", aid))
            edges += [("y:policy", aid), ("y:style", aid)]
        elif a["kind"] == "lean":  # a lean: on the futures of every length that acts
            edges += [(f"h:{hk}", aid) for hk in (horizons or {}) if hk in ACTS]
            edges += [("y:policy", aid), ("y:style", aid)]
    data = []
    for sid in sorted(used):
        s = df[sid].dropna() if sid in df else None
        data.append(
            {
                "id": sid,
                "name": cfg["series"][sid]["name"],
                "kind": "series",
                "value": r(s.iloc[-1], 4) if s is not None and len(s) else None,
                "day": (s.index[-1] - pd.Timedelta(days=cfg["series"][sid].get("lag", 0))).date().isoformat()
                if s is not None and len(s)
                else None,
                "node": lineage.latest(db, f"raw:{sid}"),
            }
        )
    present = {a[2:] for a, _ in edges if a.startswith("d:")}
    data += [{"id": k, "name": v, "kind": "source"} for k, v in SOURCES.items() if k in present]
    me = []
    if you:
        me = [
            {"id": "style", "name": f"Risk level {you['profile']} ({you['label']})"},
            {"id": "policy", "name": "Your targets" if you["own"] else f"{you['label']} base allocation"},
            {"id": "rules", "name": f"Your rules ({len(you['rules'])})"},
        ]
        if held:
            me.append({"id": "holdings", "name": "Your holdings"})
    return {
        "data": data,
        "edges": sorted(set(edges)),
        "you": me,
        "push": push,  # "s:x>h:k": how much signpost x pushes the world over length k today
    }

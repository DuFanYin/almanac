"""The world for the day (model/world.py): each signpost with expectations, read on its dial today, its measured
moves (data/timescales.json, run.py timescales) judged against its expectations as the config has them now; their
pushes combined; the data's world, and the same without the signposts with a model upstream (one the LLM proposed,
one on a text it read: "proposed"), for the control without the LLM. Recorded as the day's world node. And the
outlook: each factor's quartiles over each length the world is read at, kept daily and resolved when the window is
over, so each side of the world is scored on what it said."""

import json

import numpy as np
import pandas as pd

from ..core import lineage, store
from ..model import factors as fx
from ..model import timescale
from ..model import world as wm


def signals(cfg, evaluated, report, today, clean=False):
    """([{"id", "group", "dial", "share", "sizes", "judged"}], the stretches): every signpost with expectations and
    measures (without those with a model upstream when clean); and one on a text reading, too new to be measured,
    with the LLM's expectations sized as the measured ones typically are (model/world.py typical, assumed) and read on
    its own scale, at PROPOSED_SHARE (the owner, 2026-10-31)."""
    first = cfg["factors"].get("history_from", "1971-03-01")
    segs = [tuple(s) for s in report["stretches"]]
    out, unmeasured = [], []
    for sp in cfg["signposts"]["signpost"]:
        r = report["signposts"].get(sp["id"])
        if sp["id"] not in evaluated or not sp.get("expect") or (clean and "proposed" in sp):
            continue
        values, _ = evaluated[sp["id"]]
        if not r:
            if sp.get("reading"):
                unmeasured.append((sp, values))
            continue
        judged = timescale.judge(sp["expect"], r["factors"])
        out.append(
            {
                "id": sp["id"],
                "group": sp.get("group"),
                "dial": wm.dial_today(values, sp["red"], first, today),
                "share": wm.PROPOSED_SHARE if "proposed" in sp else 1.0,
                "sizes": wm.sizes({**r, "judged": judged}, segs),
                "judged": judged,
            }
        )
    size = wm.typical([e for s in out for e in s["judged"]])
    for sp, values in unmeasured:
        sizes, judged = wm.assumed(sp["expect"], segs, size)
        out.append(
            {
                "id": sp["id"],
                "group": sp.get("group"),
                "dial": wm.dial_scale(values, sp["red"], today),
                "share": wm.PROPOSED_SHARE,
                "sizes": sizes,
                "judged": judged,
            }
        )
    return out, segs


def build(cfg, series, evaluated, today, report=None):
    """{"data": World, "clean": World} (each with .signals and .parts: each signpost's own push today); before any
    measures, both history as it was (no pushes)."""
    report = report if report is not None else store.timescales()
    if not report.get("signposts"):
        report = {"stretches": timescale.stretches(), "signposts": {}}
    out = {}
    for name, clean in (("data", False), ("clean", True)):
        sig, segs = signals(cfg, evaluated, report, today, clean)
        push, parts = wm.pushes(sig, segs)
        w = wm.World(cfg["factors"], series, segs, push)
        w.signals, w.parts = sig, parts
        out[name] = w
    return out


def record(db, worlds, sp_today, today):
    """The day's world as a node: each factor's pushes by stretch (on the move and on the spread) and each signpost's
    dial, with the signposts' values and the measures (run.py timescales) as parents."""
    w = worlds["data"]
    value = {kind: {f: [round(float(x), 4) for x in v] for f, v in by.items()} for kind, by in w.push.items()}
    meta = {
        "day": today.isoformat(),
        "stretches": [list(s) for s in w.segs],
        "dials": {s["id"]: None if s["dial"] is None else round(s["dial"], 3) for s in w.signals},
    }
    parents = [(sp_today.get(s["id"]) or {}).get("node") for s in w.signals] + [lineage.latest(db, "timescales")]
    return lineage.record(db, "world", "world", value, meta, parents, step="world")


def outlook(db, horizons, today):
    """Each factor's quartiles over each length the world is read at, kept for the day (the outlook table): history's
    edges and each side's probabilities."""
    rows = []
    for d in (horizons or {}).values():
        for f, x in d["factors"].items():
            probs = {s: [round(p, 6) for p in x[k]] for s, k in (("model", "final"), ("no_llm", "no_llm"))}
            probs |= {s: [round(p, 6) for p in x[s]] for s in ("data", "history")}
            rows.append((today.isoformat(), d["days"], f, json.dumps(x["edges"]), json.dumps(probs)))
    db.executemany("INSERT OR REPLACE INTO outlook (day, days, factor, edges, probs) VALUES (?,?,?,?,?)", rows)
    db.commit()


def resolve_outlook(db, fcfg, series, today):
    """Each outlook whose window is over and whose factor has a value dated on or after its end: the factor's value
    over the window and the quartile it fell in. How many."""
    n = 0
    for day, days, f, edges in db.execute(
        "SELECT day, days, factor, edges FROM outlook WHERE outcome IS NULL AND date(day, '+' || days || ' days') < ?",
        (today.isoformat(),),
    ).fetchall():
        spec = fcfg["factor"].get(f)
        arr = fx.arrays(series, spec["series"]) if spec else None
        t0 = pd.Timestamp(day).value
        t1 = t0 + days * fx.DAY
        if arr is None or arr[0][-1] < t1:
            continue
        v = float(fx.window(spec["measure"], arr, np.array([t0]), np.array([t1]))[0])
        if not np.isfinite(v):
            continue
        db.execute(
            "UPDATE outlook SET outcome = ?, value = ? WHERE day = ? AND days = ? AND factor = ?",
            (int(wm.bucket(np.array([v]), json.loads(edges))[0]), v, day, days, f),
        )
        n += 1
    db.commit()
    return n

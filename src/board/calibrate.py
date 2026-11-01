"""Beside pipeline.py, as they read the data and the config like a step but write files the day reads: how long
each signpost's news lasts, measured on history (run.py timescales, and the day on Saturdays: data/timescales.json,
for the world), and the decision step walked through history (run.py walk, and the first Saturday of a month:
data/walk.json, its sense and the lengths' weights, chosen on the walk's first half)."""

import hashlib
import json
import logging
from datetime import date

import numpy as np
import pandas as pd

from .core import lineage, store
from .data import observations
from .model import backtest, calibration, decide, prob, timescale
from .model import walk as wk
from .steps import world as world_step
from .steps.proposals import signposts as signpost_proposals

log = logging.getLogger("board")


def measured(cfg, series, evaluated, today):
    """Every signpost's measures (model/timescale.py measure) on the values and series given, its expectations judged:
    the timescales report ({"day", "code", "stretches", "signposts", "tested"})."""
    segs = timescale.stretches()
    out = {"day": today.isoformat(), "code": lineage.CODE, "stretches": segs, "signposts": {}}
    for sp in cfg["signposts"]["signpost"]:
        if sp["id"] not in evaluated:
            continue
        values, states = evaluated[sp["id"]]
        r = timescale.measure(sp, values, states, cfg["factors"], series, segs)
        if r:
            out["signposts"][sp["id"]] = {"name": sp["name"], "red_when": sp["red"], **r}
    pvalues = {
        (sid, f, how): c["p"]
        for sid, x in out["signposts"].items()
        for f, by in x["factors"].items()
        for how, c in by.items()
        if c
    }
    found = calibration.discoveries(pvalues)
    for (sid, f, how), _ in pvalues.items():
        out["signposts"][sid]["factors"][f][how]["passed"] = (sid, f, how) in found
    out["tested"] = {"curves": len(pvalues), "passed": len(found), "fdr": calibration.FDR}
    by_id = {sp["id"]: sp for sp in cfg["signposts"]["signpost"]}
    for sid, x in out["signposts"].items():  # each against what it is expected to say (plan section 13.8)
        expect = by_id[sid].get("expect", [])
        x["judged"] = timescale.judge(expect, x["factors"])
        x["unexplained"] = timescale.unexplained(expect, x["factors"])
    out["_found"] = sorted(found)
    return out


def timescales(today):
    """How long each signpost's news lasts (model/timescale.py), on every factor: its pace, and the curves of its red
    days and of its dial over the stretches of the next three years, each tested once over all its stretches and the
    run's tests together kept at a false discovery rate of FDR. Written to data/timescales.json (no account data) and
    returned; the world reads it (steps/world.py)."""
    cfg, _ = store.config()
    db = store.connect()
    signpost_proposals.with_candidates(db, cfg, ("trial", "active"))
    df, series, _ = observations.load(db, cfg["series"], today)
    evaluated, _ = prob.evaluate(cfg["signposts"]["signpost"], df)
    out = measured(cfg, series, evaluated, today)
    found, segs = out.pop("_found"), out["stretches"]
    for sid, x in out["signposts"].items():
        log.info("timescales %s: from %s, %s", sid, x["from"], x["pace"])
    store.write_json(store.DATA / "timescales.json", out)
    lineage.record(
        db,
        "timescales",
        "timescales",
        {f"{sid}/{f}/{how}": out["signposts"][sid]["factors"][f][how]["holds"] for sid, f, how in found},
        {"stretches": segs, "tested": out["tested"], "file": "data/timescales.json"},
        [lineage.latest(db, f"signpost:{sid}") for sid in out["signposts"]],
        step="timescales",
    )
    db.commit()
    db.close()
    return out


WALK_FROM = 1981  # the walk's first year: the signposts' measures need years of history before it
WALK_FILE = "walk.json"


def cut(x, day):
    """A Series up to the day before `day` (what was known then)."""
    return x[x.index < pd.Timestamp(day)]


def measured_by_year(cfg, series, evaluated, years):
    """{year: the timescales report measured on what was known on its 1 January}, each kept in data/walk/ under the
    signposts' and factors' config (measured again only when they change)."""
    key = hashlib.sha1(
        json.dumps([cfg["signposts"]["signpost"], cfg["factors"]["factor"]], sort_keys=True, default=str).encode()
    ).hexdigest()[:10]
    out = {}
    for y in years:
        path = store.DATA / "walk" / f"timescales-{y}-{key}.json"
        got = store.read_json(path)
        if got is None:
            day = date(y, 1, 1)
            s = {k: cut(v, day) for k, v in series.items()}
            e = {k: (cut(v, day), cut(st, day)) for k, (v, st) in evaluated.items()}
            got = measured(cfg, s, e, day)
            got.pop("_found", None)
            store.write_json(path, got)
            log.info("walk: measures as of %s, %d signposts", y, len(got["signposts"]))
        out[y] = got
    return out


def walk(today):
    """The decision step through history (model/walk.py): the data's world on every STEP_WEEKS from WALK_FROM as it
    could be built then, each length's futures the windows already over, the senses and lengths' weights tried, the
    one kept chosen on the first half. Written to data/walk.json (no account data: your targets and limits only) and
    returned."""
    cfg, _ = store.config()
    db = store.connect()
    df, series, _ = observations.load(db, cfg["series"], today)
    db.close()
    evaluated, _ = prob.evaluate(cfg["signposts"]["signpost"], df)
    rcfg = cfg["rules"]
    n = rcfg.get("profile") if rcfg.get("profile") is not None else 3
    pr = cfg["asset_classes"]["profile"][str(n)]
    limits = {k: pr[k] for k in ("worst_loss", "cash_min", "class_cap", "tilt", "band", "step")}
    policy = {}
    for c in (rcfg.get("categories") or {}).values():
        policy[c["class"]] = policy.get(c["class"], 0.0) + c.get("target", 0.0)
    policy = {k: w for k, w in policy.items() if w > 0} or dict(pr["policy"])
    year_limit = -limits["worst_loss"] * cfg["factors"].get("year_limit", 0.8)
    first = pd.Timestamp(cfg["factors"]["history_from"]) + pd.Timedelta(days=365)
    prices = {k: backtest.total_return(c, df) for k, c in cfg["asset_classes"]["class"].items()}
    prices = {k: p for k, p in prices.items() if p is not None and p.index[0] <= first}
    keys = [k for k in policy if k in prices]
    weekly = pd.DataFrame({k: prices[k].resample("W-MON").last() for k in keys}).ffill().dropna()
    bills = df["DTB3"].ffill().dropna() / 100 if "DTB3" in df else None
    hk = {h: x["days"] for h, x in cfg["factors"]["horizon"].items() if h in decide.ACTS}
    full = world_step.build(cfg, series, {}, today, {"stretches": timescale.stretches(), "signposts": {}})["clean"]
    grid = {h: full.weights(d)["starts"] for h, d in hk.items()}
    R = {h: wk.windows({k: prices[k] for k in keys}, grid[h], d) for h, d in hk.items()}
    days = pd.date_range(f"{WALK_FROM}-01-05", pd.Timestamp(today), freq=f"{7 * wk.STEP_WEEKS}D")
    reports = measured_by_year(cfg, series, evaluated, sorted({d.year for d in days}))
    worlds, cash = {}, {}
    for d in days:
        s = {k: cut(v, d) for k, v in series.items()}
        e = {k: (cut(v, d), cut(st, d)) for k, (v, st) in evaluated.items()}
        w = world_step.build(cfg, s, e, d.date(), reports[d.year])["clean"]
        worlds[d] = {}
        for h, L in hk.items():
            a = w.weights(L)
            rows = np.searchsorted(grid[h], a["starts"])
            ok = (rows < len(grid[h])) & (grid[h][np.minimum(rows, len(grid[h]) - 1)] == a["starts"])
            rows, ww = rows[ok], a["w"][ok]
            known = np.all([np.isfinite(R[h][k][rows]) for k in keys], axis=0)
            if known.sum() >= 52:
                worlds[d][h] = (rows[known], ww[known] / ww[known].sum())
        if bills is not None:
            b = cut(bills, d + pd.Timedelta(days=1))
            cash[d] = float(b.iloc[-1]) if len(b) else None
    worlds = {d: x for d, x in worlds.items() if x}
    base = wk.run(days, worlds, R, weekly, policy, limits, year_limit, cash, 0.0, {}, band_only=True)
    out_runs = {}
    for sense in wk.SENSES:
        for name, weights in wk.WEIGHTS.items():
            if sense == 0 and name != "even":
                continue
            out_runs[(sense, name)] = wk.run(days, worlds, R, weekly, policy, limits, year_limit, cash, sense, weights)

    def judged(x):
        a, b = wk.halves(x["value"])
        return {
            "all": wk.measures(x["value"], x["turnover"]),
            "first": wk.measures(a),
            "second": wk.measures(b),
            "turnover": x["turnover"],
            "leans": x.get("leans"),
            "guarded": x.get("guarded"),
        }

    base_j = judged(base)
    runs_j = {k: judged(x) for k, x in out_runs.items()}
    sense, name = wk.choose(runs_j, base_j)
    others = {}
    style = {k: w for k, w in pr["policy"].items() if k in keys}
    plain = {k: w for k, w in cfg["asset_classes"].get("baseline", {}).get("60/40", {}).items() if k in keys}
    for label, mix in ((f"Risk level {n}'s mix", style), ("60/40 stocks and bonds", plain)):
        if mix:
            x = wk.run(days, worlds, R, weekly, mix, limits, year_limit, cash, 0.0, {}, band_only=True)
            others[label] = judged(x)
    chosen = out_runs[(sense, name)]
    mid = days[0] + (days[-1] - days[0]) / 2
    out = {
        "day": today.isoformat(),
        "code": lineage.CODE,
        "from": str(days[0].date()),
        "middle": str(mid.date()),
        "classes": keys,
        "left_out": sorted(k for k in policy if k not in keys),
        "sense": sense,
        "weights": wk.WEIGHTS[name],
        "weights_name": name,
        "chosen": runs_j[(sense, name)],
        "base": base_j,
        "others": others,
        "grid": [{"sense": k[0], "weights": k[1], **v} for k, v in runs_j.items()],
        "values": {"chosen": monthly(chosen["value"]), "base": monthly(base["value"])},
        "rules": {
            "gain_min": wk.GAIN_MIN,
            "dd_slack": wk.DD_SLACK,
            "turn_most": wk.TURN_MOST,
            "turn_plus": wk.TURN_PLUS,
        },
    }
    store.write_json(store.DATA / WALK_FILE, out)
    log.info("walk: sense %s, weights %s", sense, name)
    return out


def monthly(v):
    m = v.groupby(v.index.to_period("M")).head(1)
    return {d.strftime("%Y-%m"): round(float(x), 4) for d, x in m.items()}

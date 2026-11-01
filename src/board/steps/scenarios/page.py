"""The scenarios as the page shows them: each with its probability the set before, the notes beside each length's
set, and each scenario as a view (what it rests on, what it means for you, what speaks for and against it, the leans
it drives)."""

import json
from datetime import timedelta

import pandas as pd


def notes_for_page(sets):
    """Each length's set as the page shows it beside its scenarios: {length: {"written", "readers", "review"
    {"model", "points", "asked"}, "coach", "rejected", "trigger", "why", "periods", "covered"}}."""
    return {
        h: {
            k: s.get(k)
            for k in ("written", "readers", "review", "coach", "rejected", "trigger", "why", "periods", "covered")
        }
        for h, s in sets.items()
    }


def for_page(db, scen, today):
    """The live scenarios as the page shows them, each with its probability when the set before was written
    ("was")."""
    was = {}
    for h, spec in db.execute(
        "SELECT horizon, spec FROM scenario_sets WHERE status = 'replaced' AND written >= ? ORDER BY written",
        ((today - timedelta(days=30)).isoformat(),),
    ):
        for x in json.loads(spec)["scenarios"]:
            was[f"{h}:{x['id']}"] = x["weight"]
    out = []
    for f in scen.get("live", []):
        out.append(
            {
                **{k: v for k, v in f.items() if k != "starts"},
                "status": "live",
                "was": was.get(f["id"]),
                "p": round(f["p"], 6),
                "p_data": round(f["p_data"], 6),
            }
        )
    return out


READ_DAYS = 28  # a text reading's direction: its mean over this many days


READ_LINE = 0.25  # past this, either way, a reading leans; within it, it says nothing


def readings_now(series, rcfg, series_cfg, today):
    """{reading: {"name", "value" (mean over the READ_DAYS to today), "sign" (-1, 0, 1), "bears_on"}} of the texts
    read into numbers (config/readings.toml); a reading with no text in those days says nothing now."""
    out = {}
    for sid, r in (rcfg or {}).get("reading", {}).items():
        s = series.get(sid)
        s = s.dropna() if s is not None else None
        s = s[s.index > pd.Timestamp(today) - pd.Timedelta(days=READ_DAYS)] if s is not None else None
        if s is None or s.empty:
            continue
        v = float(s.mean())
        sign = 1 if v >= READ_LINE else -1 if v <= -READ_LINE else 0
        out[sid] = {"name": series_cfg[sid]["name"], "value": v, "sign": sign, "bears_on": r.get("bears_on", {})}
    return out


def drivers(a):
    """The scenarios behind a lean (steps/decide.py): each one's part in the class's gap from the simulation, as
    shares of the largest parts. [{"id", "share"}]; none for another action."""
    parts = a.get("llm") or {}
    total = sum(abs(x) for x in parts.values())
    if a["kind"] != "lean" or not total:
        return []
    return [{"id": p, "share": round(abs(x) / total, 4)} for p, x in parts.items()]


def explained(futures, horizons, read_now, actions):
    """Each live scenario of the page with its view (in place): how often history's windows fall in its ranges
    ("history_like", and by factor), which way it takes each factor ("leans"), what it means for your targets
    ("impact": its share of the futures, your targets' mean return in it, its share of their worst 10%), the texts
    read into numbers for or against it, the opposite view at its length (the heaviest scenario taking stocks the
    other way), and the leans it drives. Each lean gets its "drivers"."""
    for a in actions:
        if a["kind"] == "lean":
            a["drivers"] = drivers(a)
    for f in futures:
        seen = ((horizons or {}).get(f["horizon"]) or {}).get("scenarios", {}).get(f["id"])
        if not seen:
            continue
        each = seen["history_like_by_factor"]
        rare = min(each, key=each.get) if each else None
        f.update(
            history_like=seen["history_like"],
            history_like_by_factor=each,
            # its rarest assumption: the factor whose range history's windows fall in least often
            rarest={"factor": rare, "share": each[rare], "range": f["ranges"].get(rare)} if rare else None,
            leans=seen["leans"],
            impact=seen.get("targets"),
            classes=seen.get("classes"),
            drives=[a["id"] for a in actions if any(x["id"] == f["id"] for x in a.get("drivers") or [])],
        )
        said = {}  # one line a reading and a verdict, with the factors it bears on that way
        for sid, r in read_now.items():
            for factor, way in r["bears_on"].items():
                lean = (seen["leans"].get(factor) or {}).get("lean", 0)
                if r["sign"] and lean:
                    says = "for" if r["sign"] * way == lean else "against"
                    line = said.setdefault(
                        (sid, says),
                        {"series": sid, "name": r["name"], "value": round(r["value"], 2), "says": says, "factors": []},
                    )
                    line["factors"].append(factor)
        f["readings_now"] = list(said.values())
    for f in futures:
        if "leans" not in f:
            continue
        way = (f["leans"].get("stocks") or {}).get("lean", 0)
        other = [
            g
            for g in futures
            if g["horizon"] == f["horizon"]
            and g is not f
            and way
            and (g.get("leans", {}).get("stocks") or {}).get("lean", 0) == -way
        ]
        f["opposite"] = (
            {"id": g["id"], "name": g["name"], "p": g.get("p")}
            if (g := max(other, key=lambda x: x.get("p") or 0, default=None))
            else None
        )
    return futures

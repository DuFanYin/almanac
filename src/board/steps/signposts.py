"""Each signpost's value and state today, kept daily with its lineage."""

import logging

import pandas as pd

from ..core import expr, lineage
from ..data import observations

log = logging.getLogger("board")


def signposts_today(db, sps, evaluated, today, series_cfg, series):
    """Today's value and state of each signpost, stored and recorded, with the day its data are as of (the
    oldest of its series' last values, as dated by their sources); {id: {...}}."""
    out = {}
    day = pd.Timestamp(today)
    for sp in sps:
        if sp["id"] not in evaluated:
            continue
        values, states = evaluated[sp["id"]]
        v = values.get(day)
        st = states.get(day, "grey")
        since = None
        if st != "grey":
            prev = states[states.index <= day]
            changed = prev[prev != st]
            since = (
                (changed.index[-1] + pd.Timedelta(days=1)).date().isoformat()
                if len(changed)
                else prev.index[0].date().isoformat()
            )
        v = None if v is None or pd.isna(v) else round(float(v), 6)
        parents = [n for x in expr.names(sp["value"]) for n in observations.raw_nodes(db, x, series_cfg)]
        parents.append(sp.get("node"))  # a signpost the LLM proposed: its proposal, so the LLM is upstream
        node = lineage.record(
            db,
            "signpost",
            f"signpost:{sp['id']}",
            {"value": v, "state": st},
            {"expr": sp["value"], "red": sp["red"], "yellow": sp.get("yellow"), "group": sp.get("group")},
            parents,
            step="signpost",
        )
        db.execute("INSERT OR REPLACE INTO signposts VALUES (?,?,?,?)", (today.isoformat(), sp["id"], v, st))
        recent = values[values.index > day - pd.Timedelta(days=120)]
        last = [series[n].index[-1] for n in expr.names(sp["value"]) if n in series and len(series[n])]
        out[sp["id"]] = {
            "value": v,
            "state": st,
            "since": since,
            "node": node,
            "as_of": min(last).date().isoformat() if last else None,
            "spark": [None if pd.isna(x) else round(float(x), 5) for x in recent.iloc[::2]],
        }
    return out

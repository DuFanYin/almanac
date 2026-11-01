"""Each set scored once its length is over: whether the path that happened is in each scenario,
against its final probability, the simulation's share and history's when written. It never changes a factor."""

import json
from datetime import date

import numpy as np
import pandas as pd

from ...model import factors as fx
from ...model import scenarios as sm

SCORE_WAIT = 120  # days past its end a set waits for every factor's value; then it is scored on those it has


def score(s, series):
    """Where each factor of a set's scenarios ended over its length: {factor: value or None}."""
    start, end = pd.Timestamp(s["written"]), pd.Timestamp(s["expires"])
    out = {}
    for k, f in s["factor_defs"].items():
        arr = fx.arrays(series, f["series"])
        if arr is None or arr[0][-1] < end.value:
            out[k] = None
            continue
        x = float(fx.window(f["measure"], arr, np.array([start.value]), np.array([end.value]))[0])
        out[k] = x if np.isfinite(x) else None
    return out


def scored(db, series, today):
    """Each set past its end and not yet scored, live or replaced: scored once every factor it names has a value for
    its end, or SCORE_WAIT days after on those it has: {"x", "parts" [each scenario: {"id", "name", "hit" (the path
    that happened in its ranges; None without a value), "weight" (its final probability when written), "share" (the
    simulation's then), "history", "times" (the readers' factor)}]}. How many."""
    n = 0
    for i, written, expires, spec in db.execute(
        "SELECT id, written, expires, spec FROM scenario_sets WHERE how IS NULL AND expires < ?", (today.isoformat(),)
    ).fetchall():
        s = {**json.loads(spec), "written": written, "expires": expires}
        x = score(s, series)
        late = (today - date.fromisoformat(expires)).days > SCORE_WAIT
        if None in x.values() and not late:
            continue
        hits = sm.happened(x, s["scenarios"])
        how = {
            "x": x,
            "parts": [
                {
                    "id": sc["id"],
                    "name": sc["name"],
                    "hit": hit,
                    "weight": sc.get("weight"),
                    "share": sc.get("base"),
                    "history": sc.get("history"),
                    "times": sc.get("times"),
                }
                for sc, hit in zip(s["scenarios"], hits)
            ],
        }
        db.execute("UPDATE scenario_sets SET how = ? WHERE id = ?", (json.dumps(how), i))
        n += 1
    db.commit()
    return n

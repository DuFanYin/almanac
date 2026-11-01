"""The scenario sets kept: the lengths that have scenarios of their own, how many each, the live
set of each length and when the last was written."""

import json
from datetime import date

LENGTHS = ("month", "short", "long", "longer")  # the lengths with scenarios of their own


MOST = {"month": 8, "short": 8, "long": 6, "longer": 6}  # scenarios at most: each is read four times, and history at


def lengths(fcfg):
    """{length: (label, days, most)} of the lengths that have scenarios, shortest first."""
    hz = fcfg["horizon"]
    return {
        h: (hz[h]["label"], hz[h]["days"], MOST[h])
        for h in sorted(LENGTHS, key=lambda h: hz.get(h, {}).get("days", 0))
        if h in hz
    }


def live(db):
    """{length: its live set}: {"id", "horizon", "written", "expires", "node", "scenarios" [{"id", "name", "story",
    "ranges", "times", "readings", "share", "base", "history", "weight", ...}], ...}."""
    out = {}
    for i, h, written, expires, spec, node in db.execute(
        "SELECT id, horizon, written, expires, spec, node FROM scenario_sets WHERE status = 'live' ORDER BY written"
    ):
        out[h] = {**json.loads(spec), "id": i, "horizon": h, "written": written, "expires": expires, "node": node}
    return out


def last_written(db, weekly=False):
    q = "SELECT max(written) FROM scenario_sets"
    if weekly:
        q += " WHERE json_extract(spec, '$.trigger') = 'weekly'"
    row = db.execute(q).fetchone()
    return date.fromisoformat(row[0]) if row and row[0] else None

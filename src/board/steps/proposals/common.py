"""How a proposal the owner votes on is kept (the proposals table: questions.py's, and changes.py's factors and rule
changes, told apart by spec "type"): with its checks and lineage. Signposts the LLM proposes (signposts.py) are
judged by the data instead, in their own table."""

import json
from datetime import date

from ...core import lineage


def due(db, kind, every, today, last=None):
    """Whether `every` days have passed since the last round of drafts of this kind (asked), whatever it kept; last:
    the newest draft kept, for the rounds from before they were marked."""
    row = db.execute("SELECT max(day) FROM rounds WHERE kind = ?", (kind,)).fetchone()
    days = [date.fromisoformat(x) for x in (row[0], last) if x]
    return not days or (today - max(days)).days >= every


def asked(db, kind, today):
    """A round of drafts of this kind asked today: the next is `every` days on, even when none was kept."""
    db.execute("INSERT OR IGNORE INTO rounds VALUES (?, ?)", (kind, today.isoformat()))


def keep(db, pid, spec, res, out, parents, today, **detail):
    """A checked draft kept: a node with its checks, the method and its cost; open for the owner when it passed, else
    failed."""
    node = lineage.record(
        db,
        "proposal",
        f"proposal:{pid}",
        spec,
        {"checks": res, "method": out["method"], "usage": out["usage"], **detail},
        parents,
        step="propose",
    )
    db.execute(
        "INSERT OR IGNORE INTO proposals VALUES (?,?,?,?,?,?,?,?)",
        (
            pid,
            today.isoformat(),
            json.dumps(spec),
            json.dumps(res, default=float),
            int(res["passed"]),
            "open" if res["passed"] else "failed",
            node,
            None,
        ),
    )
    lineage.version(db, "proposal", pid, today)

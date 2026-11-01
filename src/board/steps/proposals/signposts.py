"""Signposts proposed: by the LLM (drafted and checked on the data until 10-31), and on the texts it reads
(config/readings.toml, steps/readings.py: accumulating their readings). Paused since the world was built: a signpost
moves the world only through what it says about the factors (expect), measured against history; the LLM's will come
back drafting that, measured like the others and used at half their size at first (it has read the history they are
measured on). Until then each is joined to the signposts (with_candidates: read and shown, with no expectations it
moves nothing) and listed on the page. A proposal is a model-made node: its signpost's daily values have it upstream,
so they stay out of the world without the LLM."""

import json


def load(db, statuses=None):
    """[(row as a dict)], oldest first."""
    out = []
    for i, d, spec, chk, status, node, judged in db.execute(
        "SELECT id, day, spec, checks, status, node, judged FROM signpost_proposals ORDER BY day, id"
    ):
        if statuses and status not in statuses:
            continue
        out.append(
            {
                "id": i,
                "day": d,
                "spec": json.loads(spec),
                "checks": json.loads(chk),
                "status": status,
                "node": node,
                "judged": json.loads(judged or "{}"),
            }
        )
    return out


def candidates(db, statuses=("accumulating", "trial", "active")):
    """The proposed signposts of these statuses, in the config's form, with the day proposed and their node."""
    out = []
    for r in load(db, statuses):
        sp = r["spec"]
        out.append(
            {
                **{k: sp[k] for k in ("id", "name", "driver", "value", "red") if k in sp},
                **({"yellow": sp["yellow"]} if sp.get("yellow") else {}),
                **({"expect": sp["expect"]} if sp.get("expect") else {}),
                **({"reading": True} if sp.get("reading") else {}),
                "proposed": r["day"],
                "node": r["node"],
            }
        )
    return out


def with_candidates(db, cfg, statuses=("accumulating", "trial", "active")):
    """cfg with the proposed signposts of these statuses joined to its signposts (those joined before replaced)."""
    own = [sp for sp in cfg["signposts"]["signpost"] if "proposed" not in sp]
    cfg["signposts"]["signpost"] = own + candidates(db, statuses)
    return cfg


def rows(db):
    """Every proposal, newest first, for the page."""
    return [
        {
            "id": r["id"],
            "day": r["day"],
            "name": r["spec"].get("name"),
            "value": r["spec"].get("value"),
            "red": r["spec"].get("red"),
            "try_on": r["spec"]["try_on"],
            "why": r["spec"].get("why"),
            "checks": r["checks"].get("checks"),
            "status": r["status"],
            "judged": r["judged"],
            "node": r["node"],
            "reading": bool(r["spec"].get("reading")),  # on a text the LLM reads, not proposed by it
        }
        for r in reversed(load(db))
    ]

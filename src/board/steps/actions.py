"""Actions: what the page lists for you to do, from one decision step (steps/decide.py: your targets leaned by the
world within the guardrails, the trades on what you hold) and the preferences beside it:
  lean          both: a class your targets lean toward or away from, at the lengths today's world stands apart from
                usual (steps/decide.py)
  guard         both: a move the guardrails make, a length's bad case past its limit
  trade         you, on what you hold: past your band from the decided share, or the decided share moved
  new_money     where new money goes first
  rule          your rules left (Bitcoin's regular purchase, its cap, its pause, its review), written in advance
  concentration both: a single stock held in size, at its own worst 10% against its class on top, past your limit
Every action is a node with parents on both sides. Only suggestions: nothing is ordered."""

from ..core import lineage
from ..model import allocation, portfolio
from ..model.rules import rule_inputs


def record(db, a, parents):
    a["node"] = lineage.record(
        db,
        "action",
        f"action:{a['id']}",
        {k: v for k, v in a.items() if k not in ("node", "detail")},
        {"detail": a.get("detail")},
        parents,
        step="act",
    )
    return a


def actions(db, cfg, you, horizons, rule_rows, held, decided=None, names=()):
    """({"public": [action], "account": [action]}): the actions the page shows to anyone (on your targets) and those
    on your holdings (kept with the account data). decided: steps/decide.py run's. An action: {"id", "kind",
    "title", "from", "status", "detail", "node", ...}."""
    public = list((decided or {}).get("public") or [])
    account = list((decided or {}).get("account") or [])
    labels = {k: c["label"] for k, c in cfg["asset_classes"]["class"].items()}
    held_mix = {k: w for k, w in you["now"].items() if k in labels} if held and you and you.get("now") else None
    if held_mix and "long" in (horizons or {}):
        account += concentration(db, cfg, names, you, held_mix, horizons["long"], labels)
    for row in rule_rows:
        rule = next((r for r in cfg["rules"].get("rule", []) if r["id"] == row["id"]), None)
        if rule is None:
            continue
        series, sps, holdings = rule_inputs(cfg["rules"], rule)
        status = "fired" if row["trigger"] else "holds" if row["holds"] else "waiting"
        a = {
            "id": f"rule:{row['id']}",
            "kind": "rule",
            "title": row["title"],
            "from": (["world"] if series or sps else []) + ["you"],
            "status": status,
            "rule": row["id"],
            "trigger": row["trigger"],
            "reads": {"series": sorted(series), "signposts": sorted(sps), "holdings": holdings},
        }
        if status == "fired":  # a firing is the action: its node already has the inputs; the you node joins them
            record(db, a, [lineage.latest(db, f"trigger:{row['trigger']}"), you["node"]])
        public.append(a)
    return {"public": public, "account": account}


def concentration(db, cfg, names, you, held_mix, d, labels):
    """The single stocks held in size (steps/account.py names_today) at their own worst 10% against their class, on
    top of what you hold's bad case a year ahead (portfolio.with_names): where that is past your limit, a trim of each
    stock that adds to it, worst first."""
    gapped = [n for n in names if n.get("p10") is not None and n["p10"] < 0]
    now = allocation.stats(held_mix, d["futures"]) if gapped else None
    if not now:
        return []
    bad = portfolio.with_names({"long": now["bad"]}, gapped)["long"]
    limit = d["limit"]
    if bad >= limit:
        return []
    out = []
    for n in sorted(gapped, key=lambda n: n["share"] * n["p10"]):
        added = n["share"] * (1 + now["bad"]) * n["p10"]
        detail = (
            f"{n['symbol']} is {n['share']:.0%} of your total. In its worst 10% of one-year periods it did "
            f"{-n['p10']:.0%} worse than {labels.get(n['class'], n['class'])} "
            f"({n['windows']:.0f} independent periods). Counting that, the worst 10% of next year's futures for what "
            f"you hold average {bad:.0%}, {added:.0%} of it from {n['symbol']}, past your limit of {limit:.0%}."
        )
        a = {
            "id": f"name:{n['symbol']}",
            "kind": "concentration",
            "horizon": "long",
            "title": f"Trim {n['symbol']} by a step ({you['limits']['step']:.1%})",
            "from": ["world", "you"],
            "status": "suggested",
            "symbol": n["symbol"],
            "share": round(n["share"], 4),
            "p10": round(n["p10"], 4),
            "added": round(added, 4),
            "bad": round(bad, 4),
            "detail": detail,
        }
        out.append(record(db, a, [d["node"], you["node"], n["node"]]))
    return out

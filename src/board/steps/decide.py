"""The decision step of the day (model/decide.py): your targets leaned by today's world against usual, both ways
alike, within the guardrails; and, with holdings, the trades it suggests on what you hold. The data's part of the
lean takes the walk's sense and the lengths' weights (data/walk.json, calibrate.py walk: chosen on history; without a
walk, none); the LLM's part (its scenarios against the simulation) takes your sense for it ([decide] llm_sense,
LLM_SENSE without one: the LLM is in the world on purpose, and its view acts both ways). Every lean and trade says
why: at which lengths the class stands apart from usual, how much of that the data and the LLM's scenarios make (the
scenarios with the largest part), and what the data moved there. Recorded as one node with the world, each length's
futures and you as parents; each suggestion its own node under it. Only suggestions: nothing is ordered."""

from datetime import date

import numpy as np

from ..core import lineage, store
from ..model import allocation
from ..model import decide as dm

LLM_PART = 0.003  # a scenario whose part in a class's gap is at least this is named in the why
LLM_SENSE = 1.0  # the LLM's part of a lean, without yours: 0.05 standard deviations lean a class its whole tilt (10-31)
SAYS = 0.0005  # a lean below this is 0.0% as the suggestion writes it: no suggestion


def params(cfg):
    """(sense, the lengths' weights, where they come from, the LLM's sense): the walk's (data/walk.json); (0, even)
    without one. Your own sense ([decide] sense in local/rules.toml) wins over the walk's: a preference you may set
    knowing the walk's record, never the LLM. The LLM's sense: yours ([decide] llm_sense), LLM_SENSE without one."""
    w = store.read_json(store.DATA / "walk.json") or {}
    even = {h: 1 / len(dm.ACTS) for h in dm.ACTS}
    mine = cfg["rules"].get("decide") or {}
    llm_sense = float(mine.get("llm_sense", LLM_SENSE))
    if mine.get("sense") is not None:
        return float(mine["sense"]), w.get("weights") or even, "yours", llm_sense
    walked = f"the walk of {w['day']}" if w.get("day") else None
    return float(w.get("sense", 0.0)), w.get("weights") or even, walked, llm_sense


def llm_parts(d, k, futures, h):
    """{scenario: its part in class k's gap at one length: what the LLM's weights moved to its windows from the
    simulation's, times the class's return there} for the scenarios with a part of LLM_PART or more, largest first."""
    fut, data = d["futures"], d.get("data_futures")
    if data is None or fut.start is None or k not in fut.returns:
        return {}
    moved = fut.w - data.w
    out = {}
    for f in futures:
        if f.get("horizon") != h:  # a length's own scenarios: another length's share its start days
            continue
        m = np.isin(fut.start, f["starts"])
        x = float(moved[m] @ fut.returns[k][m])
        if abs(x) >= LLM_PART:
            out[f["id"]] = round(x, 4)
    return dict(sorted(out.items(), key=lambda kv: -abs(kv[1])))


def moved_factors(world, days, most=3):
    """The factors today's data moved most at a length, with the signposts pushing each: [{"factor", "label",
    "moved" (in standard deviations), "by" [signpost]}]."""
    if world is None:
        return []
    a = world.weights(days)
    fcfg = world.fcfg["factor"]
    rows = sorted(zip(a["names"], a["moved"]), key=lambda x: -abs(x[1]))[:most]
    out = []
    for f, m in rows:
        if abs(m) < 0.02:
            continue
        by = [
            sid
            for sid, p in (getattr(world, "parts", None) or {}).items()
            if np.any(np.asarray(p.get("move", {}).get(f, [])) * np.sign(m) > 0)
        ]
        out.append({"factor": f, "label": fcfg[f]["label"], "moved": round(float(m), 3), "by": by})
    return out


def why_lean(k, change, got, horizons, labels, futures, names):
    """A lean's reason: at which lengths the class stands apart from usual (its expected return against history's,
    after the noise), how much of it the data and the LLM's scenarios make, the scenarios with the largest part."""
    parts, lines = {}, []
    for h, g in got["gaps"].items():
        x = g.get(k)
        if not x or not x["shrunk"]:
            continue
        lines.append(
            f"{horizons[h]['label']}: {x['gap']:+.1%} against usual (data {x['data']:+.1%}, the LLM's scenarios "
            f"{x['llm']:+.1%}; chance alone ±{x['se']:.1%})"
        )
        for fid, v in llm_parts(horizons[h], k, futures, h).items():
            parts[fid] = parts.get(fid, 0.0) + v
    top = sorted(parts.items(), key=lambda kv: -abs(kv[1]))[:3]
    named = "; ".join(f"{names.get(f, f)} {v:+.1%}" for f, v in top)
    return (
        f"{'Raise' if change > 0 else 'Lower'} {labels.get(k, k)} by {abs(change):.1%}: in today's world it does "
        f"{'better' if change > 0 else 'worse'} than usual. "
        + "; ".join(lines)
        + "."
        + (f" The LLM's scenarios behind it: {named}." if named else "")
    ), top


def run(db, cfg, you, horizons, held, futures, today, world=None):
    """{"decision" (for the page: the decided mix, the leans with their reasons, the guardrails, each length's bad
    case before and after, the walk's parameters), "public" [suggestion on your targets], "account" [trade on what
    you hold], "node"}; None without the futures or you."""
    if not you or not horizons:
        return None
    sense, weights, walked, llm_sense = params(cfg)
    labels = {k: c["label"] for k, c in cfg["asset_classes"]["class"].items()}
    names = {f["id"]: f.get("name", f["id"]) for f in futures}
    got = dm.decide(you["policy"], horizons, you["limits"], sense, weights, llm_sense)
    world_d = (world or {}).get("data") if isinstance(world, dict) else world
    node = lineage.record(
        db,
        "decision",
        "decision",
        {k: round(x, 4) for k, x in sorted(got["mix"].items())},
        {
            "lean": {k: round(x, 4) for k, x in got["lean"].items()},
            "guard": got["guard"],
            "sense": sense,
            "llm_sense": llm_sense,
            "weights": weights,
            "walk": walked,
        },
        [you["node"], *(d["node"] for h, d in horizons.items() if h in dm.ACTS)],
        step="decide",
    )
    public = []
    for k, change in sorted(got["lean"].items(), key=lambda kv: -abs(kv[1])):
        if abs(change) < SAYS:  # 0.0% as written: it says nothing
            continue
        detail, top = why_lean(k, change, got, horizons, labels, futures, names)
        public.append(
            record(
                db,
                {
                    "id": f"lean:{k}",
                    "kind": "lean",
                    "title": f"{'Raise' if change > 0 else 'Lower'} {labels.get(k, k)} by {abs(change):.1%}",
                    "from": ["world", "you"],
                    "status": "suggested",
                    "class": k,
                    "size": round(change, 4),
                    "llm": {f: round(v, 4) for f, v in top},
                    "detail": detail,
                },
                [node],
            )
        )
    for m in got["guard"]:
        h = m["horizon"]
        b = got["bad"][h]
        public.append(
            record(
                db,
                {
                    "id": f"guard:{m['from']}:{m['to']}",
                    "kind": "guard",
                    "horizon": h,
                    "title": f"Move {m['size']:.1%} from {labels.get(m['from'], m['from'])} into "
                    f"{labels.get(m['to'], m['to'])}",
                    "from": ["world", "you"],
                    "status": "suggested",
                    "class": m["from"],
                    "to": m["to"],
                    "size": round(-m["size"], 4),
                    "bad": round(b["before"], 4) if b["before"] is not None else None,
                    "after": round(b["after"], 4) if b["after"] is not None else None,
                    "limit": round(b["limit"], 4) if b["limit"] is not None else None,
                    "detail": f"Over the next {horizons[h]['label']} the worst 10% of futures for the mix would "
                    f"average {b['before']:.1%}, past your limit of {b['limit']:.1%}; with this move "
                    f"{b['after']:.1%}. The limit wins over the leans.",
                },
                [node],
            )
        )
    cats = {k: c for k, c in (cfg["rules"].get("categories") or {}).items() if c.get("target")} if you["own"] else {}
    decided = dm.by_category(got["mix"], cats, you["policy"])
    account = trades(db, you, cats, decided, held, today, node) if held and cats else []
    if held and you.get("now"):
        account += held_guard(db, you, cats, account, held, horizons, labels, node)
    after = {}
    for h, d in horizons.items():
        s = allocation.stats(got["mix"], d["futures"]) if d.get("futures") is not None else None
        if s:
            after[h] = {k: round(s[k], 4) for k in ("expected", "median", "p10", "p90", "bad", "good")}
    decision = {
        "mix": {k: round(x, 4) for k, x in sorted(got["mix"].items())},
        "categories": {k: round(x, 4) for k, x in decided.items()},
        "targets": {k: round(x, 4) for k, x in sorted(you["policy"].items())},
        "lean": {k: round(x, 4) for k, x in got["lean"].items()},
        "scores": {k: round(x, 3) for k, x in got["scores"].items()},
        "parts": {k: {p: round(v, 4) for p, v in x.items()} for k, x in got["parts"].items()},
        "gaps": {h: {k: {f: round(v, 4) for f, v in x.items()} for k, x in g.items()} for h, g in got["gaps"].items()},
        "guard": [{**m, "size": round(m["size"], 4)} for m in got["guard"]],
        "bad": {
            h: {f: (round(v, 4) if isinstance(v, float) else v) for f, v in b.items()} for h, b in got["bad"].items()
        },
        "after": after,
        "sides": {h: sides(you["policy"], d) for h, d in horizons.items() if d.get("history_futures") is not None},
        "moved": {h: moved_factors(world_d, d["days"]) for h, d in horizons.items() if h in dm.ACTS},
        "sense": sense,
        "llm_sense": llm_sense,
        "weights": weights,
        "walk": walked,
        "node": node,
    }
    return {"decision": decision, "public": public, "account": account, "node": node}


MATTERS = 0.05  # a side differs from usual only past this many of history's standard deviations at that length


def sides(mix, d):
    """Both sides of one length for your targets, against usual (history unweighted, on the same windows): the
    expected return, the good case and the bad case, each with today's world and as usual, and the word: "better" or
    "worse" (the expected return past the line), "wider" (the bad case worse and the good case better, each past it),
    "narrower", else "usual". The line: the larger of the noise a reweighting this large makes by chance on the
    expected return and MATTERS of history's spread."""
    fut, hist = d["futures"], d["history_futures"]
    now, usual = allocation.stats(mix, fut), allocation.stats(mix, hist)
    r, _ = allocation.returns(mix, fut)
    if not now or not usual or r is None:
        return None
    g = dm.gap(type(fut)(fut.X, {"mix": r}, fut.w), type(hist)(hist.X, {"mix": r}, hist.w), days=d["days"])["mix"]
    line = max(g["se"], MATTERS * g["sd"])
    e, good, bad = (now[k] - usual[k] for k in ("expected", "good", "bad"))
    word = (
        "better"
        if e > line
        else "worse"
        if e < -line
        else "wider"
        if bad < -line and good > line
        else "narrower"
        if bad > line and good < -line
        else "usual"
    )
    keys = ("expected", "good", "bad")
    return {
        "now": {k: round(now[k], 4) for k in keys},
        "usual": {k: round(usual[k], 4) for k in keys},
        "line": round(line, 4),
        "word": word,
    }


def record(db, a, parents):
    a["node"] = lineage.record(
        db,
        "action",
        f"action:{a['id']}",
        {k: v for k, v in a.items() if k not in ("node", "detail")},
        {"detail": a.get("detail")},
        parents,
        step="decide",
    )
    return a


def after_trades(you, cats, account, held):
    """What you hold by class (of everything, Bitcoin held directly too) after the suggested trades (shares of the
    account)."""
    h = held["h"]
    scale = (h["nav"] or 0) / (h["total"] or 1)
    out = dict(you["now"])
    for t in account:
        if t.get("kind") == "trade" and t.get("category") in cats:
            k = cats[t["category"]]["class"]
            out[k] = out.get(k, 0.0) + t["size"] * scale
            out[dm.CASH] = out.get(dm.CASH, 0.0) - t["size"] * scale
    return out


def held_guard(db, you, cats, account, held, horizons, labels, node):
    """The guardrails on what you hold after the trades (Bitcoin and all): where a length's bad case is past its limit,
    a step (at most your step) out of each class whose move lifts it most, into the shelters in your targets the
    moves went to, and how much in all would bring it within (model/decide.py guard): one suggestion a class."""
    mix = after_trades(you, cats, account, held)
    lengths = {h: (d["futures"], d.get("limit")) for h, d in horizons.items() if h in dm.ACTS and d.get("futures")}
    _, moves, seen = dm.guard(mix, lengths, you["policy"])
    by = {}
    for m in moves:  # a class moved into two shelters is one move out of it
        x = by.setdefault(m["from"], {"from": m["from"], "horizon": m["horizon"], "size": 0.0, "to": {}})
        x["size"] += m["size"]
        x["to"][m["to"]] = x["to"].get(m["to"], 0.0) + m["size"]
    out = []
    for m in by.values():
        b, step = seen[m["horizon"]], min(m["size"], you["limits"]["step"])
        lab = horizons[m["horizon"]]["label"]
        into = " and ".join(labels.get(k, k) for k in sorted(m["to"], key=lambda k: -m["to"][k]))
        out.append(
            record(
                db,
                {
                    "id": f"guard-held:{m['from']}",
                    "kind": "guard",
                    "horizon": m["horizon"],
                    "title": f"Move {step:.1%} of everything from {labels.get(m['from'], m['from'])} into {into}",
                    "from": ["world", "you"],
                    "status": "suggested",
                    "class": m["from"],
                    "to": max(m["to"], key=m["to"].get),
                    "into": {k: round(x / m["size"], 4) for k, x in m["to"].items()},
                    "size": round(-step, 4),
                    "all": round(m["size"], 4),
                    "bad": round(b["before"], 4),
                    "after": round(b["after"], 4),
                    "limit": round(b["limit"], 4),
                    "detail": f"Over the next {lab} the worst 10% of futures for what you hold (after the trades "
                    f"above) average {b['before']:.1%}, past your limit of {b['limit']:.1%}. About {m['size']:.1%} of "
                    f"everything moved this way would bring it within ({b['after']:.1%}); a step at a time.",
                },
                [node, held["node"]],
            )
        )
    return out


def trades(db, you, cats, decided, held, today, node):
    """The trades on what you hold (model/decide.py trades), by category against the decided shares (decided: the
    mix split by each category's share of its class's target); a category with a trade done in the last PACE days the
    same way waits (the trades table keeps each suggestion and the share held then: done once the share has moved
    half of it); a suggestion not yet done stands. And where new money goes first."""
    shares = held["shares"]
    anchor = {
        c: x
        for c, x in db.execute(
            "SELECT category, share FROM anchors WHERE category IN (%s)" % ",".join("?" * len(decided)), list(decided)
        )
    }
    cash = next((c for c, x in cats.items() if x["class"] == dm.CASH), dm.CASH)
    todo, anchor_now = dm.trades(decided, shares, anchor, you["limits"], cash)
    day = today.isoformat()
    db.executemany("INSERT OR REPLACE INTO anchors VALUES (?,?,?)", [(c, x, day) for c, x in anchor_now.items()])
    out = []
    for c, t in todo.items():
        last = db.execute(
            "SELECT id, size, day, held, node FROM trades WHERE category = ? ORDER BY day DESC LIMIT 1", (c,)
        ).fetchone()
        status, tid, since = "suggested", f"trade:{c}:{today.isoformat()}", today.isoformat()
        if last and np.sign(last[1]) == np.sign(t["size"]):
            done = (shares.get(c, 0.0) - last[3]) * np.sign(last[1]) >= abs(last[1]) / 2
            age = (today - date.fromisoformat(last[2])).days
            if done and age < dm.PACE:
                continue  # a step taken: the next one the same way after PACE days
            if not done:
                status, tid, since = "standing", last[0], last[2]
        label = cats[c].get("label", c)
        why = (
            f"you hold {shares.get(c, 0.0):.1%}, the decided share is {decided[c]:.1%}: more than your band "
            f"(±{you['limits']['band']:.1%}) away"
            if t["why"] == "band"
            else f"the decided share moved to {decided[c]:.1%} (your target {cats[c]['target']:.1%}); you hold "
            f"{shares.get(c, 0.0):.1%}"
        )
        a = {
            "id": tid,
            "kind": "trade",
            "title": f"{'Buy' if t['size'] > 0 else 'Sell'} {abs(t['size']):.1%} of {label}",
            "from": ["world", "you"] if t["why"] == "view" else ["you"],
            "status": status,
            "category": c,
            "size": round(t["size"], 4),
            "since": since,
            "decided": round(decided[c], 4),
            "held": round(shares.get(c, 0.0), 4),
            "why": t["why"],
            "detail": f"{why}. One step at a time ({you['limits']['step']:.1%}); "
            + ("into cash." if t["size"] < 0 else "from cash."),
        }
        if status == "suggested":
            record(db, a, [node, held["node"]])
            db.execute(
                "INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?)",
                (tid, c, t["size"], today.isoformat(), shares.get(c, 0.0), a["node"]),
            )
        else:
            a["node"] = last[4]
        out.append(a)
    first = dm.new_money(decided, shares, cash)
    if first:
        out.append(
            {
                "id": f"new-money:{first}",
                "kind": "new_money",
                "title": f"New money: {cats[first].get('label', first)} first",
                "from": ["world", "you"],
                "status": "note",
                "category": first,
                "detail": f"It is the furthest under its decided share ({decided[first]:.1%}; you hold "
                f"{shares.get(first, 0.0):.1%}).",
                "node": node,
            }
        )
    db.commit()
    return out

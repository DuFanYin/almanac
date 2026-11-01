"""The board's texts: the weekly brief and the case against the holdings, every number one of the data's."""

import json
import logging
import re
from datetime import timedelta

from ..core import lineage, store
from ..data import llm

log = logging.getLogger("board")


STATES = {"red": "red", "yellow": "yellow", "green": "green"}


def cited(text, data):
    """{id: "name: value"} of the data items a text cites ([D4]), for the page to show beside each number."""
    by_id = {d["id"]: d for d in data}
    ids = dict.fromkeys(re.findall(r"\[(D\d+)\]", text))
    return {i: f"{by_id[i]['name']}: {by_id[i]['value']}" for i in ids if i in by_id}


def names(cfg):
    """(question, signpost, rule) namers for the texts: a question's title and its outcomes' labels, a signpost's name,
    a rule's title, in place of their ids (the texts are for a person)."""
    cfg = cfg or {}
    qs = cfg.get("questions") or {}

    def question(qid):
        base = qid if qid in qs else qid.rsplit("-", 1)[0]
        q = qs.get(base) if isinstance(qs.get(base), dict) else {}
        title = q.get("title", qid) + (f" ({qid[len(base) + 1 :]})" if base != qid and q else "")
        return title, {b["id"]: b.get("label", b["id"]) for b in q.get("buckets") or [] if isinstance(b, dict)}

    signposts = {x["id"]: x.get("name", x["id"]) for x in (cfg.get("signposts") or {}).get("signpost") or []}
    rules = {x["id"]: x.get("title", x["id"]) for x in (cfg.get("rules") or {}).get("rule") or []}
    return question, (lambda i: signposts.get(i, i)), (lambda i: rules.get(i, i))


def story(f):
    """A scenario's story without the data ids it was written with ([D4]: an item of its own prompt, not this one's)."""
    return re.sub(r"\s*\[D\d+\]", "", f.get("story") or "")


def futures_lines(futures, data, today, labels):
    """The LLM's scenarios for a text: each length's, each with its probability (the world's), the simulation's share
    and the readers' factor as data items, and what it means for the targets (better or worse than usual, its part of
    their worst 10%). Lines; `data` grows by their items."""
    lines = []
    for f in futures:
        span = labels.get(f["horizon"], f["horizon"])
        data.append({"id": f"D{len(data) + 1}", "name": f"probability of the scenario {f['name']}", "value": f["p"]})
        line = f"- {f['name']}, next {span} [D{len(data)}]"
        data.append({"id": f"D{len(data) + 1}", "name": f"the simulation's share of {f['name']}", "value": f["p_data"]})
        line += f" (the simulation: [D{len(data)}]"
        data.append({"id": f"D{len(data) + 1}", "name": f"the readers' factor on {f['name']}", "value": f["times"]})
        line += f", times the readers' factor [D{len(data)}]"
        line += f"): {story(f)}"
        if f.get("impact"):
            side = f["impact"].get("side")
            if side:
                line += f" {side.capitalize()} than usual for the targets."
            data.append(
                {
                    "id": f"D{len(data) + 1}",
                    "name": f"share of the owner's targets' worst 10% at this length from {f['name']}",
                    "value": round(f["impact"]["worst"], 3),
                }
            )
            line += f" It makes [D{len(data)}] of the targets' worst 10%."
        lines.append(line)
    return lines


def brief(db, today, data, probs, futures=(), labels=None, cfg=None):
    """The weekly brief, in the database; returns the data it was given (the snapshot and the week's
    probabilities). futures: the LLM's scenarios (steps/scenarios/page.py for_page); labels: {horizon: label}; cfg:
    the board's config, for the questions', signposts' and rules' names."""
    data = list(data)
    question, signpost, rule = names(cfg)
    week_ago = (today - timedelta(days=7)).isoformat()
    changes = []
    for qid, v in probs.items():
        old = dict(  # the last recorded day a week ago or before
            db.execute(
                "SELECT bucket, p FROM probs WHERE question = ? AND source = 'model' AND day = (SELECT max(day) "
                "FROM probs WHERE question = ? AND source = 'model' AND day <= ?)",
                (qid, qid, week_ago),
            ).fetchall()
        )
        title, outcomes = question(qid)
        for k, p in v["model"].items():
            what = f"{title}: {outcomes.get(k, k)}"
            if k in old and abs(p - old[k]) >= 0.03:
                before = round(old[k], 3)
                data.append({"id": f"D{len(data) + 1}", "name": f"probability a week ago of {what}", "value": before})
                changes.append(f"{what}: from [D{len(data)}] to [D{len(data) + 1}]")
            data.append({"id": f"D{len(data) + 1}", "name": f"probability now of {what}", "value": round(p, 3)})
    flips = db.execute(
        "SELECT s.id, s.state, o.state FROM signposts s JOIN signposts o ON o.id = s.id AND o.day = ? "
        "WHERE s.day = ? AND s.state != o.state",
        (week_ago, today.isoformat()),
    ).fetchall()
    changes += [f"signpost {signpost(i)}: from {STATES.get(b, b)} to {STATES.get(a, a)}" for i, a, b in flips]
    fired = db.execute("SELECT rule, period FROM triggers WHERE day > ?", (week_ago,)).fetchall()
    changes += [f"rule fired: {rule(r)}" for r, _ in fired]
    named = futures_lines(futures, data, today, labels or {})
    out = llm.write(
        llm.TEXTS["brief"],
        data,
        today=today.isoformat(),
        changes="\n".join(changes) or "nothing moved by 3 points or more",
        futures="\n".join(named) or "none",
    )
    node = lineage.record(
        db,
        "brief",
        f"brief:{today}",
        out["output"],
        {
            "valid": out["valid"],
            "prompt_hash": out["prompt_hash"],
            "attempts": out["attempts"],
            "usage": out.get("usage"),
            "data": data,
        },
        [lineage.latest(db, f"prob:{q}") for q in probs],
        step="llm_brief",
    )
    db.execute(
        "INSERT OR REPLACE INTO texts VALUES (?,?,?,?)",
        (
            today.isoformat(),
            "brief",
            out["output"].get("text", ""),
            json.dumps({"valid": out["valid"], "node": node, "cites": cited(out["output"].get("text", ""), data)}),
        ),
    )
    return data


def redteam(db, today, data, account, futures):
    """The case against the current holdings (account side: data/account/redteam.json). Every number it may use
    is in its data, the holdings' shares and the holdings on each horizon's futures included, so the check can hold.
    futures: the LLM's scenarios (steps/scenarios/page.py for_page). Whether it passed its check is kept in the
    texts table, without the text (it names account shares)."""
    data = list(data)

    def cite(name, value):
        data.append({"id": f"D{len(data) + 1}", "name": name, "value": round(value, 4)})
        return f"[{data[-1]['id']}]"

    labels = {c["id"]: c["label"] for c in account["categories"]} | {"btc": "Bitcoin, of everything"}
    weights = "\n".join(
        f"{labels.get(k, k)}: {cite(f'share of the holdings: {labels.get(k, k)}', account['shares'][k])}"
        for k in labels
        if account["shares"].get(k)
    )
    named = {f["id"]: f for f in futures}
    lines = []
    for d in account["against"]["horizons"].values():
        st, span = d["holding"], d["label"]
        if not st:
            continue
        lines.append(
            f"Next {span}: expected {cite(f'the holdings expected return over {span}', st['expected'])}, the mean "
            f"of the worst 10% {cite(f'the holdings bad case over {span}', st['bad'])}, the limit "
            f"{cite(f'the limit over {span}', d['limit'])}. The worst 10% comes from:"
        )
        for part, share in list((d["worst_parts"] or {}).items())[:3]:
            f = named.get(part)
            what = f"{f['name']}: {story(f)}" if f else "history's own periods"
            lines.append(f"- {what} ({cite(f'share of the worst 10% over {span}: {part}', share)})")
    scen = "\n".join(lines)
    red = llm.write(llm.TEXTS["redteam"], data, today=today.isoformat(), weights=weights, scenarios=scen)
    points = " ".join(str(x) for x in red["output"].get("points", []))
    store.write_json(store.ACCOUNT / "redteam.json", {"day": today.isoformat(), **red, "cites": cited(points, data)})
    db.execute(
        "INSERT OR REPLACE INTO texts VALUES (?,?,?,?)",
        (today.isoformat(), "redteam", "", json.dumps({"valid": red.get("valid")})),
    )

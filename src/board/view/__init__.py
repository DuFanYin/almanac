"""The page's data, written after each run: data/scenarios.json (no account numbers: the forecasts, signposts, rules,
each horizon's futures, the actions on your targets) and, with holdings, data/account/scenarios-account.json
(holdings, targets, amounts, the holdings on each horizon's futures). Each part of the page is made in its own module
here; write puts them together. The page's server adds the owner's decisions and judgments as they are when the page
asks."""

import json
import time

from ..core import expr, lineage, store
from ..model.questions import key as key_by
from .data import indicators, stale_series
from .fmt import r, rd
from .graph import graph
from .questions import question_rows
from .record import audit, evals, scenarios_record, text_checks
from .signposts import said, signpost_rows
from .world import world_view


def write(
    db,
    cfg,
    today,
    df,
    *,
    chash,
    sp_today,
    sp_errors,
    probs,
    rule_rows,
    account,
    errors,
    score,
    resolved,
    calendar,
    lines,
    behaved,
    proposed,
    you=None,
    actions=None,
    backtest=None,
    horizons=None,
    scenarios=None,
    scenario_sets=None,
    decision=None,
    signposts_proposed=None,
    readings=None,
    system=None,
    world=None,
):
    """The page's data (data/scenarios.json) and, with holdings, the account part (data/account/)."""
    names = {
        "series": {sid: c["name"] for sid, c in cfg["series"].items()},
        "signposts": {sp["name"]: f"signpost:{sp['id']}" for sp in cfg["signposts"]["signpost"]},
    }
    # a firing stands while its period lasts and its rule still holds today: one whose condition has cleared (Bitcoin
    # back under its cap) is no longer something to do
    live = {r["trigger"] for r in rule_rows if r.get("holds") and r.get("trigger")}
    triggers = [
        {"id": i, "rule": rl, "period": p, "day": d, "standing": i in live}
        for i, rl, p, d in db.execute("SELECT id, rule, period, day FROM triggers ORDER BY day DESC LIMIT 60")
    ]
    brief = db.execute("SELECT day, body, meta FROM texts WHERE kind = 'brief' ORDER BY day DESC LIMIT 1").fetchone()
    view = {
        "updated": int(time.time()),
        "day": today.isoformat(),
        "code": lineage.CODE,
        "config": chash,
        "errors": errors,
        "stale": stale_series(df, cfg, today),
        "questions": (qrows := question_rows(db, probs, today, cfg["questions"], names)),
        "graph": graph(db, cfg, df, [q for q in qrows if not q["resolved"]], world, horizons, actions, you, today),
        "signposts": signpost_rows(cfg, sp_today, sp_errors, world),
        # the world: today's pushes, each factor's shift and spread at each length, the evidence kept
        "world": world_view(world, horizons),
        "indicators": indicators(df, cfg, today),
        "rules": [{**x, "readings": readings_said(x.get("readings"), names["series"], cfg)} for x in rule_rows],
        "triggers": triggers,
        "calendar": calendar,
        "scores": score,
        "resolved_today": resolved,
        "brief": {"day": brief[0], "text": brief[1], **json.loads(brief[2])} if brief else None,
        "class_labels": {k: c["label"] for k, c in cfg["asset_classes"]["class"].items()},
        # the names the page shows in place of ids: factors, series, signposts, questions (by template and by each
        # one registered)
        "labels": {
            "factors": {k: f["label"] for k, f in cfg["factors"]["factor"].items()},
            "series": names["series"],
            "signposts": {sp["id"]: sp["name"] for sp in cfg["signposts"]["signpost"]},
            "questions": {k: q["title"] for k, q in cfg["questions"].items() if isinstance(q, dict) and "title" in q}
            | {q["id"]: q["title"] for q in qrows},
        },
        "evals": evals(),
        "versions": lineage.versions(db),
        "behaviour": behaved,
        "rule_lines": {
            k: {
                **v,
                "rows": [{f: r(x, 4) if isinstance(x, float) else x for f, x in row.items()} for row in v["rows"]],
                "said": line_said(v, names["series"]),
            }
            for k, v in lines.items()
        },
        "audit": audit(decision),
        # the decision step: your targets leaned by today's world within the guardrails, its reasons, the walk's
        # sense and lengths' weights; the week's key questions (the furthest from usual)
        "decision": rd({k: v for k, v in (decision or {}).items() if k != "node"}) if decision else None,
        "key": [{"id": q, "apart": r(x, 4)} for q, x in key_questions(probs, qrows)],
        "proposals": proposed,
        # the signposts the LLM proposed: their checks, where they stand on history and on what came after
        "signposts_proposed": [{**x, "said": said(x, names["series"])} for x in rd(signposts_proposed or [])],
        # the latest texts the LLM read into numbers (config/readings.toml)
        "readings": rd(readings or []),
        "text_checks": text_checks(db),
        # how the LLM's scenarios turned out: the weight it gave the one that happened, against the simulation's
        "scenarios_record": scenarios_record(db),
        # the system: every stage of the day, what feeds what, how each went today (flow.py)
        "system": system,
        # you (your style, policy and rules; no holdings) and the actions where the futures meet it
        "you": rd(you),
        "actions": rd(actions or []),
        # the decision step walked through history against simple ways of holding the same things (data/walk.json)
        "backtest": rd(backtest),
        # each horizon's futures: the allocations' figures on the final world, the simulation, without the LLM and
        # history; the classes, the factors. The LLM's scenarios at each length, with their weights and the
        # simulation's share
        "scenarios": scenarios or [],
        # each length's set: who weighed it, the reviewer's points and what it asked the data, the coach's note
        "scenario_sets": rd(scenario_sets or {}),
        "horizons": {
            h: rd(
                {
                    k: v
                    for k, v in d.items()
                    if k not in ("futures", "data_futures", "no_llm_futures", "history_futures", "held")
                }
            )
            for h, d in (horizons or {}).items()
        },
    }
    store.write_json(store.VIEW, view)
    if account:
        acc = dict(account)
        # what each category has to move to reach its decided share (USD, + buy, - sell: the decision step's, else
        # its target); Bitcoin against its cap
        want = {**acc["targets"], **(acc.get("decided") or {})}
        acc["gaps"] = {c["id"]: (want[c["id"]] - acc["shares"].get(c["id"], 0)) * acc["nav"] for c in acc["categories"]}
        if "btc_max" in acc["targets"]:  # a cap on Bitcoin is the owner's choice (local/rules.toml [targets])
            acc["gaps"]["btc"] = (acc["targets"]["btc_max"] - acc["shares"].get("btc", 0)) * acc["total"]
        try:
            acc["redteam"] = json.loads((store.ACCOUNT / "redteam.json").read_text())
        except OSError:
            acc["redteam"] = None
        store.write_json(store.ACCOUNT_VIEW, acc)


def key_questions(probs, rows):
    """The week's key questions: their final probability furthest from history's base rate, one a template
    (model/questions.py key)."""
    tmpl = {q["id"]: q.get("template") or q["id"] for q in rows if not q.get("resolved")}
    got = {q: (v["model"], v["sources"].get("base")) for q, v in (probs or {}).items() if q in tmpl}
    return key_by(got, templates=tmpl)


def readings_said(xs, series, cfg):
    """A rule's readings, each condition also in words ("said": what it reads; "line_said": where it holds)."""
    sp = {x["id"]: x["name"] for x in cfg["signposts"]["signpost"]}
    out = []
    for x in xs or []:
        x = dict(x)
        if x.get("group"):
            x["group"] = [readings_said(g, series, cfg) for g in x["group"]]
        if x.get("expr"):
            words, pc = expr.say(x["expr"], series)
            x["said"] = words[:1].upper() + words[1:]
            if x.get("op") and isinstance(x.get("than"), int | float):
                x["line_said"] = expr.say_line(f"{x['op']} {x['than']}", pc)
        elif x.get("name"):
            x["said"] = sp.get(x["name"], x["name"].replace("_", " "))
        out.append(x)
    return out


def line_said(v, series):
    """A rule's line in history in words, at its own place: "AHR999 below 0.45"."""
    words, pc = expr.say(v["expr"], series)
    at = next((x for x in v["rows"] if x["shift"] == 0), None)
    where = " " + expr.say_line(f"{v['op']} {at['line']}", pc) if at else ""
    return words[:1].upper() + words[1:] + where

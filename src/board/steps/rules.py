"""The rules written in advance (preferences beside the decision step: Bitcoin's purchase, cap, pause and review):
the targets and pace they compare against, which hold today and which fire."""

import logging

from ..core import expr, lineage
from ..data import account as account_data
from ..data import observations
from ..model import behaviour, rule_history, rules

log = logging.getLogger("board")


def context(cfg, df, sp_today, held, today):
    """What the rules' conditions read today (model/rules.py Context)."""
    states = {k: v["state"] for k, v in sp_today.items()}
    return rules.Context(rules.rules_config(cfg), df, states, held and held["shares"], today)


def rules_today(db, cfg, chash, df, sp_today, held, today):
    """Which rules hold today and which fire this period, recorded; (rule rows, new firings, [(trigger id,
    rule)] standing this period). The account is read only through `held` (None without holdings): its shares
    and its purchase rhythm (a month without)."""
    rcfg = rules.rules_config(cfg)
    ctx = context(cfg, df, sp_today, held, today)
    rhythm = held["dca"]["rhythm"] if held else "month"
    rule_rows, new, standing = [], [], []
    for r, ok, readings in rules.evaluate(rcfg, ctx):
        try:
            key = rules.period_key(r["period"], today, rhythm)
        except ValueError:
            key = "invalid"
        tid = f"{r['id']}:{key}"
        fired = db.execute("SELECT day FROM triggers WHERE id = ?", (tid,)).fetchone()
        if r["period"] == "event":  # one firing per episode: the last one stands until the rule re-arms
            last = db.execute(
                "SELECT id, day FROM triggers WHERE rule = ? ORDER BY day DESC LIMIT 1", (r["id"],)
            ).fetchone()
            if last and last[1] < today.isoformat():
                days = db.execute(
                    "SELECT day, holds FROM rule_days WHERE rule = ? AND day > ? AND day < ? ORDER BY day",
                    (r["id"], last[1], today.isoformat()),
                ).fetchall()
                if not rules.rearmed(days, last[1], r.get("rearm_days", rules.REARM_DAYS)):
                    tid, fired = last[0], (last[1],)
        db.execute("INSERT OR REPLACE INTO rule_days VALUES (?,?,?)", (today.isoformat(), r["id"], int(ok)))
        if ok and not fired:
            series_in, sps_in, holdings_in = rules.rule_inputs(rcfg, r)
            parents = ([held["node"]] if holdings_in and held else []) + [
                sp_today[c]["node"] for c in sps_in if c in sp_today
            ]
            parents += [n for x in sorted(series_in) for n in observations.raw_nodes(db, x, cfg["series"])]
            lineage.record(
                db,
                "rule_trigger",
                f"trigger:{tid}",
                {"rule": r["id"], "period": key, "day": today.isoformat()},
                {"rule": r, "readings": readings, "config": chash},
                parents,
                step="rule",
            )
            db.execute("INSERT INTO triggers VALUES (?,?,?,?)", (tid, r["id"], key, today.isoformat()))
            new.append(r)
            fired = (today.isoformat(),)
        if fired:
            standing.append((tid, r))
        rule_rows.append(
            {
                "id": r["id"],
                "title": r["title"],
                "detail": r.get("detail"),
                "explain": r.get("explain"),
                "period": key,
                "holds": ok,
                "trigger": tid if fired else None,
                "buys": bool(r.get("amount")),  # a purchase: it waits while a guardrail moves out of what it buys
                "readings": readings,
            }
        )
    db.commit()
    return rule_rows, new, standing


def rule_lines(cfg, df):
    """Each condition on a value with a numeric line, walked through history at its line and at lines moved
    either side (model/rule_history.py): {condition: {"expr", "subject", "op", "rows", "verdict"}}. The subject is
    what is followed after each episode: `subject` on the condition, else the expression's first series."""
    rcfg = rules.rules_config(cfg)
    numbers = rules.Context(rcfg, df, {}, None, None)
    out = {}
    for name, c in rcfg["conditions"].items():
        if "value" not in c or c["op"] not in ("<", "<=", ">", ">="):
            continue
        subject = c.get("subject") or expr.names(c["value"])[0]
        if subject not in df:
            continue
        try:
            line = numbers.number(c["than"])
            values = expr.evaluate(c["value"], df)
        except (KeyError, ValueError) as e:
            log.warning("rule line %s: %s", name, e)
            continue
        rows, verdict = rule_history.walk(values, df[subject], c["op"], line)
        out[name] = {"expr": c["value"], "subject": subject, "op": c["op"], "rows": rows, "verdict": verdict}
    return out


def marks(db, cfg):
    """The owner's marks on the firings (done or skipped, from the page) against how each rule leans (`leans`
    in the rules): model/behaviour.py's reading, with the chosen style."""
    status = account_data.decisions()
    fired = db.execute("SELECT id, rule FROM triggers").fetchall()
    leans = {r["id"]: r["leans"] for r in cfg["rules"]["rule"] if r.get("leans")}
    out = behaviour.lean([(rule, status[t]) for t, rule in fired if t in status], leans)
    return {**out, "profile": cfg["rules"].get("profile")}

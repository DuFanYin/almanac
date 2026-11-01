"""New questions: the LLM proposes, the data judges, the owner decides (voted on and adopted with the factors and
rule changes, common.py).

Every PROPOSE_EVERY days the LLM drafts up to PER_WEEK questions (data/llm.py propose), told what the board asks,
the series it keeps, the LLM's scenarios and where the owner's targets' bad case lies; each draft is
checked on the data (model/proposals.py) and kept with its checks and lineage (the proposals table, one version
each). The page shows those that passed; the owner adopts or rejects each there. An adopted one is written to
local/questions.toml, which the config reads beside config/questions.toml, so it is registered that day and frozen
like any question; a registered question is never changed by a proposal. A draft with an id proposed before is
not checked again. No more are asked while MAX_OPEN wait for the owner."""

import json
import logging

import pandas as pd

from ...core import lineage, store
from ...data import llm
from ...model import asset_classes, backtest, describe
from ...model import proposals as checks
from .common import asked as asked_round
from .common import due, keep

log = logging.getLogger("board")

PROPOSE_EVERY = 7  # days: weekly
PER_WEEK = 3
MAX_OPEN = 6
HEADER = """# The questions the LLM proposed and the owner adopted on the scenarios page (written by the board's job,
# src/board/steps/proposals/questions.py), read beside src/board/config/questions.toml in its format. A question is
# frozen in the database once registered: editing one here changes nothing already registered.
"""


def context(horizons, futures):
    """What the board sees this week, for the LLM: the LLM's scenarios, and where the owner's targets' bad case
    lies at each horizon."""
    lines = []
    if futures:
        lines.append("The scenarios of the board's world now (written by an LLM, with their weights):")
        lines += [f"- {f['name']} ({f['horizon']}, {f['p'] or 0:.0%}): {f['story']}" for f in futures]
    for d in (horizons or {}).values():
        st = d["stats"].get("targets", {}).get("all")
        if st:
            lines.append(
                f"The owner's targets over the next {d['label']}: the mean of the worst 10% of futures {st['bad']:+.0%}"
                f" (the limit {d['limit']:+.0%})."
            )
    return "\n".join(lines) or "nothing more"


def mix_returns(cfg, you, df, days):
    """The owner's targets' return over `days` by window start ({day: r}), on the classes' total returns, over the
    classes that have them; None without targets."""
    if not you:
        return None
    defs = cfg["asset_classes"]["class"]
    rets = {}
    for k, w in you["policy"].items():
        p = backtest.total_return(defs[k], df) if w > 0 and k in defs else None
        if p is not None and len(p) > days:
            rets[k] = backtest.windows(p, days)
    return asset_classes.mix_windows({k: you["policy"][k] for k in rets}, rets) if rets else None


def run(db, cfg, you, horizons, futures, history, series, df, data, today, force=False):
    """Asks for new questions when PROPOSE_EVERY days have passed since the last (or force) and fewer than MAX_OPEN
    wait; checks and keeps each. How many passed."""
    mine = "json_extract(spec, '$.type') IS NULL"  # the questions; factors and rule changes have their own round
    last = db.execute(f"SELECT max(day) FROM proposals WHERE {mine}").fetchone()[0]
    if not force and not due(db, "questions", PROPOSE_EVERY, today, last):
        return 0
    waiting = db.execute(f"SELECT id, spec FROM proposals WHERE status = 'open' AND {mine}").fetchall()
    if len(waiting) >= MAX_OPEN and not force:
        return 0
    asked = [
        {"id": k, "title": q["title"], "outcomes": [b["label"] for b in q["buckets"]]}
        for k, q in cfg["questions"].items()
        if k != "weights"
    ]
    asked += [
        {"id": i, "title": (s := json.loads(x))["title"], "outcomes": [b["label"] for b in s["buckets"]]}
        for i, x in waiting
    ]
    out = llm.propose(
        today.isoformat(), asked, describe.series_list(cfg, series), data, context(horizons, futures), PER_WEEK
    )
    asked_round(db, "questions", today)
    tried = {r[0] for r in db.execute("SELECT id FROM proposals")}
    others = {q["id"]: outs for q, (outs, _) in history.values() if outs}
    titles = {q["id"]: q["title"] for q, _ in history.values()}
    horizon = cfg["factors"]["horizon"]["long"]["days"]
    mix = mix_returns(cfg, you, df, horizon)
    parents = [lineage.latest(db, "snapshot"), you and you["node"], *(d["node"] for d in (horizons or {}).values())]
    passed = 0
    for p in out["proposals"]:
        if p["id"] in tried:
            continue
        try:
            res = checks.check(p, series[p["series"]].dropna(), today, others, mix, horizon, titles)
        except Exception as e:  # a draft the data cannot read: kept as failed
            res = {
                "passed": False,
                "checks": [{"name": "readable", "ok": False, "say": f"{type(e).__name__}: {e}"[:200]}],
            }
        keep(db, p["id"], p, res, out, parents, today, rejected=out["rejected"])
        passed += bool(res["passed"])
    db.commit()
    log.info("proposals: %d drafted, %d passed, $%s", len(out["proposals"]), passed, out["usage"]["cost_usd"])
    return passed


def toml_entry(p, today, by=None):
    """The adopted question in questions.toml's format; a path question closes `days` after the day adopted; one the
    LLM adopted (by "llm") says so, and a periodic one stops opening a year on (until)."""
    s = lambda x: json.dumps(x, ensure_ascii=True)  # noqa: E731  (a JSON string is a TOML basic string)
    lines = [f"[{p['id']}]", f"title = {s(p['title'])}", f"kind = {s(p['kind'])}", f"series = {s(p['series'])}"]
    if p["kind"] == "path":
        lines.append(f"closes = {s((today + pd.Timedelta(days=p['days'])).date().isoformat())}")
    else:
        lines += [f"every = {s(p['every'])}", f"measure = {s(p['measure'])}"]
        if by == "llm":
            lines.append(f"until = {s((today + pd.Timedelta(days=LLM_YEAR)).date().isoformat())}")
    if by:
        lines.append(f"adopted_by = {s(by)}")
    lines += [f"detail = {s(p['detail'])}", f"purpose = {s(p['purpose'])}", f"why = {s(p['why'])}", "buckets = ["]
    lines += [
        f"    {{ id = {s(b['id'])}, label = {s(b['label'])}, when = {s(b['when'])}, tone = {float(b['tone'])} }},"
        for b in p["buckets"]
    ]
    return "\n".join([*lines, "]", ""])


def adopt_question(spec, today, by=None):
    """The question appended to local/questions.toml."""
    path = store.ADOPTED
    text = path.read_text() if path.exists() else HEADER
    path.write_text(text + "\n" + toml_entry(spec, pd.Timestamp(today), by))
    return {"file": path.name}


LLM_YEAR = 365  # days a periodic question the LLM adopted keeps opening

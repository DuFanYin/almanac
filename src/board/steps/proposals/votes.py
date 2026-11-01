"""The proposals table's decisions (questions and factors the LLM proposed and the board's checks passed): the owner's
votes from the page's inputs, taken in first; then, for a proposal the owner has left WAIT days, the LLM's (the owner,
2026-10-31: "I don't think I judge better than the LLM"): an independent reading asked JUDGES times (data/llm.py
judge_proposal), a majority adopts, at most CAP of each kind in CAP_DAYS; one it declines stays for the owner. Every
decision recorded, with whose it was and why. The owner may withdraw what was adopted (a vote "withdrawn" on it, or
deleting its table from local/questions.toml or local/factors.toml): its table is taken out and the questions
opened from it withdrawn. And the table's rows for the page."""

import json
import re
from datetime import timedelta

import pandas as pd

from ...core import lineage, store
from ...data import account as account_data
from ...data import llm
from . import changes, questions

WAIT = 3  # days the owner has to vote before the LLM decides
CAP = {"question": 2, "factor": 1}  # adopted by the LLM in CAP_DAYS, at most
CAP_DAYS = 30
OPEN = ("open", "declined by the LLM")  # the statuses the owner can still vote on


def apply(sp, today, by=None):
    kind = sp.get("type", "question")
    if kind == "question":
        return questions.adopt_question(sp, today, by)
    if kind == "factor":
        return changes.adopt_factor(sp, today, by)
    return None  # a rule change: retired with the rules that read forecasts


def adopt(db, today, use_llm=False):
    """The decisions on the open proposals: the owner's votes (the page's inputs) first, then the LLM's on those the
    owner left WAIT days (use_llm). An adopted one applied (a question written to local/questions.toml, a factor to
    local/factors.toml), each decision recorded. Run before the config is read, so what was adopted counts the same
    day. The ids decided."""
    votes = account_data.votes()
    done = []
    retired = "UPDATE proposals SET status = 'retired', decided = ? WHERE status IN (?, ?) AND "
    # a rule change still open retires with the rules that read forecasts
    db.execute(retired + "json_extract(spec, '$.type') = 'rule'", (today.isoformat(), *OPEN))
    rows = db.execute("SELECT id, spec, node, status FROM proposals WHERE status IN (?, ?, 'adopted')", OPEN).fetchall()
    for pid, spec, node, was in rows:
        if pid not in votes:
            continue
        status, note = votes[pid]
        sp, applied = json.loads(spec), {}
        if was == "adopted":
            if status != "withdrawn":
                continue
            applied = withdraw(db, sp, today)
        elif status == "adopted":
            applied = apply(sp, today)
            if applied is None:
                status = "retired"
        elif status not in ("rejected",):
            continue
        db.execute("UPDATE proposals SET status = ?, decided = ? WHERE id = ?", (status, today.isoformat(), pid))
        lineage.record(
            db,
            "decision",
            f"decision:{pid}",
            {"status": status, "note": note, "by": "owner"},
            applied or {},
            [node],
            step="owner",
        )
        done.append(pid)
    if use_llm:
        done += by_llm(db, today, votes)
    db.commit()
    return done


def adopted_lately(db, today):
    """{kind: how many the LLM adopted in the last CAP_DAYS days}."""
    since = (today - timedelta(days=CAP_DAYS)).isoformat()
    out = {}
    for (spec,) in db.execute(
        "SELECT p.spec FROM proposals p WHERE p.status = 'adopted' AND p.decided >= ? AND EXISTS (SELECT 1 FROM "
        "artifact a WHERE a.name = 'decision:' || p.id AND json_extract(a.value, '$.by') = 'llm')",
        (since,),
    ):
        k = json.loads(spec).get("type", "question")
        out[k] = out.get(k, 0) + 1
    return out


def has(db, kind):
    """What the board already has of a kind, one a line: the open questions' titles, or the factors."""
    if kind == "factor":
        cfg, _ = store.config()
        return [f"{k}: {f['label']} ({f['series']}, {f['measure']})" for k, f in cfg["factors"]["factor"].items()]
    return [
        t
        for (t,) in db.execute(
            "SELECT DISTINCT title FROM questions q WHERE NOT EXISTS (SELECT 1 FROM "
            "resolutions r WHERE r.question = q.id) ORDER BY opened DESC LIMIT 60"
        )
    ]


def by_llm(db, today, votes):
    """The LLM's decisions on the open proposals the owner left WAIT days, within CAP: the ids decided."""
    count, done = adopted_lately(db, today), []
    last = (today - timedelta(days=WAIT)).isoformat()
    for pid, spec, chk, node, day in db.execute(
        "SELECT id, spec, checks, node, day FROM proposals WHERE status = 'open' AND passed = 1 AND day <= ? "
        "ORDER BY day",
        (last,),
    ).fetchall():
        sp = json.loads(spec)
        kind = sp.get("type", "question")
        if pid in votes or kind not in CAP or count.get(kind, 0) >= CAP[kind]:
            continue
        checks = (json.loads(chk) or {}).get("checks") or []
        got = llm.judge_proposal(today.isoformat(), kind, sp, checks, has(db, kind))
        status, applied = ("adopted", apply(sp, today, "llm")) if got["adopt"] else ("declined by the LLM", {})
        if got["adopt"]:
            count[kind] = count.get(kind, 0) + 1
        db.execute("UPDATE proposals SET status = ?, decided = ? WHERE id = ?", (status, today.isoformat(), pid))
        lineage.record(
            db,
            "decision",
            f"decision:{pid}",
            {"status": status, "note": got["why"], "by": "llm"},
            {**(applied or {}), "votes": got["votes"], "method": got["method"], "usage": got["usage"]},
            [node],
            step="llm_adopt",
        )
        done.append(pid)
        db.commit()
    return done


def withdraw(db, sp, today):
    """What was adopted taken out: its table removed from local/questions.toml (and the questions opened from it
    withdrawn: no longer computed or scored) or local/factors.toml. {"removed": the table's name}."""
    kind = sp.get("type", "question")
    path, name = (store.ADOPTED_FACTORS, f"factor.{sp['id']}") if kind == "factor" else (store.ADOPTED, sp["id"])
    if path.exists():
        text = path.read_text()
        pat = re.compile(rf"^\[{re.escape(name)}\]\n.*?(?=^\[|\Z)", re.M | re.S)
        path.write_text(pat.sub("", text))
    if kind == "question":
        for (qid,) in db.execute(
            "SELECT id FROM questions WHERE json_extract(spec, '$.template') = ? AND id NOT IN (SELECT question FROM "
            "resolutions)",
            (sp["id"],),
        ).fetchall():
            db.execute(
                "INSERT OR IGNORE INTO resolutions VALUES (?,?,?,?)",
                (qid, today.isoformat(), None, "withdrawn: by you"),
            )
    return {"removed": name}


def rows(db, days=56):
    """The proposals of the last `days` days and every open one, newest first, for the page; each decided one with
    whose decision it was (the owner's or the LLM's) and why."""
    since = (pd.Timestamp.today() - pd.Timedelta(days=days)).date().isoformat()
    out = []
    for i, d, spec, chk, passed, status, node, decided in db.execute(
        "SELECT * FROM proposals WHERE day >= ? OR status IN (?, ?) ORDER BY day DESC, id", (since, *OPEN)
    ):
        who = db.execute(
            "SELECT value FROM artifact WHERE name = ? ORDER BY id DESC LIMIT 1", (f"decision:{i}",)
        ).fetchone()
        by = json.loads(who[0]) if who else {}
        out.append(
            {
                "id": i,
                "day": d,
                "spec": json.loads(spec),
                "checks": json.loads(chk),
                "passed": bool(passed),
                "status": status,
                "node": node,
                "decided": decided,
                "by": by.get("by"),
                "why": by.get("note"),
            }
        )
    return out

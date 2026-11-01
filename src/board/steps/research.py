"""Research on the week's key questions (model/questions.py key: those whose probability stands furthest from
usual on the last day recorded): every RESEARCH_EVERY days each is researched again (data/research.py) and kept with
its lineage. Its notes from the news reach the LLM's futures (the writer is told them); its own forecast is kept
beside the others as the source "research" and scored, not used."""

import logging
import time

from ..core import lineage
from ..data import research as research_data
from ..model import questions as qs

log = logging.getLogger("board")

RESEARCH_EVERY = 28  # days


def key_now(db, open_qs):
    """The week's key questions on the last day their probabilities were recorded: [question id]."""
    row = db.execute("SELECT MAX(day) FROM probs WHERE source = 'model'").fetchone()
    if not row or not row[0]:
        return []
    got = {}
    for q, src, b, p in db.execute(
        "SELECT question, source, bucket, p FROM probs WHERE day = ? AND source IN ('model', 'base')", (row[0],)
    ):
        got.setdefault(q, ({}, {}))[0 if src == "model" else 1][b] = p
    tmpl = {q["id"]: q["spec"].get("template", q["id"]) for q in open_qs}
    return [q for q, _ in qs.key({k: v for k, v in got.items() if k in tmpl}, templates=tmpl)]


def run_research(db, open_qs, data, today, key, force=False):
    """Researches each key question (key: their ids) whose last research is older than RESEARCH_EVERY days (all of
    them with force). Returns how many were kept."""
    snap = lineage.latest(db, "snapshot")
    done = 0
    for q in open_qs:
        if q["id"] not in key or q["spec"].get("rolling"):
            continue
        _, _, at = lineage.model_answer(db, f"research:{q['id']}")
        if at and not force and time.time() - at < RESEARCH_EVERY * 86400:
            continue
        out, meta = research_data.run(today.isoformat(), q, data)
        if not out["summary"]:
            log.warning("research %s: no usable answer", q["id"])
            continue
        lineage.record(
            db,
            "research",
            f"research:{q['id']}",
            {**out["summary"], "found": len(out["evidence"]), "notes": len(out["notes"])},
            {**meta, **{k: out[k] for k in ("drivers", "rounds", "evidence", "notes", "answers")}, "data": snap},
            [snap] if snap else [],
            step="research",
        )
        db.commit()
        log.info(
            "research %s: %d items, %d notes, $%s",
            q["id"],
            len(out["evidence"]),
            len(out["notes"]),
            meta["usage"]["cost_usd"],
        )
        done += 1
    return done

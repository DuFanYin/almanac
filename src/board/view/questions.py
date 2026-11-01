"""The forecasts on the page: each with its outcomes, probabilities and their sources (read off the world), research's
notes, and its recent history."""

import json
import time

import pandas as pd

from .data import HISTORY_DAYS
from .fmt import r, rounded


def tone(qcfg, template, bucket):
    """How an outcome reads for the holdings (-2..2): presentation only, so taken from the current config
    rather than the frozen registration."""
    return next((b.get("tone", 0) for b in qcfg.get(template, {}).get("buckets", []) if b["id"] == bucket), 0)


def research_notes(db, qid):
    """Research's latest notes on a question and the news behind them: {"day", "notes": [text], "found"}; None."""
    row = db.execute(
        "SELECT id, meta, created_at FROM artifact WHERE name = ? ORDER BY id DESC LIMIT 1", (f"research:{qid}",)
    ).fetchone()
    if not row:
        return None
    meta = json.loads(row[1] or "{}")
    return {
        "day": time.strftime("%Y-%m-%d", time.localtime(row[2])),
        "node": row[0],
        "notes": [n["value"] for n in meta.get("notes") or []],
        "found": len(meta.get("evidence") or []),
        "drivers": meta.get("drivers"),
    }


def question_rows(db, probs, today, qcfg, names=None):
    """Every question open or resolved in the last two months, for the page: its outcomes, today's probabilities and
    their sources, research's notes, its history."""
    rows = db.execute(
        "SELECT id, title, kind, opened, closes, buckets, spec FROM questions ORDER BY opened, id"
    ).fetchall()
    res = {q: (d, b, how) for q, d, b, how in db.execute("SELECT question, day, bucket, how FROM resolutions")}
    since = (pd.Timestamp(today) - pd.Timedelta(days=HISTORY_DAYS)).date().isoformat()
    out = []
    newest = {}  # a rolling question shows only its newest open instance; every one is still scored
    for qid, _, _, opened, _, _, spec in rows:
        t = json.loads(spec)
        if t.get("rolling") and qid not in res:
            newest[t["template"]] = max(newest.get(t["template"], ""), qid)
    for qid, title, kind, opened, closes, buckets, spec in rows:
        spec = json.loads(spec)
        resolved = res.get(qid)
        if resolved and (resolved[1] is None or (pd.Timestamp(today) - pd.Timestamp(resolved[0])).days > 60):
            continue  # resolved two months ago, or withdrawn
        if spec.get("rolling") and (resolved or newest.get(spec["template"]) != qid):
            continue
        hist = {}
        for d, src, b, p in db.execute(
            "SELECT day, source, bucket, p FROM probs WHERE question = ? "
            "AND source IN ('model', 'math', 'llm', 'futures') AND day >= ? ORDER BY day",
            (qid, since),
        ):
            hist.setdefault(d, {}).setdefault(src, {})[b] = r(p, 4)
        v = probs.get(qid) or {}
        tcfg = qcfg.get(spec.get("template"), {})
        out.append(
            {
                "id": qid,
                "title": title,
                "kind": kind,
                "template": spec.get("template"),
                "opened": opened,
                "headline": tcfg.get("headline"),
                "proposed": tcfg.get("purpose"),  # adopted from the LLM's proposals: what it was proposed for
                # the data series it is resolved and based on (a Kalshi question's `series` is Kalshi's, not ours)
                "series": spec.get("series")
                if kind in ("path", "period")
                else ("USREC" if spec.get("base_rate") == "recession_start" else None),
                "closes": closes,
                "detail": spec.get("detail"),
                "start": spec.get("start_px"),
                "buckets": [
                    {
                        "id": b["id"],
                        "label": b["label"],
                        "tone": b.get("tone", tone(qcfg, spec.get("template"), b["id"])),
                        **{k: b[k] for k in ("short", "about") if k in b},
                    }
                    for b in json.loads(buckets)
                ],
                "model": rounded(v.get("model")),
                "interval": {k: [r(x, 4) for x in iv] for k, iv in (v.get("interval") or {}).items() if iv},
                "math": rounded(v.get("math")),
                "math_interval": {k: [r(x, 4) for x in iv] for k, iv in (v.get("math_interval") or {}).items() if iv},
                "sources": {s: rounded(p) for s, p in (v.get("sources") or {}).items()},
                "weights": v.get("weights"),
                "scenarios_from": v.get("scenarios_from"),
                "research": research_notes(db, qid),
                # inputs of the computed side with a model upstream: kept out of the control without the LLM
                "node": v.get("node"),
                "math_node": v.get("math_node"),
                "history": [{"day": d, **x} for d, x in sorted(hist.items())],
                "resolved": {"day": resolved[0], "bucket": resolved[1], "how": resolved[2]} if resolved else None,
            }
        )
    return out

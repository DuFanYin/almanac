"""The questions' lifecycle: registered when due (buckets and spec frozen), their base rates and market prices,
resolved from data or the market, and the record of every source once they resolve."""

import json
import logging
import time

from ..core import lineage
from ..data import kalshi
from ..model import prob
from ..model import questions as qs

log = logging.getLogger("board")


METHOD = ["history_from"]  # how history is read, not what the question asks: taken from today's config


def open_questions(db, qcfg):
    """The open questions as registered (buckets and spec frozen), with today's method settings."""
    rows = db.execute("""SELECT q.id, q.title, q.kind, q.opened, q.closes, q.buckets, q.spec FROM questions q
                         LEFT JOIN resolutions r ON r.question = q.id WHERE r.question IS NULL
                         ORDER BY q.opened, q.id""")
    out = []
    for i, t, k, o, c, b, s in rows:
        spec = json.loads(s)
        spec.update({m: qcfg[spec["template"]][m] for m in METHOD if m in qcfg.get(spec.get("template"), {})})
        out.append({"id": i, "title": t, "kind": k, "opened": o, "closes": c, "buckets": json.loads(b), "spec": spec})
    return out


def register(db, cfg, chash, series, today):
    """Registers the question instances due today, buckets and spec frozen with the config's hash."""
    qcfg = cfg["questions"]
    existing = {r[0] for r in db.execute("SELECT id FROM questions")}
    rows = qs.due(qcfg, today, existing, series)
    events = {}
    for qid, q in qcfg.items():
        if qid != "weights" and q.get("kind") == "kalshi" and "series" in q:
            try:
                events[q["series"]] = kalshi.events(q["series"], q.get("next", 3))
            except Exception as e:
                log.warning("kalshi events %s: %s", q["series"], e)
    rows += qs.due_kalshi(qcfg, today, existing, events)  # the factors are read off the world: not registered
    for iid, title, kind, opened, closes, buckets, spec in rows:
        db.execute(
            "INSERT OR IGNORE INTO questions VALUES (?,?,?,?,?,?,?,?)",
            (iid, title, kind, opened, closes, json.dumps(buckets), json.dumps(spec), chash),
        )
    db.commit()
    return [r[0] for r in rows]


def kalshi_markets(db, open_qs, fetch=True):
    """Adds each Kalshi question's market probabilities (recorded) and its settled result, if any. Without fetch, the
    market as last recorded, and no result."""
    results = {}
    for q in open_qs:
        if q["kind"] != "kalshi":
            continue
        if not fetch:
            node = lineage.latest(db, f"market:{q['id']}")
            p = lineage.value(db, node)
            if p:
                q["market"] = (p, node)
            continue
        try:
            markets = {}
            for ev in q["spec"]["events"]:
                _, m = kalshi.event(ev)
                markets.update(m)
                time.sleep(0.2)
        except Exception as e:
            log.warning("kalshi %s: %s", q["id"], e)
            continue
        p = qs.market_bucket_probs(q, markets)
        if p:
            node = lineage.record(
                db,
                "market",
                f"market:{q['id']}",
                {k: round(v, 4) for k, v in p.items()},
                {
                    "events": q["spec"]["events"],
                    "markets": {t: {"label": m[0], "p": m[1], "status": m[2]} for t, m in markets.items()},
                },
                step="fetch",
            )
            q["market"] = (p, node)
        res = qs.kalshi_result(q, markets)
        if res:
            results[q["id"]] = (res, json.dumps({t: m[3] for t, m in markets.items()}))
    return results


def withdraw(db, qid, why, today):
    """Takes an open question out (the owner's call, run.py withdraw): no longer computed or resolved, never scored;
    what it was given stays in the database. Recorded. Whether it was open."""
    if not db.execute("SELECT 1 FROM questions WHERE id = ?", (qid,)).fetchone():
        return False
    done = db.execute(
        "INSERT OR IGNORE INTO resolutions VALUES (?,?,?,?)", (qid, today.isoformat(), None, f"withdrawn: {why}")
    ).rowcount
    if done:
        how = {"how": f"withdrawn: {why}"}
        lineage.record(db, "resolution", f"resolution:{qid}", {"bucket": None}, how, [], step="owner")
    db.commit()
    return bool(done)


def resolve(db, open_qs, series, kalshi_results, today):
    done = []
    for q in open_qs:
        r = kalshi_results.get(q["id"]) if q["kind"] == "kalshi" else qs.resolve(q, series, today)
        if r:
            bucket, how = r
            db.execute("INSERT OR IGNORE INTO resolutions VALUES (?,?,?,?)", (q["id"], today.isoformat(), bucket, how))
            lineage.record(
                db,
                "resolution",
                f"resolution:{q['id']}",
                {"bucket": bucket},
                {"how": how},
                [lineage.latest(db, f"raw:{q['spec'].get('series')}")] if q["spec"].get("series") else [],
                step="resolve",
            )
            done.append(q["id"])
    db.commit()
    return done


def scores(db):
    """The record (prob.scores) of every resolved question's daily forecasts, leaving out those whose lineage
    holds excluded data. The LLM and research are scored by version ("llm 44dcbf9c46 deepseek-v4-flash": the
    method's hash and model), so a changed prompt or model starts a record of its own."""
    forecasts, version = {}, {}

    def named(source, node):
        if source not in ("llm", "research") or not node:
            return source
        if node not in version:
            row = db.execute("SELECT meta FROM artifact WHERE id = ?", (node,)).fetchone()
            meta = json.loads(row[0]) if row else {}
            ver = " ".join(x for x in (meta.get("prompt_hash"), *(meta.get("models") or [])) if x)
            version[node] = ver or None
        return f"llm {version[node]}" if version[node] else source

    for qid, outcome in db.execute("SELECT question, bucket FROM resolutions WHERE bucket IS NOT NULL").fetchall():
        by = {}
        for day, source, bucket, p, node in db.execute(
            "SELECT day, source, bucket, p, artifact FROM probs WHERE question = ?", (qid,)
        ):
            by.setdefault((source, day), ({}, node))[0][bucket] = p
        by_source = {}
        for (source, day), (ps, node) in by.items():
            if not (node and lineage.tainted(db, node)):
                by_source.setdefault(named(source, node), []).append(ps)
        k = len(json.loads(db.execute("SELECT buckets FROM questions WHERE id = ?", (qid,)).fetchone()[0]))
        forecasts[qid] = (outcome, k, by_source)
    return prob.scores(forecasts)

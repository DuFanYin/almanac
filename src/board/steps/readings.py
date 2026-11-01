"""Texts read into numbers by the LLM (config/readings.toml): the Fed's releases and speeches, how
hawkish; the news on AI investment, cut or raised; the day's headlines, what mood. Each new text is read
(data/llm.py read: asked several times, quotes checked against the text) and kept with its reading (the readings
table, a model-made node each); the series is the mean of each day's readings, stored as observations like any data,
its snapshot with the readings as parents, so whatever reads it has the LLM upstream and stays out of the figures
without the LLM.

Only texts published from the first run on are read (at most `back` days before it): never history, whose end the
model knows. The signposts on the readings start in signpost_proposals as "accumulating" (register_signposts), each
with what the LLM expects it to say about the factors (expectations: too new to be measured, the world uses it at
half weight, sized like the measured ones, until it can be measured; the owner, 2026-10-31)."""

import hashlib
import json
import logging
import re
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from ..core import lineage, store
from ..data import llm, observations, texts

log = logging.getLogger("board")

MOST = 12  # texts read per series in one run, at most
NY = ZoneInfo("America/New_York")


def first_day(db):
    """The day the readings began (the first run that read a text, in New York), or None. Not the oldest text's
    published day: that would move the start back `back` days at every run that read an older text."""
    row = db.execute("SELECT min(read_at) FROM readings").fetchone()
    return datetime.fromtimestamp(row[0], NY).date() if row[0] else None


def last_read(db, sid):
    row = db.execute("SELECT max(day) FROM readings WHERE series = ?", (sid,)).fetchone()
    return date.fromisoformat(row[0]) if row[0] else None


def due_texts(db, sid, r, since, today):
    """The texts of reading `sid` not read yet: the Fed's and the companies' earnings releases published on or after
    `since`; the news when its `every` days have passed."""
    if r["texts"] == "fed":
        got = texts.fed(since)
    elif r["texts"] == "sec":  # only with the owner's contact, which EDGAR asks for
        contact = store.settings().get("sec_contact")
        got = texts.releases(r["companies"], since, contact) if contact else []
    else:
        last = last_read(db, sid)
        got = texts.news(r["queries"], r["every"], today) if not last or (today - last).days >= r["every"] else []
    seen = {i for (i,) in db.execute("SELECT id FROM readings WHERE series = ?", (sid,))}
    return [t for t in got if t["id"] not in seen][:MOST]


def run(db, cfg, today):
    """Reads each reading's new texts and stores its series. {series: texts read}."""
    rcfg = cfg.get("readings")
    if not rcfg:
        return {}
    since = (first_day(db) or today) - timedelta(days=rcfg.get("back", 7))
    out, known = {}, observations.latest_obs(db)
    for sid, r in rcfg["reading"].items():
        new_nodes = []
        for t in due_texts(db, sid, r, since, today):
            res = llm.read(
                t["text"], what=r["what"], scale=r["scale"], title=t["title"], day=t["day"], samples=rcfg["samples"]
            )
            value = res["value"] if res["relevant"] else None
            node = lineage.record(
                db,
                "llm_reading",
                f"reading:{sid}:{t['id']}",
                {"value": value, "relevant": res["relevant"], "spread": res["spread"]},
                {
                    "title": t["title"],
                    "url": t["url"],
                    "day": t["day"],
                    "method": res["method"],
                    "answers": res["answers"],
                    "dropped": res["dropped"],
                    "usage": res["usage"],
                },
                [],
                step="read",
            )
            detail = {"answers": res["answers"], "dropped": len(res["dropped"]), "method": res["method"]}
            db.execute(
                "INSERT OR REPLACE INTO readings VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    t["id"],
                    sid,
                    t["day"],
                    int(time.time()),
                    t["title"],
                    t["url"],
                    value,
                    res["spread"],
                    json.dumps(detail),
                    node,
                ),
            )
            new_nodes.append(node)
        if new_nodes:
            days = {}
            for d, v in db.execute("SELECT day, value FROM readings WHERE series = ? AND value IS NOT NULL", (sid,)):
                days.setdefault(d, []).append(v)
            rows = sorted((d, sum(vs) / len(vs)) for d, vs in days.items())
            observations.store_rows(
                db, sid, rows, known.get(sid, []), int(time.time()), "reading", cfg["series"][sid], new_nodes
            )
            out[sid] = len(new_nodes)
    db.commit()
    if out:
        log.info("readings: %s", out)
    return out


def register_signposts(db, cfg, today):
    """The signposts on the readings (config/readings.toml), entered in signpost_proposals as accumulating the first
    time they are seen, and kept to the config's definition after (what else their spec holds kept). How many were
    new."""
    n = 0
    for sp in (cfg.get("readings") or {}).get("signpost", []):
        row = db.execute("SELECT spec FROM signpost_proposals WHERE id = ?", (sp["id"],)).fetchone()
        if row:
            spec = json.loads(row[0])
            if any(spec.get(k) != v for k, v in sp.items()):
                db.execute(
                    "UPDATE signpost_proposals SET spec = ? WHERE id = ?", (json.dumps({**spec, **sp}), sp["id"])
                )
            continue
        spec = {**sp, "reading": True}
        db.execute(
            "INSERT INTO signpost_proposals VALUES (?,?,?,?,?,?,?)",
            (sp["id"], today.isoformat(), json.dumps(spec), "{}", "accumulating", None, "{}"),
        )
        n += 1
    db.commit()
    return n


def asked_for(sp, reading):
    """What an expectation was written for: the signpost's definition and its reading's question (a change asks
    again)."""
    parts = [sp.get(k, "") for k in ("name", "value", "red", "why")] + [reading.get(k, "") for k in ("what", "scale")]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]


def expectations(db, cfg, today, ask=None):
    """Each signpost on a reading without what the LLM expects it to say (or written for another definition): asked
    (data/llm.py expect) and kept in its spec ("expect", "expect_for"), a model-made node. How many were asked."""
    ask = ask or llm.expect
    rcfg = cfg.get("readings") or {}
    how = {"ret": "its return", "ann": "its yearly rate", "diff": "its change"}
    factors = {
        k: f"{f['label']}, {how.get(f['measure'], 'its change')} over the span"
        for k, f in cfg["factors"]["factor"].items()
    }
    n = 0
    for sp in rcfg.get("signpost", []):
        row = db.execute("SELECT spec FROM signpost_proposals WHERE id = ?", (sp["id"],)).fetchone()
        if not row:
            continue
        spec = json.loads(row[0])
        rid = next((r for r in rcfg.get("reading", {}) if re.search(rf"\b{r}\b", sp["value"])), None)
        reading = rcfg.get("reading", {}).get(rid, {})
        key = asked_for(sp, reading)
        if spec.get("expect_for") == key:
            continue
        out = ask({**sp, "what": reading.get("what", ""), "scale": reading.get("scale", "")}, factors)
        node = lineage.record(
            db,
            "llm_expect",
            f"expect:{sp['id']}",
            out["expect"],
            {"method": out["method"], "usage": out["usage"], "rejected": out["rejected"]},
            [],
            step="readings",
        )
        spec.update(expect=out["expect"], expect_for=key, expect_node=node)
        db.execute("UPDATE signpost_proposals SET spec = ? WHERE id = ?", (json.dumps(spec), sp["id"]))
        n += 1
    db.commit()
    return n


def rows(db, last=20):
    """The latest texts read, newest first, for the page: series, day, title, link, number, spread, the first quote."""
    out = []
    for i, sid, d, title, url, v, sp, detail in db.execute(
        "SELECT id, series, day, title, url, value, spread, detail FROM readings "
        "ORDER BY day DESC, read_at DESC LIMIT ?",
        (last,),
    ):
        answers = json.loads(detail or "{}").get("answers") or []
        quote = next((q for a in answers for q in a.get("quotes") or []), None)
        out.append({"series": sid, "day": d, "title": title, "url": url, "value": v, "spread": sp, "quote": quote})
    return out

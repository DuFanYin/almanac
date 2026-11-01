"""The data layer's observations: fetching every source and keeping each value as it was seen, checking it,
carrying short histories back (backfill), and the point-in-time frame the model reads. No model logic here.

  fetch(db, cfg)        every source, new or revised values checked and stored, one raw snapshot node per series
                        that changed; errors by source or series
  load(db, cfg, today)  what the model reads: the frame (a column a series, every calendar day up to today, a
                        value used only once it was out: day + lag, carried at most max_age days) and each series
                        as dated by its source, both cleaned and with backfills; and the raw values as fetched
Observations are append only: a revised value is a new row (series, day, value, seen_at). A whole history
that moved by one factor (a split, a rebased index) is fetched again from its start and stored as revisions.

Checks a value must pass (series.toml): a number; not dated after tomorrow; `min` / `max` when given;
above 0 for prices (yahoo, cboe, the price archives) unless `positive = false`; a new day's value within
`max_jump` (|log change|, default 0.7 for prices) of the last one. A value that fails is not stored and is
named in the errors. `history_from` drops what came before (a source whose early coverage means nothing)."""

import logging
import math
import statistics
import threading
import time
from datetime import date, timedelta

import pandas as pd

from ..core import expr, lineage
from . import sources

log = logging.getLogger("board.data")
FIRST = {
    "yahoo": date(1950, 1, 1),
    "fred": date(1950, 1, 1),
    "cboe": date(1990, 1, 1),
    "cnn": date(2024, 1, 1),
    "llama": date(2017, 1, 1),
    "multpl": date(1900, 1, 1),
    "archive": date(1950, 1, 1),
}
REFETCH = {
    "yahoo": 10,
    "fred": 120,
    "cboe": 10,
    "cnn": 30,
    "llama": 10,
    "multpl": 400,
    "archive": 0,
}  # days re-read for revisions
ONE_AT_A_TIME = {"fred", "cboe", "archive"}  # a request per series: each gets its own deadline
DEADLINE = {"one": 90, "all": 180}  # seconds: one series, a whole source fetched at once
PRICES = {"yahoo", "cboe"}
PRICE_ARCHIVES = {"coinmetrics", "msci", "worldbank"}
MAX_JUMP = 0.7  # |log change| a new price may make in one step before it is held out (x2.0 or x0.5)
REVISED = 1e-6  # a relative change below this is not a revision
SPLIT = 0.002  # an overlap whose every day moved by one factor within this, the factor at least 0.5% off 1


def latest_obs(db):
    """{series: [(day, value)]}: each day's latest seen value."""
    rows = db.execute("""SELECT series, day, value FROM (SELECT series, day, value, ROW_NUMBER() OVER
                         (PARTITION BY series, day ORDER BY seen_at DESC) rn FROM observations) WHERE rn = 1
                         ORDER BY series, day""")
    out = {}
    for s, d, v in rows:
        out.setdefault(s, []).append((d, v))
    return out


def fetch(db, series_cfg):
    """Fetches every source, stores new or revised values, records one snapshot node per series that changed.
    Returns ({series: node id}, {source or series: error})."""
    known = latest_obs(db)
    by_source = {}
    for sid, s in series_cfg.items():
        # derived: computed below; reading: an LLM's reading of texts, stored by steps/readings.py; an archive: once
        if s["source"] not in ("derived", "reading") and not (s["source"] == "archive" and known.get(sid)):
            by_source.setdefault(s["source"], {})[sid] = s
    errors, nodes, now = {}, {}, int(time.time())
    for source, group in by_source.items():
        last = min((known.get(sid) or [(None,)])[-1][0] or "0" for sid in group)
        since = FIRST[source] if last == "0" else date.fromisoformat(last) - timedelta(days=REFETCH[source])
        started = time.time()
        batches = [{sid: s} for sid, s in group.items()] if source in ONE_AT_A_TIME else [group]
        got = {}
        for batch in batches:
            try:
                got.update(
                    with_deadline(sources.FETCH[source], (batch, since), DEADLINE["one" if len(batches) > 1 else "all"])
                )
            except Exception as e:
                name = next(iter(batch)) if len(batch) == 1 else source
                errors[name] = f"{type(e).__name__}: {e}"[:200]
                log.warning("fetch %s: %s", name, errors[name])
        log.info("fetch %s: %d of %d series in %.1f s", source, len(got), len(group), time.time() - started)
        for sid, cfg in group.items():
            rows = got.get(sid)
            if sid not in got and (source in errors or sid in errors):
                continue
            if not rows:
                errors[sid] = "no values returned"
                continue
            if cfg.get("min_rows") and len(rows) < cfg["min_rows"]:
                errors[sid] = f"{len(rows)} values, expected at least {cfg['min_rows']}: not kept"
                continue
            factor = moved_by_one_factor(rows, known.get(sid, []))
            if not factor and cfg.get("adjusted"):  # a dividend rescales every earlier adjusted close, if only by 0.1%
                factor = rescaled(rows, known.get(sid, []))
            if factor:  # a split, a rebase or a dividend: the whole history again
                log.warning("%s: history moved by x%.4f; fetching it again from the start", sid, factor)
                try:
                    rows = with_deadline(sources.FETCH[source], ({sid: cfg}, FIRST[source]), DEADLINE["one"])[sid]
                except Exception as e:
                    errors[sid] = f"refetch after a x{factor:.4f} move: {type(e).__name__}: {e}"[:200]
                    continue
            rows, held_out = checked(rows, known.get(sid, []), cfg)
            if held_out:
                errors[sid] = "held out: " + ", ".join(f"{d} {v:g} ({why})" for d, v, why in held_out[:3])
            nodes[sid] = store_rows(db, sid, rows, known.get(sid, []), now, source, cfg)
    derived = {sid: c for sid, c in series_cfg.items() if c["source"] == "derived"}
    have = backfill(clean(latest_obs(db), series_cfg), series_cfg) if derived else {}  # with each one's earlier years
    for sid, c in derived.items():
        rows = sources.DERIVED[c["formula"]](have.get(c["from"], []), c, known.get(sid, []))
        if rows:
            nodes[sid] = store_rows(
                db, sid, rows, known.get(sid, []), now, "derived", c, parents=raw_nodes(db, c["from"], series_cfg)
            )
    db.commit()
    return nodes, errors


def with_deadline(fn, args, seconds):
    """fn(*args) in a daemon thread, given up after `seconds`: a connection the far end stops answering
    can outlast every socket timeout, and must not hold up the day (the thread dies with the process)."""
    box = {}

    def work():
        try:
            box["v"] = fn(*args)
        except Exception as e:
            box["e"] = e

    th = threading.Thread(target=work, daemon=True)
    th.start()
    th.join(seconds)
    if th.is_alive():
        raise TimeoutError(f"no answer within {seconds} s")
    if "e" in box:
        raise box["e"]
    return box["v"]


RESCALED = 1e-5  # an adjusted close's earliest day re-read moved by more than this: the history before it moved too


def rescaled(rows, known):
    """For dividend-adjusted closes: the factor the earliest day the fetch shares with what is kept moved by, when it
    moved (the provider rescaled the history before a dividend); None otherwise. A dividend too small for
    moved_by_one_factor (under 0.5%), left alone, would leave a step where the re-read days meet the kept ones."""
    have = dict(known)
    shared = sorted(d for d, _ in rows if have.get(d))
    if not shared:
        return None
    r = dict(rows)[shared[0]] / have[shared[0]]
    return r if abs(r - 1) > RESCALED else None


def moved_by_one_factor(rows, known):
    """The factor when every day the fetch shares with what is kept moved by the same one (at least three
    days, the factor at least 0.5% off 1): a split or a rebased index changes the whole history, not only
    the days re-read. None otherwise."""
    have = dict(known)
    ratios = [v / have[d] for d, v in rows if d in have and have[d]]
    if len(ratios) < 3:
        return None
    m = statistics.median(ratios)
    if abs(m - 1) < 0.005 or any(abs(r / m - 1) > SPLIT for r in ratios):
        return None
    return m


def is_price(cfg):
    return cfg.get("positive", cfg["source"] in PRICES or cfg.get("format") in PRICE_ARCHIVES)


def checked(rows, known, cfg):
    """(the rows that pass the checks, [(day, value, why)] held out)."""
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    last = known[-1] if known else None
    price = is_price(cfg)
    jump = cfg.get("max_jump", MAX_JUMP if price else None)
    keep, out = [], []
    for d, v in rows:
        why = None
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            why = "not a number"
        elif d > tomorrow:
            why = "dated in the future"
        elif "min" in cfg and v < cfg["min"] or "max" in cfg and v > cfg["max"]:
            why = "out of range"
        elif price and v <= 0:
            why = "not above 0"
        elif jump and last and d > last[0] and last[1] > 0 and v > 0 and abs(math.log(v / last[1])) > jump:
            why = f"jump x{v / last[1]:.2f}"
        if why:
            out.append((d, v, why))
            continue
        keep.append((d, v))
        if not last or d > last[0]:
            last = (d, v)
    return keep, out


def store_rows(db, sid, rows, known, now, source, cfg, parents=()):
    have = dict(known)
    new = [(sid, d, v, now) for d, v in rows if d not in have or abs(have[d] - v) > REVISED * max(1.0, abs(v))]
    db.executemany("INSERT OR IGNORE INTO observations VALUES (?,?,?,?)", new)
    merged = sorted({**have, **{d: v for _, d, v, _ in new}}.items())
    if not merged:
        return None
    value = {
        "last_day": merged[-1][0],
        "last": merged[-1][1],
        "days": len(merged),
        "first_day": merged[0][0],
        "rows_hash": lineage.digest(merged),
    }
    meta = {
        "source": source,
        "symbol": cfg.get("symbol"),
        "fetched_at": now,
        "new_or_revised": len(new),
        "fetched_days": [rows[0][0], rows[-1][0]] if rows else None,
    }
    return lineage.record(db, "raw_snapshot", f"raw:{sid}", value, meta, parents, step="fetch")


def raw_nodes(db, name, series_cfg):
    """The raw snapshot nodes a series' history is made of: its own and those of what carries it back."""
    return [lineage.latest(db, f"raw:{n}") for n in sources_of(name, series_cfg)]


def clean(obs, series_cfg):
    """obs without the values the checks would refuse (some were stored before the checks existed) and
    without what comes before a series' history_from."""
    out = {}
    for sid, rows in obs.items():
        cfg = series_cfg.get(sid)
        if not cfg:
            out[sid] = rows
            continue
        start, price = cfg.get("history_from", ""), is_price(cfg)
        out[sid] = [
            (d, v)
            for d, v in rows
            if d >= start
            and not ("min" in cfg and v < cfg["min"])
            and not ("max" in cfg and v > cfg["max"])
            and not (price and v <= 0)
        ]
    return out


def actual_series(obs):
    """{series: Series of its actual observations (dated as published by the source)}."""
    out = {}
    for sid, rows in obs.items():
        if rows:
            s = pd.Series([v for _, v in rows], index=pd.to_datetime([d for d, _ in rows]))
            out[sid] = s[~s.index.duplicated(keep="last")].sort_index()
    return out


def load(db, series_cfg, today):
    """(frame, series as dated by their sources, raw observations): the first two cleaned and with backfills."""
    raw = latest_obs(db)
    obs = backfill(clean(raw, series_cfg), series_cfg)
    return frame(obs, series_cfg, until=today), actual_series(obs), raw


def frame(obs, series_cfg, until=None):
    """One DataFrame, a column per series, every calendar day from the first value to `until` (a value whose
    day + lag falls after it is not out yet), each value carried forward from when it was out for at most
    max_age days."""
    until = pd.Timestamp(until or pd.Timestamp.today().normalize())
    cols = {}
    for key, rows in obs.items():
        if not rows:
            continue
        s = pd.Series([v for _, v in rows], index=pd.to_datetime([d for d, _ in rows]))
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s.index = s.index + pd.Timedelta(days=series_cfg.get(key, {}).get("lag", 0))
        s = s[s.index <= until]
        if len(s):
            cols[key] = s
    if not cols:
        return pd.DataFrame()
    days = pd.date_range(min(s.index[0] for s in cols.values()), until, freq="D")
    out = {}
    for key, s in cols.items():
        limit = series_cfg.get(key, {}).get("max_age", 7)
        out[key] = s.reindex(days.union(s.index)).ffill(limit=limit).reindex(days)
    return pd.DataFrame(out)


def sources_of(name, series_cfg):
    """The series whose fetched values make up `name`'s history: itself and what carries it back (backfill)."""
    out = [name]
    for b in series_cfg.get(name, {}).get("backfill", []):
        for n in [b["series"]] if "series" in b else expr.names(b["expr"]):
            out += [x for x in sources_of(n, series_cfg) if x not in out]
    return out


JOIN_DAYS = 365  # the ratio of a join: its median over the first year both series have


def backfill(obs, series_cfg):
    """obs ({series: [(day, value)]}) with each series that has `backfill` (series.toml) carried back before
    its first value: by another series scaled to meet it (join = "ratio", the default: the median ratio over
    the first year both have, not one day's), or by an expression over the others taken as it is (join =
    "level"); each entry reaches further back than the one before. The stored observations stay as
    fetched: this is the history the signposts and base rates read."""
    out = dict(obs)
    for sid, c in series_cfg.items():
        for b in c.get("backfill", []):
            rows = out.get(sid)
            if not rows:
                break
            first = rows[0][0]
            if "series" in b:
                back = out.get(b["series"]) or []
            else:
                df = frame({k: out[k] for k in expr.names(b["expr"]) if out.get(k)}, series_cfg, until=first)
                v = expr.evaluate(b["expr"], df).dropna() if not df.empty else pd.Series(dtype=float)
                back = [(d.date().isoformat(), float(x)) for d, x in v.items()]
            early = [(d, x) for d, x in back if d < first]
            if not early:
                continue
            k = 1.0
            if b.get("join", "ratio") == "ratio":
                k = join_ratio(rows, back)
                if k is None:
                    continue
            out[sid] = [(d, x * k) for d, x in held(early, first)] + rows
    return out


def join_ratio(rows, back):
    """The series over what carries it back, the median over the first JOIN_DAYS of their overlap (each back
    value against the series' value on that day or the last before it); one day's ratio when they overlap
    on no more."""
    end = (pd.Timestamp(rows[0][0]) + pd.Timedelta(days=JOIN_DAYS)).date().isoformat()
    s = pd.Series({pd.Timestamp(d): v for d, v in rows if d <= end})
    ratios = [s.asof(pd.Timestamp(d)) / x for d, x in back if rows[0][0] <= d <= end and x]
    if ratios:
        return statistics.median(ratios)
    before = [x for d, x in back if d <= rows[0][0]]
    return rows[0][1] / before[-1] if before and before[-1] else None


def held(rows, until, most=35):
    """Sparse rows (monthly, weekly) as one a calendar day, each value held until the next or `until`, for at most
    `most` days: the series they carry back is daily, and a window's ends must land on a value."""
    out = []
    for (d, x), nxt in zip(rows, [r[0] for r in rows[1:]] + [until]):
        day, stop = pd.Timestamp(d), min(pd.Timestamp(nxt), pd.Timestamp(d) + pd.Timedelta(days=most))
        while day < stop:
            out.append((day.date().isoformat(), x))
            day += pd.Timedelta(days=1)
    return out

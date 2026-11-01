"""Questions: which instances are due, how one resolves from data, and its base rates and per-start-day
outcomes in history (for the base source and for fitting signposts). Pure: the pipeline stores them.

Instances: a path question is itself (spx_2y); a period question has one
per month or quarter (spx_month-2026-11, vix_quarter-2027Q1); a Kalshi series question one per event (fed-26DEC)."""

import json
import operator
from datetime import date, timedelta

import pandas as pd

OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
STEP = 7  # days between the start days sampled in history (path questions)


def matches(conds, v):
    for c in conds:
        name, op, num = c.split()
        if v.get(name) is None or not OPS[op](v[name], float(num)):
            return False
    return True


def bucket_of(buckets, v):
    return next(b["id"] for b in buckets if matches(b.get("when", []), v))


# ---- windows and periods


def path_vars(path, start_px):
    """The window's variables; prices scaled so the window starts at start_px."""
    rel = path / path.iloc[0]
    px = rel * start_px
    return {
        "dd": float((px / px.cummax() - 1).min()),
        "ret": float(rel.iloc[-1] - 1),
        "hi": float(rel.max() - 1),
        "lo": float(rel.min() - 1),
        "px_end": float(px.iloc[-1]),
        "px_hi": float(px.max()),
        "px_lo": float(px.min()),
    }


def period_bounds(every, day):
    """(first day, last day, label) of the month or quarter containing `day`."""
    if every == "month":
        first = day.replace(day=1)
        label = f"{day:%Y-%m}"
    else:
        q = (day.month - 1) // 3
        first = date(day.year, q * 3 + 1, 1)
        label = f"{day.year}Q{q + 1}"
    nxt = date(
        first.year + (first.month + (1 if every == "month" else 3) - 1) // 12,
        (first.month + (1 if every == "month" else 3) - 1) % 12 + 1,
        1,
    )
    return first, nxt - timedelta(days=1), label


def period_vars(s, measure, first, last):
    """The period's measure from series s (actual observations), None while it is not complete."""
    if measure == "yoy_change":  # monthly series: the month before `first` vs the month before that
        m1 = pd.Timestamp(first) - pd.offsets.MonthBegin(1)
        need = [m1, m1 - pd.DateOffset(months=1), m1 - pd.DateOffset(years=1), m1 - pd.DateOffset(months=13)]
        if not all(d in s.index for d in need):
            return None
        a, b, c, d = (s[x] for x in need)
        return {"chg": float((a / c - 1) - (b / d - 1)), "yoy": float(a / c - 1)}
    before = s[s.index < pd.Timestamp(first)]
    inside = s[(s.index >= pd.Timestamp(first)) & (s.index <= pd.Timestamp(last))]
    after = s[s.index > pd.Timestamp(last)]
    if before.empty or inside.empty or after.empty:  # complete once a later value exists
        return None
    return {"ret": float(inside.iloc[-1] / before.iloc[-1] - 1), "end": float(inside.iloc[-1])}


# ---- registration


def due(cfg, today, existing, series):
    """New instances to register today: [(id, title, kind, opened, closes, buckets, spec)]. A periodic question with
    `until` (one the LLM adopted) opens no instance after that day."""
    out = []
    for qid, q in cfg.items():
        if qid == "weights":
            continue
        kind = q["kind"]
        if kind == "path" and qid not in existing:
            s = series.get(q["series"])
            if s is None or s.empty:
                continue
            spec = {**q, "template": qid, "start_day": s.index[-1].date().isoformat(), "start_px": float(s.iloc[-1])}
            out.append((qid, q["title"], kind, today.isoformat(), q["closes"], q["buckets"], spec))
        elif kind == "rolling":  # a path question opened every day over the next `days`
            iid = f"{qid}-{today.isoformat()}"
            s = series.get(q["series"])
            if iid in existing or s is None or s.empty:
                continue
            closes = (today + timedelta(days=q["days"])).isoformat()
            spec = {
                **q,
                "template": qid,
                "rolling": True,
                "closes": closes,
                "start_day": s.index[-1].date().isoformat(),
                "start_px": float(s.iloc[-1]),
            }
            out.append(
                (iid, f"{q['title']}: from {today:%d %b}", "path", today.isoformat(), closes, q["buckets"], spec)
            )
        elif kind == "period":
            if q.get("until") and today.isoformat() > q["until"]:  # one the LLM adopted stops after a year
                continue
            for d in (today, today + timedelta(days=7)):
                first, last, label = period_bounds(q["every"], d)
                iid = f"{qid}-{label}"
                if iid in existing or (today - first).days > 3:
                    continue
                spec = {**q, "template": qid, "first": first.isoformat(), "last": last.isoformat()}
                out.append(
                    (iid, f"{q['title']}: {label}", kind, today.isoformat(), last.isoformat(), q["buckets"], spec)
                )
                existing = {*existing, iid}
    return out


def due_kalshi(cfg, today, existing, events):
    """Kalshi instances: `events` is {series: [(event ticker, day)]} fetched by the caller."""
    out = []
    for qid, q in cfg.items():
        if qid == "weights" or q.get("kind") != "kalshi":
            continue
        if "event" in q:
            if qid not in existing:
                out.append(
                    (
                        qid,
                        q["title"],
                        "kalshi",
                        today.isoformat(),
                        None,
                        q["buckets"],
                        {**q, "template": qid, "events": [q["event"]]},
                    )
                )
            continue
        for ticker, day in events.get(q["series"], [])[: q.get("next", 3)]:
            iid = f"{qid}-{ticker.rsplit('-', 1)[-1]}"
            if iid not in existing:
                out.append(
                    (
                        iid,
                        f"{q['title']}: {day}",
                        "kalshi",
                        today.isoformat(),
                        day,
                        q["buckets"],
                        {**q, "template": qid, "events": [ticker]},
                    )
                )
    return out


# ---- resolution


def market_bucket_probs(q, markets):
    """Bucket probabilities from Kalshi markets {ticker: (label, p, status, result)}; None without prices. A bucket
    one of whose markets has no price (a wide spread and no trade) is unknown, not 0: a binary question takes it as
    the rest of the other; a question of more outcomes then has no market view."""
    raw, unpriced = {}, False
    for b in q["buckets"]:
        ps = [m[1] for t, m in markets.items() if any(t == k or t.endswith("-" + k) for k in b.get("markets", []))]
        raw[b["id"]] = sum(ps) if ps and None not in ps else None
        unpriced |= bool(ps) and None in ps
    if len(q["buckets"]) > 2 and unpriced:
        return None
    if len(q["buckets"]) == 2 and None in raw.values():  # a binary market: the other bucket is the rest
        known = next((k for k, v in raw.items() if v is not None), None)
        if known is None:
            return None
        other = next(k for k in raw if k != known)
        raw[other] = max(0.0, 1 - raw[known])
    total = sum(v for v in raw.values() if v)
    return {k: (v or 0) / total for k, v in raw.items()} if total else None


def kalshi_result(q, markets):
    """The settled bucket, or None while Kalshi has not settled the event."""
    settled = {
        t: m for t, m in markets.items() if m[2] in ("settled", "finalized", "determined") and m[3] in ("yes", "no")
    }
    if not settled or len(settled) < len(markets):
        return None
    for b in q["buckets"]:
        if any(
            m[3] == "yes"
            for t, m in settled.items()
            if any(t == k or t.endswith("-" + k) for k in b.get("markets", []))
        ):
            return b["id"]
    empty = [b["id"] for b in q["buckets"] if not b.get("markets")]
    return empty[0] if empty else None


def resolve(q, series, today):
    """(bucket, how) once the outcome is known from data, else None. Kalshi questions resolve in run.py."""
    spec = q["spec"]
    s = series.get(spec["series"])
    if s is None or s.empty:
        return None
    if q["kind"] == "path":
        closes = pd.Timestamp(spec["closes"])
        if s.index[-1] <= closes:  # a later value: the closing day's own is complete
            return None
        path = s[(s.index >= pd.Timestamp(spec["start_day"])) & (s.index <= closes)]
        v = path_vars(path, spec["start_px"])
        return bucket_of(q["buckets"], v), json.dumps({k: round(x, 4) for k, x in v.items()})
    if q["kind"] == "period":
        v = period_vars(s, spec["measure"], date.fromisoformat(spec["first"]), date.fromisoformat(spec["last"]))
        if v is None:
            return None
        return bucket_of(q["buckets"], v), json.dumps({k: round(x, 4) for k, x in v.items()})
    return None


# ---- history


def outcomes_now(q, s, today, usrec=None):
    """The outcomes as the question stands today, for its base rate: once it is under way, what has happened
    is kept and history supplies only the rest. A path question: the path so far, then each historical
    window of the days left, scaled on (a fall already taken stays taken). A period question measured on
    its return or its end value: the value now, moved as the series moved over the days left. Otherwise,
    and before it starts, as history_outcomes."""
    spec, kind = q["spec"], q["kind"]
    if s is None or kind not in ("path", "period") or spec.get("measure") == "yoy_change":
        return history_outcomes(q, s, usrec)
    if spec.get("history_from"):
        s = s[s.index >= pd.Timestamp(spec["history_from"])]
    now = pd.Timestamp(today)
    if kind == "path":
        start, closes = pd.Timestamp(spec["start_day"]), pd.Timestamp(spec["closes"])
        done = s[(s.index >= start) & (s.index <= now)]
        if now <= start or len(done) < 2:
            return history_outcomes(q, s, usrec)
        done = done / done.iloc[0]
        left = max((closes - done.index[-1]).days, 1)
        hist = s[s.index <= now]
        out, t = {}, hist.index[0]
        while t <= hist.index[-1] - pd.Timedelta(days=left):
            t0 = hist.index[hist.index.searchsorted(t)]
            rest = hist[(hist.index > t0) & (hist.index <= t0 + pd.Timedelta(days=left))] / hist[t0]
            path = pd.Series([*done.values, *(done.iloc[-1] * rest.values)])
            out[t0] = bucket_of(q["buckets"], path_vars(path, spec["start_px"]))
            t = t0 + pd.Timedelta(days=STEP)
        return out, min(1.0, STEP / left)
    first, last = pd.Timestamp(spec["first"]), pd.Timestamp(spec["last"])
    before, inside = s[s.index < first], s[(s.index >= first) & (s.index <= now)]
    if now < first or before.empty or inside.empty:
        return history_outcomes(q, s, usrec)
    left = max((last - inside.index[-1]).days, 1)
    v_now, v_before = inside.iloc[-1], before.iloc[-1]
    out, t = {}, s.index[0]
    while t <= now - pd.Timedelta(days=left):
        t0 = s.index[s.index.searchsorted(t)]
        moved = s.asof(t0 + pd.Timedelta(days=left)) / s[t0]
        end = v_now * moved
        out[t0] = bucket_of(q["buckets"], {"ret": float(end / v_before - 1), "end": float(end)})
        t = t0 + pd.Timedelta(days=STEP)
    return out, min(1.0, STEP / left)


def history_outcomes(q, s, usrec=None):
    """Outcome bucket by start day in history ({Timestamp: bucket}) and the effective sample size per start
    (overlapping windows count as one per window length), as the question stood when it opened: for the
    signposts' fits and the calibration. None when the question has no history form."""
    spec, kind = q["spec"], q["kind"]
    if s is not None and spec.get("history_from"):
        s = s[s.index >= pd.Timestamp(spec["history_from"])]
    if kind == "path":
        start = pd.Timestamp(spec["start_day"])
        horizon = (pd.Timestamp(spec["closes"]) - start).days
        s = s[s.index <= start]
        out = {}
        last_start = s.index[-1] - pd.Timedelta(days=horizon)
        t = s.index[0]
        while t <= last_start:
            i = s.index.searchsorted(t)
            t0 = s.index[i]
            path = s[(s.index >= t0) & (s.index <= t0 + pd.Timedelta(days=horizon))]
            out[t0] = bucket_of(q["buckets"], path_vars(path, spec["start_px"]))
            t = t0 + pd.Timedelta(days=STEP)
        return out, STEP / horizon
    if kind == "period":
        out = {}
        first = s.index[0].date()
        d = date(first.year + 1, 1, 1) if spec["measure"] == "yoy_change" else first
        end = date.fromisoformat(spec["first"])
        while True:
            f, last, _ = period_bounds(spec["every"], d)
            if f >= end:
                break
            v = period_vars(s, spec["measure"], f, last)
            if v is not None:
                out[pd.Timestamp(f) - pd.Timedelta(days=1)] = bucket_of(q["buckets"], v)
            d = last + timedelta(days=1)
        return out, 1.0
    if q["spec"].get("base_rate") == "recession_start" and usrec is not None:
        # a year that opens in a recession cannot see one start: it is left out, as the question is asked outside one
        years, inside = {}, set()
        prev = 0
        for d, v in usrec.items():
            if d.year >= 1950:
                years.setdefault(d.year, False)
                if d.month == 1 and prev == 1:
                    inside.add(d.year)
                if v == 1 and prev == 0:
                    years[d.year] = True
            prev = v
        last_full = usrec.index[-1].year - 2  # NBER dates recessions late: the last two years are not settled
        return {
            pd.Timestamp(y, 1, 1): ("yes" if hit else "no")
            for y, hit in years.items()
            if y <= last_full and y not in inside
        }, 1.0
    return None, None


def base_rate(q, outcomes, n_per):
    """{bucket: frequency}, smoothed by one observation spread evenly; and the effective n."""
    if not outcomes:
        return None, 0
    counts = {b["id"]: 0 for b in q["buckets"]}
    for b in outcomes.values():
        counts[b] += 1
    return rate_from_counts(counts, n_per)


def rate_from_counts(counts, n_per):
    n = sum(counts.values())
    n_eff = n * n_per
    return {k: (c / n * n_eff + 1 / len(counts)) / (n_eff + 1) for k, c in counts.items()}, n_eff


KEY = 4  # the week's key questions


def key(probs, n=KEY, templates=None):
    """The questions whose final probability stands furthest from history's base rate (half the sum of the buckets'
    differences), furthest first: where today's world says most that is not usual; one a template (a rolling
    question's instances are one question). probs: {question: (final {bucket: p}, base {bucket: p})}; templates:
    {question: its template}. [(question, distance)]."""
    best = {}
    for q, (final, base) in probs.items():
        if final and base:
            keys = set(final) | set(base)
            x = 0.5 * sum(abs(final.get(k, 0.0) - base.get(k, 0.0)) for k in keys)
            t = (templates or {}).get(q, q)
            if t not in best or x > best[t][1]:
                best[t] = (q, x)
    return sorted(best.values(), key=lambda x: -x[1])[:n]

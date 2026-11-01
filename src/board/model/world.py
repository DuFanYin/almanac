"""The world: from today on, a weight on every start day of history (weekly from the factors' history_from), saying
how likely the next days are to look like the days that followed it. Everything is read off the same weights at any
length: the factors, the asset classes, every question (by its own outcome after each start day).

The data moves the weights once. Each signpost reads today on its dial (its rank among its own history, toward red
up); each of its expectations that the measures agree with (model/timescale.py judge) pushes its factor over the
stretches the expectation covers, by the measured effect shrunk by its noise, times today's dial: a push on the move
(up, down) or on the spread (wider, narrower: volatility persists). Signposts of one group (measuring the same
thing) are averaged, each over the stretches its expectations cover, so one reading in the middle is a vote for usual;
across groups the strongest one way counts whole, each further one half the one before (evidence that moves together
is counted less), the two ways net, a stretch's push at most CAP_SD on the move and within SPREAD_CAP on the variance.
Over a length L the pushes add up to a shift of each factor's mean (in its standard deviations over L) and a
multiple of its variance (within VAR_RANGE). Each factor's push is its own tilt of history, the nearest to even
(least relative entropy) that moves that factor so alone, and the tilts are multiplied together: a push is evidence
about its factor, and every factor, pushed or not, also follows the others as it moved with them in history (a factor
pushed a little moves a little, not pinned where its push alone would put it).
Never on fewer than MIN_ESS of the start days' worth of evidence: past that, every shift is scaled down until it
holds.

Pure: series and the measures in, weights out."""

import zlib

import numpy as np
import pandas as pd

from . import factors as fx
from . import timescale as ts

CAP_SD = 1.0  # a stretch's push on a factor's move, all signposts together, at most this many standard deviations
SPREAD_CAP = (-0.5, 1.0)  # a stretch's push on a factor's variance, all together: from half to twice its usual
VAR_RANGE = (0.5, 2.0)  # a factor's variance over a window, against its usual: within these
HALF = 0.5  # each further signpost pushing the same way counts this much of the one before
MIN_ESS = 0.25  # the effective share of the start days the data's weights keep at least
NEAREST = 3.5  # days: a question's start day is matched to the world's nearest within this
STALE = 45  # days: a signpost whose last value is older than this reads nothing today
PROPOSED_SHARE = 0.5  # a signpost the LLM proposed pushes half as much (it has read the history it is measured on)


def dial_today(values, red_rule, first, today, step=ts.STEP):
    """Today's reading of a signpost on its dial: the rank of its latest value among its values on the start days
    sampled as the measures sample them (model/timescale.py measure), from -0.5 to 0.5, turned so toward red is up;
    None without a value in the last STALE days."""
    v = values.dropna()
    if v.empty or (pd.Timestamp(today) - v.index[-1]).days > STALE:
        return None
    t = pd.date_range(max(pd.Timestamp(first), v.index[0]), v.index[-1], freq=f"{step}D")
    hist = v.reindex(t, method="ffill").dropna().to_numpy(dtype=float)
    if not len(hist):
        return None
    now = float(v.iloc[-1])
    r = (np.sum(hist < now) + 0.5 * np.sum(hist == now)) / len(hist) - 0.5
    return float(-r if red_rule.strip().startswith("<") else r)


def sizes(report, segs):
    """{"move": {factor: [signed size per stretch]}, "spread": the same} of one signpost's measures (timescale.json's
    entry): over the stretches each expectation that agrees covers (mostly within its days), its curve there shrunk by
    its noise (timescale.shrunk), kept only the way expected; 0 elsewhere."""
    out = {"move": {}, "spread": {}}
    for e in report.get("judged") or []:
        if e["verdict"] != "agrees":
            continue
        c = ((report["factors"].get(e["factor"]) or {}).get(e["on"]) or {}).get("curve") or []
        sign = ts.WAYS[e["way"]]
        a, b = e["days"]
        kind = "spread" if e["way"] in ts.SPREAD else "move"
        arr = out[kind].setdefault(e["factor"], np.zeros(len(segs)))
        for r in c:
            if min(b, r["to"]) - max(a, r["from"]) < (r["to"] - r["from"]) / 2:
                continue
            j = segs.index((r["from"], r["to"]))
            arr[j] += sign * max(0.0, sign * ts.shrunk(r))
    return out


SCALE = 2.0  # a text reading's scale runs from -SCALE to SCALE (config/readings.toml)


def dial_scale(values, red_rule, today, most=SCALE):
    """A signpost on a text reading, too new to rank against its own history, read on its own scale: its latest value
    from -most..most to -0.5..0.5, turned so toward red is up; None without a value in the last STALE days."""
    v = values.dropna()
    if v.empty or (pd.Timestamp(today) - v.index[-1]).days > STALE:
        return None
    r = float(np.clip(v.iloc[-1] / (2 * most), -0.5, 0.5))
    return -r if red_rule.strip().startswith("<") else r


def typical(judged):
    """{"move", "spread"}: the median size of the measured expectations that agree, of each kind (judge's size, per
    unit of dial): what an expectation not yet measured is taken to be worth."""
    out = {}
    for kind in ("move", "spread"):
        xs = [e["size"] for e in judged if e["verdict"] == "agrees" and (e["way"] in ts.SPREAD) == (kind == "spread")]
        out[kind] = float(np.median(xs)) if xs else 0.0
    return out


def assumed(expect, segs, size):
    """{"move", "spread": {factor: [signed size per stretch]}} of expectations not yet measured: each the typical size
    of its kind (typical()) over the stretches mostly within its days, its way; and their judged form ("assumed")."""
    out, judged = {"move": {}, "spread": {}}, []
    for e in expect:
        kind = "spread" if e["way"] in ts.SPREAD else "move"
        a, b = e["days"]
        arr = out[kind].setdefault(e["factor"], np.zeros(len(segs)))
        rows = [j for j, (x, y) in enumerate(segs) if min(b, y) - max(a, x) >= (y - x) / 2]
        for j in rows:
            arr[j] += ts.WAYS[e["way"]] * size[kind]
        judged.append(
            {
                **e,
                "verdict": "assumed",
                "size": round(size[kind], 3),
                "on": "the LLM's",
                "stretches": [list(segs[j]) for j in rows],
            }
        )
    return out, judged


def combine(xs, lo=-CAP_SD, hi=CAP_SD):
    """Pushes on one factor in one stretch from different groups, together: each way the strongest whole and each
    further one HALF the one before, the two ways netted, within lo and hi."""
    up = sorted((x for x in xs if x > 0), reverse=True)
    down = sorted((-x for x in xs if x < 0), reverse=True)
    total = sum(x * HALF**i for i, x in enumerate(up)) - sum(x * HALF**i for i, x in enumerate(down))
    return float(np.clip(total, lo, hi))


def pushes(signals, segs):
    """Today's pushes ({"move": {factor: array per stretch, in the stretch's standard deviations}, "spread": {factor:
    array, in its usual variance}}) and each signpost's own part before combining ({signpost: the same}). signals:
    [{"id", "group", "dial", "share", "sizes"}]: a group's signposts averaged in each stretch over those whose
    expectations cover it (a reading in the middle pushes little and counts), groups combined (combine())."""
    part = {}
    for s in signals:
        if s["dial"] is None:
            continue
        part[s["id"]] = {
            kind: {f: s["share"] * s["dial"] * np.asarray(v) for f, v in by.items()} for kind, by in s["sizes"].items()
        }
    out = {}
    for kind, (lo, hi) in (("move", (-CAP_SD, CAP_SD)), ("spread", SPREAD_CAP)):
        out[kind] = {}
        for f in {f for p in part.values() for f in p.get(kind, {})}:
            by_group = {}
            for s in signals:
                x = part.get(s["id"], {}).get(kind, {}).get(f)
                if x is None:
                    continue
                has = (np.asarray(s["sizes"][kind][f]) != 0).astype(float)
                g = s.get("group") or s["id"]
                total, n = by_group.get(g, (0.0, 0.0))
                by_group[g] = (total + x * has, n + has)
            mean = [np.divide(t, n, out=np.zeros(len(segs)), where=n > 0) for t, n in by_group.values()]
            out[kind][f] = np.array([combine([v[j] for v in mean], lo, hi) for j in range(len(segs))])
    return out, part


def shift(push, segs, length, sd_seg, sd_len, how):
    """A factor's mean shifted over a window of `length` days from today, in its standard deviations over windows
    that long: each stretch's push (in its own standard deviations, sd_seg) back in the factor's units and added up
    over the part of the stretch inside the window; a yearly rate (ann) averaged over the window by days instead."""
    total = 0.0
    for j, (a, b) in enumerate(segs):
        over = max(0, min(b, length) - a)
        if not over or not push[j] or not np.isfinite(sd_seg[j]):
            continue
        moved = push[j] * sd_seg[j]
        total += moved * over / length if how == "ann" else moved * over / (b - a)
    return total / sd_len if sd_len else 0.0


def spread(push, segs, length, sd_seg, sd_len):
    """A factor's variance over a window of `length` days from today, against its usual: each stretch's push (in its
    usual variance) back in the factor's units, over the part of the stretch inside the window, added to the usual
    (moves over stretches add up in variance), within VAR_RANGE."""
    extra = 0.0
    for j, (a, b) in enumerate(segs):
        over = max(0, min(b, length) - a)
        if not over or not push[j] or not np.isfinite(sd_seg[j]):
            continue
        extra += push[j] * sd_seg[j] ** 2 * over / (b - a)
    return float(np.clip(1 + extra / sd_len**2, *VAR_RANGE)) if sd_len else 1.0


def ess(w):
    """The effective number of start days behind weights w."""
    return float(w.sum() ** 2 / (w**2).sum()) if len(w) and (w**2).sum() else 0.0


def tilt(Z, target, w0=None, iters=60):
    """The weights nearest to w0 (least relative entropy; even without it) under which the columns of Z have the
    means `target`: w ∝ w0 · exp(Z λ), λ by Newton's method on the dual, halving the step until it helps. (w, λ)."""
    n, k = Z.shape
    w0 = np.full(n, 1 / n) if w0 is None else w0 / w0.sum()
    lw0 = np.log(np.maximum(w0, 1e-300))
    lam = np.zeros(k)

    def at(lam):
        a = lw0 + Z @ lam
        top = a.max()
        e = np.exp(a - top)
        return e / e.sum(), top + np.log(e.sum()) - lam @ target

    w, obj = at(lam)
    for _ in range(iters):
        g = w @ Z - target
        if np.abs(g).max() < 1e-10:
            break
        m = w @ Z
        C = (Z * w[:, None]).T @ Z - np.outer(m, m) + 1e-10 * np.eye(k)
        d = -np.linalg.solve(C, g)
        t = 1.0
        while t > 1e-6:
            w1, obj1 = at(lam + t * d)
            if obj1 <= obj:
                break
            t /= 2
        lam, w, obj = lam + t * d, w1, obj1
    return w, lam


def tilted(Z, mu, var=None, w0=None, min_ess=MIN_ESS):
    """The weights with each column of Z (standardised) pushed: its mean by mu, its variance by the multiple var (1:
    as it is), from where w0 (even without it) has them. Each pushed column its own tilt (alone, the nearest weights
    that move it so; one whose variance moves keeps its mean), the tilts multiplied, so the columns move together as
    they did in history; a column with neither is not pushed and follows. All the tilts scaled down by one share
    until the weights keep min_ess of the start days' worth (by halving). (weights, the share kept)."""
    n, k = Z.shape
    var = np.ones(k) if var is None else np.asarray(var)
    base = np.full(n, 1 / n) if w0 is None else w0 / w0.sum()
    m0 = base @ Z
    v0 = base @ Z**2 - m0**2
    total = np.zeros(n)
    for j in range(k):
        wide = abs(var[j] - 1) > 1e-9
        if abs(mu[j]) <= 1e-12 and not wide:
            continue
        m = m0[j] + mu[j]
        F = np.column_stack([Z[:, j], Z[:, j] ** 2]) if wide else Z[:, [j]]
        goal = np.array([m, v0[j] * var[j] + m**2]) if wide else np.array([m])
        total += F @ tilt(F, goal, base)[1]
    if not total.any():
        return base, 1.0
    lw0 = np.log(np.maximum(base, 1e-300))

    def at(s):
        a = lw0 + s * total
        e = np.exp(a - a.max())
        return e / e.sum()

    if ess(at(1.0)) >= min_ess * n:
        return at(1.0), 1.0
    lo, hi = 0.0, 1.0
    for _ in range(20):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ess(at(mid)) >= min_ess * n else (lo, mid)
    return at(lo), lo


COARSE = 0.8  # a factor published less often than this share of a window's days says nothing over it


def measurable(fcfg, series, length):
    """The factors' config with only those measurable over a window of `length` days: a series published every month
    has no move within a week (left out there, not the weeks it does not span)."""
    keep = {}
    for k, f in fcfg["factor"].items():
        arr = fx.arrays(series, f["series"])
        if arr is not None and length >= COARSE * fx.spacing(arr):
            keep[k] = f
    return {**fcfg, "factor": keep}


class World:
    """The data's world: the factors (fcfg, series), today's pushes (pushes(): {"move", "spread"}) over the stretches
    `segs`. weights(length) gives the start days of every window of that length the factors cover and their weights
    (cached), with the factors over each (standardised) and the share of the pushes kept."""

    def __init__(self, fcfg, series, segs, push):
        self.fcfg, self.series, self.segs, self.push = fcfg, series, [tuple(s) for s in segs], push
        self.first = fcfg.get("history_from", "1971-03-01")
        self._sd, self._at = {}, {}

    def sd_seg(self, f):
        """A factor's standard deviation over windows as long as each stretch, in history."""
        if f not in self._sd:
            spec = self.fcfg["factor"][f]
            out = []
            for a, b in self.segs:
                _, h = fx.history(spec, self.series, b - a, self.first)
                out.append(float(np.std(h)) if len(h) >= ts.MIN_STARTS else np.nan)
            self._sd[f] = np.array(out)
        return self._sd[f]

    def mu(self, length, sd, names):
        """Each factor's mean shift over `length` days (in its standard deviations over windows that long) and its
        variance against its usual, for the factors `names`; sd: those standard deviations. A yearly rate (ann) keeps
        its spread: a window's rate is an average, its spread not the stretches' added up."""
        mv, sp = self.push.get("move", {}), self.push.get("spread", {})
        mu, var = [], []
        for j, f in enumerate(names):
            spec = self.fcfg["factor"][f]
            mu.append(shift(mv[f], self.segs, length, self.sd_seg(f), sd[j], spec["measure"]) if f in mv else 0.0)
            ok = f in sp and spec["measure"] != "ann"
            var.append(spread(sp[f], self.segs, length, self.sd_seg(f), sd[j]) if ok else 1.0)
        return np.array(mu), np.array(var)

    def weights(self, length):
        """{"starts" (int64 ns), "names" (the factors measurable over it), "X" (start days × those factors, in their
        units), "Z" (standardised), "mu", "var" (the pushes), "w", "scale", "ess", "moved" and "spread" (each
        factor's mean shift in standard deviations and its variance's multiple under w: its push and what it followed)}
        at a window of `length` days."""
        if length not in self._at:
            fcfg = measurable(self.fcfg, self.series, length)
            names = list(fcfg["factor"])
            t, vals = fx.matrix(fcfg, self.series, length, self.first)
            X = np.column_stack([vals[k] for k in names]) if len(t) else np.zeros((0, len(names)))
            mean, sd = X.mean(0), X.std(0)
            sd[sd == 0] = 1
            Z = (X - mean) / sd
            mu, var = self.mu(length, sd, names)
            w, scale = tilted(Z, mu, var)
            moved = w @ Z
            self._at[length] = {
                "days": length,
                "moved": moved,
                "spread": w @ Z**2 - moved**2,
                "starts": t,
                "names": names,
                "X": X,
                "Z": Z,
                "mu": mu,
                "var": var,
                "w": w,
                "scale": scale,
                "ess": ess(w),
            }
        return self._at[length]


def matched(starts, keys):
    """For each day in keys (Timestamps), the index of the nearest start day (int64 ns) within NEAREST days, or -1."""
    k = pd.DatetimeIndex(keys).values.astype("datetime64[ns]").astype(np.int64)
    i = np.clip(np.searchsorted(starts, k), 1, len(starts) - 1) if len(starts) > 1 else np.zeros(len(k), dtype=int)
    if len(starts) > 1:
        i = np.where(np.abs(starts[i - 1] - k) < np.abs(starts[i] - k), i - 1, i)
    near = np.abs(starts[i] - k) <= NEAREST * fx.DAY if len(starts) else np.zeros(len(k), dtype=bool)
    return np.where(near, i, -1)


def read(outcomes, n_per, starts, w, keys):
    """A question's probabilities off the world: its outcome after each start day of history ({Timestamp: bucket}),
    each matched to the world's nearest start day and weighted by it; smoothed by one observation spread evenly over
    its effective windows (the weights' effective number of start days times n_per, the independent share of one).
    ({bucket: p}, effective windows, {bucket: [lo, hi]} the 90% band from those windows); (None, 0, {}) when no
    start day matches."""
    days = list(outcomes)
    idx = matched(starts, days)
    ok = idx >= 0
    if not ok.any():
        return None, 0.0, {}
    ww = w[idx[ok]]
    got = [outcomes[d] for d, o in zip(days, ok) if o]
    total = ww.sum()
    if not total:
        return None, 0.0, {}
    n_eff = ess(ww) * n_per
    p = {}
    for k in keys:
        f = float(ww[[g == k for g in got]].sum() / total)
        p[k] = (f * n_eff + 1 / len(keys)) / (n_eff + 1)
    band = {}
    for k, x in p.items():
        half = 1.645 * np.sqrt(x * (1 - x) / max(n_eff, 1.0))
        band[k] = [max(0.0, x - half), min(1.0, x + half)]
    return p, n_eff, band


# ---- periods made for a scenario history has too few windows of (model/scenarios.py)

REACH = 0.5  # a scenario's range is kept within history's span widened by this share of it either side
LIKE = 20  # a scenario history has too few windows of is made from this many of the nearest


def made(a, names, ranges, pick):
    """A scenario history has too few windows of: its windows made from the nearest start days (pick, indexes into
    a). Each factor of its ranges that a window has outside them is moved in, spread across the range by where the
    windows stood among themselves; every other factor moves with those as they moved together in history (least
    squares over all of a's start days), so a fall in stocks made deeper brings what came with falls. {factor: each
    window's move, in the factor's units}."""
    X = a["X"]
    P = X[pick]
    shift = np.zeros((len(pick), len(names)))
    ri = [names.index(k) for k in ranges if k in names]
    for j in ri:
        lo, hi = ranges[names[j]]
        x = P[:, j]
        rank = np.argsort(np.argsort(x, kind="stable"), kind="stable")
        goal = lo + (rank + 0.5) / len(x) * (hi - lo)
        shift[:, j] = np.where((x < lo) | (x > hi), goal - x, 0.0)
    oi = [j for j in range(len(names)) if j not in ri]
    if ri and oi and len(X) > len(names):
        Xc = X - X.mean(0)
        B = np.linalg.lstsq(Xc[:, ri], Xc[:, oi], rcond=None)[0]
        shift[:, oi] = shift[:, ri] @ B
    return {k: shift[:, j] for j, k in enumerate(names)}


def made_ids(sid, n):
    """A made scenario's windows' ids: below 0 (no start day is), the same for the scenario every day."""
    base = (zlib.crc32(str(sid).encode()) % 10**9) * 1000
    return -(base + np.arange(1, n + 1, dtype=np.int64))


def extended(a, made_sets):
    """a (World.weights at its length) with the windows made for scenarios appended, at no data weight: each its
    source start day's factors moved by the shift. made_sets: [{"starts" (made_ids), "from" (the source start days),
    "shift" (made())}]. Adds "real" (how many rows are history's) and "src" (each made row's source row); a itself
    without any."""
    n = len(a["starts"])
    rows, ids, src = [], [], []
    for m in made_sets:
        if not n:
            continue
        pos = np.clip(np.searchsorted(a["starts"], m["from"]), 0, n - 1)
        for i in np.flatnonzero(a["starts"][pos] == m["from"]):
            x = a["X"][pos[i]].copy()
            for j, k in enumerate(a["names"]):
                if k in m["shift"]:
                    x[j] += m["shift"][k][i]
            rows.append(x)
            ids.append(m["starts"][i])
            src.append(pos[i])
    if not rows:
        return a
    mean, sd = a["X"].mean(0), a["X"].std(0)
    sd[sd == 0] = 1
    M = np.array(rows)
    return {
        **a,
        "starts": np.concatenate([a["starts"], np.array(ids, dtype=np.int64)]),
        "X": np.vstack([a["X"], M]),
        "Z": np.vstack([a["Z"], (M - mean) / sd]),
        "w": np.concatenate([a["w"], np.zeros(len(rows))]),
        "real": n,
        "src": np.array(src),
    }


def padded(w, a):
    """Weights over a's real start days, with a made period's (extended) at none."""
    return np.concatenate([w, np.zeros(len(a["starts"]) - len(w))])


# ---- reading the world at a length

QUARTILES = (0.25, 0.5, 0.75)


def edges(x):
    """History's quartiles of a factor over a length (its values at every start day, unweighted)."""
    return [float(v) for v in np.quantile(x, QUARTILES)]


def bucket(x, cuts):
    """The quartile each value falls in (0..3)."""
    return np.searchsorted(np.asarray(cuts), x, side="right")


def quartile_probs(x, w, cuts):
    """The weight in each of history's quartiles."""
    return [float(v) for v in np.bincount(bucket(x, cuts), weights=w, minlength=len(cuts) + 1)]

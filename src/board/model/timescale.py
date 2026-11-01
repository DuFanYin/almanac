"""How long a signpost's news lasts: for each signpost and factor, how far the factor moved, more or less than usual,
in each stretch of the next three years (the first week, the week after, ... the third year: stretches that
lengthen evenly in log) from the days the signpost was red (and, read as a dial, the more it read toward red). Its
news lasts as far as the last stretch it still moves: by stretch, not over the whole window from the day, as a move
in the first month stays in every longer window's total and would look as if it lasted. Where it is strongest and
whether it turns the other way. One test per signpost and factor over every stretch at once, so measuring many
stretches is not many chances to find something. And the signpost's own pace: how long its red spells last, how soon
its value forgets where it was.

Measured on all of history at once (not walked forward): it says how long, not yet how much to move a forecast.
Pure: series in, numbers out."""

import numpy as np
import pandas as pd

from . import factors as fx

SHORTEST, LONGEST, POINTS = 7, 1095, 9  # the stretches' ends: this many from a week to three years, evenly in log
STEP = 7  # days between the start days sampled
BLOCK_MIN = 365  # days in a bootstrap block at least (else the stretch's own): red spells cluster (2008, 2020), and
# overlapping windows are one piece of evidence, so evidence is counted by periods of time
MIN_BLOCKS = 8  # such blocks of data a stretch needs to be measured
BOOT = 1000  # bootstrap draws
EPISODE_GAP = 30  # red days further apart than this begin a new red spell
MIN_EPISODES = 5  # red spells needed to say anything about red
MIN_STARTS = 260  # start days (five years of weeks) a stretch needs to be measured


def lengths(lo=SHORTEST, hi=LONGEST, n=POINTS):
    """The stretches' ends (days from the day), from lo to hi evenly in log."""
    return [int(round(lo * (hi / lo) ** (i / (n - 1)))) for i in range(n)]


def stretches(ends=None):
    """[(from, to)] days from the day: the first week, then each end to the next."""
    ends = ends or lengths()
    return list(zip([0, *ends[:-1]], ends))


def spells(red, step=STEP, gap=EPISODE_GAP):
    """[(first index, last index)] of the red spells in a sampled boolean array (gaps of up to `gap` days joined)."""
    idx = np.flatnonzero(red)
    if not len(idx):
        return []
    out, first, last = [], idx[0], idx[0]
    for i in idx[1:]:
        if (i - last) * step > gap:
            out.append((first, last))
            first = i
        last = i
    return out + [(first, last)]


def half_life(x, step=STEP, most=LONGEST):
    """Days until a value's correlation with itself falls to one half (interpolated); None if it has not by `most`."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < MIN_STARTS or np.std(x) == 0:
        return None
    prev = 1.0
    for k in range(1, int(most / step) + 1):
        if k >= len(x) - 2:
            return None
        a, b = x[:-k], x[k:]
        r = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0
        if r < 0.5:
            return round(step * (k - 1 + (prev - 0.5) / (prev - r)), 1)
        prev = r
    return None


def pace(values, red):
    """The signpost's own pace on the sampled days: the share of days red, its red spells (how many, median length in
    days) and its value's half-life."""
    sp = spells(red)
    days = [(b - a + 1) * STEP for a, b in sp]
    return {
        "red_share": round(float(np.mean(red)), 4) if len(red) else None,
        "spells": len(sp),
        "spell_days": float(np.median(days)) if days else None,
        "half_life": half_life(values),
    }


def standardised(f, series, starts, a, b, first):
    """The factor over the stretch from `a` to `b` days after each start, in standard deviations of its history over
    windows that long (from `first`): how unusual the move was. NaN where the stretch is not covered."""
    arr = fx.arrays(series, f["series"])
    length = b - a
    if arr is None or length < 2 * fx.spacing(arr):  # a monthly series barely moves within a month: nothing to see
        return np.full(len(starts), np.nan)
    _, hist = fx.history(f, series, length, first)
    if len(hist) < MIN_STARTS or np.std(hist) == 0:
        return np.full(len(starts), np.nan)
    t0, t1 = starts + a * fx.DAY, starts + b * fx.DAY
    v = np.where(t1 <= arr[0][-1], fx.window(f["measure"], arr, t0, t1), np.nan)
    return (v - np.mean(hist)) / np.std(hist)


def slopes(x, Y, block_of, picks=None):
    """The slope of each column of Y on x (rows with both finite), from per-block sums: as measured (picks None) or
    for each bootstrap draw (picks: draws × blocks, the blocks drawn). x red/not (0/1) gives the difference in means."""
    ok = np.isfinite(Y) & np.isfinite(x)[:, None]
    xs = np.where(ok, x[:, None], 0.0)
    ys = np.where(ok, Y, 0.0)
    nb = block_of.max() + 1
    sums = [
        np.stack([np.bincount(block_of, weights=c[:, j], minlength=nb) for j in range(Y.shape[1])], axis=1)
        for c in (ok.astype(float), xs, ys, xs * ys, xs * xs)
    ]
    if picks is not None:
        sums = [s[picks].sum(axis=1) for s in sums]
    else:
        sums = [s.sum(axis=0) for s in sums]
    n, sx, sy, sxy, sxx = sums
    den = n * sxx - sx * sx
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((den > 0) & (n >= MIN_STARTS), (n * sxy - sx * sy) / den, np.nan)


def curve(x, Y, segs, step=STEP, boot=BOOT, seed=7):
    """The move along the stretches and its test. x: the signpost at each start (0/1 red, or a dial in -0.5..0.5);
    Y: starts × stretches, the factor's standardised move. {"curve": [{"from", "to", "effect", "lo", "hi"}], "p",
    "strongest": [from, to], "sign", "holds": [[from, to]] the stretches that move the strongest one's way past their
    90% band (joined where they follow one another), "lasts": [from, to] the joined ones around the strongest, "flips":
    [[from, to]] the stretches that move the other way past it} or None. A move the other way is against the usual
    move from a day that was not red, which may hold red ahead: lower than usual, not a fall. Each stretch is
    drawn again by blocks of its own length (at least BLOCK_MIN days: overlapping windows and clustered red spells are
    one piece of evidence) and measured only on MIN_BLOCKS blocks' span of data. The test: the largest |effect / its
    spread| over the stretches, against the same over the draws centred on the measured curve (one test however many
    stretches; the stretches drawn apart, which errs safe)."""
    rng = np.random.default_rng(seed)
    est, sd, lo, hi, null = [], [], [], [], []
    for j, (a, b) in enumerate(segs):
        y = Y[:, [j]]
        has = np.isfinite(y[:, 0]) & np.isfinite(x)
        block = max(BLOCK_MIN, b - a)
        block_of = (np.arange(len(x)) * step) // block
        nb = block_of.max() + 1
        at = np.flatnonzero(has)
        if not len(at) or (at[-1] - at[0]) * step < MIN_BLOCKS * block:
            est.append(np.nan), sd.append(np.nan), lo.append(np.nan), hi.append(np.nan), null.append(None)
            continue
        e = float(slopes(x, y, block_of)[0])
        drawn = slopes(x, y, block_of, rng.integers(nb, size=(boot, nb)))[:, 0]
        drawn = drawn[np.isfinite(drawn)]
        s = float(np.std(drawn)) if len(drawn) > boot // 2 else np.nan
        est.append(e), sd.append(s)
        lo.append(float(np.quantile(drawn, 0.05)) if np.isfinite(s) else np.nan)
        hi.append(float(np.quantile(drawn, 0.95)) if np.isfinite(s) else np.nan)
        null.append(np.abs(drawn[: boot // 2] - e) / s if np.isfinite(e) and s > 0 else None)
    est, sd, lo, hi = map(np.array, (est, sd, lo, hi))
    use = np.isfinite(est) & (sd > 0)
    if not use.any():
        return None
    t = np.abs(est[use] / sd[use])
    t_null = np.max(np.stack([null[j] for j in np.flatnonzero(use)]), axis=0)
    p = (np.sum(t_null >= t.max()) + 1) / (len(t_null) + 1)
    i_best = np.flatnonzero(use)[int(np.argmax(t))]
    sign = np.sign(est[i_best])
    past = [j for j in range(len(segs)) if use[j] and (lo[j] > 0 if sign > 0 else hi[j] < 0)]
    against = [list(segs[j]) for j in range(len(segs)) if use[j] and (hi[j] < 0 if sign > 0 else lo[j] > 0)]
    runs = []  # the stretches moving its way, joined where they follow one another
    for j in past:
        if runs and runs[-1][-1] == j - 1:
            runs[-1].append(j)
        else:
            runs.append([j])
    main = next((r for r in runs if i_best in r), None)
    return {
        "curve": [
            {"from": s[0], "to": s[1], "effect": round(float(e), 3), "lo": round(float(a), 3), "hi": round(float(b), 3)}
            for s, e, a, b, u in zip(segs, est, lo, hi, use)
            if u
        ],
        "p": round(float(p), 4),
        "strongest": list(segs[i_best]),
        "sign": int(sign),
        "holds": [[segs[r[0]][0], segs[r[-1]][1]] for r in runs],
        "lasts": [segs[main[0]][0], segs[main[-1]][1]] if main else None,
        "flips": against,
    }


def dial(values, red_rule):
    """The signpost's value as a dial from -0.5 to 0.5, its rank among the days sampled, turned so that toward red is
    up (a signpost red below its line reads reversed)."""
    v = pd.Series(values)
    r = v.rank(pct=True).to_numpy() - 0.5
    return -r if red_rule.strip().startswith("<") else r


def measure(sp, values, states, fcfg, series, segs=None):
    """One signpost against every factor: its pace and, for each factor, the curves of its red days and of its dial
    over the stretches: of the move (red, dial) and of its spread (red_spread, dial_spread: the squared standardised
    move less its usual 1, so 0.5 is a variance half again as large). values, states: the signpost's daily values
    and states (prob.evaluate), dated as known."""
    segs = segs or stretches()
    first = pd.Timestamp(fcfg.get("history_from", "1971-03-01"))
    v = values.dropna()
    if v.empty:
        return None
    lo = max(first, v.index[0])
    t = pd.date_range(lo, v.index[-1], freq=f"{STEP}D")
    x_val = v.reindex(t, method="ffill").to_numpy(dtype=float)
    red = (states.reindex(t, method="ffill") == "red").to_numpy()
    ok = np.isfinite(x_val)
    t, x_val, red = t[ok], x_val[ok], red[ok]
    if len(t) < MIN_STARTS:
        return None
    starts = t.values.astype("datetime64[ns]").astype(np.int64)
    out = {"from": t[0].date().isoformat(), "pace": pace(x_val, red), "factors": {}}
    enough_red = out["pace"]["spells"] >= MIN_EPISODES
    x_dial = dial(x_val, sp["red"])
    for k, f in fcfg["factor"].items():
        Y = np.stack([standardised(f, series, starts, a, b, first) for a, b in segs], axis=1)
        V = Y**2 - 1  # how much wider than usual: the squared move against its usual 1
        out["factors"][k] = {
            "red": curve(red.astype(float), Y, segs) if enough_red else None,
            "dial": curve(x_dial, Y, segs),
            "red_spread": curve(red.astype(float), V, segs) if enough_red else None,
            "dial_spread": curve(x_dial, V, segs),
        }
    return out


Z90 = 3.29  # a 90% band's width in standard errors


def shrunk(row):
    """A stretch's effect pulled toward 0 by its own noise: e × e² / (e² + se²). Weak evidence counts for little,
    none for nothing; nothing is cut off."""
    e, se = row["effect"], (row["hi"] - row["lo"]) / Z90
    return e * e * e / (e * e + se * se) if e or se else 0.0


SPREAD = {"wider": 1, "narrower": -1}  # the ways of an expectation about the spread; up and down are about the move
WAYS = {"up": 1, "down": -1, **SPREAD}


def judge(expect, curves):
    """Each expectation of a signpost (config/signposts.toml expect) against its measured curves (measure()'s factors):
    the stretches mostly within its days, on the dial (red alone where there is no dial), the move's curves for up
    and down, the spread's for wider and narrower, each stretch shrunk; their mean signed so that positive is the way
    expected ("size": in standard deviations of the factor's usual move per stretch, or for the spread in its usual
    variance). "agrees": it goes the way expected; "against": it goes the other way past the 90% band of a stretch (not
    used); "flat": the other way within the noise (its size 0); "unmeasured": no stretch of it measured."""
    out = []
    for e in expect:
        by = curves.get(e["factor"]) or {}
        tail = "_spread" if e["way"] in SPREAD else ""
        c, how = (
            (by.get("dial" + tail), "dial" + tail) if by.get("dial" + tail) else (by.get("red" + tail), "red" + tail)
        )
        a, b = e["days"]
        rows = [
            r for r in (c or {}).get("curve", []) if min(b, r["to"]) - max(a, r["from"]) >= (r["to"] - r["from"]) / 2
        ]
        sign = WAYS[e["way"]]
        if not rows:
            out.append({**e, "verdict": "unmeasured", "size": None, "on": how})
            continue
        size = sign * float(np.mean([shrunk(r) for r in rows]))
        wrong = any((r["hi"] < 0) if sign > 0 else (r["lo"] > 0) for r in rows)
        verdict = "agrees" if size > 0 else "against" if wrong else "flat"
        out.append(
            {
                **e,
                "verdict": verdict,
                "size": round(size if size > 0 else 0.0, 3),
                "on": how,
                "stretches": [[r["from"], r["to"]] for r in rows],
            }
        )
    return out


def unexplained(expect, curves):
    """The curves that passed the false discovery control whose run around their strongest stretch no expectation of
    that factor and way covers: a move in the data without a mechanism (not used). [{"factor", "how", "lasts",
    "sign"}]."""
    out = []
    for f, by in curves.items():
        for how, c in by.items():
            if not c or not c.get("passed") or not c.get("lasts"):
                continue
            a, b = c["lasts"]
            way = next(
                w for w, s in (SPREAD if how.endswith("_spread") else {"up": 1, "down": -1}).items() if s == c["sign"]
            )
            if not any(
                e["factor"] == f and e["way"] == way and min(b, e["days"][1]) > max(a, e["days"][0]) for e in expect
            ):
                out.append({"factor": f, "how": how, "lasts": c["lasts"], "sign": c["sign"]})
    return out

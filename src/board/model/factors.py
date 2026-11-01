"""Factors: the few measures that describe the world (config/factors.toml), each one series measured over a window:
ret (end / start - 1), diff (end - start) or ann (the yearly rate); their values over every window of history
(history, matrix: the joint history the world weighs, model/world.py), and a daily factor's unusual move.

Pure: series in, numbers out. Thousands of windows are read, so on plain arrays."""

import numpy as np
import pandas as pd

DAY = 86_400_000_000_000  # a day in nanoseconds
STEP = 7  # days between the window starts sampled in history


def arrays(series, sid):
    """(days as int64 nanoseconds, values) of a series, NaNs dropped; None without one."""
    s = series.get(sid)
    if s is None:
        return None
    s = s.dropna()
    return (s.index.values.astype("datetime64[ns]").astype(np.int64), s.values.astype(float)) if len(s) else None


def at(arr, t):
    """The values at days t (int64 ns, an array): the last value on or before each; NaN before the first."""
    days, x = arr
    i = np.searchsorted(days, t, side="right") - 1
    out = np.where(i >= 0, x[np.clip(i, 0, None)], np.nan)
    return out


def when(arr, t):
    """The days (int64 ns) of the values `at` takes at days t; the first day before the first."""
    days, _ = arr
    return days[np.clip(np.searchsorted(days, t, side="right") - 1, 0, None)]


def window(how, arr, t0, t1):
    """The factor's value over windows from t0 to t1 (int64 ns arrays). A yearly rate is over the days between the
    values it takes: a monthly series' 91-day window spans two, three or four of its months, not 91 days."""
    a, b = at(arr, t0), at(arr, t1)
    if how != "ann":
        return measure(how, a, b, (t1 - t0) / DAY)
    days = (when(arr, t1) - when(arr, t0)) / DAY
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(days > 0, measure(how, a, b, np.maximum(days, 1)), np.nan)


def spacing(arr):
    """The days between a series' values, as it is published now (the median of its last 24 gaps)."""
    return float(np.median(np.diff(arr[0][-25:]))) / DAY if len(arr[0]) > 1 else 1.0


def measure(how, a, b, days):
    """A window's value from its start and end values (arrays)."""
    if how == "ret":
        return b / a - 1
    if how == "diff":
        return b - a
    if how == "ann":
        return (b / a) ** (365 / days) - 1
    raise ValueError(f"factor: unknown measure {how!r}")


def starts(arr, first, length, step=STEP):
    """The window starts (int64 ns) from `first` every `step` days whose window of `length` days the series covers."""
    days, _ = arr
    lo = max(pd.Timestamp(first).value, days[0])
    hi = days[-1] - length * DAY
    return np.arange(lo, hi + 1, step * DAY, dtype=np.int64) if hi >= lo else np.array([], dtype=np.int64)


def history(f, series, length, first):
    """(window starts, the factor's value over each window of `length` days) in history from `first`."""
    arr = arrays(series, f["series"])
    if arr is None:
        return np.array([], dtype=np.int64), np.array([])
    t = starts(arr, first, length)
    if not len(t):
        return t, np.array([])
    v = window(f["measure"], arr, t, t + length * DAY)
    ok = np.isfinite(v)
    return t[ok], v[ok]


def matrix(fcfg, series, length, first=None):
    """(window starts, {factor: values}) for the windows of `length` days every factor covers: the joint history."""
    first = first or fcfg.get("history_from", "1900-01-01")
    arrs = {k: arrays(series, f["series"]) for k, f in fcfg["factor"].items()}
    if any(a is None for a in arrs.values()):
        missing = [k for k, a in arrs.items() if a is None]
        raise ValueError(f"factors: no series for {missing}")
    lo = max(pd.Timestamp(first).value, *(a[0][0] for a in arrs.values()))
    hi = min(a[0][-1] for a in arrs.values()) - length * DAY
    t = np.arange(lo, hi + 1, STEP * DAY, dtype=np.int64)
    vals = {k: window(f["measure"], arrs[k], t, t + length * DAY) for k, f in fcfg["factor"].items()}
    ok = np.all([np.isfinite(v) for v in vals.values()], axis=0) if len(t) else np.array([], dtype=bool)
    return t[ok], {k: v[ok] for k, v in vals.items()}


MOVE_DAYS = 5  # a move over this many days, against history's moves of the same length
MOVE_TAIL = 0.01  # beyond history's 1st or 99th percentile is unusual


def unusual_move(f, series, today, first, days=MOVE_DAYS, tail=MOVE_TAIL):
    """The factor's move over the last `days` to today when it is beyond history's `tail` quantiles of such moves
    (from `first`): (move, low edge, high edge); None when it is not, or the series is not daily."""
    arr = arrays(series, f["series"])
    if arr is None or len(arr[0]) < 260 or np.median(np.diff(arr[0][-260:])) > 3 * DAY:
        return None
    now = pd.Timestamp(today).value
    x = float(window(f["measure"], arr, np.array([now - days * DAY]), np.array([now]))[0])
    _, past = history(f, series, days, first)
    if not np.isfinite(x) or len(past) < 100:
        return None
    lo, hi = (float(np.quantile(past, q)) for q in (tail, 1 - tail))
    return (x, lo, hi) if x < lo or x > hi else None


# ---- outcomes


def fmt(f, x):
    """A factor's value as the page writes it: a percent for ret and ann, points for a rate, the index otherwise."""
    if f["measure"] in ("ret", "ann"):
        return f"{x * 100:+.1f}%"
    return f"{x:+.2f} pts" if f.get("unit") == "pts" else f"{x:+.2f}"

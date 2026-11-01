"""The asset classes through history: each one's total return (its stand-in's price, the dividends of each day at that
day's yield, the bill's interest for cash), its windows over a horizon, and the measures of a value path (the
decision step walked through history: model/walk.py). Pure: no files."""

import numpy as np
import pandas as pd

from . import asset_classes


def total_return(c, frame, k=None):
    """A class's total return index by day (a Series starting at 1): cash accrues the bill's yield; a stock class its
    stand-in's price times the dividends at each day's yield (asset_classes.dividend_yield); the rest its stand-in."""
    ids = asset_classes.proxy_ids(c)
    if ids is asset_classes.CASH:
        y = asset_classes.stand_in(frame, ids, scale=False)
        if y is None:
            return None
        y = y.ffill().dropna()
        return np.exp((y / 100 / 365).cumsum())
    s = asset_classes.stand_in(frame, ids)
    if s is None:
        return None
    if c.get("history_from"):
        s = s[s.index >= c["history_from"]]
    s = s.ffill().dropna()
    y, _ = asset_classes.dividend_yield(c, frame)
    if y is not None:
        y = y.reindex(s.index).ffill().fillna(0)
        s = s * np.exp((y / 365).cumsum())
    return s / s.iloc[0]


def measures(v):
    """{"start", "years", "cagr", "vol" (yearly, of monthly returns), "max_dd", "worst_2y", "multiple"} of a value
    series by month."""
    r = v.pct_change().dropna()
    years = (v.index[-1] - v.index[0]).days / 365
    worst = None
    two = v.shift(-24) / v - 1
    if two.notna().any():
        worst = float(two.min())
    return {
        "start": v.index[0].date().isoformat(),
        "years": round(years, 1),
        "cagr": float(v.iloc[-1] ** (1 / years) - 1) if years else None,
        "vol": float(r.std() * 12**0.5),
        "max_dd": float((v / v.cummax() - 1).min()),
        "worst_2y": worst,
        "multiple": float(v.iloc[-1]),
    }


def windows(prices, days, step=7):
    """The return over `days` of a total return index (a Series), one window every `step` days: {start day: r}."""
    p = prices.resample("D").last().ffill()
    starts = p.index[: len(p) - days : step]
    end = p.reindex(starts + pd.Timedelta(days=days)).values
    return {d: float(e / p[d] - 1) for d, e in zip(starts, end) if np.isfinite(e)}

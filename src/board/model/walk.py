"""The decision step (model/decide.py) walked through history: every STEP_WEEKS from the first
year the signposts can be measured, the data's world as it could be built then (calibrate.py walk: the signposts'
measures from the years before, the values known then; the LLM's futures and the markets cannot be rebuilt and are
left out), each length's futures the windows of history already over then, each class's return in each its actual
total return, cash at that day's bill rate. On each day the decision step gives a mix and the trades it would
suggest on what the walk holds (the band, the anchor, a step at a time); between them the holdings drift with the
markets. Against your targets rebalanced past your band, your risk level's mix and 60/40, on the same days.

The sense and the lengths' weights are chosen on the first half of the walk (choose) and judged on the second, so
the choice is not fitted to what it is judged on. Pure: arrays in, numbers out."""

import numpy as np
import pandas as pd

from . import backtest, decide
from .distribution import Futures

STEP_WEEKS = 4  # weeks between the walk's decisions
SENSES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)  # the senses tried: a score of 0.1 standard deviations at sense 0.5 leans 5%
WEIGHTS = {  # the lengths' weights tried
    "short": {"month": 0.5, "short": 0.5},
    "even": {"month": 0.25, "short": 0.25, "long": 0.25, "longer": 0.25},
    "long": {"long": 0.5, "longer": 0.5},
    "year": {"long": 1.0},
}
GAIN_MIN = 0.002  # a sense is kept only if it adds at least this much a year on the first half
DD_SLACK = 0.02  # and its deepest fall is at most this much deeper than the targets rebalanced
TURN_MOST = 2.0  # and it trades at most this many times as much (plus TURN_PLUS a year)
TURN_PLUS = 0.10
LEAN_SHOWN = 0.01  # a day counts as leaning when a class leans this much


def windows(prices, starts, days):
    """Each class's return over `days` from each start day (int64 ns): {class: array}, NaN where the window is not
    over in the data. prices: {class: total return index (a Series by day)}."""
    t0 = pd.to_datetime(starts)
    t1 = t0 + pd.Timedelta(days=days)
    out = {}
    for k, p in prices.items():
        p = p.resample("D").last().ffill()
        a = p.reindex(t0).to_numpy(dtype=float)
        b = p.reindex(t1).to_numpy(dtype=float)
        out[k] = b / a - 1
    return out


def futures_at(R, starts, w, keys, cash_rate=None, days=None):
    """The futures of one day at one length: the start days `starts` (indexes into R's rows) weighed by w, the
    classes `keys`; cash at the day's bill rate (cash_rate, yearly) over the length. (world, history) Futures."""
    ret = {k: R[k][starts] for k in keys if k != decide.CASH}
    if decide.CASH in keys and cash_rate is not None:
        ret[decide.CASH] = np.full(len(starts), cash_rate * days / 365)
    X = np.zeros((len(starts), 0))
    return Futures(X, ret, w), Futures(X, ret)


def limits_at(policy, history, year_limit):
    """Each length's limit on the day from the history known then: a year's year_limit, every other length's the
    year's times your targets' bad case at it over theirs at a year (steps/distribution.py limits)."""
    bad = {h: decide.bad(policy, f) for h, f in history.items()}
    base = bad.get("long")
    if not base:
        return {h: year_limit for h in bad}
    return {h: year_limit * b / base if b is not None else None for h, b in bad.items()}


def run(days, worlds, R, weekly, policy, limits, year_limit, cash, sense, weights, band_only=False):
    """One walk: the decision step on each day of `days` ({day: {length: (row indexes into R[length], weights)}}:
    worlds), trades on the holdings; between days the holdings drift on `weekly` (a DataFrame of total return
    indexes by week, a column per class). band_only: your targets rebalanced past your band (no lean, no
    guardrail). {"value" (a Series by week), "turnover" (a year), "leans" (the share of days leaning LEAN_SHOWN),
    "guarded" (the share with a guardrail move), "trades"}."""
    keys = [k for k in policy if k in weekly.columns]
    pol = {k: policy[k] / sum(policy[k] for k in keys) for k in keys}
    held = dict(pol)
    anchor = {}
    value, traded, leaned, guarded, n_trades = 1.0, 0.0, 0, 0, 0
    out = {}
    weeks = weekly.index
    rel = weekly.pct_change().fillna(0.0)
    day_list = sorted(worlds)
    nxt = 0
    for t in weeks[weeks >= day_list[0]]:
        r = rel.loc[t]
        grow = {k: held[k] * (1 + r[k]) for k in held}
        total = sum(grow.values())
        value *= total
        held = {k: x / total for k, x in grow.items()}
        if nxt < len(day_list) and t >= day_list[nxt]:
            d = day_list[nxt]
            nxt += 1
            if band_only:
                mix = dict(pol)
            else:
                lengths = {}
                for h, (rows, w) in worlds[d].items():
                    fut, hist = futures_at(R[h], rows, w, keys, cash.get(d), weights_days(h))
                    lengths[h] = {"futures": fut, "history_futures": hist, "days": weights_days(h)}
                lims = limits_at(pol, {h: x["history_futures"] for h, x in lengths.items()}, year_limit)
                for h in lengths:
                    lengths[h]["limit"] = lims.get(h)
                got = decide.decide(pol, lengths, limits, sense, weights)
                mix = got["mix"]
                leaned += any(abs(x) >= LEAN_SHOWN for x in got["lean"].values())
                guarded += bool(got["guard"])
            target = {k: mix.get(k, 0.0) for k in keys}
            todo, anchor = decide.trades(target, held, anchor, limits)
            for k, x in todo.items():
                held[k] = held.get(k, 0.0) + x["size"]
                held[decide.CASH] = held.get(decide.CASH, 0.0) - x["size"]
                traded += abs(x["size"])
                n_trades += 1
        out[t] = value
    v = pd.Series(out)
    years = max((v.index[-1] - v.index[0]).days / 365, 1e-9)
    return {
        "value": v,
        "turnover": traded / years,
        "leans": leaned / max(len(day_list), 1),
        "guarded": guarded / max(len(day_list), 1),
        "trades": n_trades,
    }


LENGTH_DAYS = {"week": 7, "month": 30, "short": 91, "long": 365, "longer": 1095}


def weights_days(h):
    return LENGTH_DAYS[h]


def measures(v, turnover=None):
    """backtest.measures on the walk's values by month, with the turnover a year."""
    m = v.groupby(v.index.to_period("M")).head(1)
    out = backtest.measures(m)
    if turnover is not None:
        out["turnover"] = turnover
    return out


def halves(v):
    """The walk's values split at its middle day: (first half, second half), each from 1."""
    mid = v.index[0] + (v.index[-1] - v.index[0]) / 2
    a, b = v[v.index <= mid], v[v.index >= mid]
    return a / a.iloc[0], b / b.iloc[0]


def choose(results, base):
    """The sense and weights kept: on the first half, the one that adds most a year over your targets rebalanced
    (base) while its deepest fall is at most DD_SLACK deeper and it trades at most TURN_MOST times as much plus
    TURN_PLUS a year; none adding GAIN_MIN: sense 0 (no lean: the guardrails and the band alone). results: {(sense,
    weights name): {"first": measures, "second": measures, "turnover"}}."""
    b = base["first"]
    best, gain = None, GAIN_MIN
    for key, x in results.items():
        f = x["first"]
        if key[0] == 0 or f["cagr"] is None:
            continue
        ok = f["max_dd"] >= b["max_dd"] - DD_SLACK and x["turnover"] <= TURN_MOST * base["turnover"] + TURN_PLUS
        if ok and f["cagr"] - b["cagr"] >= gain:
            best, gain = key, f["cagr"] - b["cagr"]
    return best or (0.0, "even")

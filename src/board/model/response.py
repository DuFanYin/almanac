"""How each asset class's return over a horizon follows the factors: a shrunk linear fit on
every window of history, and what it leaves unexplained (the residuals), kept by window so the classes' unexplained
parts stay as they moved together. In log returns (log(1 + r)): they add up over time and a fall can never pass
100%, so a class with a few huge years (Bitcoin) does not reach an impossible loss when the fit is carried to
history's other windows (10-31: a year's worst Bitcoin windows at -700% on simple returns). A future's return is
exp(fit + residual) - 1 (model/distribution.py windows).

The fit is on total returns (each day's dividends, the bills' interest: model/backtest.py). What a window starts from
is today's, not history's average: cash and Treasuries earn today's yield (a class's `start` in
config/asset_classes.toml), stocks today's dividend yield; the intercept is moved by that difference over the horizon.
Checked on history (10-29): out of sample the fit explains 0.5-0.9 of the stock classes and Treasuries, a third of
cash, little of gold; Bitcoin's history is too short to fit, so it keeps its own windows' spread.

Each class's path through each window is kept too (its value each week from the start, as a multiple of the start),
in the residuals' order: a future's fall along the way is read from the window it came from (model/allocation.py
drawdown).

Pure."""

import numpy as np
import pandas as pd

from . import asset_classes, backtest, factors

SHRINK = 0.05  # ridge penalty per window, on standardised factors
MIN_FIT = 8  # independent windows a class needs to be fitted; below, its own windows' returns stand as they were


class Response:
    """The fit at one horizon: factor names, their means and spreads, and per class (intercept, slopes in the
    factors' units), its residuals by window start, its fit's share explained, its independent windows and its path
    through each window (windows x weeks, in the residuals' order)."""

    def __init__(self, names, mean, sd, days):
        self.names, self.mean, self.sd, self.days = names, mean, sd, days
        self.coef, self.resid, self.r2, self.n_eff, self.own, self.paths = {}, {}, {}, {}, {}, {}

    def predict(self, k, X):
        """Class k's log return for factor rows X (N x factors): the fit, or for a class too short to fit, its
        mean."""
        if k in self.own:
            return np.full(len(X), self.own[k].mean())
        a, b = self.coef[k]
        return a + X @ b


def ridge(Z, y, lam):
    zc, yc = Z - Z.mean(0), y - y.mean()
    b = np.linalg.solve(zc.T @ zc + lam * np.eye(Z.shape[1]), zc.T @ yc)
    return y.mean() - Z.mean(0) @ b, b


def start_shift(c, frame, starts_ns, days):
    """How much today's start adds over the horizon against the windows' own starts (a return): a yield series
    (`start` of cash and Treasuries, in percent) today against its mean at the window starts; for stocks, today's
    dividend yield against the windows'."""
    years = days / 365
    st = asset_classes.starts_of(c)
    if c.get("proxy") == "cash" or st and st[0] in ("DGS10", "DTB3"):
        sid = "DTB3" if c.get("proxy") == "cash" else st[0]
        if sid not in frame:
            return 0.0
        s = frame[sid].ffill().dropna()
        then = s.reindex(pd.to_datetime(starts_ns), method="ffill").mean()
        return float((s.iloc[-1] - then) / 100 * years)
    y, _ = asset_classes.dividend_yield(c, frame)
    if y is None:
        return 0.0
    y = y.ffill().dropna()
    then = y.reindex(pd.to_datetime(starts_ns), method="ffill").mean()
    return float((y.iloc[-1] - then) * years)


def paths(arr, starts, days, step=7):
    """Each window's path (windows x points): the value every `step` days from its start and at its end, over the
    start's."""
    grid = np.minimum(np.arange(0, days + step, step), days) * factors.DAY
    grid = np.unique(grid)
    v = factors.at(arr, (starts[:, None] + grid[None, :]).ravel()).reshape(len(starts), len(grid))
    return v / v[:, :1]


def windows(c, frame, t, days):
    """A class's windows among the starts t: (the starts it covers, its total return over each, the total return
    index as (days, values), the mask of t kept); None without a total return series."""
    p = backtest.total_return(c, frame)
    if p is None or p.empty:
        return None
    idx = p.index.values.astype("datetime64[ns]").astype(np.int64)
    ok = (t >= idx[0]) & (t + days * factors.DAY <= idx[-1])
    arr = (idx, p.values.astype(float))
    r = factors.at(arr, t[ok] + days * factors.DAY) / factors.at(arr, t[ok]) - 1
    keep = np.zeros(len(t), dtype=bool)
    keep[np.flatnonzero(ok)[np.isfinite(r)]] = True
    return t[keep], r[np.isfinite(r)], arr, keep


def fit(fcfg, series, frame, classes, days):
    """The Response at a horizon of `days`, over the joint history of the factors, for each class with a total
    return series."""
    t, vals = factors.matrix(fcfg, series, days)
    names = list(vals)
    X = np.column_stack([vals[k] for k in names]) if len(t) else np.zeros((0, len(names)))
    mean, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    out = Response(names, mean, sd, days)
    for k, c in classes.items():
        got = windows(c, frame, t, days)
        if got is None or not len(got[0]):
            continue
        tt, r, arr, keep = got
        r = np.log1p(r)
        Z = (X[keep] - mean) / sd
        out.paths[k] = paths(arr, tt, days)
        n_eff = len(r) * factors.STEP / days
        shift = start_shift(c, frame, tt, days)  # a yield's difference over the horizon: small, the same in logs
        out.n_eff[k] = n_eff
        if n_eff < MIN_FIT:  # too short to fit: its own windows, moved to today's start
            out.own[k] = r + shift
            out.resid[k] = dict(zip(tt.tolist(), (r - r.mean()).tolist()))
            out.r2[k] = None
            continue
        a, b = ridge(Z, r, len(r) * SHRINK)
        e = r - a - Z @ b
        out.coef[k] = (a - (mean / sd) @ b + shift, b / sd)
        out.resid[k] = dict(zip(tt.tolist(), e.tolist()))
        out.r2[k] = float(1 - (e**2).sum() / ((r - r.mean()) ** 2).sum())
    out.starts = t
    out.X = X
    return out


FOLDS = 5  # stretches of history held out in turn


def holdout_r2(fcfg, series, frame, classes, days, folds=FOLDS):
    """{class: share of its return the fit explains out of sample}: history cut into `folds` stretches, each held out
    in turn and predicted by a fit on the rest, one horizon left out either side of it so overlapping windows do not
    leak; the classes with MIN_FIT independent windows. How a proposed factor is judged (model/proposals.py)."""
    t, vals = factors.matrix(fcfg, series, days)
    if not len(t):
        return {}
    X = np.column_stack(list(vals.values()))
    mean, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    gap = int(np.ceil(days / factors.STEP))
    out = {}
    for k, c in classes.items():
        got = windows(c, frame, t, days)
        if got is None or len(got[0]) * factors.STEP / days < MIN_FIT:
            continue
        _, r, _, keep = got
        r = np.log1p(r)
        Z, n = (X[keep] - mean) / sd, len(r)
        edges = np.linspace(0, n, folds + 1).astype(int)
        pred = np.full(n, np.nan)
        for lo, hi in zip(edges[:-1], edges[1:]):
            train = np.r_[0 : max(lo - gap, 0), min(hi + gap, n) : n]
            if len(train) < 2 * gap:
                continue
            a, b = ridge(Z[train], r[train], len(train) * SHRINK)
            pred[lo:hi] = a + Z[lo:hi] @ b
        ok = np.isfinite(pred)
        if ok.sum() > gap:
            out[k] = float(1 - ((r[ok] - pred[ok]) ** 2).sum() / ((r[ok] - r[ok].mean()) ** 2).sum())
    return out

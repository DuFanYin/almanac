"""The futures at a length, read off the world (model/world.py): every start day of history once, weighed as the world
weighs it, with its factor values and the asset classes' returns that follow from them: the fitted response to its
factors (model/response.py) plus the unexplained part of the same window, so the classes' extremes stay as they
happened together. Each future keeps which window each class came from (src), so its path along the way can be read
from that window (model/allocation.py drawdown), and its start day (start). The same futures under other weights
(weighed) differ only by their weights, so two sides of the world compare on the same windows, and a day's figures
move only as the weights do: nothing is drawn at random. A window made for a scenario (model/world.py made,
extended) is its source window's with the factors moved: the fit to the moved factors plus the source window's
unexplained part, along the source window's path. And which way a scenario takes each factor against history
(leans).

Pure."""

import zlib

import numpy as np

SEED = 7  # a class without a start day's window takes one of its own, the same one every day


class Futures:
    """N futures: their factor values (N x factors), the classes' returns ({class: N}), their weights (w, summing to
    1; even without them), their start days (start, int64 ns), and for each class the window its path comes from
    ({class: N row indexes into paths[class]}, the response's paths: windows x weeks)."""

    def __init__(self, X, returns, w=None, src=None, paths=None, start=None):
        self.X, self.returns = X, returns
        n = len(X)
        self.w = np.full(n, 1 / n) if w is None else np.asarray(w, dtype=float) / np.sum(w)
        self.src, self.paths, self.start = src or {}, paths or {}, start

    def weighed(self, w):
        """The same futures under other weights."""
        return Futures(self.X, self.returns, w, self.src, self.paths, self.start)


def windows(resp, w=None, made=None):
    """Every window of history the response was fitted on, once, weighed by w (even without it); and after them the
    periods made from them (made: model/world.py extended, its rows past "real"). Each class's return is the fit to
    the window's factors plus the unexplained part of the same window, so a window's extremes stay together (in
    2000-02 tech fell further than its fit, as the market fell); a class without that window takes one of its own,
    fixed by the class (SEED), so it is the same one on every side and every day. A made period: the fit to its moved
    factors plus its source window's unexplained part and path. Fit and unexplained part in log returns (model/
    response.py), each future's return exp(their sum) - 1."""
    F, win = resp.X, resp.starts
    n = len(F)
    extra = made is not None and made.get("real") is not None
    if extra:
        F = np.vstack([F, made["X"][made["real"] :]])
        win = np.concatenate([win, made["starts"][made["real"] :]])
    out, src = {}, {}
    for k in [*resp.coef, *resp.own]:
        rng = np.random.default_rng([SEED, zlib.crc32(k.encode())])
        e = resp.resid.get(k, {})
        keys = np.fromiter(e.keys(), dtype=np.int64, count=len(e))
        vals = np.fromiter(e.values(), dtype=float, count=len(e))
        if k in resp.own:  # too short to fit: its own windows as they were
            src[k] = rng.integers(len(resp.own[k]), size=n)
            if extra:
                src[k] = np.concatenate([src[k], src[k][made["src"]]])
            out[k] = np.expm1(resp.own[k][src[k]])
            continue
        base = resp.predict(k, F)
        if len(keys):
            real = win[:n]
            pos = np.clip(np.searchsorted(keys, real), 0, len(keys) - 1)
            src[k] = np.where(keys[pos] == real, pos, rng.integers(len(vals), size=n))
            if extra:
                src[k] = np.concatenate([src[k], src[k][made["src"]]])
            base = base + vals[src[k]]
        out[k] = np.expm1(base)
    paths = {k: resp.paths[k] for k in src if k in getattr(resp, "paths", {})}
    return Futures(F, out, w, src, paths, np.asarray(win))


LEAN = 0.4  # a range whose middle sits below history's 40th percentile leans down; above its 60th, up


def leans(base, names, ranges):
    """Which way a scenario takes each factor against history (base: history's windows, each once): {factor:
    {"rank": where its range sits among history's values (the middle of history's ranks inside it; an end's rank
    when none is), "lean": -1, 0 or 1}}. A range open at an end (None) runs to history's end."""
    out = {}
    for j, k in enumerate(names):
        if k not in ranges:
            continue
        x = np.sort(base.X[:, j])
        lo, hi = ranges[k]
        a = 0 if lo is None else np.searchsorted(x, lo, side="left")
        b = len(x) if hi is None else np.searchsorted(x, hi, side="right")
        rank = float((a + b) / 2 / len(x)) if len(x) else 0.5
        out[k] = {"rank": rank, "lean": -1 if rank <= LEAN else 1 if rank >= 1 - LEAN else 0}
    return out

"""An allocation on a set of futures (model/distribution.py), each future weighed as the world weighs its start day:
its return in each, and what the page and the decision step read: the expected return, the median, the 10th
percentile, the bad case (the mean of the worst 10% of the weight: tail), the good case (the mean of the best 10%), the
chance of a loss past a line, which classes the worst 10% loses in; and, for the page, its deepest fall along the way
(drawdown). Computed on the weights exactly, so a figure moves only as the world does.

Pure."""

import numpy as np

TAIL = 0.10


def returns(weights, fut):
    """The allocation's return in each future, over the classes the futures have ({class: share}, rescaled to those);
    and the share of the allocation that had none (left as it is)."""
    held = {k: w for k, w in weights.items() if w > 0}
    have = {k: w for k, w in held.items() if k in fut.returns}
    total = sum(have.values())
    if not total:
        return None, 1.0
    r = sum(w * fut.returns[k] for k, w in have.items()) / total
    return r, 1 - total / sum(held.values())


def quantile(r, w, q):
    """The weighted q-quantile of r: the least value with at least q of the weight at or below it."""
    order = np.argsort(r, kind="stable")
    c = np.cumsum(w[order])
    return float(r[order][min(int(np.searchsorted(c, q - 1e-12)), len(r) - 1)])


def tail(r, w, q=TAIL):
    """Each future's part of the worst q of the weight (summing to 1): whole for those below the cut, the share that
    fits for the one at it. The bad case is tail @ r."""
    order = np.argsort(r, kind="stable")
    ws = w[order]
    take = np.clip(q - (np.cumsum(ws) - ws), 0, ws)
    out = np.zeros(len(r))
    out[order] = take / take.sum()
    return out


def stats(weights, fut, lines=(-0.10, -0.20, -0.30), bins=None):
    """{"expected", "median", "p10", "p90", "bad", "good" (the mean of the best 10%), "p_below" {line: chance},
    "losses_in" {class: its part of the bad case},
    "left_out", "hist" (the shares over bins: (lo, hi, step), else histogram's default)}; None when no class can be
    judged."""
    r, left = returns(weights, fut)
    if r is None:
        return None
    w, t = fut.w, tail(r, fut.w)
    held = {k: x for k, x in weights.items() if x > 0 and k in fut.returns}
    total = sum(held.values())
    return {
        "expected": float(w @ r),
        "median": quantile(r, w, 0.5),
        "p10": quantile(r, w, TAIL),
        "p90": quantile(r, w, 0.9),
        "bad": float(t @ r),
        "good": float(tail(-r, w) @ r),
        "p_below": {str(x): float(w[r < x].sum()) for x in lines},
        "losses_in": {k: float(x / total * (t @ fut.returns[k])) for k, x in held.items()},
        "left_out": left,
        "hist": histogram(r, w, *(bins or ())),
    }


def histogram(r, w, lo=-0.6, hi=1.0, step=0.05):
    """The share of the weight in each bin of `step` from lo to hi (the ends take the rest): for the page's chart."""
    edges = np.arange(lo, hi + step / 2, step)
    c = np.histogram(np.clip(r, lo, hi - 1e-9), bins=edges, weights=w)[0]
    return {"lo": round(lo, 6), "step": step, "share": [round(float(x), 5) for x in c]}


STEPS = (0.01, 0.02, 0.025, 0.05, 0.1, 0.2)  # a chart's bin widths to choose from
BINS = 44  # bins a chart has, at most


def chart_bins(returns, marks=(), keep=0.005):
    """(lo, hi, step) for a chart of these sets of returns: from the 0.5th to the 99.5th percentile of them all, the
    marks (0, the limit) inside, in at most BINS bins of the narrowest round width that allows, on whole bins."""
    r = np.concatenate([x for x in returns if x is not None])
    lo = min([float(np.quantile(r, keep)), *marks])
    hi = max([float(np.quantile(r, 1 - keep)), *marks])
    step = next((s for s in STEPS if (hi - lo) / s <= BINS), STEPS[-1])
    return float(round(np.floor(lo / step + 1e-9) * step, 6)), float(round(np.ceil(hi / step - 1e-9) * step, 6)), step


def class_stats(fut):
    """Each class's expected return, 10th percentile and bad case on the futures."""
    w = fut.w
    return {
        k: {"expected": float(w @ r), "p10": quantile(r, w, TAIL), "bad": float(tail(r, w) @ r)}
        for k, r in fut.returns.items()
    }


def drawdown(weights, fut, lines=(-0.20, -0.30)):
    """The allocation's deepest fall along the way in each future, held without rebalancing: each class follows its
    path through the window the future came from, its drift moved so it ends at the future's return (the window's
    swings, the future's end). {"median", "bad" (the mean of the worst 10%), "p_below" {line: chance}}; None without
    the paths."""
    held = {k: w for k, w in weights.items() if w > 0 and k in fut.src and k in fut.paths}
    total = sum(held.values())
    if not total:
        return None
    value = 0.0
    for k, w in held.items():
        path = fut.paths[k][fut.src[k]]
        drift = np.maximum(1 + fut.returns[k], 1e-3) / path[:, -1]
        value = value + w / total * path * drift[:, None] ** np.linspace(0, 1, path.shape[1])[None, :]
    dd = (value / np.maximum.accumulate(value, axis=1) - 1).min(axis=1)
    return {
        "median": quantile(dd, fut.w, 0.5),
        "bad": float(tail(dd, fut.w) @ dd),
        "p_below": {str(x): float(fut.w[dd < x].sum()) for x in lines},
    }


SHELTER = ("cash", "treasuries", "gold")  # where a guardrail's move may go (model/decide.py guard)

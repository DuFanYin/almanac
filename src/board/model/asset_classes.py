"""The asset classes (config/asset_classes.toml): each one's stand-in series carried back in time, its dividend
yield, its starting point; and the helpers on window-by-window returns the proposals' check uses. How a class's
return follows the factors is model/response.py; its return over each horizon's futures, model/distribution.py.

Pure: no files."""

import numpy as np
import pandas as pd

from ..core import expr

CASH = ["DGS3MO", "DTB3"]  # the 3-month bill: constant maturity from 1981, the secondary market rate from 1954
MIN_WINDOWS = 3  # fewer independent windows than this say nothing
BOOT = 200  # bootstrap draws behind an interval


def proxy_ids(c):
    """An asset class's stand-in series, the first one first: each next one carries it further back in time."""
    p = c["proxy"]
    return CASH if p == "cash" else [p] if isinstance(p, str) else list(p)


def stand_in(frame, ids, scale=True):
    """The first series of ids, carried back before its first value by each next one, scaled to meet it
    (yields are joined as they are)."""
    ids = [i for i in ids if i in frame]
    if not ids:
        return None
    s = frame[ids[0]].copy()
    for sid in ids[1:]:
        start, back = s.first_valid_index(), frame[sid]
        if start is None:
            s = back.copy()
            continue
        k = 1.0
        if scale:
            if pd.isna(back.get(start)) or not back[start]:
                continue
            k = s[start] / back[start]
        early = s.index < start
        s[early] = back[early] * k
    return s


def dividend_yield(c, frame):
    """A class's yearly dividend yield on each day (a Series) and its multiple of the base, or (None, None) without
    `dividend`: the base yield (an expression: the S&P 500's), times how its fund (`like`) yielded against the base
    over the days both are known (the fund's yearly gain with dividends reinvested less without, over the base's
    mean then)."""
    d = c.get("dividend")
    if not d:
        return None, None
    base = expr.evaluate(d["yield"], frame).ffill()
    k = 1.0
    like = d.get("like")
    if like and like in frame and f"{like}_TR" in frame:
        both = pd.concat([frame[like], frame[f"{like}_TR"], base], axis=1, keys=["p", "tr", "y"]).dropna()
        years = (both.index[-1] - both.index[0]).days / 365 if len(both) > 1 else 0
        if years >= 3:
            first, last = both.iloc[0], both.iloc[-1]
            gained = (np.log(last["tr"] / first["tr"]) - np.log(last["p"] / first["p"])) / years
            k = max(0.0, float(gained / both["y"].mean()))
    return base * k, k


def starts_of(c):
    """A class's candidate starting points: its `start`, one expression or a list."""
    st = c.get("start")
    return [st] if isinstance(st, str) else list(st or [])


def resample(rets, horizon_days, seed, n=BOOT):
    """(medians, worst tenths) of n bootstrap draws of the windows ({day: r}): whole stretches of windows that
    start within one horizon of each other are drawn, so overlapping windows are not separate evidence."""
    stretches, first = [], None
    for d in sorted(rets):
        if first is None or (d - first).days >= horizon_days:
            stretches.append([])
            first = d
        stretches[-1].append(rets[d])
    arrs = [np.array(x) for x in stretches]
    rng = np.random.default_rng(seed)
    med, p10 = np.empty(n), np.empty(n)
    for i in range(n):
        x = np.sort(np.concatenate([arrs[j] for j in rng.integers(len(arrs), size=len(arrs))]))
        med[i], p10[i] = np.median(x), x[len(x) // 10]
    return med, p10


def mix_windows(weights, rets_by_class):
    """The mix's return window by window over the windows its classes share ({day: r}); None when one has none."""
    held = [k for k, w in weights.items() if w > 0]
    if not held or any(k not in rets_by_class for k in held):
        return None
    common = set.intersection(*(set(rets_by_class[k]) for k in held))
    total = sum(weights[k] for k in held)
    return {d: sum(weights[k] * rets_by_class[k][d] for k in held) / total for d in common}

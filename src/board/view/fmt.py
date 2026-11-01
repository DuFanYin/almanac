"""Numbers for the page: rounded, NaN as null, the bootstrap's draws left out."""

import pandas as pd


def r(x, d=6):
    return None if x is None or (isinstance(x, float) and pd.isna(x)) else round(float(x), d)


def rounded(p, d=4):
    return {k: r(v, d) for k, v in p.items()} if p else None


def rd(x, d=4):
    """x for the page: floats rounded, the bootstrap's draws left out."""
    if isinstance(x, dict):
        return {k: rd(v, d) for k, v in x.items() if k != "median_draws"}
    if isinstance(x, (list, tuple)):
        return [rd(v, d) for v in x]
    if isinstance(x, float):
        return r(x, d)
    return x

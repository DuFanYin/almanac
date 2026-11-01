"""Values and series described for a reader, the LLM included: a figure as a person would write it, how often a
series is published, the series the board keeps.

Pure."""

import pandas as pd


def figure(x):
    """A value as a reader would write it: whole above 1,000, two decimals above 10, else four."""
    return round(x) if abs(x) >= 1000 else round(x, 2) if abs(x) >= 10 else round(x, 4)


def every(s):
    gap = pd.Series(s.index).diff().dt.days.median()
    return "daily" if gap <= 3 else "weekly" if gap <= 8 else "monthly" if gap <= 40 else "quarterly"


def series_list(cfg, series):
    """The series the board keeps, for the LLM: [{"id", "name", "every", "since", "last"}]."""
    out = []
    for sid, sc in cfg["series"].items():
        s = series.get(sid)
        if s is None or len(s) < 24:
            continue
        out.append(
            {
                "id": sid,
                "name": sc["name"],
                "every": every(s),
                "since": s.index[0].year,
                "last": figure(float(s.iloc[-1])),
            }
        )
    return out

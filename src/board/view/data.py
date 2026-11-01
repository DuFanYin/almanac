"""The data on the page: the series past their age, and the indicators with a month's change and a sparkline."""

import pandas as pd

from .fmt import r

INDICATORS = [
    "GSPC",
    "IXIC",
    "DJI",
    "BTC",
    "ETH",
    "GOLD",
    "FG",
    "AHR999",
    "VIX",
    "VIX3M",
    "SKEW",
    "MOVE",
    "DGS10",
    "DGS3MO",
    "T10Y3M",
    "HY",
    "NFCI",
    "DFF",
    "DFII10",
    "CAPE",
    "DXY",
    "USDSGD",
    "OIL",
    "SOX",
    "STABLES",
    "UNRATE",
    "CPI",
    "ICSA",
]
HISTORY_DAYS = 180
HISTORY_DAYS = 180


def stale_series(df, cfg, today):
    out = []
    for sid, s in cfg["series"].items():
        if s["source"] == "archive":  # a one-time history ends where it ends
            continue
        if sid not in df or df[sid].isna().all():
            out.append(sid)
            continue
        last = df[sid].last_valid_index()
        if (pd.Timestamp(today) - last).days > s.get("max_age", 7):
            out.append(sid)
    return out


def indicators(df, cfg, today):
    out = []
    for sid in INDICATORS:
        if sid not in df or df[sid].isna().all():
            continue
        s = df[sid].dropna()
        last = s.index[-1]
        month = s[s.index <= last - pd.Timedelta(days=30)]
        recent = s[s.index > last - pd.Timedelta(days=HISTORY_DAYS)]
        out.append(
            {
                "id": sid,
                "name": cfg["series"][sid]["name"],
                "value": r(s.iloc[-1], 4),
                "day": (last - pd.Timedelta(days=cfg["series"][sid].get("lag", 0))).date().isoformat(),
                "month_ago": r(month.iloc[-1], 4) if len(month) else None,
                "spark": [r(x, 4) for x in recent.iloc[::3]],
            }
        )
    return out

"""Kalshi's public market data, no key: event probabilities for the market source of a question, settled
results for resolving it, and event dates for the calendar."""

import logging
import time

from .sources import get_json

log = logging.getLogger("board.data")
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"


def get(path):
    return get_json(f"{KALSHI}/{path}", {"Accept": "application/json"})


def price(m):
    """A market's probability: the bid/ask mid while the spread is tight, else the last trade."""

    def f(k):
        return float(m[k]) if m.get(k) not in (None, "") else None

    bid, ask, last = f("yes_bid_dollars"), f("yes_ask_dollars"), f("last_price_dollars")
    if bid is not None and ask is not None and ask > 0 and ask - bid <= 0.1:
        return (bid + ask) / 2
    return last


def event(ticker):
    """{market ticker: (label, probability, status, result)} of one event, with its strike date."""
    e = get(f"events/{ticker}?with_nested_markets=true")
    ev = e.get("event", e)
    markets = ev.get("markets") or e.get("markets") or []
    return ev.get("strike_date"), {
        m["ticker"]: (
            m.get("yes_sub_title") or m.get("subtitle") or m["ticker"],
            price(m),
            m.get("status"),
            m.get("result"),
        )
        for m in markets
    }


def events(series, n=3):
    """The next `n` open events of a series: [(event ticker, strike date or the first market's close)]."""
    evs = get(f"events?series_ticker={series}&status=open&limit=20&with_nested_markets=true")["events"]
    out = []
    for e in evs:
        when = e.get("strike_date") or min(
            (m.get("close_time") for m in e.get("markets", []) if m.get("close_time")), default=None
        )
        if when:
            out.append((e["event_ticker"], when[:10]))
    return sorted(out, key=lambda x: x[1])[:n]


def calendar():
    """Upcoming FOMC decisions, CPI and payroll releases, from Kalshi's event dates: [(day, what)]."""
    out = []
    for series, what in (("KXFEDDECISION", "FOMC decision"), ("KXCPI", "CPI"), ("KXPAYROLLS", "Payrolls")):
        try:
            out += [(d, what) for _, d in events(series, 4)]
        except Exception as e:
            log.warning("calendar %s: %s", series, e)
        time.sleep(0.2)
    return sorted(out)

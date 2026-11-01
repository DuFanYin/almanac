"""The one place the account comes in (holdings_today), and the account part of the page."""

import logging

from ..data import account as account_data
from ..model import allocation, asset_classes, portfolio, rules

log = logging.getLogger("board")


def holdings_today(db, cfg, df):
    """The one place the account comes in: the positions by category against the rules' categories, their
    shares, the regular purchase, the positions' lineage node. None without a positions file: everything else
    runs the same, a rule that reads a share cannot be judged, and no account view is written."""
    exposure, node = account_data.exposure(db)
    if not exposure:
        return None
    btc_px = float(df["BTC"].dropna().iloc[-1]) if "BTC" in df and df["BTC"].notna().any() else None
    h = portfolio.holdings(exposure, cfg["rules"].get("symbols", {}), btc_px, cfg["rules"])
    usd, rhythm = portfolio.dca(exposure)
    return {
        "exposure": exposure,
        "node": node,
        "h": h,
        "shares": portfolio.shares(h, cfg["rules"]),
        "dca": {"usd": usd, "rhythm": rhythm},
    }


def names_today(db, cfg, held, df, days, today):
    """The single names held in size (portfolio.concentrated), each against its class's index over every window of
    `days` (portfolio.name_gap): [{"symbol", "category", "class", "share", "p10", "median", "worst", "windows",
    "node"}]; a name without three independent windows is listed without its gap."""
    if not held:
        return []
    rcfg = cfg["rules"]
    big = portfolio.concentrated(held["h"], rcfg)
    prices = account_data.name_prices(db, [s for s, _, _ in big], today) if big else {}
    out = []
    for sym, cat, share in big:
        cls = rcfg["categories"][cat]["class"]
        index = asset_classes.stand_in(df, asset_classes.proxy_ids(cfg["asset_classes"]["class"][cls]))
        series, node = prices.get(sym, (None, None))
        gap = portfolio.name_gap(series, index.dropna(), days) if series is not None else None
        out.append({"symbol": sym, "category": cat, "class": cls, "share": share, "node": node, **(gap or {})})
    return out


def account_view(cfg, held, standing, today, you, horizons, decision=None, trades=()):
    """The account part of the page (data/account/): the holdings against the targets and the decided shares (the
    decision step's: steps/decide.py), each standing firing's amount, and the holdings by asset class against your
    targets, with both on each horizon's futures (steps/distribution.py): the expected return, the bad case against
    the limit, what the worst 10% is made of; and what you hold after the suggested trades on the same futures,
    beside it as it is."""
    rcfg, h, sh = rules.rules_config(cfg), held["h"], held["shares"]
    tg, exposure = rcfg["targets"], held["exposure"]
    amounts = {}
    for tid, r in standing:
        a = portfolio.amounts(r["amount"], h, sh, tg, rcfg, today, held["dca"]["usd"]) if r.get("amount") else None
        if a:
            amounts[tid] = {"usd": a[0], "how": a[1]}
    against = None
    if you and horizons:
        now = you["now"] or portfolio.by_class(h, rcfg)
        after = dict(now)  # the trades are shares of the account; what you hold by class is of everything
        scale = (h["nav"] or 0) / (h["total"] or 1)
        cats = portfolio.categories(rcfg)
        for t in trades:
            if t.get("kind") == "trade" and t.get("category") in cats:
                k = cats[t["category"]]["class"]
                after[k] = after.get(k, 0.0) + t["size"] * scale
                after["cash"] = after.get("cash", 0.0) - t["size"] * scale
        judged = set.intersection(*(set(d["futures"].returns) for d in horizons.values()))
        against = {
            "profile": you["profile"],
            "label": you["label"],
            "now": now,
            "policy": you["policy"],
            "mix": (decision or {}).get("mix") or you["policy"],
            "after": after,
            "outside": sorted(k for k in now if k not in judged),
            "horizons": {
                hk: {
                    "label": d["label"],
                    "limit": d["limit"],
                    "holding": (d["held"] or {}).get("all"),
                    "targets": d["stats"]["targets"]["all"],
                    "holding_no_llm": (d["held"] or {}).get("no_llm"),
                    "after": allocation.stats(after, d["futures"]) if after != now else None,
                    "worst_parts": (d["held"] or {}).get("worst_parts"),
                }
                for hk, d in horizons.items()
            },
        }
    return {
        "date": exposure.get("date"),
        "csv_end": exposure.get("csv_end"),
        "missing": exposure.get("missing"),
        "by": h["by"],
        "rows": h["rows"],
        "nav": h["nav"],
        "total": h["total"],
        "btc_value": h["btc_value"],
        "shares": sh,
        "targets": tg,
        "categories": [{"id": k, **c} for k, c in portfolio.categories(rcfg).items()],
        "unclassified": h["unclassified"],
        "dca": held["dca"],
        "amounts": amounts,
        "against": against,
        "decided": (decision or {}).get("categories") or {},
        "exposure_node": held["node"],
    }

"""The holdings by category, against the targets, by asset class, and the single names held in size.

Inputs: the positions (data/account.py: a file outside this repository); each symbol's category, set by hand
([symbols] in local/rules.toml; an option counts under its underlying); BTC's daily close.
The categories and their targets are in the rules (local/rules.toml); each names its asset class
(config/asset_classes.toml), on whose futures it is judged (model/allocation.py). Pure: no files."""

import statistics

import pandas as pd

UNCLASSIFIED = "unclassified"
BTC_CLASS = "bitcoin"  # the asset class of Bitcoin held directly


def categories(rcfg):
    return rcfg["categories"]


def targets(rcfg):
    """Every target the rules and the page compare against: each category's, the tech total, the caps, Bitcoin's."""
    cats = categories(rcfg)
    out = {k: c.get("target", 0.0) for k, c in cats.items()}
    out["tech"] = sum(c.get("target", 0.0) for c in cats.values() if c.get("tech"))
    out.update({f"{k}_cap": c["cap"] for k, c in cats.items() if "cap" in c})
    out.update(rcfg.get("targets", {}))
    return out


def holdings(exposure, kinds, btc_price, rcfg):
    """Values by category (USD), each holding with its category ({symbol: category} as the owner set it), the
    IBKR NAV, the total with BTC, and the symbols not yet put in a category."""
    cats = categories(rcfg)
    by = {k: 0.0 for k in [*cats, UNCLASSIFIED]}
    options = {}
    rows = {}
    for p in exposure["ibkr"]["positions"]:
        sym = p["underlying"]
        cat = kinds.get(sym) if kinds.get(sym) in cats else UNCLASSIFIED
        by[cat] += p["mv"] or 0.0
        if p["asset"] == "Option":
            options[cat] = options.get(cat, 0.0) + (p["mv"] or 0.0)
        row = rows.setdefault(sym, {"symbol": sym, "category": cat, "value": 0.0, "options": False})
        row["value"] += p["mv"] or 0.0
        row["options"] |= p["asset"] == "Option"
    if "cash" in by:
        by["cash"] += exposure["ibkr"]["cash"]
    btc = exposure.get("btc") or {}
    btc_value = (btc.get("qty") or 0) * btc_price if btc_price else 0.0
    nav = exposure["ibkr"]["nav"]
    return {
        "by": by,
        "options": options,
        "rows": sorted(rows.values(), key=lambda r: -r["value"]),
        "nav": nav,
        "total": nav + btc_value,
        "btc_value": btc_value,
        "unclassified": sorted(s for s, r in rows.items() if r["category"] == UNCLASSIFIED),
    }


def shares(h, rcfg):
    """Each category's share of the IBKR NAV; tech, the tech categories together; btc, Bitcoin held directly
    and through the btc categories, of everything."""
    cats, nav, total = categories(rcfg), h["nav"] or 1, h["total"] or 1
    out = {k: v / nav for k, v in h["by"].items()}
    out["tech"] = sum(out[k] for k, c in cats.items() if c.get("tech"))
    out["btc"] = (h["btc_value"] + sum(h["by"][k] for k, c in cats.items() if c.get("btc"))) / total
    return out


def months_left(pace, today):
    last = pd.Period(pace["trim_last"], "M")
    now = max(pd.Period(today, "M"), pd.Period(pace["trim_start"][:7], "M"))
    return max(1, (last - now).n + 1)


def dca(exposure):
    """The regular purchase: the median of the last six purchases in USD, and its rhythm (week / month)."""
    buys = (exposure.get("btc") or {}).get("buys") or []
    if not buys:
        return None, "month"
    last = buys[-6:]
    days = [(pd.Timestamp(b["date"]) - pd.Timestamp(a["date"])).days for a, b in zip(last, last[1:])]
    gap = statistics.median(days) if days else 30
    return statistics.median(b["usd"] for b in last), ("week" if gap <= 10 else "month")


def amounts(kind, h, sh, tg, rcfg, today, dca_usd):
    """A rule's suggested amount in USD with a line saying how it was reached; None for rules without one."""
    nav, pace, cats = h["nav"], rcfg["pace"], categories(rcfg)
    if kind == "trim_month":
        n = months_left(pace, today)
        excess = max(0.0, sh["tech"] - tg["tech"]) * nav
        first = [
            f"{c['label'].lower()} first ({max(0.0, sh[k] - c['cap']) * nav:,.0f} above their cap)"
            for k, c in cats.items()
            if "cap" in c and sh.get(k, 0) > c["cap"]
        ]
        return excess / n, f"{excess:,.0f} above the tech target, spread over {n} months" + (
            "; " + "; ".join(first) if first else ""
        )
    if kind == "new_high_step":
        return pace["new_high_step"] * nav, f"{pace['new_high_step']:.0%} of the account"
    if kind == "buyback_step":
        cash = max(0.0, h["by"].get("cash", 0.0))
        step = pace["buyback_step"] * nav
        return min(step, cash), (
            f"{pace['buyback_step']:.0%} of the account, from cash and short Treasuries"
            + (f"; only {cash:,.0f} of it there" if cash < step else "")
        )
    if kind == "fear_step":
        cash = max(0.0, h["by"].get("cash", 0.0))
        step = pace["fear_step"] * nav
        return min(step, cash), (
            f"{pace['fear_step']:.0%} of the account, from cash"
            + (f"; only {cash:,.0f} of it there" if cash < step else "")
        )
    if kind == "to_cash_step":
        # from the stock category most above its target into cash and short Treasuries (Bitcoin has its own rules)
        safe = ("cash", "treasuries", "gold")
        stocks = [k for k, c in cats.items() if c.get("class") not in safe and not c.get("btc")]
        over = {k: sh.get(k, 0) - tg.get(k, 0) for k in stocks}
        k = max(over, key=over.get, default=None)
        if k is None:
            return None
        held = max(0.0, h["by"].get(k, 0.0))
        step = pace["to_cash_step"] * nav
        return min(step, held), (
            f"{pace['to_cash_step']:.0%} of the account, into cash from {cats[k]['label'].lower()}"
            + " (the most above its target)"
            + (f"; only {held:,.0f} held there" if held < step else "")
        )
    if kind == "dca" and dca_usd:
        return dca_usd, "the median of the last six purchases"
    if kind == "dca_half" and dca_usd:
        return dca_usd / 2, "half the median of the last six purchases"
    return None


def by_class(h, rcfg):
    """The holdings by asset class, as shares of everything: each category under its class, Bitcoin held directly
    under BTC_CLASS, options and the unclassified apart (their value says nothing about a class)."""
    cats, total = categories(rcfg), h["total"] or 1
    opts = h.get("options", {})
    out = {}
    for k, v in h["by"].items():
        if not v:
            continue
        key = cats[k]["class"] if k in cats else k
        out[key] = out.get(key, 0.0) + (v - opts.get(k, 0.0)) / total
    if sum(opts.values()):
        out["options"] = sum(opts.values()) / total
    if h["btc_value"]:
        out[BTC_CLASS] = out.get(BTC_CLASS, 0.0) + h["btc_value"] / total
    return {k: x for k, x in out.items() if x}


CONCENTRATED = 0.10  # a single name at least this share of everything is judged on its own too


def concentrated(h, rcfg, threshold=CONCENTRATED):
    """[(symbol, its category, share of everything)] for the single names held at least `threshold` (options on it
    counted at their value)."""
    total, cats = h["total"] or 1, categories(rcfg)
    return [
        (r["symbol"], r["category"], r["value"] / total)
        for r in h["rows"]
        if r["category"] in cats and r["value"] / total >= threshold
    ]


def name_gap(name, index, horizon_days, step=7):
    """How a single name fared against its class's index over every window of the horizon both cover: its return
    over the index's, (1 + r_name) / (1 + r_index) - 1, one window every `step` days. {"p10", "median", "windows"
    (independent), "worst"}; None with under three independent windows."""
    both = (
        pd.concat([name, index], axis=1, keys=["n", "i"]).dropna().resample("D").last().ffill(limit=7)
        if len(name) and len(index)
        else None
    )
    if both is None or both.empty:
        return None
    gaps, t = [], both.index[0]
    last = both.index[-1] - pd.Timedelta(days=horizon_days)
    while t <= last:
        end = t + pd.Timedelta(days=horizon_days)
        a, b = both.loc[t], both.loc[end] if end in both.index else None
        if b is not None and a.notna().all() and b.notna().all():
            gaps.append((b["n"] / a["n"]) / (b["i"] / a["i"]) - 1)
        t += pd.Timedelta(days=step)
    n_eff = len(gaps) * step / horizon_days
    if n_eff < 3:
        return None
    xs = sorted(gaps)
    return {"p10": xs[len(xs) // 10], "median": statistics.median(xs), "windows": round(n_eff, 1), "worst": xs[0]}


def with_names(bad_by, names):
    """Each horizon's bad case with the single names at their own worst tenth against their class as well (as if
    both happened at once: a harsh case): bad + sum of share x (1 + bad) x the name's p10 gap, where it is a loss.
    names: [{"share", "p10"}]."""
    return {
        g: (b + sum(n["share"] * (1 + b) * min(0.0, n["p10"]) for n in names) if b is not None else None)
        for g, b in bad_by.items()
    }

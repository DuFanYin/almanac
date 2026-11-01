"""What the scenarios' LLMs are told, in words: the factors and where today's signposts move them,
what the simulation is built from, the scenarios standing and how earlier ones turned out, each scenario's share of
the simulation, the paths no scenario covers, what the market's prices say, and the reviewer's tools on the board's
own data."""

import json

import numpy as np
import pandas as pd

from ...data import llm
from ...model import scenarios as sm
from ...model.allocation import quantile
from .sets import lengths

RECORD = 8  # how many scored sets the writers and readers are told


QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)  # history's spread of each factor over each length, as the LLMs are told


UNITS = {  # how the LLMs are told each kind of factor's value over a length
    "ret": "its return over the length, decimal: -0.25 for a 25% fall",
    "ann": "its yearly rate over the length, decimal: 0.03 for 3% a year; a level, not a change",
    "diff": "its change over the length, in its points: 0.8 for +0.8",
}


COVER = 0.10  # more of the simulation than this in no scenario: the writer is told where, and adds scenarios once


def factors_text(fcfg, world, days=None):
    """The factors as the LLMs read them: each one's unit; over each length with scenarios (or `days` only),
    history's QUANTILES, and where today's signposts move the simulation (its mean shift and spread)."""
    lines = []
    days = [x[1] for x in lengths(fcfg).values()] if days is None else [days]
    for k, f in fcfg["factor"].items():
        unit = UNITS[f["measure"]]
        parts, lean = [], []
        for d in days:
            a = world.weights(d)
            if not len(a["X"]) or k not in a["names"]:
                continue
            j = a["names"].index(k)
            q = np.quantile(a["X"][:, j], QUANTILES)
            parts.append(f"{d} days " + " / ".join(f"{x:.3g}" for x in q))
            if abs(a["moved"][j]) >= 0.05 or abs(a["spread"][j] - 1) >= 0.05:
                lean.append(f"{d} days: mean {a['moved'][j]:+.2f} sd, spread x{a['spread'][j]:.2f}")
        if not parts:
            continue
        said = "; today's signposts: " + ", ".join(lean) if lean else "; today's signposts: no lean"
        lines.append(f"- {k}: {f['label']} ({unit}); history " + "; ".join(parts) + said)
    return "\n".join(lines)


def simulation_text(world, scfg):
    """What the simulation is built from, in words: each signpost read today (its dial, model/world.py dial_today)
    and which way its own push takes each factor (up or down; its spread wider or narrower)."""
    names = {s["id"]: s.get("name", s["id"]) for s in (scfg or {}).get("signpost", [])}
    parts = getattr(world, "parts", None) or {}
    lines = []
    for s in getattr(world, "signals", None) or []:
        if s["dial"] is None:
            continue
        p = parts.get(s["id"], {})
        moves = [
            f"{f} {'up' if m > 0 else 'down'}"
            for f, v in p.get("move", {}).items()
            if abs(m := float(np.mean(v))) >= 1e-3
        ]
        moves += [
            f"{f}'s spread {'wider' if m > 0 else 'narrower'}"
            for f, v in p.get("spread", {}).items()
            if abs(m := float(np.mean(v))) >= 1e-3
        ]
        lines.append(f"- {names.get(s['id'], s['id'])}: {s['dial']:+.2f}; " + (", ".join(moves) or "no push today"))
    return "\n".join(lines) or "(no signpost read today: history unweighted)"


def standing_text(sets, labels):
    """The scenarios standing, in words, for the writer: each length's, with its factor and share when written."""
    lines = []
    for h, s in sets.items():
        lines.append(f"{h} ({labels.get(h, h)}), written {s['written']}:")
        for x in s["scenarios"]:
            r = "; ".join(f"{k} {llm.span(lo, hi)}" for k, (lo, hi) in x["ranges"].items())
            lines.append(
                f"  - {x['name']} (the simulation {x.get('base', 0):.1%}, x{x.get('times', 1):.3g}, so "
                f"{x['weight']:.1%}): {x['story']} Ranges: {r}."
            )
    return "\n".join(lines)


def record_text(db, labels, n=RECORD):
    """How the latest scored sets turned out, in words: each scenario, whether it happened, the factor it was given,
    the simulation's share and its final probability when written."""
    lines = []
    for h, written, how in db.execute(
        "SELECT horizon, written, how FROM scenario_sets WHERE how IS NOT NULL ORDER BY expires DESC LIMIT ?", (n,)
    ):
        got = json.loads(how)
        parts = [x for x in got.get("parts", []) if x.get("hit") is not None]
        if not parts:
            continue
        lines.append(f"- {labels.get(h, h)} from {written}:")
        lines += [
            f"  - {x['name']}: {'happened' if x['hit'] else 'did not happen'}; factor x{x.get('times') or 1:.3g} on "
            f"the simulation's {x.get('share') or 0:.1%}, so {x.get('weight') or 0:.1%}"
            for x in parts
        ]
    return "\n".join(lines)


def share_words(world, lengths_, sets_):
    """{length: {scenario: its share of the simulation and of history unweighted, in words}} (for the readers)."""
    out = {}
    for h, scs in sets_.items():
        got = sm.at_length(world.weights(lengths_[h][1]), scs)
        out[h] = {}
        for x, e in zip(scs, (got or {}).get("each", [{}] * len(scs))):
            if not e:
                out[h][x["id"]] = "unknown (no window of history at this length)"
                continue
            made = (
                f"; history has {e['episodes']} such windows that do not overlap, so it is made from the nearest "
                "ones moved into its ranges"
                if e["made"]
                else ""
            )
            out[h][x["id"]] = (
                f"the simulation {e['base']:.2%} of its paths (as estimated from the {got['periods']} windows of "
                f"history at this length that do not overlap), history unweighted {e['history']:.2%}{made}"
            )
    return out


def uncovered(world, lengths_, sets_):
    """{length: in words, the simulated paths in no scenario's ranges, for the writer to tell them too} for the lengths
    with more than COVER of the simulation outside: how much, how much of the S&P 500's worst and best tenth of paths
    is among them, and what every factor did over them (the weighted median, 10th to 90th, against all of history's
    median)."""
    out = {}
    for h, scs in sets_.items():
        a = world.weights(lengths_[h][1])
        if not len(a["X"]):
            continue
        m = ~sm.inside(a["X"], a["names"], scs).any(1)
        left = float(a["w"][m].sum())
        if left <= COVER:
            continue
        w = a["w"][m] / left
        said = [f"{left:.0%} of the simulated paths are in no scenario's ranges"]
        if "stocks" in a["names"]:
            x = a["X"][:, a["names"].index("stocks")]
            lo, hi = np.quantile(x, [0.1, 0.9])
            for name, tail in (("worst", x <= lo), ("best", x >= hi)):
                said.append(
                    f"{a['w'][m & tail].sum() / max(a['w'][tail].sum(), 1e-12):.0%} of the S&P 500's {name} tenth"
                )
        seen = []
        for j, k in enumerate(a["names"]):
            v = a["X"][m, j]
            q = [float(quantile(v, w, p)) for p in (0.1, 0.5, 0.9)]
            seen.append(f"{k} {q[1]:.3g} ({q[0]:.3g} to {q[2]:.3g}; all {float(np.median(a['X'][:, j])):.3g})")
        out[h] = ", ".join(said) + "; over them: " + ", ".join(seen)
    return out


MARKET = ("VIX", "VIX3M", "SKEW")  # the series the market's prices are read from, for the readers and the reviewer


def market_text(series, today):
    """What the options market prices, in words: the VIX (and the one-month move it implies), the three-month VIX,
    SKEW, each against its own history (its percentile)."""
    out = []
    for sid in MARKET:
        x = series.get(sid)
        x = x.dropna() if x is not None else None
        if x is None or x.empty or (pd.Timestamp(today) - x.index[-1]).days > 10:
            continue
        v, rank = float(x.iloc[-1]), float((x <= x.iloc[-1]).mean())
        line = f"- {sid} {v:.1f} on {x.index[-1].date()} (its {rank:.0%} percentile since {x.index[0].year})"
        if sid == "VIX":
            month = v / 100 / np.sqrt(12)
            line += f": the options price a one-month standard deviation of about {month:.1%} for the S&P 500"
        out.append(line)
    return "\n".join(out) or "(no prices today)"


def tools(world, lengths_):
    """The reviewer's tools: the board's own data, read-only. {name: (what, function of the arguments)}."""
    names = list(world.fcfg["factor"])

    def at(args):
        h = args.get("length")
        if h not in lengths_:
            raise ValueError(f"length must be one of {sorted(lengths_)}")
        return world.weights(lengths_[h][1])

    def inside(a, ranges):
        cut = sm.clamp({k: v for k, v in (ranges or {}).items() if k in a["names"]}, a["X"], a["names"])
        if not cut:
            raise ValueError(f"ranges on the factors {a['names']}")
        m = np.ones(len(a["X"]), dtype=bool)
        for j, (lo, hi) in cut.items():
            m &= (a["X"][:, j] >= lo) & (a["X"][:, j] <= hi)
        return m

    def share(args):
        a = at(args)
        m = inside(a, args.get("ranges"))
        years = sorted({pd.Timestamp(t).year for t in a["starts"][m]})
        return (
            f"history {m.mean():.1%} of the windows ({m.sum()} of {len(m)}), the simulation {a['w'][m].sum():.1%}; "
            f"{sm.episodes(a['starts'][m], a['days'])} that do not overlap; their years: "
            + (", ".join(map(str, years[:30])) or "none")
        )

    def alongside(args):
        a = at(args)
        m = inside(a, args.get("ranges"))
        if not m.any():
            return "no window of history is in those ranges"
        return "; ".join(
            f"{k}: median {np.median(a['X'][m, j]):.3g} there, {np.median(a['X'][:, j]):.3g} in all history"
            for j, k in enumerate(a["names"])
        )

    def factor(args):
        a = at(args)
        k = args.get("factor")
        if k not in a["names"]:
            raise ValueError(f"factor must be one of {a['names']}")
        x, w = a["X"][:, a["names"].index(k)], a["w"]
        order = np.argsort(x)
        cum = np.cumsum(w[order])
        sim = [float(x[order][np.searchsorted(cum, q)]) for q in QUANTILES]
        hist = np.quantile(x, QUANTILES)
        return (
            f"{k} at {QUANTILES} history "
            + " / ".join(f"{v:.3g}" for v in hist)
            + "; the simulation "
            + " / ".join(f"{v:.3g}" for v in sim)
        )

    rng = 'args {"length": "<length id>", "ranges": {"<factor>": [low or null, high or null]}}'
    one = 'args {"length": "<length id>", "factor": "<factor>"}'
    return {
        "share": (
            f"how often history's windows of a length fell in some ranges, weighted and not, and when; {rng}",
            share,
        ),
        "alongside": (f"what every factor did in those windows against all history (medians); {rng}", alongside),
        "factor": (f"one factor's spread at a length, history and the simulation; {one} ({', '.join(names)})", factor),
    }

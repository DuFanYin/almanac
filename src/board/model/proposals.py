"""Checks on a question the LLM proposes, before the owner sees it; pure, on the board's data.

1. history's base rate can be computed: at least MIN_EFF independent windows, and no outcome rarer than EDGE or
   commoner than 1 - EDGE (a question whose answer history already knows tells nothing);
2. the board does not ask it already: on the start days it shares with each question the board asks, its outcomes
   are not nearly a function of the other's (normalised mutual information below MAX_OVERLAP);
3. for a "decision" question, it would change a decision: the owner's targets' return over the asset classes'
   horizon differs between its outcomes past the noise (the 90% interval of the gap between its best and worst
   outcome's medians, a bootstrap over separate stretches of windows, excludes 0). A "score" question resolves
   within SCORE_DAYS so the board's forecasters are scored sooner, and needs only 1 and 2.
The shape (fields, conditions, a last bucket that takes the rest) is checked where it is drafted
(src/llm/methods/propose.py).

And checks on a signpost the LLM proposes (check_signpost), before it is tried: it can be computed from the series the
board keeps, it has SP_YEARS of history, its red line is crossed on SP_RED days, and it is not nearly a signpost the
board has (the absolute correlation of their values below SP_SAME).

And on a factor it proposes (check_factor), before the owner sees it: its series covers the factors' whole history,
it is not nearly a factor the board has, and the asset classes' fit to the factors explains more out of sample with it
(model/response.py holdout_r2)."""

import copy
import math
from datetime import timedelta

import numpy as np
import pandas as pd

from ..core import expr
from . import asset_classes, factors, response
from . import questions as qs

MIN_EFF = 8.0  # independent windows behind its base rate
EDGE = 0.03  # no outcome's base rate under this or over 1 - this
MAX_OVERLAP = 0.8  # normalised mutual information with a question the board asks
SCORE_DAYS = 92  # a "score" question resolves within this
MIN_SHARED = 20  # start days two questions must share to be compared
TOL = pd.Timedelta(days=7)  # start days this close count as the same


def as_question(p, s, today):
    """The proposal as the board would register it today (its spec frozen as questions.due does): {"id", "kind",
    "buckets", "spec"}, and its horizon in days. s: its series."""
    spec = {k: p[k] for k in ("series", "detail") if k in p} | {"template": p["id"]}
    if p["kind"] == "path":
        spec |= {
            "closes": (today + timedelta(days=p["days"])).isoformat(),
            "start_day": s.index[-1].date().isoformat(),
            "start_px": float(s.iloc[-1]),
        }
        horizon = p["days"]
    else:
        first, last, _ = qs.period_bounds(p["every"], today)
        spec |= {"every": p["every"], "measure": p["measure"], "first": first.isoformat(), "last": last.isoformat()}
        horizon = (last - first).days + 1
    return {"id": p["id"], "kind": p["kind"], "buckets": p["buckets"], "spec": spec}, horizon


def align(a, b, tol=TOL):
    """[(a's value, b's value)] on the start days of a ({day: x}) that b has within tol."""
    if not a or not b:
        return []
    bs = pd.Series(b).sort_index()
    idx = bs.index.get_indexer(list(a), method="nearest", tolerance=tol)
    return [(x, bs.iloc[i]) for x, i in zip(a.values(), idx) if i >= 0]


def nmi(pairs):
    """Normalised mutual information of two label sequences [(a, b)]: 0 unrelated, 1 one fixes the other."""
    n = len(pairs)
    if not n:
        return 0.0
    pa, pb, pab = {}, {}, {}
    for a, b in pairs:
        pa[a], pb[b], pab[(a, b)] = pa.get(a, 0) + 1, pb.get(b, 0) + 1, pab.get((a, b), 0) + 1
    h = lambda c: -sum(v / n * math.log(v / n) for v in c.values())  # noqa: E731
    ha, hb = h(pa), h(pb)
    if not ha or not hb:
        return 0.0
    mi = sum(v / n * math.log(v * n / (pa[a] * pb[b])) for (a, b), v in pab.items())
    return mi / math.sqrt(ha * hb)


def overlap(outcomes, others):
    """(the highest normalised mutual information with a question the board asks, its id); others: {id: {day:
    bucket}} as they stood when each opened."""
    best = (0.0, None)
    for qid, outs in others.items():
        pairs = align(outcomes, outs)
        if len(pairs) >= MIN_SHARED:
            best = max(best, (nmi(pairs), qid), key=lambda x: x[0])
    return best


def say_history(n_eff, base, labels):
    """The "enough history" check in words: how many separate cases history has, and how often each outcome came."""
    if not base:
        return "history cannot answer it"
    rates = "; ".join(f"{labels.get(k, k)}: {x:.0%}" for k, x in base.items())
    return (
        f"about {n_eff:.0f} separate periods in history to count from (at least {MIN_EFF:.0f}). How often each "
        f"outcome came in them (none under {EDGE:.0%} or over {1 - EDGE:.0%}): {rates}"
    )


def say_overlap(v, title):
    """The "not a repeat" check in words: the forecast already asked that it overlaps most."""
    if not title:
        return "no forecast already asked is close"
    return (
        f"closest forecast already asked: {title}, overlapping {v:.0%} (100% is the same question; at most "
        f"{MAX_OVERLAP:.0%})"
    )


def say_alike(kind, name, c, most):
    """A "not a copy" check in words: the closest indicator or factor the board has, by how alike their values move."""
    if not name:
        return f"no close {kind}"
    return f"closest {kind}: {name}, its values moving alike at a correlation of {c:.2f} (at most {most:.2f})"


def decision_value(outcomes, mix, horizon_days, seed=11):
    """How the owner's targets fared by the proposal's outcome: {"by_outcome": {bucket: (median, windows)}, "gap"
    (the best outcome's median minus the worst's), "gap_90", "between": (best, worst)}; None unless two outcomes
    have MIN_WINDOWS separate windows. mix: the targets' return over the classes' horizon by start day ({day: r});
    each of the proposal's start days takes the nearest within TOL."""
    ms = pd.Series(mix).sort_index()
    idx = ms.index.get_indexer(list(outcomes), method="nearest", tolerance=TOL)
    by = {}
    for (d, b), i in zip(outcomes.items(), idx):
        if i >= 0:
            by.setdefault(b, {})[d] = float(ms.iloc[i])
    by = {b: r for b, r in by.items() if round(len(r) * 7 / horizon_days) >= asset_classes.MIN_WINDOWS}
    if len(by) < 2:
        return None
    med = {b: float(np.median(list(r.values()))) for b, r in by.items()}
    hi, lo = max(med, key=med.get), min(med, key=med.get)
    draws = (
        asset_classes.resample(by[hi], horizon_days, seed)[0]
        - asset_classes.resample(by[lo], horizon_days, seed + 1)[0]
    )
    return {
        "by_outcome": {b: (med[b], len(r)) for b, r in by.items()},
        "gap": med[hi] - med[lo],
        "gap_90": [float(x) for x in np.quantile(draws, [0.05, 0.95])],
        "between": (hi, lo),
    }


def check(p, s, today, others, mix, mix_horizon, titles=None):
    """{"passed", "checks": [{"name", "ok", "say"}], "base", "n_eff", "overlap", "value", "horizon"} for a shaped
    proposal p on its series s; others: the board's questions' outcomes in history ({id: {day: bucket}}); mix: the
    owner's targets' return by start day over mix_horizon days (None without targets); titles: {id: title} of the
    questions in others, for the owner to read."""
    labels = {b["id"]: b.get("label", b["id"]) for b in p["buckets"]}
    q, horizon = as_question(p, s, today)
    out = {"checks": [], "horizon": horizon}

    def add(name, ok, say):
        out["checks"].append({"name": name, "ok": bool(ok), "say": say})
        return ok

    if p["purpose"] == "score" and not add(
        "resolves soon",
        horizon <= SCORE_DAYS,
        f"resolves in {horizon} days (a forecast to score by must resolve within {SCORE_DAYS})",
    ):
        out["passed"] = False
        return out
    outcomes, n_per = qs.history_outcomes(q, s)
    p_base, n_eff = qs.base_rate(q, outcomes or {}, n_per or 0)
    out["base"], out["n_eff"] = p_base, n_eff
    ok = add(
        "enough history",
        p_base and n_eff >= MIN_EFF and all(EDGE <= x <= 1 - EDGE for x in p_base.values()),
        say_history(n_eff, p_base, labels),
    )
    if not ok:
        out["passed"] = False
        return out
    v, qid = overlap(outcomes, others)
    out["overlap"] = (v, qid)
    ok = add("not a repeat", v < MAX_OVERLAP, say_overlap(v, (titles or {}).get(qid, qid) if qid else None))
    if p["purpose"] == "decision":
        val = decision_value(outcomes, mix, mix_horizon) if mix else None
        out["value"] = val
        say = "too few separate periods of your targets in its outcomes to tell"
        if val:
            (hi, lo), m = val["between"], val["by_outcome"]
            say = (
                f"your targets' typical {round(mix_horizon / 365)}-year return: {labels.get(hi, hi)} {m[hi][0]:+.0%}, "
                f"{labels.get(lo, lo)} {m[lo][0]:+.0%} (the gap, 90% sure: {val['gap_90'][0]:+.0%} to "
                f"{val['gap_90'][1]:+.0%}; it must not include 0)"
            )
        ok = add("changes a decision", val and (val["gap_90"][0] > 0 or val["gap_90"][1] < 0), say) and ok
    out["passed"] = bool(ok)
    return out


SP_YEARS = 10  # years of history a proposed signpost needs
SP_RED = (0.02, 0.5)  # the share of days it is red, at least and at most
SP_SAME = 0.9  # the absolute correlation with a signpost the board has, at most


def check_signpost(sp, df, existing, years=SP_YEARS, names=None):
    """{"passed", "checks": [{"name", "ok", "say"}], "red_share", "same_as"} for a proposed signpost (its board form)
    on the frame df; existing: {id: values Series} of the signposts the board has; years: of history it needs (one
    on a reading, which starts the day the readings did); names: {id: name} of those signposts."""
    out = {"checks": []}

    def add(name, ok, say):
        out["checks"].append({"name": name, "ok": bool(ok), "say": say})
        return ok

    missing = sorted(set(expr.names(sp["value"])) - set(df.columns))
    if not add("can be computed", not missing, f"unknown series: {missing}" if missing else "every series is kept"):
        out["passed"] = False
        return out
    try:
        v = expr.evaluate(sp["value"], df)
        st = expr.states(v, sp["red"], sp.get("yellow"))
    except Exception as e:
        add("can be computed", False, f"{type(e).__name__}: {e}"[:200])
        out["passed"] = False
        return out
    v = v.replace([np.inf, -np.inf], np.nan).dropna()
    have = (v.index[-1] - v.index[0]).days / 365 if len(v) else 0
    add("enough history", have >= years, f"{have:.0f} years of values (at least {years:g})")
    seen = st[st != "grey"]
    share = float((seen == "red").mean()) if len(seen) else 0.0
    out["red_share"] = round(share, 3)
    add(
        "red now and then",
        SP_RED[0] <= share <= SP_RED[1],
        f"red on {share:.0%} of days (between {SP_RED[0]:.0%} and {SP_RED[1]:.0%})",
    )
    weekly = v.resample("7D").last()
    same, best = None, 0.0
    for sid, other in existing.items():
        both = pd.concat([weekly, other.replace([np.inf, -np.inf], np.nan).resample("7D").last()], axis=1).dropna()
        if len(both) >= 156:
            c = abs(float(both.iloc[:, 0].corr(both.iloc[:, 1])))
            if np.isfinite(c) and c > best:
                same, best = sid, c
    out["same_as"] = same
    add("not a copy", best < SP_SAME, say_alike("indicator", (names or {}).get(same, same), best, SP_SAME))
    out["passed"] = all(c["ok"] for c in out["checks"])
    return out


F_SAME = 0.9  # the absolute correlation with a factor the board has, at most (over the one-year windows)
F_GAIN = 0.02  # the out-of-sample share explained a factor must add, on average over the classes, at some horizon
F_LOSS = 0.01  # and lose at no horizon more than this


def check_factor(f, fcfg, series, frame, classes, horizon="long"):
    """{"passed", "checks", "gains": {horizon: mean change}, "by_class": {horizon: {class: [without, with]}}} for a
    proposed factor ({"id", "series", "measure", ...}); the mean over the classes, a share explained below 0 counted as
    0 (a fit worse than the mean explains nothing: Bitcoin's noise moves nothing either way, a class the factor comes to
    explain counts in full)."""
    out = {"checks": []}

    def add(name, ok, say):
        out["checks"].append({"name": name, "ok": bool(ok), "say": say})
        return ok

    if not add(
        "new",
        f["id"] not in fcfg["factor"],
        f"{f.get('label', f['id'])} {'is' if f['id'] in fcfg['factor'] else 'is not'} a factor already",
    ):
        out["passed"] = False
        return out
    first = pd.Timestamp(fcfg["history_from"])
    t, v = factors.history(f, series, fcfg["horizon"][horizon]["days"], fcfg["history_from"])
    starts = pd.to_datetime(t[0]) if len(t) else None
    if not add(
        "covers the history",
        starts is not None and starts <= first + pd.Timedelta(days=365),
        f"its data from {starts.year if starts is not None else 'nowhere'} (the factors' history starts {first.year})",
    ):
        out["passed"] = False
        return out
    with_f = copy.deepcopy(fcfg)
    with_f["factor"][f["id"]] = {k: f[k] for k in ("series", "measure", "label", "up", "unit") if k in f}
    tt, vals = factors.matrix(with_f, series, fcfg["horizon"][horizon]["days"])
    same, best = None, 0.0
    for k, x in vals.items():
        if k != f["id"] and len(tt) > 10:
            c = abs(float(np.corrcoef(vals[f["id"]], x)[0, 1]))
            if np.isfinite(c) and c > best:
                same, best = k, c
    label = with_f["factor"].get(same, {}).get("label", same)
    add("not a copy", best < F_SAME, say_alike("factor", label, best, F_SAME))
    out["gains"], out["by_class"] = {}, {}
    for hk, h in fcfg["horizon"].items():
        before = response.holdout_r2(fcfg, series, frame, classes, h["days"])
        after = response.holdout_r2(with_f, series, frame, classes, h["days"])
        both = {k: [round(before[k], 4), round(after[k], 4)] for k in before if k in after}
        used = [max(a, 0.0) - max(b, 0.0) for b, a in both.values()]
        out["by_class"][hk] = both
        out["gains"][hk] = round(float(np.mean(used)), 4) if used else None
    g = [x for x in out["gains"].values() if x is not None]
    add(
        "explains more",
        g and max(g) >= F_GAIN and min(g) >= -F_LOSS,
        "share of the classes' returns explained, in points: "
        + ", ".join(
            f"{fcfg['horizon'][h].get('label', h)} {x * 100:+.1f}" for h, x in out["gains"].items() if x is not None
        )
        + f" (needs +{F_GAIN * 100:.0f} at some length and nowhere below -{F_LOSS * 100:.0f})",
    )
    out["passed"] = all(c["ok"] for c in out["checks"])
    return out

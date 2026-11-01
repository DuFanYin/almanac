"""Signposts and probabilities. Signposts: each one's value and state (green / yellow / red) on every day of history
(core/expr.py). Probabilities: sources pooled by weight (the world and a market), the owner's view on top, and the
Brier and log scores of each source's daily forecasts once a question resolves. Pure: no database."""

import math
import statistics

from ..core import expr


def evaluate(signposts, df):
    """{id: (values Series, states Series)}; a signpost whose value cannot be computed is left out, with its error."""
    out, errors = {}, {}
    for sp in signposts:
        try:
            v = expr.evaluate(sp["value"], df)
            out[sp["id"]] = (v, expr.states(v, sp["red"], sp.get("yellow")))
        except Exception as e:  # a missing series, a bad expression
            errors[sp["id"]] = f"{type(e).__name__}: {e}"
    return out, errors


def weights_for(qcfg, all_weights, has_market):
    return qcfg.get("weights") or all_weights["with_market" if has_market else "without_market"]


def normalised(p, keys):
    total = sum(max(p.get(k, 0), 0) for k in keys)
    return {k: max(p.get(k, 0), 0) / total for k in keys} if total else None


def pool(sources, weights, keys):
    """Weighted mean of the sources present ({source: {bucket: p}}, each normalised over the question's
    buckets); weights of absent sources drop out."""
    sources = {s: normalised(p, keys) for s, p in sources.items() if p}
    present = {s: w for s, w in weights.items() if sources.get(s)}
    total = sum(present.values())
    if not total:
        return None, {}
    p = {k: sum(w * sources[s][k] for s, w in present.items()) / total for k in keys}
    return p, {s: round(w / total, 4) for s, w in present.items()}


Z90 = 3.29  # a normal's 90% interval, in standard deviations


def final(computed, llm, w, own, w_own, keys):
    """The computed probability moved toward the panel's by w, then the owner's view pooled on top by w_own (each
    side absent: its weight drops out)."""
    p = {k: computed[k] + (w * (llm[k] - computed[k]) if llm else 0.0) for k in keys}
    if own:
        o = normalised(own, keys)
        p = {k: (1 - w_own) * p[k] + w_own * o[k] for k in keys}
    return p


def brier(probs, outcome):
    return sum((p - (1.0 if k == outcome else 0.0)) ** 2 for k, p in probs.items())


def log_score(probs, outcome):
    return -math.log(max(probs.get(outcome, 0), 1e-6))


def scores(forecasts):
    """Per source: the mean over resolved questions of the daily Brier and log scores, plus the even guess as
    the baseline. forecasts: {question: (outcome, number of buckets, {source: [{bucket: p}, one a day]})}."""
    per = {}
    for outcome, k, by_source in forecasts.values():
        uniform = {str(i): 1 / k for i in range(k)}
        days = {s: [(brier(ps, outcome), log_score(ps, outcome)) for ps in xs] for s, xs in by_source.items() if xs}
        days["uniform"] = [(brier(uniform, "0"), log_score(uniform, "0"))]
        for source, xs in days.items():
            per.setdefault(source, []).append([statistics.mean(c) for c in zip(*xs)])
    return {
        s: {"brier": statistics.mean(x[0] for x in xs), "log": statistics.mean(x[1] for x in xs), "questions": len(xs)}
        for s, xs in per.items()
    }

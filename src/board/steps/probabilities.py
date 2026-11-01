"""Each open question's probability, read off the world (model/world.py read): its own outcome after every start day
of history since the factors' history_from (what has happened in it so far kept: questions.outcomes_now), weighted as
the world weighs that start day over the days the question has left. Kept daily with its lineage, by source:
  base     history's start days unweighted: the base rate, what usual is
  math     the data's world (the signposts and how long each one's news lasts), pooled with Kalshi's market where it
           has one (weights.with_market)
  futures  the final world (the LLM's scenarios over the data's, model/scenarios.py), pooled with the market as math
           is (before your own view)
  model    the final: that, your own view on top (weights.own); what the page reads and what is scored
  no_llm   the same without the LLM: no scenarios, and none of the signposts with a model upstream
  market, own   as they came
Each with its independent windows (n_eff) and a 90% band from them. A forecast drives no suggestion: the questions
score the world and show it; those standing furthest from usual are the week's key ones (model/questions.py key)."""

from datetime import date

import numpy as np

from ..core import lineage
from ..data import account as account_data
from ..data import observations
from ..model import prob
from ..model import questions as qs
from ..model import scenarios as sm
from ..model import world as wm


def judgment(db, qid):
    """The owner's latest own view of a question (from the page's inputs), recorded."""
    j = account_data.judgment(qid)
    if not j:
        return None, None
    probs, note, at = j
    node = lineage.record(db, "judgment", f"own:{qid}", {"p": probs, "at": at, "note": note}, {}, step="judgment")
    return probs, node


def days_left(q, today):
    """The days a question has left: to its close (a period's: to its last day); one with neither (a recession
    starting in a year) a year. At least a week."""
    spec = q["spec"]
    end = spec.get("last") if q["kind"] == "period" else spec.get("closes") or q.get("closes")
    return max((date.fromisoformat(end) - today).days if end else 365, 7)


def outcomes(q, series, today):
    """(its outcome after each start day of history as it stands today, the independent share of one start day, its
    outcomes as it stood when it opened); (None, None, None) without a history form."""
    spec = q["spec"]
    s = series.get(spec.get("series")) if spec.get("series") else None
    if s is None and not spec.get("base_rate"):
        return None, None, None
    outs, n_per = qs.outcomes_now(q, s, today, series.get("USREC"))
    opened = qs.history_outcomes(q, s, series.get("USREC"))
    return outs, n_per, (opened if opened and opened[0] else None)


def band(p, n_eff):
    """{bucket: [lo, hi]}: the 90% band of each probability from its independent windows."""
    out = {}
    for k, x in p.items():
        half = 1.645 * np.sqrt(x * (1 - x) / max(n_eff, 1.0))
        out[k] = [max(0.0, x - half), min(1.0, x + half)]
    return out


def read(db, cfg, chash, open_qs, series, worlds, scen, today, world_node=None):
    """({question: {"sources", "weights", "math", "model", "no_llm", "n_eff", "interval", "math_interval", "node",
    "math_node", ...}}, {template: (question, its outcomes as it opened)}): every open question read off the world.
    worlds: {"data", "clean"} (steps/world.py build); scen: the LLM's scenarios (steps/scenarios/world.py
    for_world), the world's last layer, carried to the question's days from the nearest length that has them."""
    sets = (scen or {}).get("sets") or {}
    wcfg = cfg["questions"]["weights"]
    out, history = {}, {}
    for q in open_qs:
        if q["kind"] == "factor":
            continue
        keys = [b["id"] for b in q["buckets"]]
        tmpl = q["spec"]["template"]
        sources, nodes = {}, {}
        if q.get("market"):
            sources["market"], nodes["market"] = q["market"]
        outs, n_per, opened = outcomes(q, series, today)
        if opened and tmpl not in history:
            history[tmpl] = (q, opened)
        sides, n_eff, length, w_llm, came = {}, 0.0, days_left(q, today), None, None
        if outs:
            a, c = worlds["data"].weights(length), worlds["clean"].weights(length)
            # a window made for a scenario has no outcome of the question: its weight is on the window it was made from
            w_llm, came = sm.on_length(sets, a, length)
            flat = np.full(len(a["w"]), 1 / max(len(a["w"]), 1))
            for name, w, at in (("base", flat, a), ("math", a["w"], a), ("futures", w_llm, a), ("no_llm", c["w"], c)):
                p, n, _ = wm.read(outs, n_per, at["starts"], w, keys)
                if p:
                    sides[name] = p
                    if name == "futures":
                        n_eff = n
        if not sides and "market" not in sources:
            continue
        if sides.get("base"):
            sources["base"] = sides["base"]
            nodes["base"] = lineage.record(
                db,
                "base_rate",
                f"base:{q['id']}",
                {"p": sides["base"], "n_eff": round(n_eff, 1)},
                {
                    "windows": len(outs),
                    "from": min(outs).date().isoformat(),
                    "to": max(outs).date().isoformat(),
                    "days": length,
                    "series": q["spec"].get("series") or "USREC",
                    "config": chash,
                },
                observations.raw_nodes(db, q["spec"].get("series") or "USREC", cfg["series"]),
                step="base_rate",
            )
        weights = prob.weights_for(cfg["questions"].get(tmpl, {}), wcfg, "market" in sources)

        def pooled(side):
            """A side of the world pooled with the market, the world in history's place."""
            got = {"base": sides.get(side), "market": sources.get("market")}
            return prob.pool({k: v for k, v in got.items() if v}, weights, keys)

        math, used = pooled("math")
        if math is None:
            continue
        with_llm, clean = pooled("futures")[0] or math, pooled("no_llm")[0] or math
        own_p, own_node = judgment(db, q["id"])
        w_own = float(cfg["questions"].get(tmpl, {}).get("own", wcfg["own"])) if own_p else 0.0
        if own_p:
            sources["own"], nodes["own"] = own_p, own_node
        model = prob.final(with_llm, None, 0.0, own_p, w_own, keys)
        no_llm = prob.final(clean, None, 0.0, own_p, w_own, keys)
        math_node = lineage.record(
            db,
            "computed",
            f"math:{q['id']}",
            {k: round(v, 6) for k, v in math.items()},
            {"weights": used, "n_eff": round(n_eff, 1), "days": length, "config": chash},
            [nodes.get("base"), nodes.get("market"), world_node],
            step="compute",
        )
        node = lineage.record(
            db,
            "probability",
            f"prob:{q['id']}",
            {k: round(v, 6) for k, v in model.items()},
            {"futures": sides.get("futures"), "scenarios_from": came, "own": w_own, "config": chash},
            [math_node, *{x["set"]["node"] for x in sets.values()}, nodes.get("own")],
            step="balance",
        )
        if sides.get("futures"):  # the final world, pooled with the market as the data's is
            sources["futures"] = with_llm
        day = today.isoformat()
        rows = [(day, q["id"], k, s, float(p), chash, nodes.get(s)) for s, ps in sources.items() for k, p in ps.items()]
        rows += [(day, q["id"], k, "math", float(v), chash, math_node) for k, v in math.items()]
        rows += [(day, q["id"], k, "model", float(v), chash, node) for k, v in model.items()]
        rows += [(day, q["id"], k, "no_llm", float(v), chash, None) for k, v in no_llm.items()]
        db.executemany("INSERT OR REPLACE INTO probs VALUES (?,?,?,?,?,?,?)", rows)
        out[q["id"]] = {
            "sources": sources,
            "weights": {k: v for k, v in (("world", used.get("base")), ("market", used.get("market"))) if v},
            "scenarios_from": came,
            "math": math,
            "model": model,
            "no_llm": no_llm,
            "n_eff": n_eff,
            "interval": band(model, n_eff) if n_eff else {},
            "math_interval": band(math, n_eff) if n_eff else {},
            "node": node,
            "math_node": math_node,
        }
    db.commit()
    return out, history

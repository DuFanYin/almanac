"""The futures at each length the world is read at (config/factors.toml horizon), built daily from the world: every
start day of history once (and the windows made for a scenario history has too few of), weighed by the final world:
the LLM's scenarios over the simulation (steps/scenarios/, model/scenarios.py; a length without scenarios of its
own, a week, carried from the nearest that has them); each class's return over the length from the factors
(model/response.py) plus the unexplained part of the same window. The allocations judged on them exactly
(model/allocation.py): your targets (or your risk level's base allocation), your holdings, the base allocation,
60/40; beside them the simulation alone (the data's world), the same without anything the LLM made and history
unweighted, on the same windows. The bad-case limit of each length: a year's from your risk level, every other
length's as strict on your targets' own history. Each scenario seen at its length; each factor's quartiles over each
length, for the outlook (steps/world.py). Recorded per length, with the world and the scenario sets as parents."""

import numpy as np

from ..core import lineage
from ..model import allocation, response
from ..model import distribution as dist
from ..model import scenarios as sm
from ..model import world as wm


def limits(cfg, you, bad_history):
    """{horizon: the bad-case limit}: a year's is your risk level's worst_loss times year_limit; every other
    length's the year's times your targets' bad case in history at that length over theirs at a year
    (bad_history: {horizon: that bad case}), so each length allows the same depth past what is usual for them."""
    if not you:
        return {}
    year = -you["limits"]["worst_loss"] * cfg["factors"].get("year_limit", 0.8)
    base = bad_history.get("long")
    if not base:
        return {h: year for h in bad_history}
    return {h: year * b / base for h, b in bad_history.items()}


def in_scenarios(fut, scenarios):
    """{scenario: whether each future is one of its windows}."""
    return {f["id"]: np.isin(fut.start, f["starts"]) for f in scenarios}


def worst_parts(weights, fut, inside, best=False):
    """{scenario: the share of the allocation's worst 10% (best: its best 10%) in its windows}, largest first (a
    window may be in several scenarios or in none, so the shares need not add up to 1)."""
    r, _ = allocation.returns(weights, fut)
    if r is None or not inside:
        return {}
    t = allocation.tail(-r if best else r, fut.w)
    out = {k: float(t[m].sum()) for k, m in inside.items()}
    return dict(sorted(out.items(), key=lambda x: -x[1]))


def mean_in(fut, x, m):
    """The weighted mean of x over the futures in m, or None."""
    s = float(fut.w[m].sum())
    return float(fut.w[m] @ x[m]) / s if s > 0 else None


def seen(a, f, fut, data, inside, mix, usual=None):
    """A scenario as a view (a: World.weights at its length): how often history's windows fall in its ranges, in all
    and factor by factor; which way it takes each factor; each class's mean return in its windows, and how much of the
    world the LLM moved to them (moved: their final weight less the simulation's); and for the allocation (your
    targets), their share of the world, its mean return in them, their share of its worst 10%, and its side: better
    than usual (usual: its mean return over history unweighted at this length) or worse. The side is computed, never
    asked: what a scenario means for you is a result of what happens in it."""
    names = a["names"]
    each = {}
    for k, (lo, hi) in f["ranges"].items():
        if k in names and len(a["X"]):
            x = a["X"][:, names.index(k)]
            each[k] = float(((lo is None or x >= lo) & (hi is None or x <= hi)).mean())
    m = inside[f["id"]]
    out = {
        "history_like": f["p_history"],
        "weight": f["p"],
        "history_like_by_factor": each,
        "leans": dist.leans(type("H", (), {"X": a["X"]}), names, f["ranges"]) if len(a["X"]) else {},
        "classes": {k: mean_in(fut, r, m) for k, r in fut.returns.items()} if fut.w[m].sum() > 0 else {},
        "p_data": f["p_data"],
        "p": f["p"],
        "moved": float(fut.w[m].sum() - data.w[m].sum()),
    }
    r, _ = allocation.returns(mix, fut) if mix else (None, None)
    if r is not None:
        mean = mean_in(fut, r, m)
        out["targets"] = {
            "share": float(fut.w[m].sum()),
            "mean": mean,
            "worst": float(allocation.tail(r, fut.w)[m].sum()),
            "side": None if mean is None or usual is None else "better" if mean > usual else "worse",
        }
    return out


def sides(r, usual, fut, data, inside):
    """Your targets over a length, path by path: {"worse" (their return below usual), "better" (at it or above),
    "none" (the paths in no scenario)}, each {"p" (the final world's), "data" (the simulation's)}. Read off the paths,
    not added up from the scenarios: they may overlap, and some paths are in none. {} without returns."""
    if r is None or usual is None:
        return {}
    low = r < usual
    out = {"worse": low, "better": ~low}
    if inside:
        out["none"] = ~np.any(np.array(list(inside.values())), axis=0)
    return {k: {"p": round(float(fut.w[m].sum()), 4), "data": round(float(data.w[m].sum()), 4)} for k, m in out.items()}


def build(db, cfg, series, frame, worlds, scen, you, held_mix=None, today=None, world_node=None):
    """{horizon: {"label", "days", "limit", "r2", "n_eff", "classes", "classes_data", "classes_no_llm",
    "classes_history", "stats" {allocation: {"all", "data", "no_llm", "history", "worst_parts", "best_parts"}}, "held",
    "llm_moved", "from" (the length its scenarios are, or carried from), "scenarios" {scenario: its view}, "sides"
    (sides(): your targets below usual or not, and the paths in no scenario), "factors"
    {factor: {"label", "edges", "final", "data", "no_llm", "history"}}, "node", "futures", "data_futures",
    "no_llm_futures", "history_futures"}}. worlds: steps/world.py build; scen: steps/scenarios/world.py for_world."""
    fcfg = cfg.get("factors")
    if not fcfg:
        return {}
    classes = cfg["asset_classes"]["class"]
    own = (scen or {}).get("sets") or {}
    live = (scen or {}).get("live") or []
    mixes = {"targets": you["policy"]} if you else {}
    base = cfg["asset_classes"]["profile"][str(you["profile"])]["policy"] if you else None
    if base:
        mixes["base"] = dict(base)
    if "60/40" in cfg["asset_classes"].get("baseline", {}):
        mixes["60/40"] = dict(cfg["asset_classes"]["baseline"]["60/40"])
    shown_key = "targets" if "targets" in mixes else next(iter(mixes), None)
    shown = mixes.get(shown_key)
    out = {}
    for hk, h in sorted(fcfg["horizon"].items(), key=lambda kv: kv[1]["days"]):
        length = h["days"]
        a, c = worlds["data"].weights(length), worlds["clean"].weights(length)
        names = a["names"]
        resp = response.fit(wm.measurable(fcfg, series, length), series, frame, classes, length)
        if not len(resp.X) or len(resp.X) != len(a["w"]):
            continue
        if len(c["w"]) != len(a["w"]):
            continue
        if hk in own:  # its own scenarios: the windows made for them appended, the final weights
            ext, w_llm, came = own[hk]["ext"], own[hk]["w"], hk
            mine = [f for f in live if f["horizon"] == hk]
        else:  # a week: the nearest length's scenarios carried over by start day
            w_llm, came = sm.on_length(own, a, length)
            ext, mine = a, []
        # the scenarios its futures are in: its own, or a week's start days in those of the length it is carried from
        parts_of = mine or [f for f in live if f["horizon"] == came]
        flat_w = wm.padded(np.full(len(a["w"]), 1 / len(a["w"])), ext)
        base_w = dist.windows(resp, flat_w, ext)  # history unweighted; the other sides the same windows weighed
        fut, data, no_llm = (
            base_w.weighed(w_llm),
            base_w.weighed(wm.padded(a["w"], ext)),
            base_w.weighed(wm.padded(c["w"], ext)),
        )
        flat = base_w
        inside = in_scenarios(fut, parts_of)
        bins = (
            allocation.chart_bins([allocation.returns(shown, fut)[0], allocation.returns(shown, no_llm)[0]], [0.0])
            if shown
            else None
        )

        def judged(w):
            got = {
                "all": allocation.stats(w, fut, bins=bins),
                "data": allocation.stats(w, data, bins=bins),
                "no_llm": allocation.stats(w, no_llm, bins=bins),
                "history": allocation.stats(w, flat, bins=bins),
            }
            for key, f in (("all", fut), ("no_llm", no_llm)):  # the deepest fall along the way, for the page
                if got[key]:
                    got[key]["dd"] = allocation.drawdown(w, f)
            return got

        stats = {
            name: {
                **judged(w),
                "worst_parts": worst_parts(w, fut, inside),
                "best_parts": worst_parts(w, fut, inside, True),
            }
            for name, w in mixes.items()
        }
        held = {**judged(held_mix), "worst_parts": worst_parts(held_mix, fut, inside)} if held_mix else None
        usual = (stats.get(shown_key, {}).get("history") or {}).get("expected")
        views = {f["id"]: seen(a, f, fut, data, inside, shown, usual) for f in mine}
        facs = {}
        for j, k in enumerate(names):
            cuts = wm.edges(a["X"][:, j])
            facs[k] = {
                "label": fcfg["factor"][k]["label"],
                "edges": cuts,
                "final": wm.quartile_probs(ext["X"][:, j], w_llm, cuts),
                "no_llm": wm.quartile_probs(c["X"][:, j], c["w"], wm.edges(c["X"][:, j])),
                "data": wm.quartile_probs(a["X"][:, j], a["w"], cuts),
                "history": wm.quartile_probs(ext["X"][:, j], flat_w, cuts),
            }
        moved = 0.5 * float(np.abs(fut.w - data.w).sum())  # how much of the world the LLM's scenarios moved
        node = lineage.record(
            db,
            "distribution",
            f"distribution:{hk}",
            {
                n: {k: round(v, 4) for k, v in (x["all"] or {}).items() if isinstance(v, float)}
                for n, x in stats.items()
            },
            {"days": length, "llm_moved": round(moved, 4), "from": came, "r2": resp.r2},
            [world_node, *{x["set"]["node"] for x in own.values()}],
            step="distribution",
        )
        out[hk] = {
            "label": h["label"],
            "days": length,
            "r2": {k: (round(v, 3) if v is not None else None) for k, v in resp.r2.items()},
            "n_eff": {k: round(v, 1) for k, v in resp.n_eff.items()},
            "classes": allocation.class_stats(fut),
            "classes_data": allocation.class_stats(data),
            "classes_no_llm": allocation.class_stats(no_llm),
            "classes_history": allocation.class_stats(flat),
            "stats": stats,
            "held": held,
            "llm_moved": round(moved, 4),
            "from": came,
            "scenarios": views,
            "sides": sides(
                allocation.returns(shown, fut)[0] if shown else None, usual, fut, data, inside if mine else {}
            ),
            "factors": facs,
            "ess": round(wm.ess(w_llm), 1),
            "node": node,
            "futures": fut,  # for the actions; not written to the page
            "data_futures": data,
            "no_llm_futures": no_llm,
            "history_futures": flat,
        }
    if shown:
        bad = {hk: (d["stats"]["targets"]["history"] or {}).get("bad") for hk, d in out.items()}
        lim = limits(cfg, you, {hk: b for hk, b in bad.items() if b is not None})
        for hk, d in out.items():
            d["limit"] = lim.get(hk)
    return out

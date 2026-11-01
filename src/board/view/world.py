"""The world on the page (model/world.py): the stretches its pushes are over, today's pushes on each factor, each
factor's mean shift and spread at each length the world is read at (its own push and what it followed of the others),
how much evidence the weights keep (the simulation's, and the final world's with the LLM's scenarios)."""

from .fmt import r


def world_view(world, horizons):
    """{"stretches", "push" {"move"|"spread": {factor: [per stretch]}}, "factors" {factor: {"label", "mu"
    {horizon: sd}, "var" {horizon: multiple}}}, "ess" {horizon: {"data", "all", "of"}}}."""
    if not world:
        return None
    w = world["data"]
    names = list(w.fcfg["factor"])
    factors = {k: {"label": w.fcfg["factor"][k]["label"], "mu": {}, "var": {}} for k in names}
    ess = {}
    for hk, d in (horizons or {}).items():
        a = w.weights(d["days"])
        for j, k in enumerate(a["names"]):
            factors[k]["mu"][hk] = r(float(a["moved"][j]), 3)
            factors[k]["var"][hk] = r(float(a["spread"][j]), 3)
        ess[hk] = {"data": r(a["ess"], 0), "all": d.get("ess"), "of": len(a["w"])}
    return {
        "stretches": [list(s) for s in w.segs],
        "push": {kind: {f: [r(float(x), 4) for x in v] for f, v in by.items()} for kind, by in w.push.items()},
        "factors": factors,
        "ess": ess,
    }

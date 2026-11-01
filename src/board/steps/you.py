"""Layer 5, you: what the view is turned into decisions for. Your style (its limits), your policy (the categories'
targets by asset class; without them, your style's own mix), your rules, and, when there are holdings, what you hold
by class. Without holdings everything else is the same: the policy stands in for the holdings. Recorded as one node,
so an action's lineage runs back to you as well as to the world (steps/actions.py)."""

from ..core import lineage
from ..model import portfolio, rules


def you(db, cfg, held):
    """{"mode" ("holdings" or "policy"), "profile", "label", "limits", "policy" {class: share}, "own" (the policy is
    your targets, not a style's), "now" {class: share} (holdings only), "rules", "node"}."""
    rcfg = cfg["rules"]
    n = rcfg.get("profile") if rcfg.get("profile") is not None else 3  # no style chosen: the middle one
    pr = rules.profile(cfg) or cfg["asset_classes"]["profile"][str(n)]
    policy = {}
    for c in (rcfg.get("categories") or {}).values():
        policy[c["class"]] = policy.get(c["class"], 0.0) + c.get("target", 0.0)
    policy = {k: w for k, w in policy.items() if w > 0}
    own = bool(policy)
    if not own:
        policy = dict(pr["policy"])
    now = portfolio.by_class(held["h"], rules.rules_config(cfg)) if held else None
    limits = {k: pr[k] for k in ("worst_loss", "cash_min", "class_cap", "tilt", "band", "step")}
    node = lineage.record(
        db,
        "you",
        "you",
        {"profile": int(n), "policy": {k: round(w, 4) for k, w in sorted(policy.items())}, "limits": limits},
        {"rules": rules.version(cfg), "own": own, "holdings": bool(held)},
        [held["node"]] if held else [],
        step="you",
    )
    return {
        "mode": "holdings" if held else "policy",
        "profile": int(n),
        "label": pr["label"],
        "limits": limits,
        "policy": policy,
        "own": own,
        "now": now,
        "rules": [{"id": r["id"], "title": r["title"]} for r in rcfg.get("rule", [])],
        "node": node,
    }

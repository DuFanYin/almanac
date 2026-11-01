"""The page's record: what the decision step rests on (audit), how often the LLM's writing passed its checks, each
version of the panel's method as measured."""

import json
import math

from ..core import store
from .fmt import r


def audit(decision):
    """What the decision step rests on, class by class: at each length that acts, how far today's world stands from
    usual (gap), what chance alone would give a reweighting as large (se), what is left after the noise (shrunk),
    the lengths together (score) and the lean it made; and whether a guardrail took it back. {"classes": [...],
    "counts": {status: n}, "sense", "weights"}."""
    if not decision:
        return {"classes": [], "counts": {}}
    rows = []
    for k, score in sorted(decision["scores"].items(), key=lambda kv: -abs(kv[1])):
        lean = decision["lean"].get(k, 0.0)
        by = {h: g[k] for h, g in decision["gaps"].items() if k in g}
        status = (
            "cash: takes what the others give up"
            if k == "cash"
            else "not in your targets"
            if k not in (decision.get("targets") or {})
            else "moved"
            if abs(lean) >= 0.001
            else "too small to move"
        )
        rows.append(
            {
                "class": k,
                "score": r(score, 3),
                "lean": r(lean, 4),
                "lengths": {h: {f: r(x[f], 4) for f in ("gap", "se", "shrunk", "data", "llm")} for h, x in by.items()},
                "status": status,
            }
        )
    counts = {}
    for x in rows:
        counts[x["status"]] = counts.get(x["status"], 0) + 1
    return {
        "classes": rows,
        "counts": counts,
        "guard": decision.get("guard") or [],
        "sense": decision.get("sense"),
        "weights": decision.get("weights"),
    }


def text_checks(db, last=12):
    """How often the LLM's writing passed its checks, over the last runs: the briefs and the red team (every number one
    of the data's),
    research's notes (points kept against dropped), the panel's answers (usable against failed)."""
    briefs, red = (
        [
            json.loads(m or "{}").get("valid")
            for (m,) in db.execute("SELECT meta FROM texts WHERE kind = ? ORDER BY day DESC LIMIT ?", (kind, last))
        ]
        for kind in ("brief", "redteam")
    )
    kept = dropped = 0
    for (m,) in db.execute("SELECT meta FROM artifact WHERE kind = 'research' ORDER BY id DESC LIMIT ?", (last,)):
        for rnd in json.loads(m or "{}").get("rounds") or []:
            kept, dropped = kept + rnd.get("kept", 0), dropped + len(rnd.get("dropped") or [])
    ok = bad = 0
    for (m,) in db.execute("SELECT meta FROM artifact WHERE kind = 'llm_panel' ORDER BY id DESC LIMIT ?", (last * 4,)):
        for a in json.loads(m or "{}").get("answers") or []:
            ok, bad = (ok, bad + 1) if a.get("error") else (ok + 1, bad)
    share = lambda a, b: r(a / (a + b), 3) if a + b else None  # noqa: E731
    return {
        "brief": {"passed": share(sum(1 for x in briefs if x), sum(1 for x in briefs if x is False)), "n": len(briefs)},
        "redteam": {"passed": share(sum(1 for x in red if x), sum(1 for x in red if x is False)), "n": len(red)},
        "notes": {"passed": share(kept, dropped), "n": kept + dropped},
        "panel": {"passed": share(ok, bad), "n": ok + bad},
    }


def evals():
    """Each version of the LLM's method as measured on the same snapshots (data/eval/), newest first."""
    out = []
    for f in sorted((store.DATA / "eval").glob("*.json")) if (store.DATA / "eval").exists() else []:
        try:
            e = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        out.append(
            {
                "method": e["method"],
                "day": e["day"],
                "cost_usd": e.get("cost_usd"),
                "overall": {k: r(v, 4) for k, v in e["overall"].items()},
            }
        )
    return sorted(out, key=lambda e: e["day"], reverse=True)


def scenarios_record(db):
    """How the LLM's scenarios turned out: each set scored once its length was over (live or replaced: a record of
    only the sets kept would flatter it), each scenario in it happened or not, against its final probability when
    written (the simulation's share times the readers' factor), the simulation's share and history's. {"n" (scenarios
    scored), "llm", "data", "history" (the mean log score of each, log p when it happened and log (1 - p) when not:
    higher is better), "by_horizon": {length: the same}, "sets": [the latest RECORD_SHOWN scenarios: {"horizon",
    "written", "name", "hit", "weight", "share", "history", "times"}]}; n 0 before any."""
    rows = []
    for h, written, how in db.execute(
        "SELECT horizon, written, how FROM scenario_sets WHERE how IS NOT NULL ORDER BY written DESC"
    ):
        for x in json.loads(how).get("parts", []):
            if x.get("hit") is None:
                continue
            rows.append(
                {
                    "horizon": h,
                    "written": written,
                    "name": x["name"],
                    "hit": x["hit"],
                    "times": x.get("times"),
                    **{k: x.get(k) for k in SIDES},
                }
            )

    def summary(xs):
        n = len(xs)
        out = {"n": n}
        for k in SIDES:
            ok = [(x[k], x["hit"]) for x in xs if x.get(k) is not None]
            out[k if k != "weight" else "llm"] = (
                r(sum(math.log(max(p if hit else 1 - p, 1e-6)) for p, hit in ok) / len(ok), 4) if ok else None
            )
        return out

    return {
        **summary(rows),
        "by_horizon": {
            h: summary([x for x in rows if x["horizon"] == h]) for h in sorted({x["horizon"] for x in rows})
        },
        "sets": [{k: (r(v, 4) if isinstance(v, float) else v) for k, v in x.items()} for x in rows[:RECORD_SHOWN]],
    }


SIDES = ("weight", "share", "history")  # the final probability, the simulation's share, history's
RECORD_SHOWN = 20

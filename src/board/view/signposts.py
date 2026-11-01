"""The signposts on the page: today's state and reading, where it reads against its own history (its dial), what it
says about the factors (its expectations, each against what was measured: agrees, flat, against, unmeasured, and its
size) and what it pushes today (model/world.py)."""

from ..core import expr
from .fmt import r


def said(sp, names):
    """A signpost's definition in words: what it reads and where it turns yellow and red."""
    words, pc = expr.say(sp["value"], names)
    lines = [f"red {expr.say_line(sp['red'], pc)}"]
    if sp.get("yellow"):
        lines.append(f"yellow {expr.say_line(sp['yellow'], pc)}")
    return f"{words[:1].upper()}{words[1:]}; {', '.join(lines)}"


def signpost_rows(cfg, sp_today, sp_errors, world=None):
    """[signpost]: as the page shows it; world: steps/world.py build (its data world's signals and their parts)."""
    names = {sid: c["name"] for sid, c in cfg["series"].items()}
    signals = {s["id"]: s for s in (world["data"].signals if world else [])}
    parts = world["data"].parts if world else {}
    out = []
    for sp in cfg["signposts"]["signpost"]:
        t = sp_today.get(sp["id"], {})
        s = signals.get(sp["id"]) or {}
        push = {
            kind: {f: [r(x, 4) for x in v] for f, v in by.items() if any(abs(x) > 1e-6 for x in v)}
            for kind, by in (parts.get(sp["id"]) or {}).items()
        }
        out.append(
            {
                "id": sp["id"],
                "driver": sp["driver"],
                "name": sp["name"],
                "expr": sp["value"],
                "said": said(sp, names),
                "red": sp["red"],
                "yellow": sp.get("yellow"),
                "group": sp.get("group"),
                "value": t.get("value"),
                "state": t.get("state", "grey"),
                "since": t.get("since"),
                "spark": t.get("spark"),
                "as_of": t.get("as_of"),
                "node": t.get("node"),
                "dial": r(s.get("dial"), 3),
                "share": s.get("share"),
                "expect": [
                    {k: (r(v, 3) if isinstance(v, float) else v) for k, v in e.items()} for e in s.get("judged") or []
                ]
                or sp.get("expect", []),
                "push": {k: v for k, v in push.items() if v},
                "error": sp_errors.get(sp["id"]),
                "proposed": sp.get("proposed"),  # proposed by the LLM on this day (steps/proposals/signposts.py)
            }
        )
    return out

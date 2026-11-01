"""The rules written in advance (local/rules.toml, else config/rules.example.toml): which hold today, and which
fire this period. Since 10-31 they are preferences beside the decision step (model/decide.py): Bitcoin's regular
purchase, its cap, its pause and its review. A rule reads values, signposts, dates and account shares; never a
forecast (forecasts do not drive suggestions).

A rule fires at most once per period: its trigger id is rule:period (tech_trim:2026-11). A firing is kept
in the database (rule, period, day: no amounts) until the owner marks it done or skipped on the page; its
amount is account data and goes to data/account/ with the rest of the account view."""

import hashlib
import json
from datetime import date

import pandas as pd

from ..core import expr
from . import portfolio
from .questions import OPS

PERIODS = {"week", "month", "quarter", "dca", "once", "event"}
REARM_DAYS = 20  # an event rule fires again only after its condition has not held this many days running


def period_key(period, today, dca_rhythm):
    """The firing's period: a rule fires at most once in it. An event rule's key is its day: whether it may
    fire is up to rearmed()."""
    if period not in PERIODS:
        raise ValueError(f"unknown period {period!r}")
    if period == "event":
        return today.isoformat()
    if period == "dca":
        period = dca_rhythm
    if period == "week":
        y, w, _ = today.isocalendar()
        return f"{y}-W{w:02d}"
    if period == "month":
        return f"{today:%Y-%m}"
    if period == "quarter":
        return f"{today.year}Q{(today.month - 1) // 3 + 1}"
    return "once"


class Context:
    """What the conditions read: today's frame row, signpost states, account shares."""

    def __init__(self, cfg, df, states, shares, today):
        self.cfg, self.df, self.states, self.shares, self.today = cfg, df, states, shares, today
        self.values = {}

    def number(self, x, key=None):
        if isinstance(x, (int, float)):
            return float(x)
        if x == "target":
            return float(self.cfg["targets"][key])
        return float({**self.cfg["targets"], **self.cfg.get("pace", {})}[x])

    def value(self, text):
        if text not in self.values:
            v = expr.evaluate(text, self.df)
            self.values[text] = float(v.iloc[-1]) if pd.notna(v.iloc[-1]) else None
        return self.values[text]

    def holds(self, name):
        """True, False, or None when it cannot be judged today (a stale value, a signpost without data, no
        holdings)."""
        groups = self.cfg.get("groups", {})
        if name in groups:
            return any_of([all_of([self.holds(c) for c in alt]) for alt in groups[name]["any"]])
        c = self.cfg["conditions"][name]
        if "signpost" in c:
            st = self.states.get(c["signpost"])
            return None if st in (None, "grey") else st in c["state"]
        if "value" in c:
            v = self.value(c["value"])
            return None if v is None else OPS[c["op"]](v, self.number(c["than"]))
        if "share" in c:
            if self.shares is None:
                return None
            key = c["share"] if c["than"] == "target" else None
            return OPS[c["op"]](self.shares[c["share"]], self.number(c["than"], key))
        if "date" in c:
            return OPS[c["date"]](self.today, date.fromisoformat(str(self.number_or_text(c["than"]))))
        if "prob" in c:
            raise ValueError(f"condition {name}: a rule no longer reads a forecast")
        raise ValueError(f"condition {name}: unknown kind")

    def number_or_text(self, x):
        return self.cfg.get("pace", {}).get(x, x) if isinstance(x, str) else x

    def explain(self, name):
        """A condition's current reading, for the page."""
        groups = self.cfg.get("groups", {})
        if name in groups:
            return {
                "name": name,
                "holds": self.holds(name),
                "group": [[self.explain(c) for c in alt] for alt in groups[name]["any"]],
            }
        c = self.cfg["conditions"][name]
        out = {"name": name, "holds": self.holds(name)}
        if "value" in c:
            out.update(value=self.value(c["value"]), expr=c["value"], op=c["op"], than=self.number(c["than"]))
        if out.get("value") is not None and isinstance(out.get("than"), float):
            out["gap"] = out["than"] - out["value"]  # how far the value must move for the condition to flip
        elif "signpost" in c:
            out.update(value=self.states.get(c["signpost"]), than=c["state"])
        elif "date" in c:
            out.update(than=str(self.number_or_text(c["than"])))
        return out


def all_of(xs):
    return False if False in xs else None if None in xs else True


def any_of(xs):
    return True if True in xs else None if None in xs else False


def evaluate(cfg, ctx):
    """[(rule, holds, readings)] for every rule. A rule holds only when it can be judged: every `all`
    condition true, one `any` true, and every `none` condition known to be false (a check that cannot be
    read today, a thesis-failure condition on stale data, stops the rule rather than letting it through)."""
    out = []
    for r in cfg["rule"]:
        names = [*r.get("all", []), *r.get("any", []), *r.get("none", [])]
        try:
            if r.get("period", "once") not in PERIODS:
                raise ValueError(f"unknown period {r.get('period')!r}")
            ok = (
                all_of([ctx.holds(c) for c in r.get("all", [])]) is True
                and (not r.get("any") or any_of([ctx.holds(c) for c in r["any"]]) is True)
                and all(ctx.holds(c) is False for c in r.get("none", []))
            )
            readings = [ctx.explain(c) for c in names]
        except Exception as e:
            ok, readings = False, [{"name": "error", "holds": False, "error": f"{type(e).__name__}: {e}"}]
        out.append((r, ok, readings))
    return out


def rearmed(days, last_fired, rearm_days=REARM_DAYS):
    """Whether an event rule may fire again: never fired, or since it last fired its condition has not held
    for rearm_days recorded days running. days: [(day, held)] in order, after last_fired."""
    if last_fired is None:
        return True
    run = best = 0
    for _, held in days:
        run = 0 if held else run + 1
        best = max(best, run)
    return best >= rearm_days


def rule_inputs(rcfg, r):
    """What a rule reads: (series, signposts, holdings?), through its named groups."""
    series, sps, holdings = set(), set(), False
    todo = [*r.get("all", []), *r.get("any", []), *r.get("none", [])]
    while todo:
        name = todo.pop()
        if name in rcfg.get("groups", {}):
            todo += [c for alt in rcfg["groups"][name]["any"] for c in alt]
            continue
        c = rcfg["conditions"][name]
        if "value" in c:
            series.update(expr.names(c["value"]))
        elif "signpost" in c:
            sps.add(c["signpost"])
        elif "share" in c:
            holdings = True
    return series, sps, holdings


# ---- the rules' configuration: the owner's style, the targets, the version


def profile(cfg):
    """The owner's style (the styles in config/asset_classes.toml), when the rules name one with `profile = N`;
    else None."""
    n = cfg["rules"].get("profile")
    return cfg["asset_classes"]["profile"][str(n)] if n is not None else None


def rules_config(cfg):
    """The rules with every target they compare against, the category ones included, and the profile's band and
    step as default values of their pace."""
    prof = profile(cfg) or {}
    pace = {**{k: prof[k] for k in ("band", "step") if k in prof}, **cfg["rules"].get("pace", {})}
    rules = cfg["rules"]
    return {**rules, "pace": pace, "targets": portfolio.targets(rules)}


def version(cfg):
    """The rules' version: their hash without the symbols' categories (a symbol filed is not a new rule)."""
    rules_only = {k: v for k, v in cfg["rules"].items() if k != "symbols"}
    return hashlib.sha1(json.dumps(rules_only, sort_keys=True, default=str).encode()).hexdigest()[:10]

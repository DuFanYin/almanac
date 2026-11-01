"""The expression language of signposts, rule conditions and backfills, evaluated on every day at once
over a frame of series (one column a series on a calendar-day index; data/observations.py builds it).
Windows are in calendar days.

  ret(x, d)   x / x d days ago - 1          chg(x, d)   x - x d days ago
  dd(x, d)    x / its d-day high - 1        gap(x, d)   x / its d-day mean - 1
  yoy(x)      ret(x, 365)                   ann(x, d)   ret(x, d) annualised
  mean, high, low (x, d): rolling           rank(x, d)  x's percentile in its last d days, 0-1
  avg(x, d)   the mean of the values in its last d days, however few (a series just begun)
  + - * / and numbers. Parsed with ast, only these names and operators: never eval."""

import ast
import operator

import numpy as np
import pandas as pd

FUNCS = {
    "ret": lambda x, d: x / x.shift(int(d)) - 1,
    "chg": lambda x, d: x - x.shift(int(d)),
    "dd": lambda x, d: x / x.rolling(int(d), min_periods=1).max() - 1,
    "gap": lambda x, d: x / x.rolling(int(d)).mean() - 1,
    "yoy": lambda x: x / x.shift(365) - 1,
    "ann": lambda x, d: (x / x.shift(int(d))) ** (365 / d) - 1,
    "mean": lambda x, d: x.rolling(int(d)).mean(),
    "avg": lambda x, d: x.rolling(int(d), min_periods=1).mean(),
    "high": lambda x, d: x.rolling(int(d)).max(),
    "low": lambda x, d: x.rolling(int(d)).min(),
    "rank": lambda x, d: x.rolling(int(d)).rank(pct=True),
}
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def evaluate(text, df):
    """The expression's value on every day of df (a Series); raises ValueError on anything not allowed."""

    def walk(n):
        if isinstance(n, ast.Expression):
            return walk(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.Name):
            if n.id not in df:
                raise ValueError(f"unknown series {n.id}")
            return df[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](walk(n.left), walk(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -walk(n.operand)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCS and not n.keywords:
            return FUNCS[n.func.id](*[walk(a) for a in n.args])
        raise ValueError(f"not allowed in a signpost: {ast.dump(n)[:80]}")

    v = walk(ast.parse(text, mode="eval"))
    if not isinstance(v, pd.Series):
        raise ValueError("a signpost must depend on a series")
    return v.replace([np.inf, -np.inf], np.nan)


def names(text):
    """The series an expression reads (not the functions it calls)."""
    nodes = list(ast.walk(ast.parse(text, mode="eval")))
    calls = {n.func.id for n in nodes if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    return sorted({n.id for n in nodes if isinstance(n, ast.Name)} - calls)


def condition(text):
    """'< -0.15' -> a test on a value."""
    op, _, num = text.strip().partition(" ")
    ops = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
    if op not in ops:
        raise ValueError(f"bad condition {text!r}")
    limit = float(num)
    return lambda v: ops[op](v, limit)


def states(values, red, yellow=None):
    """green / yellow / red for each day: red needs the red line crossed, and holds until the value is back
    a quarter of the way to the yellow line (so it does not flicker on the line); grey where it is missing."""
    is_red, is_yellow = condition(red), condition(yellow) if yellow else None
    hold = is_red
    if yellow:
        op, _, num = red.strip().partition(" ")
        r, y = float(num), float(yellow.strip().partition(" ")[2])
        hold = condition(f"{op} {r + (y - r) / 4}")
    out, cur = [], "green"
    for v in values:
        if v is None or not np.isfinite(v):
            out.append("grey")
            cur = "green"
            continue
        if is_red(v) or (cur == "red" and hold(v)):
            cur = "red"
        elif is_yellow and is_yellow(v):
            cur = "yellow"
        else:
            cur = "green"
        out.append(cur)
    return pd.Series(out, index=values.index)


def span(d, adjective=False):
    """A window of d calendar days in words: a week, 3 weeks, 2 months, a year, 200 days; as an adjective 1-week,
    3-week, 2-month, 1-year, 200-day."""
    d = int(d)
    if d % 365 == 0:
        n, unit = d // 365, "year"
    elif d % 7 == 0 and d <= 56:
        n, unit = d // 7, "week"
    elif 1 <= (m := round(d / 30.4)) <= 24 and abs(d - m * 30.4) <= 3:
        n, unit = m, "month"
    else:
        n, unit = d, "day"
    if adjective:
        return f"{n}-{unit}"
    return f"a {unit}" if n == 1 else f"{n} {unit}s"


PERCENT = {"ret", "dd", "gap", "yoy", "ann"}  # functions whose value is a fraction, said as a percentage
WORDS = {
    "ret": "% change in {x} over {d}",
    "chg": "change in {x} over {d}",
    "dd": "{x} below its {a} high",
    "gap": "{x} against its {a} average",
    "yoy": "% change in {x} over a year",
    "ann": "yearly rate of {x} over {d}",
    "mean": "the {a} average of {x}",
    "avg": "{x}, averaged over {d}",
    "high": "the {a} high of {x}",
    "low": "the {a} low of {x}",
    "rank": "where {x} stands in its last {d} (0 lowest, 1 highest)",
}
SAID_OPS = {ast.Add: "plus", ast.Sub: "minus", ast.Mult: "times", ast.Div: "over"}
SAID_TESTS = {"<": "below", "<=": "at or below", ">": "above", ">=": "at or above"}


def say(text, names=None):
    """(the expression in words, whether its value is a percentage): `chg(DFII10, 63)` -> ("change in 10-year real
    yield over 2 months", False); names: {series id: name}."""
    names = names or {}

    def walk(n, inner=False):
        if isinstance(n, ast.Expression):
            return walk(n.body)
        if isinstance(n, ast.Constant):
            return f"{n.value:g}", False
        if isinstance(n, ast.Name):
            return names.get(n.id, n.id), False
        if isinstance(n, ast.UnaryOp):
            x, pc = walk(n.operand, True)
            return f"minus {x}", pc
        if isinstance(n, ast.Call):
            f, args = n.func.id, n.args
            x = walk(args[0], True)[0]
            d = args[1].value if len(args) > 1 and isinstance(args[1], ast.Constant) else None
            words = WORDS.get(f, f + "({x}, {d})")
            return words.format(x=x, d=span(d) if d else "", a=span(d, True) if d else ""), f in PERCENT
        if isinstance(n, ast.BinOp):
            # a / b - 1: a against b, as a percentage
            if (
                isinstance(n.op, ast.Sub)
                and isinstance(n.left, ast.BinOp)
                and isinstance(n.left.op, ast.Div)
                and isinstance(n.right, ast.Constant)
                and n.right.value == 1
            ):
                return f"{walk(n.left.left, True)[0]} against {walk(n.left.right, True)[0]}", True
            if isinstance(n.op, ast.Mult) and isinstance(n.right, ast.UnaryOp | ast.Constant):
                k = walk(n.right)[0]
                if k == "minus 1":
                    x, pc = walk(n.left, True)
                    return f"the opposite of {x}", pc
            (a, pa), (b, pb) = walk(n.left, True), walk(n.right, True)
            said = f"{a} {SAID_OPS[type(n.op)]} {b}"
            return (f"({said})" if inner else said), pa and pb and isinstance(n.op, ast.Add | ast.Sub)
        return ast.unparse(n), False

    return walk(ast.parse(text, mode="eval"))


def say_line(test, percent=False):
    """A line in words: '> 0.04' -> 'above 4%' (percent) or 'above 0.04'."""
    op, _, num = test.strip().partition(" ")
    v = float(num)
    return f"{SAID_TESTS.get(op, op)} {f'{v * 100:g}%' if percent else f'{v:g}'}"

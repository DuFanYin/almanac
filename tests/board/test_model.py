"""The model layer, pure: probabilities, questions, rules, asset classes, the computed side and the panel balanced,
draws and many tests at once, coherence, rules' lines in history, the owner's marks, dividends and the decision
layer, factors, the futures and an allocation on them. (No network, no account data.)"""

from datetime import date

import numpy as np
import pandas as pd
from src.board.core import store
from src.board.model import asset_classes, behaviour, calibration, portfolio, prob, rule_history, rules
from src.board.model import questions as qs
from tests.board.support import check, tmp


def test_sources_pooled_and_scored():
    """Sources pooled by weight, each normalised over the question's buckets, a missing one dropping out; the owner's
    view on top of the final; a sure right answer scores 0."""
    pooled, used = prob.pool(
        {"base": {"a": 0.2, "b": 0.8}, "market": {"a": 0.6, "b": 0.4}, "own": None},
        {"base": 0.35, "market": 0.65, "own": 0.3},
        ["a", "b"],
    )
    check(
        "missing source drops out of the pool", set(used) == {"base", "market"} and abs(sum(used.values()) - 1) < 1e-9
    )
    check("the pool by weight", abs(pooled["a"] - (0.35 * 0.2 + 0.65 * 0.6)) < 1e-9)
    p2, _ = prob.pool({"base": {"a": 2.0, "b": 2.0}}, {"base": 1.0}, ["a", "b"])
    check("each source normalised over the question's buckets", p2 == {"a": 0.5, "b": 0.5})
    f = prob.final({"a": 0.4, "b": 0.6}, None, 0.0, {"a": 1.0, "b": 0.0}, 0.2, ["a", "b"])
    check("the owner's view on top", abs(f["a"] - 0.52) < 1e-9)
    check("Brier of a sure right answer is 0", prob.brier({"a": 1.0, "b": 0.0}, "a") == 0)


def test_questions():
    """Questions."""
    buckets = [{"id": "crash", "when": ["dd <= -0.30"]}, {"id": "up", "when": ["ret >= 0"]}, {"id": "down", "when": []}]
    path = pd.Series([100, 120, 80, 130])
    check("deepest fall from the running high", abs(qs.path_vars(path, 100)["dd"] - (80 / 120 - 1)) < 1e-12)
    check("first matching bucket wins", qs.bucket_of(buckets, qs.path_vars(path, 100)) == "crash")
    check("absolute levels scale with the start", qs.path_vars(pd.Series([1.0, 2.0]), 50000)["px_hi"] == 100000)
    first, last, label = qs.period_bounds("quarter", date(2026, 11, 5))
    check("quarter bounds", (first, last, label) == (date(2026, 10, 1), date(2026, 12, 31), "2026Q4"))
    s = pd.Series([100, 101, 99, 105], index=pd.to_datetime(["2026-10-30", "2026-11-02", "2026-11-30", "2026-12-01"]))
    check(
        "month return from the last close before it",
        abs(qs.period_vars(s, "ret", date(2026, 11, 1), date(2026, 11, 30))["ret"] + 0.01) < 1e-12,
    )
    check(
        "period open until a later value exists",
        qs.period_vars(s[:-1], "ret", date(2026, 11, 1), date(2026, 11, 30)) is None,
    )
    mk = {
        "KXFED-26DEC-H0": ("Hold", 0.2, "active", ""),
        "KXFED-26DEC-H25": ("Hike", 0.7, "active", ""),
        "KXFED-26DEC-C25": ("Cut", 0.05, "active", ""),
    }
    q = {
        "buckets": [
            {"id": "cut", "markets": ["C25"]},
            {"id": "hold", "markets": ["H0"]},
            {"id": "hike", "markets": ["H25"]},
        ]
    }
    mp = qs.market_bucket_probs(q, mk)
    check("Kalshi markets map to buckets, renormalised", abs(mp["hike"] - 0.7 / 0.95) < 1e-9)
    check("unsettled Kalshi event does not resolve", qs.kalshi_result(q, mk) is None)
    settled = {k: (v[0], v[1], "settled", "yes" if k.endswith("H25") else "no") for k, v in mk.items()}
    check("settled Kalshi event resolves", qs.kalshi_result(q, settled) == "hike")
    binary = {"buckets": [{"id": "yes", "markets": ["KXREC-27"]}, {"id": "no", "markets": []}]}
    check(
        "binary market: the other bucket is the rest",
        abs(qs.market_bucket_probs(binary, {"KXREC-27": ("Yes", 0.2, "active", "")})["no"] - 0.8) < 1e-9,
    )


def test_rules():
    """Rules."""
    days = pd.date_range("2020-01-01", periods=400, freq="D")
    df = pd.DataFrame({"A": np.linspace(100, 200, 400), "B": np.linspace(10, 5, 400)}, index=days)
    cfg = {
        "targets": {"tech": 0.55, "btc_max": 0.15},
        "pace": {"review": "2028-03-01"},
        "conditions": {
            "tech_over": {"share": "tech", "op": ">", "than": "target"},
            "cheap": {"value": "B", "op": "<", "than": 6},
            "lit": {"signpost": "s1", "state": ["red"]},
            "late": {"date": ">=", "than": "review"},
        },
        "groups": {"either": {"any": [["cheap", "lit"], ["late"]]}},
        "rule": [
            {"id": "r1", "title": "t", "period": "month", "all": ["tech_over"], "none": ["either"]},
            {"id": "r2", "title": "t", "period": "week", "any": ["cheap"]},
        ],
    }
    ctx = rules.Context(cfg, df, {"s1": "green"}, {"tech": 0.9, "btc": 0.1}, date(2026, 11, 3))
    out = {r["id"]: ok for r, ok, _ in rules.evaluate(cfg, ctx)}
    check("rule holds: share over target, group false", out == {"r1": True, "r2": True})
    ctx = rules.Context(cfg, df, {"s1": "red"}, {"tech": 0.9, "btc": 0.1}, date(2026, 11, 3))
    check("group of conditions blocks a rule", not dict((r["id"], ok) for r, ok, _ in rules.evaluate(cfg, ctx))["r1"])
    check(
        "no holdings: a share condition cannot be judged",
        rules.Context(cfg, df, {}, None, date(2026, 11, 3)).holds("tech_over") is None,
    )
    check(
        "a check that cannot be read stops the rule",
        not dict(
            (r["id"], ok)
            for r, ok, _ in rules.evaluate(
                cfg, rules.Context(cfg, df, {"s1": "grey"}, {"tech": 0.9, "btc": 0.1}, date(2026, 11, 3))
            )
        )["r1"],
    )
    check(
        "an event rule re-arms only after a run of days without",
        rules.rearmed([], None)
        and not rules.rearmed([("d1", 0), ("d2", 1), ("d3", 0)], "d0", 2)
        and rules.rearmed([("d1", 0), ("d2", 0), ("d3", 1)], "d0", 2),
    )
    check(
        "an unknown period is an error",
        rules.evaluate({**cfg, "rule": [{**cfg["rule"][0], "period": "mnth"}]}, ctx)[0][1] is False,
    )
    check(
        "period keys",
        rules.period_key("week", date(2026, 11, 3), "month") == "2026-W45"
        and rules.period_key("dca", date(2026, 11, 3), "month") == "2026-11",
    )


def test_an_open_question_keeps_what_has_happened():
    """An open question keeps what has happened."""
    pq = {
        "kind": "path",
        "buckets": [{"id": "bear", "when": ["dd <= -0.20"]}, {"id": "other", "when": []}],
        "spec": {"template": "p", "start_day": "2026-01-01", "closes": "2026-03-01", "start_px": 100.0},
    }
    hist_days = pd.date_range("2000-01-01", "2026-01-01", freq="D")
    flat = pd.Series(100.0, index=hist_days)
    fell = pd.concat([flat, pd.Series([90.0, 75.0], index=pd.to_datetime(["2026-01-10", "2026-01-20"]))])
    check("before it starts, history decides", set(qs.outcomes_now(pq, flat, "2025-12-01")[0].values()) == {"other"})
    check("a fall already taken stays taken", set(qs.outcomes_now(pq, fell, "2026-01-20")[0].values()) == {"bear"})

    roll_cfg = {"r": {"title": "R", "kind": "rolling", "series": "S", "days": 28, "buckets": [{"id": "x", "when": []}]}}
    roll = qs.due(roll_cfg, date(2026, 11, 3), set(), {"S": pd.Series([100.0], index=[pd.Timestamp("2026-11-02")])})
    check(
        "a rolling question opens a path instance each day",
        len(roll) == 1
        and roll[0][0] == "r-2026-11-03"
        and roll[0][2] == "path"
        and roll[0][4] == "2026-12-01"
        and roll[0][6]["rolling"]
        and roll[0][6]["start_px"] == 100.0,
    )
    check(
        "and only once a day",
        qs.due(
            roll_cfg, date(2026, 11, 3), {"r-2026-11-03"}, {"S": pd.Series([100.0], index=[pd.Timestamp("2026-11-02")])}
        )
        == [],
    )


def test_an_asset_class_s_stand_in():
    """An asset class's stand-in."""
    frame_cls = pd.DataFrame(
        {"A": [np.nan, np.nan, 10.0, 11.0], "B": [1.0, 2.0, 4.0, 5.0]}, index=pd.date_range("2000-01-01", periods=4)
    )
    check(
        "a stand-in carried back by the next series, scaled to meet it",
        list(asset_classes.stand_in(frame_cls, ["A", "B"])) == [2.5, 5.0, 10.0, 11.0],
    )


def test_how_far_a_rule_is_from_flipping():
    """How far a rule is from flipping; a rule no longer reads a forecast."""
    cfg = {
        "targets": {},
        "conditions": {"v": {"value": "X", "op": "<", "than": 0.5}, "p": {"prob": "q", "buckets": ["a"], "op": ">"}},
        "rule": [{"id": "r", "title": "t", "period": "week", "all": ["p"]}],
    }
    ctx = rules.Context(cfg, pd.DataFrame({"X": [0.8]}), {}, None, date(2026, 11, 3))
    check("a value condition says how far it is from its line", abs(ctx.explain("v")["gap"] + 0.3) < 1e-9)
    (_, ok, readings), *_ = rules.evaluate(cfg, ctx)
    check("a forecast condition is an error: the rule does not fire", not ok and "forecast" in readings[0]["error"])


def test_many_tested_at_once():
    """Many things tested at once: Benjamini-Hochberg keeps what passes at its rank."""
    found = calibration.discoveries({"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5}, q=0.1)
    check("Benjamini-Hochberg keeps what passes at its rank", found == {"a", "b", "c"})


def test_a_rule_s_line_walked_through_history():
    """A rule's line walked through history, and its neighbours."""
    hd = pd.date_range("2000-01-01", periods=3000, freq="D")
    wave = pd.Series(100 + 30 * np.sin(np.arange(3000) / 120), index=hd)
    ep = rule_history.episodes(wave, "<", 80.0)
    check(
        "episodes: one each time the line is crossed after 20 days away",
        len(ep) == len([1 for i in range(1, 3000) if wave.iloc[i] < 80 <= wave.iloc[i - 1]]),
    )
    rows_h, verdict = rule_history.walk(wave, wave, "<", 80.0)
    check(
        "a line walked with its neighbours: five lines, a verdict",
        len(rows_h) == 5 and verdict in ("steady", "cliff", "unclear"),
    )
    check("a line never reached: rare", rule_history.walk(wave, wave, "<", 69.0)[1] == "rare")


def test_the_owner_s_marks_against_how_each():
    """The owner's marks against how each rule leans."""
    lean_of = {"trim": "safer", "buy": "bolder"}
    marks = [("trim", "skipped")] * 3 + [("buy", "done")] * 3
    check(
        "skipping the safer rules and doing the bolder: acts bolder", behaviour.lean(marks, lean_of)["says"] == "bolder"
    )
    check("too few marks say nothing", behaviour.lean(marks[:2], lean_of)["says"] is None)
    check(
        "skipping as often on both sides: as chosen",
        behaviour.lean(
            [("trim", "skipped"), ("trim", "done")] * 2 + [("buy", "skipped"), ("buy", "done")] * 2, lean_of
        )["says"]
        == "as chosen",
    )


def test_dividends_single_names_the_decision_layer_through():
    """Dividends, single names, the decision layer through history."""
    fr = pd.DataFrame(
        {"Y": 2.0, "P": np.exp(np.arange(2000) * 0.0), "P_TR": np.exp(np.arange(2000) * 0.01 / 365)},
        index=pd.date_range("2000-01-01", periods=2000, freq="D"),
    )
    y, k = asset_classes.dividend_yield({"dividend": {"yield": "Y / 100", "like": "P"}}, fr)
    check("a fund yielding half the base: half its yield", abs(k - 0.5) < 0.01 and abs(float(y.iloc[-1]) - 0.01) < 1e-3)
    idx = pd.Series(np.linspace(100, 200, 4000), index=pd.date_range("2000-01-01", periods=4000, freq="D"))
    nm = idx * np.where(np.arange(4000) < 2000, 1.0, 0.5)  # the name halves against its index one day
    g = portfolio.name_gap(nm, idx, 365)
    check("a single name's gap to its index", g and abs(g["worst"] + 0.5) < 1e-9 and g["median"] == 0.0)
    check(
        "its worst tenth on top of the class's bad case",
        abs(portfolio.with_names({"long": -0.3}, [{"share": 0.2, "p10": -0.5}])["long"] - (-0.3 - 0.2 * 0.7 * 0.5))
        < 1e-9,
    )
    from src.board.model import backtest  # noqa: E402

    days = pd.date_range("2000-01-01", periods=800, freq="D")
    up = pd.Series(np.linspace(1, 2, 800), index=days)
    held = up.groupby(up.index.to_period("M")).head(1)
    check(
        "the measures of a value path", backtest.measures(held)["max_dd"] == 0 and backtest.measures(held)["cagr"] > 0
    )
    wins = backtest.windows(up, 365)
    check(
        "a total return's windows, one a week, each over the horizon",
        len(wins) == len(range(0, 800 - 365, 7)) and abs(wins[days[0]] - (up[days[365]] / up[days[0]] - 1)) < 1e-12,
    )


def test_factors_the_futures_an_allocation_on_them():
    """The futures, an allocation on them, a question withdrawn."""
    from src.board.model import allocation  # noqa: E402
    from src.board.model import distribution as distm  # noqa: E402
    from src.board.model import factors as fxm  # noqa: E402

    fut = distm.Futures(np.zeros((1000, 1)), {"a": np.linspace(-0.5, 0.5, 1000), "cash": np.full(1000, 0.02)})
    st = allocation.stats({"a": 1.0}, fut)
    check(
        "the bad case is the mean of the worst 10%",
        abs(st["bad"] - np.linspace(-0.5, 0.5, 1000)[:100].mean()) < 1e-3 and st["bad"] < st["p10"],
    )
    w = np.ones(1000)
    w[:50] = 2.0  # the 50 worst twice as likely: they are 100 of the 1050 weight, the worst 10% 105 of it
    heavy = allocation.stats({"a": 1.0}, fut.weighed(w))
    x = np.linspace(-0.5, 0.5, 1000)
    check(
        "under weights, the worst 10% of the weight exactly: whole futures, and what fits of the one at the cut",
        abs(heavy["bad"] - (2 * x[:50].sum() + x[50:55].sum()) / 105) < 1e-12,
    )
    w2 = w.copy()
    w2[0] += 1e-6
    check(
        "a figure moves only as the weights do: nothing drawn",
        abs(allocation.stats({"a": 1.0}, fut.weighed(w2))["bad"] - heavy["bad"]) < 1e-6,
    )
    from src.board.model import response as respm  # noqa: E402

    pdays = np.arange(0, 30) * fxm.DAY
    pw = respm.paths((pdays, np.array([100.0] * 10 + [50.0] * 10 + [100.0] * 10)), np.array([0]), 21, step=7)
    check("a window's path: each week from its start, as a multiple of it", np.allclose(pw, [[1.0, 1.0, 0.5, 1.0]]))
    two = distm.Futures(
        np.zeros((2, 1)), {"a": np.array([0.0, 0.2])}, None, {"a": np.array([0, 0])}, {"a": np.array([[1.0, 0.5, 1.0]])}
    )
    ddw = allocation.drawdown({"a": 1.0}, two)
    ddh = allocation.drawdown({"a": 1.0}, two.weighed([1.0, 3.0]))
    check(
        "the deepest fall along the way: the window's swings, its drift moved to each future's end",
        abs(ddw["bad"] + 0.5) < 1e-9 and abs(ddh["median"] - (0.5 * 1.2**0.5 - 1)) < 1e-9,
    )
    from src.board.steps import questions as q_step  # noqa: E402

    wdb = store.connect(tmp / "data" / "w.db")
    wdb.execute("INSERT INTO questions VALUES ('gone', 'G', 'path', '2026-10-01', '2028-10-31', '[]', '{}', NULL)")
    wdb.execute("INSERT INTO probs VALUES ('2026-10-29', 'gone', 'a', 'model', 1.0, NULL, NULL)")
    check("a question withdrawn", q_step.withdraw(wdb, "gone", "retired", date(2026, 10, 30)))
    check("once", not q_step.withdraw(wdb, "gone", "retired", date(2026, 10, 31)))
    check("is no longer open, and not scored", q_step.open_questions(wdb, {}) == [] and q_step.scores(wdb) == {})


def test_a_scenario_against_history_and_what_each_part_does():
    """Which way a scenario takes each factor against history; the share of an allocation's worst 10% that came from
    each scenario's windows."""
    from src.board.model import distribution as distm
    from src.board.steps import distribution as dist_step

    X = np.column_stack([np.linspace(-0.3, 0.3, 100), np.linspace(-1, 1, 100)])
    lean = distm.leans(type("H", (), {"X": X}), ["stocks", "unemp"], {"stocks": [-0.3, -0.18], "unemp": [-0.1, 0.1]})
    check(
        "a range low in history leans down; one in the middle does not lean",
        lean["stocks"]["lean"] == -1 and lean["unemp"]["lean"] == 0,
    )
    fut = distm.Futures(X, {"a": X[:, 0], "cash": np.full(100, 0.02)}, start=np.arange(100, dtype=np.int64))
    inside = {"bust": fut.start < 20, "rest": fut.start >= 20}
    parts = dist_step.worst_parts({"a": 1.0}, fut, inside)
    check("the lowest returns' scenario: the whole worst 10%", parts["bust"] == 1.0 and parts["rest"] == 0.0)
    best = dist_step.worst_parts({"a": 1.0}, fut, inside, best=True)
    check("and none of the best 10%", best["rest"] == 1.0 and list(best)[0] == "rest")


def test_how_long_a_signpost_s_news_lasts():
    """A signpost's timescale (model/timescale.py): a red spell that moves the factor over the next month is found at
    the first stretches only; one that moves nothing is not found; its pace."""
    from src.board.model import timescale as ts

    rng = np.random.default_rng(1)
    days = pd.date_range("1971-01-01", "2026-10-01", freq="D")
    red = np.zeros(len(days), dtype=bool)
    for s in rng.choice(len(days) - 60, 40, replace=False):  # 40 red spells of 20 days
        red[s : s + 20] = True
    after = np.convolve(red.astype(float), np.ones(30), "full")[: len(days)] > 0  # the 30 days from a red day on
    steps = rng.normal(0, 0.01, len(days)) + np.where(after, 0.004, 0.0)
    series = {"X": pd.Series(np.exp(np.cumsum(steps)), index=days)}
    fcfg = {"history_from": "1971-03-01", "factor": {"x": {"series": "X", "measure": "ret"}}}
    states = pd.Series(np.where(red, "red", "green"), index=days)
    values = pd.Series(red.astype(float) + rng.normal(0, 0.01, len(days)), index=days)
    out = ts.measure({"red": "> 0.5"}, values, states, fcfg, series)
    c = out["factors"]["x"]["red"]
    check("the lengths: a week to three years, evenly in log", ts.lengths()[0] == 7 and ts.lengths()[-1] == 1095)
    check("a month's move is found", c["p"] < 0.01 and c["sign"] == 1)
    check("strongest within the month after", c["strongest"][1] <= 47)
    check("and lasts no further than three months", c["lasts"] and c["lasts"][1] <= 88)
    check("its red spells counted", 30 <= out["pace"]["spells"] <= 40)
    calm = pd.Series(np.where(np.roll(red, 5000), "red", "green"), index=days)  # red at days unrelated to the moves
    c0 = ts.measure({"red": "> 0.5"}, values, calm, fcfg, series)["factors"]["x"]["red"]
    check("red days unrelated to the moves: nothing found", c0["p"] > 0.05)
    short = {"X": series["X"]["2015":]}  # a decade: too few independent windows for the third year
    c1 = ts.measure({"red": "> 0.5"}, values, states, fcfg, short)["factors"]["x"]["red"]
    check("too little data: the far stretches are not measured", max(r["to"] for r in c1["curve"]) < 1095)
    check("nor anything with ten start days", ts.curve(np.ones(10), np.zeros((10, 1)), [(0, 7)]) is None)


def test_a_signpost_s_expectations_against_its_measures():
    """What a signpost is expected to say against what was measured: the way expected sized (shrunk by its noise),
    the other way past its band not used, within the noise flat; a measured move no expectation covers listed."""
    from src.board.model import timescale as ts

    rows = [
        {"from": 0, "to": 7, "effect": 0.4, "lo": 0.2, "hi": 0.6},
        {"from": 7, "to": 13, "effect": -0.3, "lo": -0.5, "hi": -0.1},
        {"from": 13, "to": 25, "effect": -0.05, "lo": -0.4, "hi": 0.3},
    ]
    curves = {"x": {"dial": {"curve": rows, "passed": True, "lasts": [7, 13], "sign": -1}, "red": None}}
    up = {"factor": "x", "way": "up", "days": [0, 7]}
    j = ts.judge([up, {**up, "days": [7, 13]}, {**up, "days": [13, 25]}, {**up, "factor": "y"}], curves)
    check("the way expected: sized below its measure", j[0]["verdict"] == "agrees" and 0 < j[0]["size"] < 0.4)
    check("the other way past its band: against", j[1]["verdict"] == "against" and j[1]["size"] == 0)
    check("the other way within the noise: flat", j[2]["verdict"] == "flat")
    check("nothing measured", j[3]["verdict"] == "unmeasured")
    check("a strong move down no expectation covers", ts.unexplained([up], curves)[0]["lasts"] == [7, 13])
    check("covered: not listed", ts.unexplained([{**up, "way": "down", "days": [0, 30]}], curves) == [])


def test_the_world_weights_on_history():
    """The world (model/world.py): history's start days reweighted as little as it takes to move the pushed factors'
    means and spreads; the evidence capped and counted once a group; read at any length; a question read off it."""
    from src.board.model import world as wm

    rng = np.random.default_rng(3)
    Z = rng.normal(size=(4000, 2))
    Z = (Z - Z.mean(0)) / Z.std(0)
    w, scale = wm.tilted(Z, np.array([0.3, 0.0]), np.array([1.0, 1.0]))
    check("the pushed mean moves to its target", abs(w @ Z[:, 0] - 0.3) < 1e-6 and scale == 1.0)
    check("the other left free (no other data to keep it)", abs(w @ Z[:, 1]) < 0.05)
    w, _ = wm.tilted(Z, np.array([0.0, 0.0]), np.array([1.5, 1.0]))
    check("a spread moved, its mean held", abs(w @ Z[:, 0] ** 2 - 1.5) < 1e-6 and abs(w @ Z[:, 0]) < 1e-6)
    w, scale = wm.tilted(Z, np.array([3.0, 0.0]), min_ess=0.25)
    check("never on fewer than a quarter: the shift scaled down", scale < 1 and wm.ess(w) >= 0.25 * 4000 - 1)
    check("no push: even", np.allclose(wm.tilted(Z, np.zeros(2))[0], 1 / 4000))
    C = np.column_stack([Z[:, 0], 0.8 * Z[:, 0] + 0.6 * Z[:, 1]])  # two factors that move together
    free, _ = wm.tilted(C, np.array([0.0, 0.5]))
    nudged, _ = wm.tilted(C, np.array([0.001, 0.5]))
    check(
        "a factor not pushed follows one that is, as they moved together; pushed a little, it is not pinned there",
        abs(free @ C[:, 0] - 0.4) < 0.05 and abs(nudged @ C[:, 0] - free @ C[:, 0]) < 0.01,
    )

    check("each way the strongest whole, the next half", abs(wm.combine([0.4, 0.2, -0.1]) - 0.4) < 1e-12)
    check("capped", wm.combine([0.9, 0.9]) == wm.CAP_SD and wm.combine([-0.9, -0.9], *wm.SPREAD_CAP) == -0.5)
    sig = [
        {"id": "a", "group": "g", "dial": 0.5, "share": 1.0, "sizes": {"move": {"x": [0.4, 0.0]}, "spread": {}}},
        {"id": "b", "group": "g", "dial": 0.5, "share": 1.0, "sizes": {"move": {"x": [0.2, 0.0]}, "spread": {}}},
        {"id": "c", "group": None, "dial": -0.5, "share": 1.0, "sizes": {"move": {"x": [0.2, 0.0]}, "spread": {}}},
    ]
    push, parts = wm.pushes(sig, [(0, 7), (7, 30)])
    check("a group averaged; the other way nets", abs(push["move"]["x"][0] - ((0.2 + 0.1) / 2 - 0.1)) < 1e-12)
    calm = {"id": "d", "group": "g", "dial": 0.0, "share": 1.0, "sizes": {"move": {"x": [0.4, 0.0]}, "spread": {}}}
    silent = {"id": "e", "group": "g", "dial": 0.5, "share": 1.0, "sizes": {"move": {"x": [0.0, 0.3]}, "spread": {}}}
    push2, _ = wm.pushes([*sig, calm, silent], [(0, 7), (7, 30)])
    check(
        "one of the group reading in the middle is a vote for usual; one with nothing to say there is not counted",
        abs(push2["move"]["x"][0] - ((0.2 + 0.1 + 0.0) / 3 - 0.1)) < 1e-12
        and abs(push2["move"]["x"][1] - 0.15) < 1e-12,
    )
    check("each signpost's own part kept", abs(parts["c"]["move"]["x"][0] + 0.1) < 1e-12)

    segs = [(0, 7), (7, 30)]
    check("a push within the window counts", abs(wm.shift([0.0, 1.0], segs, 30, [1, 2], 2.0, "ret") - 1.0) < 1e-12)
    check(
        "part of the stretch inside, that part of the push",
        abs(wm.shift([0.0, 1.0], segs, 18, [1, 2], 2.0, "ret") - 11 / 23) < 1e-12,
    )
    check(
        "a yearly rate averaged over the window", abs(wm.shift([1.0, 0.0], segs, 28, [1, 2], 1.0, "ann") - 0.25) < 1e-12
    )
    check("variance added, within its range", wm.spread([0.0, 1.0], segs, 30, [1, 1], 1.0) == 2.0)

    days = pd.date_range("2000-01-01", "2026-10-01", freq="D")
    v = pd.Series(np.arange(len(days), dtype=float), index=days)
    check(
        "today's value at the top of its history: the dial up",
        wm.dial_today(v, "> 1", "2000-01-01", "2026-10-02") > 0.49,
    )
    check("red below its line: turned", wm.dial_today(v, "< 1", "2000-01-01", "2026-10-02") < -0.49)
    check("a stale value reads nothing", wm.dial_today(v, "> 1", "2000-01-01", "2027-01-01") is None)

    starts = pd.date_range("1971-03-01", periods=10, freq="7D").values.astype("datetime64[ns]").astype(np.int64)
    outs = {pd.Timestamp("1971-03-02") + pd.Timedelta(days=7 * i): ("a" if i < 5 else "b") for i in range(10)}
    w = np.array([1.0] * 5 + [3.0] * 5)
    p, n_eff, band = wm.read(outs, 1.0, starts, w / w.sum(), ["a", "b"])
    check(
        "each outcome matched to its nearest start day and weighted", p["b"] > 0.7 and abs(sum(p.values()) - 1) < 1e-12
    )
    check("its band from the effective windows", band["b"][0] < p["b"] < band["b"][1] and 6 < n_eff < 10)


def test_a_signpost_moves_the_world_where_its_news_lasts():
    """The world end to end on made-up history: a signpost measured to raise a factor over the first month moves its
    mean over a month and a year from today, and nothing over a week (no push there)."""
    from src.board.model import world as wm

    rng = np.random.default_rng(2)
    days = pd.date_range("1971-01-01", "2026-10-01", freq="D")
    series = {"X": pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, len(days)))), index=days)}
    fcfg = {"history_from": "1971-03-01", "factor": {"x": {"series": "X", "measure": "ret"}}}
    segs = [(0, 7), (7, 30), (30, 365)]
    W = wm.World(fcfg, series, segs, {"move": {"x": np.array([0.0, 0.5, 0.0])}, "spread": {}})
    a7, a30, a365 = W.weights(7), W.weights(30), W.weights(365)
    check("over a week: no push, even weights", a7["mu"][0] == 0 and np.allclose(a7["w"], 1 / len(a7["w"])))
    check("over a month: its mean raised", a30["mu"][0] > 0.3 and a30["w"] @ a30["Z"][:, 0] > 0.3)
    check("over a year: still raised, less in a year's spread", 0 < a365["mu"][0] < a30["mu"][0])


def test_the_scenarios_reweigh_the_simulation():
    """The LLM's scenarios are the world's last layer, each in a direction of its own: a window is in every scenario
    whose ranges it is in, several or none. Each one should have the simulation's share (as estimated from history's
    windows that do not overlap) times the readers' factor; the windows are reweighed until each has it, the
    simulation moved least: inside and outside a lone scenario the data's proportions kept, every factor 1 the
    simulation itself, scenarios at odds as near as they go. Carried to another length by start day; a path that
    happened is in every scenario whose ranges it is in."""
    from src.board.model import scenarios as sm

    rng = np.random.default_rng(4)
    n = 2000
    X = rng.normal(size=(n, 2))
    w = np.exp(0.5 * X[:, 0])  # the simulation: the data leans up on x
    a = {"starts": np.arange(n, dtype=np.int64) * 7 * 86_400_000_000_000, "X": X, "w": w / w.sum(), "names": ["x", "y"]}
    a["days"] = 30
    scs = [
        {"id": "down", "ranges": {"x": [None, -0.5]}, "times": 2.0},  # open below
        {"id": "y-up", "ranges": {"y": [0.5, None]}, "times": 0.5},  # another direction, overlapping the first
    ]
    inside = sm.inside(X, a["names"], scs)
    check(
        "in its ranges, whichever else",
        (inside[:, 0] == (X[:, 0] <= -0.5)).all() and (inside[:, 1] == (X[:, 1] >= 0.5)).all(),
    )
    check("a window in both, and one in none", (inside.all(1)).any() and (~inside.any(1)).any())
    got = sm.at_length(a, scs)
    check("the periods: windows that do not overlap", got["periods"] == sm.episodes(a["starts"], 30))
    share = inside.T @ a["w"]
    base = (share * got["periods"] + 0.5) / (got["periods"] + 1)
    check("the simulation's share as estimated from them", np.allclose([e["base"] for e in got["each"]], base))
    check(
        "each one has its share times its factor",
        np.allclose([e["weight"] for e in got["each"]], base * [2.0, 0.5], atol=1e-4)
        and np.allclose([e["target"] for e in got["each"]], base * [2.0, 0.5]),
    )
    cell = inside[:, 0] & ~inside[:, 1]
    check(
        "inside a cell of the scenarios, the simulation's proportions kept",
        np.allclose(got["w"][cell] / got["w"][cell].sum(), a["w"][cell] / a["w"][cell].sum()),
    )
    plain = sm.at_length(a, [{**x, "times": 1.0} for x in scs])
    check("every factor 1: the simulation's own (near)", np.allclose(plain["w"], a["w"], atol=2e-5))
    odds = sm.at_length(
        a,
        [
            {"id": "a", "ranges": {"x": [0.0, None]}, "times": 1.5},
            {"id": "b", "ranges": {"x": [0.0, None]}, "times": 0.5},
        ],
    )
    flip = sm.at_length(
        a,
        [
            {"id": "b", "ranges": {"x": [0.0, None]}, "times": 0.5},
            {"id": "a", "ranges": {"x": [0.0, None]}, "times": 1.5},
        ],
    )
    check(
        "two scenarios at odds: between the two, each short of its own, whatever their order",
        odds["each"][1]["target"] < odds["each"][0]["weight"] < odds["each"][0]["target"]
        and abs(odds["each"][0]["weight"] - flip["each"][1]["weight"]) < 1e-9,
    )
    w_hist = sm.on_history(got["ext"], got["w"], n)
    b = {**a, "w": np.full(n, 1 / n)}
    carried = sm.carry(a, w_hist, b)
    check(
        "carried by start day: the final over the data's weight, times the other length's",
        np.allclose(carried, (w_hist / a["w"]) / (w_hist / a["w"]).sum()),
    )
    check("the nearest length by ratio", sm.nearest({"month": 30, "long": 365}, 7) == "month")
    check("none without scenarios", sm.nearest({}, 30) is None)
    check("a path that happened: in each it is in", sm.happened({"x": -0.7, "y": 0.9}, scs) == [True, True])
    check("past any history: in the open end", sm.happened({"x": -50.0, "y": 0.0}, scs) == [True, False])
    check("a value missing: unknown", sm.happened({"x": None, "y": 0.0}, scs) == [None, False])
    check("episodes: windows that do not overlap", sm.episodes(np.array([0, 7, 14, 40]) * 86_400_000_000_000, 30) == 2)


def test_a_scenario_history_has_not_seen_is_made():
    """A scenario with fewer than three windows in its ranges that do not overlap is made from the nearest: its
    factors moved into its ranges, the others moved with them as they went together; its share half a window's worth
    when history has none; its probability its share times its factor, on the windows made for it; with some history
    in it, real and made in their (sources') data proportions; their returns from their moved factors, in logs,
    never below -100%."""
    from src.board.model import distribution as distm
    from src.board.model import scenarios as sm
    from src.board.model import timescale as ts
    from src.board.model import world as wm

    rng = np.random.default_rng(9)
    days = pd.date_range("1971-03-01", "2026-10-30", freq="D")
    a_ = np.cumsum(rng.normal(0.0003, 0.01, len(days)))
    b_ = 0.8 * a_ + np.cumsum(rng.normal(0, 0.006, len(days)))  # moves with the first
    series = {"A": pd.Series(100 * np.exp(a_), index=days), "B": pd.Series(100 * np.exp(b_), index=days)}
    fcfg = {
        "history_from": "1971-03-01",
        "factor": {k.lower(): {"series": k, "measure": "ret", "label": k} for k in ("A", "B")},
    }
    world = wm.World(fcfg, series, ts.stretches(), {})
    a = world.weights(30)
    scs = [
        {"id": "deep", "ranges": {"a": [-0.6, -0.5]}, "times": 4.0},  # a fall no month in history had
        {"id": "other", "ranges": {"a": [-0.5, 1.0]}},
    ]
    got = sm.at_length(a, scs)
    ext, n = got["ext"], len(a["starts"])
    deep = got["each"][0]
    check("too few like it: made from the nearest", deep["episodes"] < sm.EPISODES and deep["made"] == sm.MADE)
    check("appended, in it", len(ext["starts"]) == n + sm.MADE and got["inside"][n:, 0].all())
    made_a = ext["X"][n:, 0]
    check("its factor moved into its range", ((made_a >= -0.6) & (made_a <= -0.5)).all())
    check("the other moved with it, the same way", (ext["X"][n:, 1] < ext["X"][ext["src"], 1]).all())
    k = got["periods"]
    check("none in history: half a window's worth", deep["share"] == 0 and abs(deep["base"] - 0.5 / (k + 1.0)) < 1e-12)
    check(
        "its probability near that times its factor (the other, nearly everything else, asks for its own too)",
        abs(deep["weight"] - 4 * deep["base"]) < 0.5 * 4 * deep["base"] and deep["weight"] > 2 * deep["base"],
    )
    check("all of it on the windows made for it", abs(got["w"][n:].sum() - deep["weight"]) < 1e-12)
    half = sm.at_length(a, [{**scs[0], "ranges": {"a": [-0.6, -0.14]}}, {**scs[1], "ranges": {"a": [-0.1, 1.0]}}])
    m, real = half["inside"][:, 0], np.arange(len(half["w"])) < n
    check(
        "with some history in it: its real windows in the data's proportions, the made ones too",
        half["each"][0]["made"]
        and half["each"][0]["share"] > 0
        and np.allclose(half["w"][m & real] / half["w"][m & real].sum(), a["w"][m[:n]] / a["w"][m[:n]].sum()),
    )
    w_hist = sm.on_history(ext, got["w"], n)
    check("on history: a made window's weight on its source", abs(w_hist.sum() - 1) < 1e-12 and len(w_hist) == n)

    class R:  # a response with one class, the factor's own return, nothing unexplained
        X, starts, coef, own, resid, paths = a["X"], a["starts"], {"a": None}, {}, {}, {}

        @staticmethod
        def predict(k, X):
            return X[:, 0]

    fut = distm.windows(R, wm.padded(np.full(n, 1 / n), ext), ext)
    check(
        "its returns from its moved factors (a log return)",
        np.allclose(fut.returns["a"][n:], np.expm1(ext["X"][n:, 0])),
    )

    class Steep(R):  # a fit that, carried far, says a log return of -8: still a loss short of everything
        @staticmethod
        def predict(k, X):
            return np.full(len(X), -8.0)

    check("no return below -100%", (distm.windows(Steep, None, ext).returns["a"] > -1).all())


def test_a_reading_too_new_to_measure():
    """A signpost on a text reading: read on its own scale (no history to rank it in); the LLM's expectations sized as
    the measured ones that agree typically are, each kind its own, over the stretches its days cover."""
    from src.board.model import world as wm

    days = pd.date_range("2026-10-01", "2026-10-30", freq="D")
    v = pd.Series(np.linspace(0, 1.0, len(days)), index=days)
    check("on its scale, toward red up", wm.dial_scale(v, "> 0.75", date(2026, 10, 30)) == 0.25)
    check("red below its line: reversed", wm.dial_scale(v, "< -0.5", date(2026, 10, 30)) == -0.25)
    check("stale: nothing", wm.dial_scale(v, "> 0.75", date(2027, 1, 30)) is None)
    judged = [
        {"way": "up", "verdict": "agrees", "size": 0.2},
        {"way": "down", "verdict": "agrees", "size": 0.4},
        {"way": "up", "verdict": "flat", "size": 0.0},
        {"way": "wider", "verdict": "agrees", "size": 1.0},
    ]
    size = wm.typical(judged)
    check("typical: the median of those that agree, by kind", abs(size["move"] - 0.3) < 1e-12 and size["spread"] == 1.0)
    segs = [(0, 7), (7, 13), (13, 25), (25, 47)]
    sizes, j = wm.assumed([{"factor": "rate10", "way": "down", "days": [7, 30], "why": "w"}], segs, size)
    check(
        "the LLM's expectation over the stretches its days cover, its way, the typical size",
        np.allclose(sizes["move"]["rate10"], [0, -0.3, -0.3, 0]) and j[0]["verdict"] == "assumed",
    )


def test_the_decision_step():
    """Your targets leaned by the world against usual, both ways; the noise; the guardrails; the trades on what you
    hold; the decided shares by category within their caps."""
    from src.board.model import decide as dm
    from src.board.model import distribution as distm

    n = 1000
    rng = np.random.default_rng(3)
    a = rng.normal(0.08, 0.2, n)
    flat = distm.Futures(np.zeros((n, 0)), {"a": a, "b": rng.normal(0.05, 0.1, n), "cash": np.full(n, 0.03)})
    good = np.exp(3 * (a - a.mean()) / a.std())  # a world that puts a's good years forward
    g = dm.gap(flat.weighed(good), flat, days=365)
    check("a class the world puts above usual: a gap up, a positive score", g["a"]["gap"] > 0 and g["a"]["score"] > 0)
    tiny = np.ones(n)
    tiny[:3] = 1.01
    t = dm.gap(flat.weighed(tiny), flat, days=365)
    check("a reweighting chance alone would make: its gap shrunk to almost nothing", abs(t["a"]["shrunk"]) < 1e-5)
    check("cash at a fixed rate: no gap", abs(g["cash"]["gap"]) < 1e-12)
    limits = {"tilt": 0.05, "class_cap": 0.6, "cash_min": 0.05, "band": 0.075, "step": 0.075}
    policy = {"a": 0.5, "b": 0.3, "cash": 0.2}
    mix, lean = dm.lean(policy, {"a": 1.0, "b": -1.0}, limits, 0.5)
    check(
        "both ways alike: up to the lean allowed, cash the difference",
        abs(lean["a"] - 0.05) < 1e-12 and abs(lean["b"] + 0.05) < 1e-12 and abs(mix["cash"] - 0.2) < 1e-12,
    )
    mix, lean = dm.lean(policy, {"a": 1.0, "b": 1.0}, {**limits, "cash_min": 0.15}, 0.5)
    check("the cash floor holds: the leans up scaled down", abs(mix["cash"] - 0.15) < 1e-12)
    mix, lean = dm.lean(policy, {"a": 1.0}, limits, 0.0)
    check("sense 0: your targets as they are", mix == policy and lean == {})
    llm_good = good * np.where(a > np.quantile(a, 0.5), 1.5, 1.0)  # the LLM puts more on a's better half
    parts = dm.gap(flat.weighed(llm_good), flat, flat.weighed(good), days=365)["a"]
    check(
        "the gap in two parts: the data's and the LLM's, each scored on its own",
        abs(parts["data"] + parts["llm"] - parts["gap"]) < 1e-12 and parts["score_llm"] > 0 and parts["score_data"] > 0,
    )
    horizons = {
        "long": {
            "futures": flat.weighed(llm_good),
            "data_futures": flat.weighed(good),
            "history_futures": flat,
            "days": 365,
            "limit": None,
        }
    }
    only_llm = dm.decide(policy, horizons, limits, 0.0, {"long": 1.0}, llm_sense=1.0)
    check(
        "the data's sense 0, the LLM's 1: the LLM's part alone leans",
        only_llm["lean"].get("a", 0) > 0 and only_llm["parts"]["a"]["data"] == 0,
    )
    check("both 0: no lean", dm.decide(policy, horizons, limits, 0.0, {"long": 1.0})["lean"] == {})
    gm, moves, seen = dm.guard({"a": 0.9, "cash": 0.1}, {"long": (flat, -0.2)}, policy)
    check(
        "past a length's limit: moved into a shelter you hold until within it",
        moves and moves[0]["to"] == "cash" and seen["long"]["after"] >= -0.2 - 1e-9 and seen["long"]["before"] < -0.2,
    )
    check("within it: nothing", dm.guard({"a": 0.2, "cash": 0.8}, {"long": (flat, -0.2)}, policy)[1] == [])
    lim = {"band": 0.075, "step": 0.075}
    todo, anchor = dm.trades({"a": 0.5, "cash": 0.5}, {"a": 0.56, "cash": 0.44}, {}, lim)
    check("within your band and the view unmoved: no trade", todo == {} and anchor["a"] == 0.5)
    todo, _ = dm.trades({"a": 0.5, "cash": 0.5}, {"a": 0.7, "cash": 0.3}, {}, lim)
    check("past your band: a step toward the decided share", abs(todo["a"]["size"] + 0.075) < 1e-12)
    todo, anchor = dm.trades({"a": 0.45, "cash": 0.55}, {"a": 0.5, "cash": 0.5}, {"a": 0.5}, lim)
    check("the decided share moved half a step: a trade on the view", todo["a"]["why"] == "view")
    todo, anchor = dm.trades({"a": 0.5, "cash": 0.5}, {"a": 0.45, "cash": 0.55}, {"a": 0.45}, lim)
    check("the view gone: back again", todo["a"]["why"] == "view" and todo["a"]["size"] > 0)
    todo, _ = dm.trades({"a": 0.9, "cash": 0.1}, {"a": 0.7, "cash": 0.0}, {}, lim)
    check("buys only from the cash there is", todo == {})
    cats = {"x": {"class": "a", "target": 0.5, "cap": 0.4}, "c": {"class": "cash", "target": 0.5}}
    by = dm.by_category({"a": 0.55, "cash": 0.45}, cats, {"a": 0.5, "cash": 0.5})
    check("a category at most its cap, the rest in cash", abs(by["x"] - 0.4) < 1e-12 and abs(by["c"] - 0.6) < 1e-12)
    check("new money goes first where you are furthest under", dm.new_money(by, {"x": 0.3, "c": 0.7}) == "x")


def test_the_decision_walked_through_history():
    """The walk: the decision step on each day, the holdings drifting between; sense 0 with no limit binding is your
    targets rebalanced; the choice keeps a sense only if it adds on the first half within the drawdown and turnover
    it may cost."""
    from src.board.model import walk as wk

    weeks = pd.date_range("2000-01-03", periods=520, freq="W-MON")
    rng = np.random.default_rng(5)
    weekly = pd.DataFrame(
        {"a": np.cumprod(1 + rng.normal(0.002, 0.02, 520)), "cash": np.cumprod(np.full(520, 1.0005))}, index=weeks
    )
    starts = np.arange(200)
    R = {"long": {"a": rng.normal(0.08, 0.2, 200), "cash": np.full(200, 0.03)}}
    days = weeks[::4]
    worlds = {d: {"long": (starts, np.full(200, 1 / 200))} for d in days}
    limits = {"tilt": 0.05, "class_cap": 1.0, "cash_min": 0.0, "band": 0.075, "step": 0.075}
    policy = {"a": 0.8, "cash": 0.2}
    base = wk.run(days, worlds, R, weekly, policy, limits, -0.9, {}, 0.0, {}, band_only=True)
    same = wk.run(days, worlds, R, weekly, policy, limits, -0.9, {}, 0.0, {"long": 1.0})
    check("no lean and no limit binding: the targets rebalanced", np.allclose(base["value"], same["value"]))
    first = {"cagr": 0.08, "max_dd": -0.3}
    b = {"first": first, "turnover": 0.05}
    runs = {
        (0.5, "year"): {"first": {"cagr": 0.09, "max_dd": -0.31}, "turnover": 0.1},
        (1.0, "year"): {"first": {"cagr": 0.12, "max_dd": -0.5}, "turnover": 0.1},
    }
    check("a sense kept when it adds within what it may cost", wk.choose(runs, b) == (0.5, "year"))
    check(
        "none adding enough: sense 0",
        wk.choose({(0.5, "year"): runs[(0.5, "year")]}, {**b, "first": {**first, "cagr": 0.1}})[0] == 0.0,
    )

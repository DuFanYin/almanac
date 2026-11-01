"""The scenarios step (steps/scenarios/) and the LLM roles behind it: written, checked, put on the world, scored,
explained for the page, the news of them, what no scenario covers told to the writer."""

import json as _json
from datetime import date

import numpy as np
import pandas as pd
from src.board.core import store
from tests.board.support import check, tmp


def test_the_scenarios_written_checked_and_scored():
    """The LLM's scenarios: written weekly (all lengths in one answer); each one's factor the median of its readers'
    (on the logarithms), 1 for one no reader answered; the live set replaced, what changed said; written again only
    when the week is up, something happened (not within two days) or asked; on today's simulation each scenario's
    probability its share times its factor, the paths in none as the simulation has them; scored when its length is
    over, each scenario happened or not, the writer told."""
    from src.board.model import timescale as tsm
    from src.board.model import world as wm
    from src.board.steps import scenarios as sc

    db = store.connect(tmp / "data" / "scen.db")
    days = pd.date_range("1971-01-01", "2027-01-15", freq="D")
    gspc = pd.Series(np.exp(np.cumsum(np.random.default_rng(5).normal(0.0003, 0.01, len(days)))), index=days)
    cfg = {
        "factors": {
            "history_from": "1971-03-01",
            "horizon": {"month": {"label": "1 month", "days": 30}},
            "factor": {"stocks": {"label": "S&P 500", "series": "GSPC", "measure": "ret"}},
        }
    }
    world = wm.World(cfg["factors"], {"GSPC": gspc[:"2026-10-29"]}, tsm.stretches(), {"move": {}, "spread": {}})
    told = []
    drafts = [  # name, range, each reader's factor (None: it did not answer)
        ("Calm", [-0.02, 0.02], [1.0, 1.0, None]),
        ("Fall", [-0.5, -0.05], [1.0, 4.0, 2.0]),
        ("Rise", [0.05, 0.5], [0.5, 0.5, 0.5]),
        ("Mute", [0.6, None], [None, None, None]),
    ]  # between -0.05 and -0.02, and 0.02 and 0.05: in no scenario

    def write(**kw):
        told.append(kw)
        scs = [{"id": n.lower(), "name": n, "story": "s", "ranges": {"stocks": r}} for n, r, _ in drafts]
        readings = {
            "month": {
                n.lower(): [
                    {"first": a, "final": a}
                    for a in ({"times": x, "missing": "m", "why": "w"} if x else None for x in xs)
                ]
                for n, _, xs in drafts
            }
        }
        review = {
            "points": [{"length": "month", "scenario": "fall", "says": "already in the simulation", "why": "w"}],
            "asked": [],
            "model": "m",
        }
        return {
            "sets": {"month": scs},
            "rejected": [],
            "readers": [{"view": "v", "model": "m"}] * 3,
            "readings": readings,
            "review": review,
            "usage": {},
            "method": "m",
        }

    real = sc.llm.scenarios
    sc.llm.scenarios = write
    try:
        before, news = sc.write(db, cfg, world, [], [], date(2026, 10, 30))
        live = sc.live(db)["month"]
        got = {x["id"]: x for x in live["scenarios"]}
        check("written: the first time is the week's", set(before) == {"month"} and live["trigger"] == "weekly")
        check(
            "the readers told each scenario's share of the simulation, the tools answer from the board's data",
            "the simulation"
            in told[0]["shares"]({"month": [{"id": "calm", "ranges": {"stocks": [-0.02, 0.02]}}]})["month"]["calm"]
            and "history" in told[0]["tools"]["share"][1]({"length": "month", "ranges": {"stocks": [None, -0.1]}})
            and "stocks: median"
            in told[0]["tools"]["alongside"][1]({"length": "month", "ranges": {"stocks": [-0.02, 0.02]}})
            and "the simulation" in told[0]["tools"]["factor"][1]({"length": "month", "factor": "stocks"})
            and told[0]["market"] == "(no prices today)"
            and "history unweighted" in told[0]["simulation"]
            and "30 days" in told[0]["factors"]["month"],
        )
        check("the review kept with its length", live["review"]["points"][0]["scenario"] == "fall")
        check(
            "each factor the median of its readers' on the logarithms; one no reader answered at 1",
            set(got) == {"calm", "fall", "rise", "mute"}
            and got["mute"]["times"] == 1.0
            and got["calm"]["times"] == 1.0
            and got["fall"]["times"] == 2.0
            and got["rise"]["times"] == 0.5,
        )
        check(
            "the probabilities when written: each its share times its factor, the paths in none the rest",
            all(abs(x["weight"] - x["base"] * x["times"]) < 1e-4 for x in got.values())
            and sum(x["weight"] for x in got.values()) < 1,
        )
        check(
            "each with every reading",
            [r["final"]["times"] if r["final"] else None for r in got["fall"]["readings"]] == [1.0, 4.0, 2.0]
            and got["calm"]["readings"][2]["final"] is None,
        )
        check("nothing to write the day after", sc.write(db, cfg, world, [], [], date(2026, 10, 31)) == ({}, []))
        check(
            "something happened, but within two days: not yet",
            sc.write(db, cfg, world, [], [], date(2026, 10, 31), happened=["x"]) == ({}, []),
        )
        drafts[2] = ("Rise", [0.05, 0.5], [6.0, 6.0, 6.0])
        _, news = sc.write(db, cfg, world, [], [], date(2026, 11, 2), happened=["S&P 500 moved -6%"])
        check(
            "on an event: written again, revising the standing ones, what changed said",
            "S&P 500 moved" in told[-1]["why"]
            and "x0.5" in told[-1]["previous"]
            and any("Rise" in x for x in news)
            and sc.live(db)["month"]["trigger"] == "event",
        )
        n = dict(db.execute("SELECT status, count(*) FROM scenario_sets GROUP BY status").fetchall())
        check("the set before replaced, one live", n == {"live": 1, "replaced": 1})
    finally:
        sc.llm.scenarios = real
    scen = sc.for_world(world, sc.live(db))
    month = [f for f in scen["live"] if f["horizon"] == "month"]
    by = {f["sid"]: f for f in month}
    check(
        "on today's simulation: the probability each should have, its share today times its factor",
        all(abs(f["p_target"] - min(f["p_data"] * f["times"], 1 - 1e-4)) < 1e-9 for f in month),
    )
    check(
        "at odds (Rise x6 leaves too little for the others): each short of its own",
        by["rise"]["p"] < by["rise"]["p_target"] and by["calm"]["p"] < by["calm"]["p_target"],
    )
    page = sc.for_page(db, scen, date(2026, 11, 2))
    check(
        "the page: each with its probability in the set before", {f["sid"]: f["was"] for f in page}["calm"] is not None
    )
    labels = {"month": "1 month"}
    check("before its length is over: not scored", sc.scored(db, {"GSPC": gspc}, date(2026, 11, 20)) == 0)
    check("over: scored, both sets", sc.scored(db, {"GSPC": gspc}, date(2026, 12, 10)) == 2)
    how = _json.loads(db.execute("SELECT how FROM scenario_sets WHERE status = 'live'").fetchone()[0])
    x = gspc.asof(pd.Timestamp("2026-12-02")) / gspc.asof(pd.Timestamp("2026-11-02")) - 1
    edges = {"calm": (-0.02, 0.02), "fall": (-0.5, -0.05), "rise": (0.05, 0.5), "mute": (0.6, 9)}
    parts = {p["id"]: p for p in how["parts"]}
    check(
        "each scenario happened or not, with its probability, the simulation's share and the factor when written",
        all(parts[k]["hit"] == (lo <= x <= hi) for k, (lo, hi) in edges.items())
        and parts["fall"]["weight"] > 0
        and parts["fall"]["share"] > 0
        and parts["fall"]["times"] > 0,
    )
    check(
        "the writer is told",
        "happen; factor x" in sc.record_text(db, labels) or "happened; factor x" in sc.record_text(db, labels),
    )


def test_how_the_scenarios_turned_out_and_the_brief_s_scenarios():
    """The scenarios' record: each scenario of each scored set, happened or not, against its final probability, the
    simulation's share and history's, as mean log scores (log p when it happened, log 1 - p when not); the scenarios
    the weekly brief is given, each probability, share and factor a data item."""
    from src.board.steps import texts as tx
    from src.board.view.record import scenarios_record

    db = store.connect(tmp / "data" / "fr.db")
    spec = {"scenarios": [{"id": "bust", "name": "Bust", "weight": 0.5}, {"id": "calm", "name": "Calm", "weight": 0.3}]}

    def part(i, hit, weight, share, history, times):
        return {
            "id": i,
            "name": i.title(),
            "hit": hit,
            "weight": weight,
            "share": share,
            "history": history,
            "times": times,
        }

    rows = [
        (
            "s1",
            "month",
            "replaced",
            {"parts": [part("bust", True, 0.5, 0.25, 0.2, 3.0), part("calm", False, 0.3, 0.3, 0.3, 1.0)]},
        ),
        ("s3", "month", "replaced", {"parts": [part("calm", None, 0.2, 0.25, 0.3, 1.0)]}),
        ("s2", "month", "live", None),
    ]
    for i, h, status, how in rows:
        db.execute(
            "INSERT INTO scenario_sets VALUES (?,?,?,?,?,?,?,?)",
            (i, h, "2026-10-01", "2026-10-31", _json.dumps(spec), status, None, _json.dumps(how) if how else None),
        )
    fr = scenarios_record(db)
    check(
        "two scored (one without a value left out): the final log score against the simulation's and history's",
        fr["n"] == 2
        and abs(fr["llm"] - (np.log(0.5) + np.log(0.7)) / 2) < 1e-4
        and abs(fr["share"] - (np.log(0.25) + np.log(0.7)) / 2) < 1e-4
        and {(x["name"], x["hit"]) for x in fr["sets"]} == {("Bust", True), ("Calm", False)},
    )
    data = []
    futures = [
        {
            "name": "AI bust",
            "horizon": "long",
            "p": 0.2,
            "p_data": 0.1,
            "times": 2.0,
            "story": "s [D3]",
            "impact": {"side": "worse", "worst": 0.5},
        }
    ]
    lines = tx.futures_lines(futures, data, date(2026, 10, 30), {"long": "1 year"})
    check(
        "the brief is told each scenario with its probability, the simulation's share and the factor as data, and "
        "its side for the targets",
        lines[0].startswith(
            "- AI bust, next 1 year [D1] (the simulation: [D2], times the readers' factor [D3]): s Worse"
        )
        and [d["value"] for d in data] == [0.2, 0.1, 2.0, 0.5],
    )


def test_each_scenario_as_a_view_and_the_leans_it_drives():
    """A scenario with its rarest assumption, the texts for and against it (one line a reading and a verdict), the
    opposite view at its length, and the actions it drives; a reading within its line says nothing."""
    from src.board.steps import scenarios as ls_

    horizons = {
        "long": {
            "classes_history": {"a": {"expected": 0.08}},
            "scenarios": {
                "bust": {
                    "history_like": 0.0,
                    "history_like_by_factor": {"stocks": 0.05, "unemp": 0.2},
                    "leans": {"stocks": {"rank": 0.05, "lean": -1}, "rate10": {"rank": 0.2, "lean": -1}},
                    "classes": {"a": -0.3},
                    "targets": {"share": 0.2, "mean": -0.25, "worst": 0.8},
                    "moved": 0.05,
                },
                "boom": {
                    "history_like": 0.1,
                    "history_like_by_factor": {"stocks": 0.3},
                    "leans": {"stocks": {"rank": 0.9, "lean": 1}},
                    "classes": {"a": 0.3},
                    "targets": {"share": 0.1, "mean": 0.3, "worst": 0.0},
                    "moved": 0.0,
                },
            },
        }
    }
    futures = [
        {
            "id": "bust",
            "horizon": "long",
            "name": "Bust",
            "p": 0.2,
            "ranges": {"stocks": [-0.4, -0.2]},
            "signals": ["s"],
        },
        {"id": "boom", "horizon": "long", "name": "Boom", "p": 0.1, "ranges": {"stocks": [0.2, 0.4]}, "signals": []},
        {"id": "old", "horizon": "long", "name": "Ended", "p": 0.1, "ranges": {}, "signals": []},
    ]
    actions = [
        {"id": "lean:a", "kind": "lean", "class": "a", "size": -0.02, "llm": {"bust": -0.03, "boom": 0.01}},
        {"id": "guard:a:cash", "kind": "guard", "horizon": "long", "class": "a", "size": -0.01},
        {"id": "rule:x", "kind": "rule"},
    ]
    read_now = {
        "AI": {"name": "AI investment", "value": 1.5, "sign": 1, "bears_on": {"stocks": 1}},
        "FED": {"name": "Fed", "value": 1.0, "sign": 1, "bears_on": {"rate10": 1, "stocks": -1}},
        "MOOD": {"name": "Mood", "value": 0.1, "sign": 0, "bears_on": {"stocks": 1}},
    }
    ls_.explained(futures, horizons, read_now, actions)
    bust = futures[0]
    check("its rarest assumption", bust["rarest"] == {"factor": "stocks", "share": 0.05, "range": [-0.4, -0.2]})
    said = {(r["series"], r["says"]): r["factors"] for r in bust["readings_now"]}
    check(
        "AI investment rising is against a bust; the Fed against it on rates, for it on stocks; mood says nothing",
        said == {("AI", "against"): ["stocks"], ("FED", "against"): ["rate10"], ("FED", "for"): ["stocks"]},
    )
    check("the opposite view: the boom", bust["opposite"]["id"] == "boom" and futures[1]["opposite"]["id"] == "bust")
    check(
        "a lean is driven by each scenario's part in the class's gap from the simulation",
        actions[0]["drivers"] == [{"id": "bust", "share": 0.75}, {"id": "boom", "share": 0.25}]
        and bust["drives"] == ["lean:a"],
    )
    check(
        "one not on the page's lengths is left as it was; a guardrail and a rule have no drivers",
        "impact" not in futures[2] and "drivers" not in actions[1] and "drivers" not in actions[2],
    )


def test_the_scenarios_by_several_llms():
    """The roles in order: one writes; each reader reads each scenario on its own; the reviewer checks their reasons;
    only the readers of a scenario it found fault with read it again, told their answer and what it said; the
    readers on the roster's models in turn, the reviewer on another when there is one; each factor the median of the
    readers' last (fake clients)."""
    from src.board.data import llm as L

    asked = []

    class Fake:
        def __init__(self, model):
            self.model = model

        async def complete(self, user, system=None, node=None):
            asked.append((node, self.model, user))
            if node == "scenarios.write":
                cut = {
                    "month": [
                        {"name": "Up", "ranges": {"stocks": [0, None]}},
                        {"name": "Down", "ranges": {"stocks": [None, 0]}},
                    ]
                }
                return _json.dumps({"lengths": cut})
            if node == "scenarios.review":
                return _json.dumps(
                    {
                        "points": [
                            {"length": "month", "scenario": "down", "says": "already in the simulation", "why": "w"}
                        ]
                    }
                )
            if "What the reviewer said" in user:
                return _json.dumps({"times": 1.0, "missing": "none", "why": "w"})
            return _json.dumps({"times": 3.0 if "Down." in user else 0.5, "missing": "m", "why": "w"})

    real = L.client, L.ready
    L.client = lambda spec, calls, t, m: Fake(spec)
    L.ready = lambda: None
    try:
        out = L.scenarios(
            roster=["a:flash", "b:other"],
            lengths={"month": ("1 month", 30, 3)},
            today="2026-10-31",
            simulation="- VIX high: +0.40; stocks down",
            factors={None: "- stocks: S&P (its return over the length, decimal: -0.25); ...", "month": "- stocks"},
            previous="",
            record="",
            coach="",
            why="",
            notes="",
            data=[],
            shares=lambda s: {h: {x["id"]: f"share of {x['id']}" for x in scs} for h, scs in s.items()},
            market="",
            tools={},
        )
    finally:
        L.client, L.ready = real
    order = [n for n, _, _ in asked]
    check(
        "written, each scenario read by four, reviewed, the one found at fault read again by four",
        order == ["scenarios.write"] + ["scenarios.read"] * 8 + ["scenarios.review"] + ["scenarios.read"] * 4,
    )
    reads = [u for n, _, u in asked[1:9]]
    check(
        "each reader told one scenario and its share, not the others",
        all(("Up." in u) != ("Down." in u) for u in reads) and sum("share of down" in u for u in reads) == 4,
    )
    check(
        "the readers on the roster in turn, the reviewer on the second model",
        [r["model"] for r in out["readers"]] == ["a:flash", "b:other", "a:flash", "b:other"]
        and asked[9][1] == "b:other",
    )
    check(
        "the reviewer told every reader's factor and reason; the second readings their answer and its point",
        "x3" in asked[9][2]
        and "x0.5" in asked[9][2]
        and all("Down." in u and "already in the simulation" in u and "x3" in u for _, _, u in asked[10:]),
    )
    check(
        "each factor the median of the readers' last", L.times(out["readings"]) == {"month": {"up": 0.5, "down": 1.0}}
    )


def test_the_scenarios_news_is_a_probability_moved():
    """Only a scenario whose probability moved ten points is news; one new or gone is not (every writing revises some,
    and the page shows them all), so a rewriting sends no message of every scenario new and gone."""
    from src.board.steps.scenarios import changes

    before = {
        "scenarios": [
            {"id": "recession", "name": "Recession", "weight": 0.11},
            {"id": "calm", "name": "Calm", "weight": 0.3},
        ]
    }
    now = [{"id": "recession", "name": "Recession", "weight": 0.22}, {"id": "steady", "name": "Steady", "weight": 0.35}]
    check("a move of ten points", changes(before, now, "year") == ["Next year: Recession 11% → 22%"])
    check("nothing to compare, no news", changes(None, now, "year") == [])


def test_the_writer_tells_what_no_scenario_covers():
    """Much of the simulation in no scenario's ranges: the writer is told how much and what those paths look like, and
    writes again once, adding scenarios (fake client); the board says how much and where (steps/scenarios.py
    uncovered), nothing for a length its scenarios cover."""
    from src.board.data import llm as L
    from src.board.model import timescale as tsm
    from src.board.model import world as wm
    from src.board.steps import scenarios as sc

    days = pd.date_range("1971-01-01", "2026-10-30", freq="D")
    gspc = pd.Series(np.exp(np.cumsum(np.random.default_rng(5).normal(0.0003, 0.01, len(days)))), index=days)
    fcfg = {
        "history_from": "1971-03-01",
        "horizon": {"month": {"label": "1 month", "days": 30}},
        "factor": {"stocks": {"label": "S&P 500", "series": "GSPC", "measure": "ret"}},
    }
    world = wm.World(fcfg, {"GSPC": gspc}, tsm.stretches(), {"move": {}, "spread": {}})
    L_ = sc.lengths(fcfg)
    narrow = {"month": [{"id": "fall", "ranges": {"stocks": [None, -0.08]}}]}
    said = sc.uncovered(world, L_, narrow)["month"]
    check(
        "how much is in no scenario, the S&P 500's tails among it, every factor over it",
        "of the simulated paths are in no scenario" in said and "best tenth" in said and "stocks " in said,
    )
    check(
        "nothing for a length its scenarios cover",
        sc.uncovered(world, L_, {"month": [{"id": "all", "ranges": {"stocks": [-9, 9]}}]}) == {},
    )

    asked = []

    class Fake:
        async def complete(self, user, system=None, node=None):
            asked.append((node, user))
            if node == "scenarios.write":
                more = [{"name": "Rise", "ranges": {"stocks": [0, None]}}] if len(asked) > 1 else []
                return _json.dumps(
                    {"lengths": {"month": [{"name": "Fall", "ranges": {"stocks": [None, -0.08]}}, *more]}}
                )
            if node == "scenarios.review":
                return _json.dumps({"points": []})
            return _json.dumps({"times": 1.0, "missing": "m", "why": "w"})

    real = L.client, L.ready
    L.client = lambda spec, calls, t, m: Fake()
    L.ready = lambda: None
    try:
        out = L.scenarios(
            roster=["a:flash"],
            lengths={"month": ("1 month", 30, 3)},
            today="2026-10-31",
            simulation="",
            factors={None: "- stocks: S&P (its return over the length, decimal: -0.25); ...", "month": "- stocks"},
            previous="",
            record="",
            coach="",
            why="",
            notes="",
            data=[],
            shares=lambda s: {h: {x["id"]: "share" for x in scs} for h, scs in s.items()},
            market="",
            tools={},
            uncovered=lambda s: (
                {"month": "70% of the simulated paths are in no scenario's ranges"} if len(s["month"]) < 2 else {}
            ),
        )
    finally:
        L.client, L.ready = real
    writes = [u for n, u in asked if n == "scenarios.write"]
    check(
        "written again once, told what is left out, four more allowed, the scenarios it wrote kept as standing; the "
        "added ones read",
        len(writes) == 2
        and "70% of the simulated paths" in writes[1]
        and "at most 7 scenarios" in writes[1]
        and "Fall." in writes[1]
        and [s["id"] for s in out["sets"]["month"]] == ["fall", "rise"]
        and set(out["readings"]["month"]) == {"fall", "rise"},
    )

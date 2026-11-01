"""Unit tests for the methods any caller can use: forecast, proposals, grounded writing,
indicators (fake clients)."""

from __future__ import annotations

import asyncio
import json

from src.llm.methods import forecast, grounded, indicators, propose, readings

QUESTION = {
    "id": "q",
    "title": "Where does X end?",
    "outcomes": [{"id": "up", "label": "Higher"}, {"id": "down", "label": "Lower"}],
}
DATA = [{"id": "D1", "name": "X", "value": 101.5}, {"id": "D2", "name": "X a month ago", "value": 0.25}]


class FakeClient:
    """Answers by prompt: the listed wording 70/30, each single outcome 0.6 (a sum of 1.2), or fails."""

    def __init__(self, fail=False):
        self.fail = fail

    async def complete(self, user, *, system=None, node=""):
        if self.fail:
            raise RuntimeError("provider down")
        if node == "forecast.single":
            return json.dumps({"probability": 0.6, "reason": "[D1]"})
        return json.dumps({"probabilities": {"up": 0.7, "down": 0.3}, "reason": "[D1]"})


def test_forecast_summary_over_wordings_and_samples():
    out = asyncio.run(
        forecast.forecast(
            [QUESTION], DATA, today="2026-10-29", models=["m"], samples=2, client_for=lambda m: FakeClient()
        )
    )
    sm = out["summary"]["q"]
    assert len(out["answers"]) == 6 and sm["n"] == 6 and sm["failed"] == 0
    assert abs(sum(sm["p"].values()) - 1) < 1e-9
    assert sm["p"]["up"] > sm["p"]["down"]
    assert abs(sm["incoherence"] - 0.2) < 1e-9  # the single wording summed to 1.2
    assert out["method"] == forecast.method_hash()


def test_forecast_failed_answers_left_out():
    out = asyncio.run(
        forecast.forecast(
            [QUESTION], DATA, today="2026-10-29", models=["m"], samples=1, client_for=lambda m: FakeClient(fail=True)
        )
    )
    assert out["summary"] == {} and all("error" in a for a in out["answers"])


def test_unknown_numbers_in_chinese_text():
    data = [*DATA, {"id": "D3", "name": "200-day mean", "value": 0.3512}]
    assert grounded.unknown_numbers("占比0.3512[D3]，高出200日均线，上涨25%[D2]", data) == []
    assert grounded.unknown_numbers("占比0.4512[D3]", data) == ["0.4512 [D3]"]


def test_a_number_must_be_the_item_it_cites():
    assert grounded.unknown_numbers("X is 101.5 [D1]", DATA) == []
    assert grounded.unknown_numbers("X is 101.5 [D2]", DATA) == ["101.5 [D2]"]
    assert grounded.unknown_numbers("up 25% [D2]", DATA) == []


def test_unknown_numbers():
    assert grounded.unknown_numbers("X is 101.5 [D1], up 25% [D2] in 3 days in 2026", DATA) == []
    assert grounded.unknown_numbers("X is 140 [D1]", DATA) == ["140 [D1]"]


class Scripted:
    def __init__(self, answers):
        self.answers, self.users = list(answers), []

    async def complete(self, user, *, system=None, node=""):
        self.users.append((system, user))
        return json.dumps(self.answers.pop(0))


def test_grounded_asks_again_until_the_numbers_are_the_datas():
    client = Scripted([{"text": "X is 140 [D1]."}, {"text": "X is 101.5 [D1]."}])
    out = asyncio.run(grounded.write("You write a note.", "Say where X is.", DATA, client=client))
    assert out["valid"] and len(out["attempts"]) == 2 and out["attempts"][0]["unknown_numbers"] == ["140 [D1]"]
    system, user = client.users[1]
    assert (
        system.startswith("You write a note.")
        and "numbers not in the data, or not the value of the item cited after them: 140 [D1]" in user
        and "[D1] X: 101.5" in user
    )


def test_grounded_points_and_gives_up_after_three():
    client = Scripted([{"points": ["X is 140 [D1]"]}] * 3)
    out = asyncio.run(grounded.write("Argue.", "Argue against.", DATA, kind="points", client=client))
    assert not out["valid"] and len(out["attempts"]) == 3


def test_indicators_prompt_lists_outcomes():
    client = Scripted([{"drivers": []}])
    out = asyncio.run(indicators.propose("Where does X end?", QUESTION["outcomes"], client=client))
    assert out["output"] == {"drivers": []} and "- up: Higher" in client.users[0][1]


def test_views_by_turns_and_data_left_out_repeatably():
    seen = []

    class Recording(FakeClient):
        async def complete(self, user, *, system=None, node=""):
            seen.append((system, user))
            return await super().complete(user, system=system, node=node)

    data = [{"id": f"D{i}", "name": f"x{i}", "value": i} for i in range(1, 11)]

    def run():
        return asyncio.run(
            forecast.forecast(
                [QUESTION],
                data,
                today="t",
                models=["m"],
                samples=2,
                views=["A", "B"],
                drop=0.3,
                seed="s",
                client_for=lambda m: Recording(),
            )
        )

    out = run()
    assert {a["view"] for a in out["answers"]} == {"A", "B"} and any("Think as A would." in s for s, _ in seen)
    assert any(a["dropped"] for a in out["answers"]) and out["summary"]["q"]["views"].keys() == {"A", "B"}
    assert [a["dropped"] for a in run()["answers"]] == [a["dropped"] for a in out["answers"]]


def test_a_comma_after_a_year_is_punctuation():
    assert grounded.unknown_numbers("In 2008, and again in 2022, gold rose; 1,250 [D1] is not", DATA) == ["1,250 [D1]"]


def test_propose_keeps_the_drafts_in_shape():
    good = {
        "id": "cpi_2y",
        "title": "CPI over two years",
        "purpose": "decision",
        "kind": "path",
        "series": "CPI",
        "days": 730,
        "buckets": [
            {"id": "hot", "label": "Hot", "when": ["ret >= 0.0816"]},
            {"id": "calm", "label": "Calm", "when": []},
        ],
    }
    client = Scripted([{"questions": [good, {**good, "id": "spx_2y"}, {**good, "id": "x2", "series": "NOPE"}]}])
    out = asyncio.run(
        propose.propose(
            [{"id": "spx_2y", "title": "S&P", "outcomes": ["up", "down"]}],
            [{"id": "CPI", "name": "CPI", "every": "monthly", "since": 1947, "last": 330}],
            DATA,
            today="2026-10-29",
            context="nothing",
            client=client,
        )
    )
    assert [p["id"] for p in out["proposals"]] == ["cpi_2y"] and len(out["rejected"]) == 2
    assert "- spx_2y: S&P (up; down)" in client.users[0][1] and "- CPI: CPI, monthly, since 1947" in client.users[0][1]


def test_indicators_for_board_keep_the_drafts_in_shape():
    good = {
        "id": "oil_spike",
        "name": "Oil, 3-month change",
        "driver": "Inflation",
        "value": "ret(OIL, 91)",
        "red": "> 0.3",
        "yellow": "> 0.15",
        "try_on": ["infl_short"],
        "why": "y",
    }
    drafts = [
        good,
        {**good, "id": "vix_high"},
        {**good, "id": "oil_x", "red": "big"},
        {**good, "id": "oil_y", "try_on": ["nope"]},
    ]
    client = Scripted([{"signposts": drafts}])
    out = asyncio.run(
        indicators.for_board(
            today="2026-10-30",
            targets="- infl_short: CPI over 3 months",
            series=["OIL: crude oil, daily, since 1986"],
            language="ret(x, d)",
            existing=["vix_high: VIX = VIX, red > 30"],
            data=DATA,
            n=3,
            client=client,
        )
    )
    assert [s["id"] for s in out["signposts"]] == ["oil_spike"] and len(out["rejected"]) == 3
    assert "- infl_short: CPI over 3 months" in client.users[0][1] and "vix_high: VIX" in client.users[0][1]


def test_readings_keep_the_answers_whose_quotes_are_in_the_text():
    text = "The Committee decided to maintain the target range.\nInflation remains somewhat elevated."
    client = Scripted(
        [
            {"relevant": True, "value": 1.0, "quotes": ["inflation remains  somewhat elevated"], "why": "w"},
            {"relevant": True, "value": 0.0, "quotes": ["decided to maintain the target range"], "why": "w"},
            {"relevant": True, "value": 1.5, "quotes": ["rates will rise"], "why": "not in the text"},
            {"relevant": True, "value": 7, "quotes": ["decided to maintain"], "why": "off the scale"},
        ]
    )
    out = asyncio.run(
        readings.read(
            text, what="hawkish", scale="-2..2", title="FOMC", day="2026-10-28", samples=4, client_for=lambda i: client
        )
    )
    assert out["relevant"] and out["value"] == 0.5 and out["spread"] == 0.5 and len(out["dropped"]) == 2
    assert "hawkish" in client.users[0][1] and "FOMC, 2026-10-28" in client.users[0][1]


def test_readings_a_text_most_answers_call_irrelevant_has_no_value():
    client = Scripted(
        [{"relevant": False, "value": None, "quotes": []}] * 2 + [{"relevant": True, "value": 1, "quotes": ["x"]}]
    )
    out = asyncio.run(
        readings.read("x y", what="w", scale="s", title="t", day="d", samples=3, client_for=lambda i: client)
    )
    assert not out["relevant"] and out["value"] is None


def test_propose_factors_and_the_judge_keep_their_shape():
    good = {"id": "credit", "label": "Credit spread", "series": "BAA", "measure": "diff", "unit": "pts", "up": -1}
    client = Scripted([{"factors": [good, {**good, "id": "stocks"}, {**good, "id": "x_y", "measure": "level"}]}])
    out = asyncio.run(
        propose.factors(
            today="2026-10-30",
            existing=["stocks: S&P 500, GSPC, ret"],
            series=[{"id": "BAA", "name": "Baa yield", "every": "daily", "since": 1953}],
            data=DATA,
            n=3,
            client=client,
        )
    )
    assert [f["id"] for f in out["factors"]] == ["credit"] and len(out["rejected"]) == 2
    client = Scripted([{"adopt": True, "why": "it adds a short horizon"}])
    out = asyncio.run(
        propose.judge(
            today="2026-10-30",
            kind="factor",
            proposal=good,
            checks=[{"name": "new", "ok": True, "say": "not a copy"}],
            has=["stocks: S&P 500"],
            client=client,
        )
    )
    assert out["adopt"] is True and out["why"]
    client = Scripted([{"adopt": "yes"}])
    out = asyncio.run(propose.judge(today="d", kind="question", proposal={}, checks=[], has=[], client=client))
    assert out["adopt"] is False  # only a JSON true adopts


def test_a_scenario_set_s_shape():
    """A length's scenarios (scenarios.sets): one to three ranges on known factors, an open end, a rate in percent read
    as a decimal, a name once, at most the length's count; no weights and no sides from the writer."""
    from src.llm.methods import scenarios

    factors = (
        "- stocks: S&P (its return over the length, decimal: -0.25); ...\n- unemp: Unemployment (its change); ...\n"
        "- infl: CPI (its yearly rate over the length, decimal: 0.03); ..."
    )
    lengths = {"month": ("1 month", 30, 2), "long": ("1 year", 365, 3)}
    raw = {
        "lengths": {
            "month": [
                {"name": "Jobs crack", "story": "s", "ranges": {"stocks": [-15, -5], "unemp": [0.3, None]}},
                {"name": "Calm", "story": "s", "side": "better", "ranges": {"stocks": [-0.05, 0.05]}},
                {"name": "Calm", "story": "again", "ranges": {"stocks": [0, 0.1]}},
                {"name": "Nowhere", "ranges": {"gold": [0, 1]}},
                {"name": "Upside down", "ranges": {"stocks": [0.1, 0.0]}},
                {"name": "Four", "ranges": {"stocks": [0.1, None], "unemp": [0, 1], "infl": [0, 1], "gold": [0, 1]}},
                {"name": "Third", "ranges": {"stocks": [0.1, None]}},
            ],
            "other": [{"name": "x", "ranges": {"stocks": [0, 1]}}],
        }
    }
    got, rejected = scenarios.sets(raw, lengths, factors)
    month = got["month"]
    assert [s["id"] for s in month] == ["jobs-crack", "calm"], (
        "a name once; unknown factors, bad ranges refused; at most the length's count"
    )
    assert month[0]["ranges"] == {"stocks": [-0.15, -0.05], "unemp": [0.3, None]}, "percent read; an end open"
    assert "weight" not in month[0] and "side" not in month[1], "no weights, no sides"
    assert "other" not in got and "long" not in got and len(rejected) == 4


def test_each_scenario_read_on_its_own_reviewed_and_coached():
    """A reader's answer on one scenario: a factor above 0, what the simulation misses, why (else none); told its
    first answer and the review when it reads again; the reviewer asking the board's data before its points, a point
    on no scenario of the length dropped, a failing tool said so, no more asks than allowed; the coach's note (fake
    clients)."""
    from src.llm.methods import scenarios

    sets_ = {
        "month": [
            {"id": "calm", "name": "Calm", "story": "s", "ranges": {"stocks": [0, None]}},
            {"id": "fall", "name": "Fall", "story": "s", "ranges": {"stocks": [None, 0]}},
        ]
    }
    assert scenarios.answer({"times": "2", "missing": "a tariff", "why": "w"}) == {
        "times": 2.0,
        "missing": "a tariff",
        "why": "w",
    }
    assert scenarios.answer({"times": 0}) is None and scenarios.answer({"times": "x"}) is None, "a factor above 0"

    class Fake:
        def __init__(self, answers):
            self.answers, self.told = list(answers), []

        async def complete(self, user, system=None, node=None):
            self.told.append(user)
            return json.dumps(self.answers.pop(0))

    read = dict(today="2026-10-31", length="1 month (30 days)", scenario=sets_["month"][1], share="the simulation 3%")
    read |= dict(simulation="- VIX high: +0.40; stocks down", factors="", view="a pessimist", market="", record="")
    read |= dict(coach="", notes="", data=[])
    reader = Fake([{"times": 1.5, "missing": "m", "why": "w"}, {"times": 1.1, "missing": "m", "why": "w2"}])
    first = asyncio.run(scenarios.read(**read, client=reader))
    again = asyncio.run(scenarios.read(**read, client=reader, first=first, review="- already in the simulation: VIX"))
    assert first["times"] == 1.5 and again["times"] == 1.1
    assert "Fall" in reader.told[0] and "Calm" not in reader.told[0], "one scenario, the others unseen"
    assert "x1.5" in reader.told[1] and "already in the simulation" in reader.told[1], "told its answer and the review"

    calls = []

    def share(args):
        calls.append(args)
        return "history 2.0%"

    def broken(args):
        raise ValueError("no such length")

    tools = {"share": ("how often", share), "broken": ("fails", broken)}
    point = {"length": "month", "scenario": "fall", "says": "already in the simulation", "why": "the VIX is a signpost"}
    fake = Fake(
        [
            {"ask": {"tool": "share", "args": {"length": "month"}}},
            {"ask": {"tool": "broken", "args": {}}},
            {"points": [point, {**point, "scenario": "nope"}]},
        ]
    )
    kw = dict(today="2026-10-31", lengths={"month": ("1 month", 30, 2)}, simulation="", factors="", readings="")
    kw |= dict(sets_=sets_, market="", record="", tools=tools)
    out = asyncio.run(scenarios.review(**kw, client=fake))
    assert calls == [{"length": "month"}] and "history 2.0%" in fake.told[1], "an answer told before it asks again"
    assert out["asked"][1]["answer"].startswith("cannot answer"), "a failing tool said so"
    assert out["points"] == [point], "a point on no scenario of the length dropped"
    greedy = Fake([{"ask": {"tool": "share", "args": {}}}] * 3 + [{"points": []}])
    out = asyncio.run(scenarios.review(**kw, client=greedy, asks=2))
    assert len(out["asked"]) == 2 and "Questions left: 0" in greedy.told[-1], "no more asks than allowed"
    note = asyncio.run(scenarios.coach(today="2026-10-31", record="r", client=Fake([{"note": "too even"}])))
    assert note["note"] == "too even"

"""Research for one question (fake client and fake searches, no network): the rounds, the checked notes, the
forecast on the data and the notes."""

from __future__ import annotations

import asyncio
import json
import re

from src.llm.methods import grounded
from src.research import research, sources

QUESTION = {
    "id": "q",
    "title": "Does X fall 20% by 2028?",
    "outcomes": [{"id": "yes", "label": "Yes"}, {"id": "no", "label": "No"}],
}
DATA = [{"id": "D1", "name": "X", "value": 101.5}]


class FakeClient:
    """A plan, notes that cite the first news item (one point with a number not in it, one citing nothing that
    exists), a judge that wants one more round and then is satisfied, and a forecast of 30/70."""

    def __init__(self):
        self.judged = 0
        self.forecast_data = []

    async def complete(self, user, *, system=None, node=""):
        if node == "research.plan":
            return json.dumps({"drivers": ["earnings"], "queries": ["x earnings", "x guidance"]})
        if node == "grounded.points":
            first = re.search(r"\[(D\d+)\] news", user).group(1)
            return json.dumps(
                {
                    "points": [
                        f"X cut its outlook by 12% [{first}], so yes is likelier.",
                        f"X sales fell 99% [{first}].",
                        "Analysts expect a rebound [D999].",
                    ]
                }
            )
        if node == "research.judge":
            self.judged += 1
            return json.dumps({"enough": self.judged > 1, "missing": "guidance", "queries": ["x layoffs"]})
        if node.startswith("forecast"):
            self.forecast_data.append(user)
            if node == "forecast.single":
                return json.dumps({"probability": 0.3 if "Yes" in user else 0.7, "reason": ""})
            return json.dumps({"probabilities": {"yes": 0.3, "no": 0.7}, "reason": ""})
        raise AssertionError(node)


def fake_search(query):
    return [
        {
            "kind": "news",
            "title": f"X cuts outlook by 12% ({query})",
            "source": "Wire",
            "day": "2026-10-28",
            "url": f"https://a/{query}",
            "text": "",
        },
        {
            "kind": "news",
            "title": "Same story",
            "source": "Wire",
            "day": "2026-10-28",
            "url": "https://a/same",
            "text": "",
        },
    ]


def run(client):
    return asyncio.run(
        research.research(
            QUESTION,
            DATA,
            today="2026-10-29",
            client=client,
            forecast_kw={"models": ["m"], "samples": 1, "client_for": lambda m: client},
            search=fake_search,
        )
    )


def test_rounds_notes_and_forecast():
    client = FakeClient()
    out = run(client)
    assert [r["queries"] for r in out["rounds"]] == [["x earnings", "x guidance"], ["x layoffs"]]
    assert [r["found"] for r in out["rounds"]] == [3, 1]  # the same link once only
    assert [e["id"] for e in out["evidence"]] == ["D2", "D3", "D4", "D5"]
    # the point with a number not in its item and the one citing nothing that exists are dropped
    assert [n["value"] for n in out["notes"]] == ["X cut its outlook by 12% [D2], so yes is likelier."]
    dropped = out["rounds"][-1]["dropped"]
    assert dropped[0]["unknown_numbers"] == ["99% [D2]"] and dropped[1]["unknown_ids"] == ["D999"]
    assert out["summary"]["p"]["yes"] < out["summary"]["p"]["no"]
    assert all("Research note: X cut its outlook" in u for u in client.forecast_data)
    assert out["method"] == research.method_hash()


def test_numbers_in_an_items_text_count():
    items = [{"id": "D7", "name": "news, Wire", "value": "«X cuts outlook by 12.5%»"}]
    assert grounded.unknown_numbers("outlook cut 12.5% [D7]", items) == []
    assert grounded.unknown_numbers("outlook cut 14% [D7]", items) == ["14% [D7]"]


def test_a_failing_source_gives_nothing(monkeypatch):
    def down(query):
        raise OSError("feed down")

    monkeypatch.setattr(sources, "google_news", down)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert sources.search("x") == []

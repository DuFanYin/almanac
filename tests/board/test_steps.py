"""The steps, on throwaway databases: holdings, a question's sides, a fresh clone, proposed questions, what the LLM
cited, the LLM's scenarios, proposed signposts, texts read into numbers, factors proposed. (No network, no account
data.)"""

import json as _json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from src.board.core import lineage, store
from src.board.data import account, observations
from src.board.model import allocation, portfolio, rules
from src.board.model import proposals as pchk
from src.board.model import questions as qs
from src.board.steps.proposals import signposts as spp
from tests.board.support import check, tmp


def test_holdings():
    """Holdings."""
    rcfg = {
        "targets": {"btc_max": 0.35},
        "pace": {"trim_start": "2026-11-01", "trim_last": "2027-10"},
        "categories": {
            "semis": {"label": "Semiconductors", "tech": True, "target": 0.10, "cap": 0.15, "class": "semis"},
            "platforms": {"label": "Big tech platforms", "tech": True, "target": 0.30, "class": "us_tech"},
            "cash": {"label": "Cash", "target": 0.60, "class": "cash"},
        },
    }
    exposure = {
        "ibkr": {
            "nav": 1000.0,
            "cash": 100.0,
            "positions": [
                {"symbol": "XYZ", "underlying": "XYZ", "asset": "Stock", "qty": 1, "mv": 500.0},
                {"symbol": "AAPL", "underlying": "AAPL", "asset": "Stock", "qty": 1, "mv": 300.0},
                {"symbol": "NEW", "underlying": "NEW", "asset": "Stock", "qty": 1, "mv": 100.0},
            ],
        },
        "btc": {"qty": 0.01, "buys": [{"date": "2026-10-01", "usd": 100}, {"date": "2026-10-08", "usd": 120}]},
    }
    h = portfolio.holdings(exposure, {"XYZ": "semis", "AAPL": "platforms"}, 10000.0, rcfg)
    sh = portfolio.shares(h, rcfg)
    tg = portfolio.targets(rcfg)
    check(
        "categories from the hand-kept file",
        h["by"]["semis"] == 500 and h["by"]["platforms"] == 300 and h["by"]["cash"] == 100,
    )
    check(
        "a symbol not in the file is unclassified, and named",
        h["by"]["unclassified"] == 100 and h["unclassified"] == ["NEW"],
    )
    check(
        "tech share of the account, BTC share of everything",
        abs(sh["tech"] - 0.8) < 1e-9 and abs(sh["btc"] - 100 / 1100) < 1e-9,
    )
    check(
        "tech target is the tech categories' sum; caps kept", abs(tg["tech"] - 0.40) < 1e-9 and tg["semis_cap"] == 0.15
    )
    usd, how = portfolio.amounts("trim_month", h, sh, tg, rcfg, date(2026, 11, 3), None)
    check(
        "monthly trim: the excess over the months left, capped categories first",
        abs(usd - 400 / 12) < 1e-6 and "semiconductors first" in how,
    )
    check("regular purchase and its rhythm", portfolio.dca(exposure) == (110.0, "week"))
    rcfg_cash = {**rcfg, "pace": {**rcfg["pace"], "to_cash_step": 0.03}}
    usd, how = portfolio.amounts("to_cash_step", h, sh, tg, rcfg_cash, date(2026, 11, 3), None)
    check("a step into cash comes from the stocks most above their target", abs(usd - 30) < 1e-9 and "semicond" in how)
    check(
        "half the regular purchase", portfolio.amounts("dca_half", h, sh, tg, rcfg, date(2026, 11, 3), 110.0)[0] == 55.0
    )
    check(
        "account data is written private",
        (store.write_json(store.ACCOUNT_VIEW, {}) or True) and oct(store.ACCOUNT_VIEW.stat().st_mode)[-3:] == "600",
    )


def test_the_settings():
    """The settings read from local/settings.toml."""
    check(
        "a URL in the settings stays a URL",
        (store.LOCAL / "settings.toml").write_text('notify = "http://h:1"\npositions = "~/x"\n')
        or (store.settings()["notify"] == "http://h:1" and isinstance(store.settings()["positions"], Path)),
    )
    (store.LOCAL / "settings.toml").unlink()


def test_a_question_read_off_the_world():
    """A question read off the world: its own outcome after each start day, weighed as the world weighs it; history
    unweighted as its base; pooled with the market; the owner's view on top; a band from its independent windows.
    What the LLM cited, traced to its sources; a node with a model upstream knows it."""
    from src.board.model import timescale as tsm
    from src.board.model import world as wm
    from src.board.steps import probabilities as prob_step  # noqa: E402

    pdb = store.connect(tmp / "data" / "p.db")
    days = pd.date_range("1971-01-01", "2026-10-29", freq="D")
    gspc = pd.Series(np.exp(np.cumsum(np.random.default_rng(6).normal(0.0003, 0.01, len(days)))), index=days)
    fcfg = {"history_from": "1971-03-01", "factor": {"stocks": {"label": "S&P", "series": "GSPC", "measure": "ret"}}}
    flat = wm.World(fcfg, {"GSPC": gspc}, tsm.stretches(), {"move": {}, "spread": {}})
    up = wm.World(fcfg, {"GSPC": gspc}, tsm.stretches(), {"move": {"stocks": np.full(9, 0.5)}, "spread": {}})
    buckets = [{"id": "up", "label": "Up", "when": ["ret > 0"]}, {"id": "down", "label": "Down", "when": []}]
    q = {
        "id": "q1",
        "kind": "path",
        "closes": "2027-10-29",
        "buckets": buckets,
        "spec": {"template": "t", "series": "GSPC", "start_day": "2026-10-29", "closes": "2027-10-29", "start_px": 1.0},
    }
    pcfg = {
        "series": {"GSPC": {"source": "yahoo"}},
        "questions": {
            "weights": {"own": 0.2, "without_market": {"base": 1.0}, "with_market": {"base": 0.35, "market": 0.65}}
        },
    }
    got, hist = prob_step.read(pdb, pcfg, "h", [q], {"GSPC": gspc}, {"data": up, "clean": flat}, {}, date(2026, 10, 29))
    r = got["q1"]
    check("history unweighted as its base", abs(sum(r["sources"]["base"].values()) - 1) < 1e-9)
    check("a world leaning up reads more of up than history", r["model"]["up"] > r["sources"]["base"]["up"] + 0.02)
    check(
        "without the LLM's side (here: history), the base", abs(r["no_llm"]["up"] - r["sources"]["base"]["up"]) < 1e-9
    )
    check("a band from its independent windows", r["interval"]["up"][0] < r["model"]["up"] < r["interval"]["up"][1])
    check("its outcomes as it opened kept for proposals", "t" in hist)
    qm = {**q, "id": "q2", "market": ({"up": 0.9, "down": 0.1}, None)}
    m = prob_step.read(pdb, pcfg, "h", [qm], {"GSPC": gspc}, {"data": flat, "clean": flat}, {}, date(2026, 10, 29))
    m = m[0]["q2"]
    check(
        "a market pooled with the world (0.65 to 0.35)",
        abs(m["model"]["up"] - (0.65 * 0.9 + 0.35 * m["sources"]["base"]["up"])) < 1e-9,
    )
    rows = dict(pdb.execute("SELECT source, count(*) FROM probs WHERE question = 'q1' GROUP BY source").fetchall())
    check("each side kept daily", {"base", "math", "model", "no_llm"} <= set(rows))
    check("days left: to its close, a week at least", prob_step.days_left(q, date(2026, 10, 29)) == 365)
    from src.board.model import scenarios as sm

    a_year = up.weights(365)
    scs = [
        {"id": "falls", "ranges": {"stocks": [-0.9, 0.0]}, "times": 16.0},
        {"id": "rises", "ranges": {"stocks": [0.0, 2.0]}},
    ]
    at = sm.at_length(a_year, scs)
    scen = {"sets": {"long": {"a": a_year, "set": {"node": None}, **at}}}
    s_ = prob_step.read(
        pdb, pcfg, "h", [{**q, "id": "q3"}], {"GSPC": gspc}, {"data": up, "clean": flat}, scen, date(2026, 10, 29)
    )
    s_ = s_[0]["q3"]
    rises = at["each"][1]["weight"]
    check(
        "the LLM's scenarios the last layer: a year up near the probability of the rises (the falls x16), the "
        "data's world far above",
        abs(s_["model"]["up"] - rises) < 0.15 and s_["scenarios_from"] == "long" and s_["math"]["up"] > rises + 0.3,
    )

    # what the LLM cited, traced to its sources; a node with a model upstream knows it
    from src.board.steps import llm_data as panel_step  # noqa: E402

    items = [
        {"id": "D1", "name": "S&P 500", "value": 1, "source": "series:GSPC"},
        {"id": "D2", "name": "VIX high (signpost, red)", "value": 2, "source": "signpost:vix_high"},
        {"id": "D3", "name": "News note, research of 2026-10-29", "value": "x"},
    ]
    refs = panel_step.cited(
        [{"reason": "VIX [D2] and [D2], S&P [D1], news [D3], nothing [D9]"}, {"reason": "[D2]"}], items
    )
    check(
        "citations counted by source, unknown ids left out",
        refs == {"signpost:vix_high": 3, "series:GSPC": 1, "news": 1},
    )
    ldb = store.connect(tmp / "data" / "llm.db")
    made = lineage.record(ldb, "llm_series", "raw:FEDTONE", {"last": 0.4})
    base_node = lineage.record(ldb, "base_rate", "base:q2", {"p": {"a": 0.5}}, {}, [made])
    check("a node with a model upstream knows it", lineage.llm_upstream(ldb, base_node) == ["raw:FEDTONE"])
    check(
        "one without does not", lineage.llm_upstream(ldb, lineage.record(ldb, "base_rate", "base:q3", {"p": 1})) == []
    )


def test_a_fresh_clone_no_local_the_example():
    """A fresh clone: no local/, the example rules, each category on an asset class."""
    cfg0, _ = store.config()
    check(
        "without local/rules.toml the example rules are read",
        store.rules_file().name == "rules.example.toml" and cfg0["rules"]["rule"],
    )
    check(
        "every category of the example names an asset class",
        all(c["class"] in cfg0["asset_classes"]["class"] for c in cfg0["rules"]["categories"].values()),
    )
    check(
        "no positions file: the board runs without holdings",
        account.exposure(store.connect(tmp / "data" / "x.db")) == (None, None),
    )
    ctx0 = rules.Context(
        {**cfg0["rules"], "targets": portfolio.targets(cfg0["rules"])},
        pd.DataFrame({"GSPC": [100.0]}),
        {},
        None,
        date(2026, 11, 3),
    )
    check("without holdings a rule on a share cannot be judged", ctx0.holds("stocks_over_cap") is None)


def test_a_proposed_question_checked_on_the_data():
    """A proposed question: checked on the data, adopted by the owner into local/questions.toml."""
    from src.board.model import proposals as pchk  # noqa: E402
    from src.board.steps.proposals import questions as pstep  # noqa: E402
    from src.board.steps.proposals import votes as pvotes  # noqa: E402
    from src.llm.methods import propose as pmethod  # noqa: E402

    pdays = pd.date_range("1950-01-01", "2026-10-01", freq="MS")
    rng_p = np.random.default_rng(9)
    cpi = pd.Series(np.cumprod(1 + rng_p.normal(0.003, 0.004, len(pdays))) * 100, index=pdays)
    draft = {
        "id": "cpi_2y",
        "title": "CPI over two years",
        "purpose": "decision",
        "kind": "path",
        "series": "CPI",
        "days": 730,
        "detail": "x",
        "why": "y",
        "buckets": [
            {"id": "hot", "label": "Above 4% a year", "when": ["ret >= 0.0816"]},
            {"id": "calm", "label": "The rest", "when": []},
        ],
    }
    shaped, why_not = pmethod.shaped(draft, {"CPI"})
    check("a draft in shape", shaped is not None and why_not is None and shaped["buckets"][0]["tone"] == 0)
    check(
        "one on a series the board does not keep is not", pmethod.shaped({**draft, "series": "X"}, {"CPI"})[0] is None
    )
    check(
        "nor one whose last bucket has a condition",
        pmethod.shaped({**draft, "buckets": [{**b, "when": ["ret > 0"]} for b in draft["buckets"]]}, {"CPI"})[0]
        is None,
    )
    check(
        "nor one reading a name its kind does not have",
        pmethod.shaped(
            {**draft, "buckets": [{**draft["buckets"][0], "when": ["chg > 0"]}, draft["buckets"][1]]}, {"CPI"}
        )[0]
        is None,
    )
    hot = qs.history_outcomes(pchk.as_question(shaped, cpi, date(2026, 10, 29))[0], cpi)[0]
    mix_p = {d: (-0.2 if b == "hot" else 0.15) + rng_p.normal(0, 0.03) for d, b in hot.items()}
    res = pchk.check(shaped, cpi, date(2026, 10, 29), {}, mix_p, 730)
    check(
        "it passes when history can say, it is new and your targets fare differently by its outcome",
        res["passed"] and res["value"]["between"] == ("calm", "hot") and res["value"]["gap_90"][0] > 0,
    )
    res_same = pchk.check(shaped, cpi, date(2026, 10, 29), {"cpi_old": hot}, mix_p, 730)
    check("not when the board asks it already", not res_same["passed"] and res_same["overlap"][1] == "cpi_old")
    flat_mix = {d: 0.05 + rng_p.normal(0, 0.03) for d in hot}
    check(
        "nor when your targets fare the same either way",
        not pchk.check(shaped, cpi, date(2026, 10, 29), {}, flat_mix, 730)["passed"],
    )
    rare = {**shaped, "buckets": [{**shaped["buckets"][0], "when": ["ret >= 2"]}, shaped["buckets"][1]]}
    check(
        "nor when history already knows the answer",
        not pchk.check(rare, cpi, date(2026, 10, 29), {}, mix_p, 730)["passed"],
    )
    check(
        "a question to score by must resolve soon",
        not pchk.check({**shaped, "purpose": "score"}, cpi, date(2026, 10, 29), {}, None, 730)["passed"],
    )
    entry = pstep.toml_entry(shaped, pd.Timestamp("2026-10-29"))
    import tomllib  # noqa: E402

    parsed = tomllib.loads(entry)["cpi_2y"]
    check(
        "adopted, it is a question in questions.toml's format",
        parsed["kind"] == "path"
        and parsed["closes"] == "2028-10-28"
        and parsed["buckets"][0]["when"] == ["ret >= 0.0816"],
    )

    # the owner's vote on the page: an adopted question written to local/questions.toml and read with the config
    import json as _json  # noqa: E402
    import sqlite3  # noqa: E402

    inputs_db = store.INPUTS  # the page writes them (src/web)
    inputs_db.unlink(missing_ok=True)
    con = sqlite3.connect(inputs_db)
    con.execute("CREATE TABLE proposals (id TEXT PRIMARY KEY, status TEXT, note TEXT, at INTEGER)")
    con.execute("INSERT INTO proposals VALUES ('cpi_2y', 'adopted', '', 0)")
    con.commit()
    con.close()
    adb = store.connect(tmp / "data" / "adopt.db")
    adb.execute(
        "INSERT INTO proposals VALUES (?,?,?,?,?,?,?,?)",
        ("cpi_2y", "2026-10-29", _json.dumps(shaped), _json.dumps(res), 1, "open", None, None),
    )
    done = pvotes.adopt(adb, date(2026, 10, 30))
    cfg_a, _ = store.config()
    check(
        "an adopted question joins the config the same day, and is no longer open",
        done == ["cpi_2y"]
        and cfg_a["questions"]["cpi_2y"]["closes"] == "2028-10-29"
        and adb.execute("SELECT status FROM proposals").fetchone()[0] == "adopted",
    )
    check("a vote counts once", pvotes.adopt(adb, date(2026, 10, 31)) == [])
    store.ADOPTED.unlink()
    # left by the owner three days: the LLM decides (stubbed here), within its cap; one it declines stays votable
    from src.board.data import llm as llm_data  # noqa: E402

    asked = []

    def judge(today, kind, proposal, checks, has, n=3):
        asked.append(proposal["id"])
        yes = proposal["id"] != "no_q"
        return {"method": "m", "votes": [], "adopt": yes, "why": "w", "usage": {}}

    real, llm_data.judge_proposal = llm_data.judge_proposal, judge
    try:
        for pid, day in (
            ("q_a", "2026-10-20"),
            ("q_b", "2026-10-21"),
            ("no_q", "2026-10-19"),
            ("q_c", "2026-10-23"),
            ("q_new", "2026-10-30"),
        ):
            adb.execute(
                "INSERT INTO proposals VALUES (?,?,?,?,?,?,?,?)",
                (pid, day, _json.dumps({**shaped, "id": pid}), _json.dumps(res), 1, "open", None, None),
            )
        got = pvotes.adopt(adb, date(2026, 11, 1), use_llm=True)
        status = dict(adb.execute("SELECT id, status FROM proposals").fetchall())
        check(
            "the LLM decides what the owner left three days, at most two questions a month",
            got == ["no_q", "q_a", "q_b"]
            and status["q_a"] == status["q_b"] == "adopted"
            and status["no_q"] == "declined by the LLM"
            and status["q_c"] == "open"
            and status["q_new"] == "open",
        )
        cfg_l, _ = store.config()
        check("what it adopted says so", cfg_l["questions"]["q_a"].get("adopted_by") == "llm")
        rows = {r["id"]: r for r in pvotes.rows(adb)}
        check("the page sees whose decision it was", rows["q_a"]["by"] == "llm" and rows["q_a"]["why"] == "w")
    finally:
        llm_data.judge_proposal = real
    con = sqlite3.connect(inputs_db)
    con.execute("INSERT INTO proposals VALUES ('q_a', 'withdrawn', '', 0)")
    con.commit()
    con.close()
    pvotes.adopt(adb, date(2026, 11, 2))
    check(
        "withdrawn: its table out of local/questions.toml, the rest kept",
        "q_a" not in store.config()[0]["questions"] and "q_b" in store.config()[0]["questions"],
    )
    store.ADOPTED.unlink()


def test_a_signpost_the_llm_proposes_checked_on():
    """A signpost the LLM proposes: checked on the data; joined to the signposts (paused since the world)."""
    from src.board.steps.proposals import signposts as spp  # noqa: E402

    sdays = pd.date_range("2000-01-01", "2026-10-01", freq="D")
    sfr = pd.DataFrame(
        {"A": np.sin(np.arange(len(sdays)) / 200.0) + 2, "B": np.cos(np.arange(len(sdays)) / 300.0)}, index=sdays
    )
    draft_sp = {"id": "a_high", "name": "A high", "value": "A", "red": "> 2.8", "try_on": ["stocks_short"]}
    ok_sp = pchk.check_signpost(draft_sp, sfr, {"b": sfr["B"]})
    check("a proposed signpost on the data: computable, long enough, red some days, new", ok_sp["passed"])
    check(
        "one on a series the board does not keep fails",
        not pchk.check_signpost({**draft_sp, "value": "Z"}, sfr, {})["passed"],
    )
    check("one never red fails", not pchk.check_signpost({**draft_sp, "red": "> 9"}, sfr, {})["passed"])
    check(
        "a near copy of one the board has fails", not pchk.check_signpost(draft_sp, sfr, {"a": sfr["A"] * 2})["passed"]
    )
    pdb2 = store.connect(tmp / "data" / "sp.db")
    for sid, status in (("a_high", "trial"), ("gone", "failed")):
        pdb2.execute(
            "INSERT INTO signpost_proposals VALUES (?,?,?,?,?,?,?)",
            (sid, "2026-01-05", _json.dumps({**draft_sp, "id": sid}), "{}", status, None, "{}"),
        )
    spcfg = {"signposts": {"signpost": [{"id": "own", "name": "o", "value": "B", "red": "> 0"}]}}
    spp.with_candidates(pdb2, spcfg)
    spp.with_candidates(pdb2, spcfg)
    cand = spcfg["signposts"]["signpost"][-1]
    check(
        "joined to the signposts once, with the day proposed (paused: with no expectations it moves nothing)",
        [x["id"] for x in spcfg["signposts"]["signpost"]] == ["own", "a_high"]
        and cand["proposed"] == "2026-01-05"
        and "expect" not in cand,
    )


def test_texts_read_into_numbers_the_fed_s():
    """Texts read into numbers: the Fed's page parsed, new texts read once, the day's mean stored as the series; a."""
    # signpost on it accumulates, then is checked
    from src.board.data import texts as texts_data  # noqa: E402
    from src.board.steps import readings as rd_step  # noqa: E402

    page = '<nav>menu</nav><div id="article"><p>The Committee decided</p><script>x()</script><p>to hold.</p></div>'
    check(
        "the Fed's page: the article alone, no scripts",
        texts_data.article_text(page) == "The Committee decided\nto hold.",
    )
    index = (
        '<tr><td>1</td><td>8-K</td><td><a href="/Archives/edgar/data/1/2/x-8k.htm">x-8k.htm</a></td><td>8-K</td></tr>'
        '<tr><td>2</td><td>Press release</td><td><a href="/Archives/edgar/data/1/2/ex991.htm">ex991.htm</a></td>\n'
        "<td>EX-99.1</td></tr>"
    )
    check(
        "an EDGAR filing's press release found by its type",
        texts_data.press_release(index) == "/Archives/edgar/data/1/2/ex991.htm",
    )
    check(
        "a release's text: blocks a line, cells spaced, no styles",
        texts_data.document_text("<style>p{}</style><p>Capex</p><table><tr><td>Q3</td><td>$35bn</td></tr></table>")
        == "Capex\nQ3 $35bn",
    )
    rss = (
        "<rss><channel><item><title>FOMC statement</title><link>https://f/1</link>"
        "<pubDate>Wed, 28 Oct 2026 18:00:00 GMT</pubDate></item></channel></rss>"
    )
    check(
        "a feed's items, dated in New York",
        texts_data.feed_items(rss) == [("FOMC statement", "https://f/1", "2026-10-28")],
    )
    rdb = store.connect(tmp / "data" / "r.db")
    fed_docs = [
        {"id": "u1", "day": "2026-10-28", "title": "FOMC", "url": "u1", "text": "t"},
        {"id": "u2", "day": "2026-10-28", "title": "Speech", "url": "u2", "text": "t"},
    ]
    real_fed, real_read = texts_data.fed, rd_step.llm.read
    texts_data.fed = lambda since: fed_docs
    values = iter([1.0, 0.0, 2.0])
    rd_step.llm.read = lambda text, **kw: {
        "method": "m",
        "relevant": True,
        "value": next(values),
        "spread": 0.1,
        "answers": [],
        "dropped": [],
        "usage": {},
    }
    rcfg = {
        "readings": {
            "samples": 1,
            "back": 7,
            "reading": {"FED_HAWK": {"texts": "fed", "what": "w", "scale": "s"}},
            "accumulate": 30,
            "signpost": [
                {
                    "id": "fed_reads_hawkish",
                    "name": "n",
                    "driver": "d",
                    "value": "mean(FED_HAWK, 2)",
                    "red": "> 0.75",
                    "try_on": ["rate10_short"],
                }
            ],
        },
        "series": {"FED_HAWK": {"source": "reading", "name": "n"}},
    }
    check(
        "each new text read once",
        rd_step.run(rdb, rcfg, date(2026, 10, 30)) == {"FED_HAWK": 2}
        and rd_step.run(rdb, rcfg, date(2026, 10, 30)) == {},
    )
    check("the series: the day's mean", observations.latest_obs(rdb)["FED_HAWK"] == [("2026-10-28", 0.5)])
    texts_data.fed, rd_step.llm.read = real_fed, real_read
    check(
        "a signpost on a reading starts accumulating, once",
        rd_step.register_signposts(rdb, rcfg, date(2026, 10, 30)) == 1
        and rd_step.register_signposts(rdb, rcfg, date(2026, 10, 30)) == 0,
    )
    acfg = {"signposts": {"signpost": []}, **rcfg}
    spp.with_candidates(rdb, acfg)
    check("accumulating: shown, moving nothing", "expect" not in acfg["signposts"]["signpost"][0])

    cb = allocation.chart_bins([np.linspace(-0.15, 0.2, 1000)], [0.0, -0.175])
    check(
        "a chart's bins: the returns and the limit inside, a round width, at most BINS of them",
        cb[0] <= -0.175
        and cb[1] >= 0.2
        and cb[2] in allocation.STEPS
        and (cb[1] - cb[0]) / cb[2] <= allocation.BINS + 1e-9,
    )


def test_a_factor_the_llm_proposes():
    """A factor the LLM proposes: checked on the data, adopted into local/."""
    from src.board.steps.proposals import changes as chp  # noqa: E402

    fdays = pd.date_range("1970-01-01", "2026-10-01", freq="D")
    rng_f = np.random.default_rng(8)
    cdrv = pd.Series(np.cumsum(rng_f.normal(0, 0.01, len(fdays))), index=fdays)  # what the class follows
    noise = pd.Series(np.cumsum(rng_f.normal(0, 0.01, len(fdays))), index=fdays)
    fseries = {"C": np.exp(cdrv), "N": np.exp(noise), "A": np.exp(cdrv * 1.5 + 0.2 * noise)}
    fframe = pd.DataFrame(fseries)
    ffcfg = {
        "history_from": "1971-03-01",
        "horizon": {"long": {"days": 365}},
        "factor": {"n": {"label": "N", "series": "N", "measure": "ret"}},
    }
    fcls = {"a": {"proxy": "A"}}
    chk_f = pchk.check_factor({"id": "c", "series": "C", "measure": "ret", "label": "C"}, ffcfg, fseries, fframe, fcls)
    check("a factor the class follows explains more: passes", chk_f["passed"] and chk_f["gains"]["long"] > 0.2)
    check(
        "one that adds nothing does not",
        not pchk.check_factor(
            {"id": "n2", "series": "N", "measure": "ret", "label": "N2"}, ffcfg, fseries, fframe, fcls
        )["passed"],
    )
    chp.adopt_factor(
        {"id": "credit", "label": "Credit", "series": "BAA10Y", "measure": "diff", "unit": "pts", "up": -1}
    )
    check("an adopted factor joins the factors", "credit" in store.config()[0]["factors"]["factor"])
    store.ADOPTED_FACTORS.unlink()


def test_the_day_s_news():
    """A rule fired, a signpost red today, a headline that moved ten points in a week (sent once), the futures' news;
    the decision step's suggestions new, changed and gone."""
    from src.board.steps import alerts as al

    db = store.connect(tmp / "data" / "alerts.db")
    db.execute(
        "INSERT INTO questions VALUES ('spx', 't', 'path', '2026-01-01', '2028-01-01', '[]', ?, NULL)",
        (_json.dumps({"template": "spx"}),),
    )
    for d, p in (("2026-10-20", 0.2), ("2026-10-21", 0.2), ("2026-10-28", 0.2)):
        db.execute("INSERT INTO probs VALUES (?, 'spx', 'bear', 'model', ?, NULL, NULL)", (d, p))
    cfg = {"questions": {"spx": {"headline": {"buckets": ["bear"], "short": "S&P bear"}}}}
    sps = [{"id": "s", "name": "Spreads wide"}]
    today = date(2026, 10, 29)
    lines = al.alerts(
        db,
        cfg,
        sps,
        {"s": {"state": "red", "since": "2026-10-29"}},
        {"spx": {"model": {"bear": 0.35}}},
        [{"title": "Sell tech"}],
        today,
        ["Future denied: x"],
    )
    check(
        "a rule, the futures' news, a signpost red today, a headline 15 points up in a week",
        lines == ["Rule: Sell tech", "Future denied: x", "Signpost red: Spreads wide", "S&P bear: 20% → 35% in a week"],
    )
    db.execute("INSERT INTO probs VALUES ('2026-10-29', 'spx', 'bear', 'model', 0.35, NULL, NULL)")
    later = al.alerts(db, cfg, sps, {}, {"spx": {"model": {"bear": 0.36}}}, [], date(2026, 10, 30))
    check("the move was sent yesterday: not again", later == [])
    lean = {"id": "lean:us_tech", "kind": "lean", "title": "Lower Big tech by 2.0%", "size": -0.02}
    check("a suggestion new", al.suggestions([], [lean]) == ["Suggested: Lower Big tech by 2.0%"])
    bigger = {**lean, "title": "Lower Big tech by 4.0%", "size": -0.04}
    check(
        "changed by a point or more",
        al.suggestions([lean], [bigger]) == ["Suggested now: Lower Big tech by 4.0% (was 2.0%)"],
    )
    check("changed by less: nothing", al.suggestions([lean], [{**lean, "size": -0.025}]) == [])
    check("gone", al.suggestions([lean], []) == ["No longer suggested: Lower Big tech by 2.0%"])
    check(
        "a rule's line is not a suggestion", al.suggestions([], [{"id": "rule:x", "kind": "rule", "title": "x"}]) == []
    )


def test_a_text_reading_with_nothing_lately_says_nothing():
    """A text reading with nothing in the last weeks says nothing now."""
    from src.board.steps import scenarios as ls

    reading = pd.Series([1.5, 1.5], index=pd.to_datetime(["2026-08-01", "2026-08-02"]))
    rc = {"reading": {"AI": {"bears_on": {"stocks": 1}}}}
    sc = {"AI": {"name": "AI investment"}}
    check("an old reading says nothing now", ls.readings_now({"AI": reading}, rc, sc, date(2026, 10, 30)) == {})
    check("a recent one does", ls.readings_now({"AI": reading}, rc, sc, date(2026, 8, 20))["AI"]["sign"] == 1)


def test_the_outlook_kept_and_resolved():
    """Each factor's quartiles over a length kept for the day; resolved once the window is over and the factor has a
    value at its end."""
    from src.board.steps import world as ws

    db = store.connect(tmp / "data" / "outlook.db")
    probs = {"final": [0.1, 0.2, 0.3, 0.4], "no_llm": [0.25] * 4, "data": [0.25] * 4, "history": [0.25] * 4}
    horizons = {"week": {"days": 7, "factors": {"stocks": {"edges": [-0.01, 0.0, 0.01], **probs}}}}
    ws.outlook(db, horizons, date(2026, 10, 1))
    fcfg = {"factor": {"stocks": {"series": "GSPC", "measure": "ret"}}}
    days = pd.date_range("2026-09-01", "2026-10-20", freq="D")
    gspc = pd.Series(np.where(days > pd.Timestamp("2026-10-01"), 110.0, 100.0), index=days)  # up 10% in the week
    check("not over yet: unresolved", ws.resolve_outlook(db, fcfg, {"GSPC": gspc}, date(2026, 10, 5)) == 0)
    check("over: resolved", ws.resolve_outlook(db, fcfg, {"GSPC": gspc}, date(2026, 10, 12)) == 1)
    out, value = db.execute("SELECT outcome, value FROM outlook").fetchone()
    check("in its top quartile", out == 3 and abs(value - 0.1) < 1e-9)


def test_each_length_s_limit_and_the_trades_on_what_you_hold():
    """A year's limit from your risk level; each other length's as strict on your targets' own history. Each side's
    probability: the paths your targets end below usual in or not, and those in no scenario. A trade on what you hold
    stands until done; once done, the next step the same way waits PACE days."""
    from src.board.model import decide as dm
    from src.board.steps import decide as dec_step
    from src.board.steps import distribution as dist_step

    you = {"limits": {"worst_loss": 0.35, "step": 0.075, "band": 0.075}, "policy": {"a": 0.9, "cash": 0.1}, "own": True}
    cfg = {"factors": {"year_limit": 0.8}}
    lim = dist_step.limits(cfg, you, {"week": -0.04, "long": -0.2})
    check("a year's: worst_loss times year_limit", abs(lim["long"] + 0.28) < 1e-12)
    check("a week's: as far past its usual", abs(lim["week"] + 0.28 * 0.2) < 1e-12)

    class F:  # four paths, the targets' return in each, the final world's weights and the simulation's
        w = np.array([0.4, 0.3, 0.2, 0.1])

    class D:
        w = np.array([0.25, 0.25, 0.25, 0.25])

    r = np.array([-0.1, 0.02, 0.08, 0.2])
    inside = {"m:a": np.array([True, True, False, False]), "m:b": np.array([False, True, False, False])}
    check(
        "your targets below usual or not, path by path, and the paths in no scenario: not added up from the "
        "scenarios, which overlap",
        dist_step.sides(r, 0.05, F, D, inside)
        == {"worse": {"p": 0.7, "data": 0.5}, "better": {"p": 0.3, "data": 0.5}, "none": {"p": 0.3, "data": 0.5}},
    )
    db = store.connect(tmp / "data" / "trades.db")
    cats = {"x": {"class": "a", "target": 0.9, "label": "X"}, "c": {"class": "cash", "target": 0.1, "label": "Cash"}}
    decided = {"x": 0.9, "c": 0.1}

    def day(d, x):
        held = {"shares": {"x": x, "c": 1 - x}, "node": None}
        return [a for a in dec_step.trades(db, you, cats, decided, held, d, None) if a["kind"] == "trade"]

    one = day(date(2026, 11, 2), 0.7)
    check(
        "past your band: a step to buy", len(one) == 1 and one[0]["status"] == "suggested" and one[0]["size"] == 0.075
    )
    two = day(date(2026, 11, 3), 0.7)
    check("not done: it stands as it was", two[0]["status"] == "standing" and two[0]["id"] == one[0]["id"])
    check("done: the next step waits", day(date(2026, 11, 5), 0.78) == [])
    later = day(date(2026, 11, 5) + pd.Timedelta(days=dm.PACE).to_pytimedelta(), 0.78)
    check("after PACE days, the next step", len(later) == 1 and later[0]["status"] == "suggested")


def test_what_the_llm_expects_of_a_reading():
    """A signpost on a text reading is asked once what it says about the factors (and again when its definition
    changes); the answer kept in its spec and carried to the signposts the world reads."""
    from src.board.steps import readings as rd
    from src.board.steps.proposals import signposts as sp_props

    db = store.connect(tmp / "data" / "expect.db")
    sp = {"id": "fed_hawk", "name": "Fed hawkish", "value": "mean(FED_HAWK, 60)", "red": "> 0.75", "why": "w"}
    cfg = {
        "readings": {"reading": {"FED_HAWK": {"what": "how hawkish", "scale": "-2..2"}}, "signpost": [sp]},
        "factors": {"factor": {"rate10": {"label": "10-year", "measure": "diff"}}},
        "signposts": {"signpost": []},
    }
    rd.register_signposts(db, cfg, date(2026, 10, 31))
    asked = []

    def ask(indicator, factors):
        asked.append((indicator["what"], sorted(factors)))
        e = {"factor": "rate10", "way": "up", "days": [0, 180], "why": "a hawkish Fed lifts yields"}
        return {"method": "m", "expect": [e], "rejected": [], "usage": {}}

    check(
        "asked once", rd.expectations(db, cfg, date(2026, 10, 31), ask) == 1 and asked == [("how hawkish", ["rate10"])]
    )
    check("not again", rd.expectations(db, cfg, date(2026, 11, 1), ask) == 0)
    cfg["readings"]["signpost"] = [{**sp, "red": "> 1.0"}]
    check("its definition changed: asked again", rd.expectations(db, cfg, date(2026, 11, 2), ask) == 1)
    joined = sp_props.with_candidates(db, cfg)["signposts"]["signpost"]
    check(
        "carried to the signposts, marked a reading",
        joined[0]["expect"][0]["way"] == "up" and joined[0]["reading"] is True and "proposed" in joined[0],
    )


def test_research_asks_its_forecast_with_its_own_settings():
    """Research on a question: the news read by research's model, then the forecast with its own views and share of
    the data left out (a stub for the research itself: every name it is given must exist)."""
    from src.board.data import llm as L
    from src.board.data import research as rd

    got = {}

    async def fake(question, data, *, today, client, forecast_kw):
        got.update(forecast_kw, question=question)
        return {"method": "m", "notes": []}

    real = rd.research.research, L.ready
    rd.research.research, L.ready = fake, lambda: None
    try:
        out, meta = rd.run("2026-10-31", {"id": "q", "title": "t", "buckets": []}, [])
    finally:
        rd.research.research, L.ready = real
    check(
        "the forecast told research's views and share left out",
        got["views"] == rd.VIEWS and got["drop"] == rd.DROP and got["question"]["id"] == "q" and meta["samples"] == 3,
    )

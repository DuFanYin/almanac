"""The system on one page: every stage of the day (pipeline.py), what it does, how often, in which modules, where the
LLM enters and what it must pass, what feeds what, and how each stage went today. The page's System tab draws it;
tests/board/test_board_layers.py keeps it whole (every step module and every step that may fail is in a stage, every
module named exists).

A stage: id, phase (pipeline.py's), side (data: computed from the data; llm: the LLM writes it; you: your settings
and choices; out: what you read), name, what, when, modules (under src/board/), checks (what the LLM's work must
pass), attempts (the keys under which pipeline.py's Day.attempt keeps its failures)."""

PHASES = [
    ("start", "Your choices"),
    ("data", "Data"),
    ("forecasts", "Forecasts"),
    ("futures", "Futures"),
    ("decisions", "Decisions"),
    ("publish", "What you read"),
]

STAGES = [
    {
        "id": "votes",
        "phase": "start",
        "side": "you",
        "name": "Your votes",
        "what": "Your adopt or reject on each proposal, taken in first, so what you adopted counts the same day.",
        "when": "daily",
        "modules": ["steps/proposals/votes.py"],
    },
    {
        "id": "definitions",
        "phase": "start",
        "side": "you",
        "name": "Definitions and your settings",
        "what": "The series, forecasts, indicators, factors and asset classes (config/), and yours: risk level, "
        "targets, rules (local/).",
        "when": "daily",
        "modules": ["core/store.py"],
    },
    {
        "id": "fetch",
        "phase": "data",
        "side": "data",
        "name": "Market and economic data",
        "what": "Every series from free sources; each value checked, and kept as it was first seen.",
        "when": "daily",
        "modules": ["data/observations.py", "data/sources.py", "data/kalshi.py"],
    },
    {
        "id": "readings",
        "phase": "data",
        "side": "llm",
        "name": "Texts read into numbers",
        "what": "The Fed's releases, news headlines and the largest AI investors' earnings releases, each read onto a "
        "scale from -2 to +2. Each indicator on them is given what the LLM expects it to say about the factors; the "
        "world uses that at half weight, sized as measured expectations typically are, until it can be measured.",
        "when": "daily, as texts are published; the expectations once, and when an indicator's definition changes",
        "checks": "Each text read three times; an answer counts only if what it quotes is in the text word for word; "
        "only texts from 30 October 2026 on. An expectation must name a factor, a way and days the board knows.",
        "modules": ["steps/readings.py", "data/texts.py", "data/llm.py"],
        "attempts": ["readings", "reading_expect"],
    },
    {
        "id": "signposts",
        "phase": "data",
        "side": "data",
        "name": "Indicators",
        "what": "Indicators computed from the series: each one green, yellow or red, and where it reads today against "
        "its own history (its dial). Each says what it means for the factors, as a mechanism (expect: which factor, "
        "which way or how widely, over which days).",
        "when": "daily",
        "modules": ["steps/signposts.py", "core/expr.py"],
    },
    {
        "id": "forecasts",
        "phase": "forecasts",
        "side": "data",
        "name": "Forecasts",
        "what": "Each forecast registered when due, Kalshi's market where there is one, and its outcome once known.",
        "when": "daily",
        "modules": ["steps/questions.py", "model/questions.py"],
    },
    {
        "id": "world",
        "phase": "forecasts",
        "side": "data",
        "name": "The world",
        "what": "A weight on every week of history since 1971: each indicator, read on its dial today, pushes the "
        "factors it is expected to move (or the spread of their moves) over the stretches where its measures agree, "
        "and history's periods are reweighted as little as that takes. How long each indicator's news lasts is "
        "measured every Saturday, a week to three years ahead.",
        "when": "daily; how long each indicator's news lasts measured on Saturdays",
        "checks": "Only expectations with a mechanism, and only where the measures go the way expected; weak "
        "evidence shrunk, indicators of one group averaged (one in the middle a vote for usual), every push capped, "
        "never on fewer than a quarter of history's weeks.",
        "modules": ["steps/world.py", "model/world.py", "model/timescale.py", "calibrate.py"],
        "attempts": ["world"],
    },
    {
        "id": "research",
        "phase": "forecasts",
        "side": "llm",
        "name": "Research",
        "what": "News searched and read into notes on the week's key forecasts (those furthest from usual), for the "
        "LLM that writes the scenarios.",
        "when": "every four weeks",
        "checks": "Every number in a note must be the cited news item's, and the item real; a failing note is dropped.",
        "modules": ["steps/research.py", "data/research.py"],
        "attempts": ["research"],
    },
    {
        "id": "scenarios",
        "phase": "futures",
        "side": "llm",
        "name": "Scenarios the LLMs write and read",
        "what": "The world's last layer. At a month, three months, a year and three years one LLM writes, from "
        "today's data and the simulation, the scenarios worth telling apart now, each in a direction of its own (a "
        "story with the ranges of one or two factors); a path may be in several or in none; no probabilities, no "
        "sides. Four readers (a macroeconomist, a market historian, an optimist, a pessimist) each read one "
        "scenario at a time and say how much likelier or less likely it is than the simulation's share, for what "
        "the simulation does not include; a reviewer that may ask the board's data checks their reasons, giving no "
        "numbers, and the readers of a scenario it found fault with read it again. Each one should have the "
        "simulation's share times the median factor: the paths are reweighed until each has it, the simulation "
        "moved least. A week and a forecast's days take the nearest length's.",
        "when": "weekly, revising the week before; again when a factor moves unusually; the coach every four weeks",
        "checks": "Each scenario's ranges on one to three known factors, low below high or open at one end; at most "
        "eight scenarios a length (six at a year and three years). Each scenario is scored when its length is over: "
        "whether it happened, against its probability, the simulation's share and history's. The scores go back to "
        "the LLMs through the coach; nothing changes a factor.",
        "modules": [
            "steps/scenarios/sets.py",
            "steps/scenarios/context.py",
            "steps/scenarios/write.py",
            "steps/scenarios/world.py",
            "steps/scenarios/page.py",
            "model/scenarios.py",
            "steps/llm_data.py",
            "data/llm.py",
        ],
        "attempts": ["scenarios"],
    },
    {
        "id": "final",
        "phase": "futures",
        "side": "data",
        "name": "Forecasts read off the world",
        "what": "Each forecast's outcome after every week of history, weighed as the world weighs it over the days it "
        "has left; pooled with the market; your own view on top; the same without the LLM kept beside.",
        "when": "daily",
        "modules": ["steps/probabilities.py", "model/prob.py"],
    },
    {
        "id": "distribution",
        "phase": "futures",
        "side": "data",
        "name": "Futures at each length",
        "what": "The world read at a week, a month, three months, a year and three years: each asset class's return "
        "from the factors on every week of history once, weighed as the world weighs it (nothing drawn at random); "
        "your targets, the base mix and 60/40 on it, on the final world, the simulation alone, without the LLM and on "
        "history; the bad-case limit of each length; each scenario as a view (how often history falls in it, its "
        "share of your worst 10%). Each factor's quartiles kept daily as the outlook, scored when the window is over.",
        "when": "daily",
        "modules": ["steps/distribution.py", "model/distribution.py", "model/response.py", "model/asset_classes.py"],
        "attempts": ["distribution"],
    },
    {
        "id": "you",
        "phase": "futures",
        "side": "you",
        "name": "You",
        "what": "Your risk level and its limits, your targets, and your holdings by asset class.",
        "when": "daily",
        "modules": ["steps/you.py", "steps/account.py", "data/account.py", "model/portfolio.py"],
    },
    {
        "id": "q_proposals",
        "phase": "futures",
        "side": "llm",
        "name": "Proposed forecasts",
        "what": "New forecasts the LLM suggests, told what is asked, the futures and where your bad case lies.",
        "when": "weekly",
        "checks": "History can answer it, it is new, and it changes a decision or resolves within three months.",
        "modules": ["steps/proposals/questions.py", "steps/proposals/common.py", "model/proposals.py"],
        "attempts": ["llm_propose"],
    },
    {
        "id": "sp_proposals",
        "phase": "futures",
        "side": "llm",
        "name": "Proposed indicators",
        "what": "New indicators the LLM suggests. Paused since the world was built: to come back drafting what each "
        "means for the factors, measured like the others and used at half its size at first.",
        "when": "paused",
        "checks": "Computable from the series kept, ten years of history, red on 2-50% of days, not a copy of one.",
        "modules": ["steps/proposals/signposts.py"],
    },
    {
        "id": "decide",
        "phase": "decisions",
        "side": "data",
        "name": "The decision",
        "what": "One step from the world to what to do: your targets leaned toward the classes today's world puts "
        "above usual and away from those it puts below, both ways alike, at a month, three months, a year and three "
        "years (a week is shown, not acted on), after the noise a reweighting that large makes by chance; within "
        "your risk level's lean and caps; then the guardrails (each length's bad case within its limit); then the "
        "trades on what you hold: past your band, or the decided share moved half a step, one step at a time.",
        "when": "daily",
        "modules": ["steps/decide.py", "model/decide.py", "model/allocation.py", "steps/actions.py"],
        "attempts": ["decide"],
    },
    {
        "id": "rules",
        "phase": "decisions",
        "side": "you",
        "name": "Your rules left",
        "what": "Bitcoin's regular purchase (doubled when it is cheap), its cap, its pause when the thesis fails, its "
        "review at the halving; each fires at most once a period. The rules that sold or bought on forecasts or "
        "falls retired into the decision and the indicators.",
        "when": "daily",
        "modules": ["steps/rules.py", "model/rules.py", "model/rule_history.py", "model/behaviour.py"],
    },
    {
        "id": "changes",
        "phase": "decisions",
        "side": "llm",
        "name": "Proposed factors",
        "what": "New factors for the futures.",
        "when": "every four weeks",
        "checks": "A factor must cover the factors' history, not copy one, and explain more of the asset classes' "
        "returns on held-out history.",
        "modules": ["steps/proposals/changes.py", "model/proposals.py", "model/describe.py"],
        "attempts": ["llm_changes"],
    },
    {
        "id": "backtest",
        "phase": "publish",
        "side": "data",
        "name": "Through history",
        "what": "The decision step replayed through history every four weeks since 1981 on the data's world as it "
        "could be built then (the indicators measured on the years before), against your targets rebalanced, your "
        "risk level's mix and 60/40. How strongly it leans on the data (its sensitivity) and how much each length "
        "counts are chosen on the first half and judged on the second.",
        "when": "the first Saturday of a month",
        "modules": ["calibrate.py", "model/walk.py", "model/backtest.py"],
    },
    {
        "id": "texts",
        "phase": "decisions",
        "side": "llm",
        "name": "Weekly brief and the case against",
        "what": "A short note on what changed, and the strongest case against your holdings.",
        "when": "Saturdays",
        "checks": "Every number must be a data item it cites, or it is written again.",
        "modules": ["steps/texts.py"],
        "attempts": ["llm_brief"],
    },
    {
        "id": "scores",
        "phase": "publish",
        "side": "data",
        "name": "Scores",
        "what": "Every source scored on every forecast that resolved, every outlook, and every scenario set whose "
        "length is over.",
        "when": "daily, as forecasts, outlooks and scenarios resolve",
        "modules": ["steps/questions.py", "steps/world.py", "steps/scenarios/score.py"],
    },
    {
        "id": "page",
        "phase": "publish",
        "side": "out",
        "name": "This page",
        "what": "Everything above, with its sources; your holdings only in the account part.",
        "when": "daily",
        "modules": ["view/__init__.py"],
    },
    {
        "id": "alerts",
        "phase": "publish",
        "side": "out",
        "name": "Notification",
        "what": "Rules that fired, indicators turned red, a headline that moved, the scenarios' news, and a suggestion "
        "new, changed or gone, through your notify command (local/settings.toml).",
        "when": "daily, when there is news",
        "modules": ["steps/alerts.py"],
    },
]

# what feeds what; an edge into an earlier phase is read the next day
EDGES = [
    ("votes", "definitions"),
    ("definitions", "fetch"),
    ("definitions", "forecasts"),
    ("definitions", "you"),
    ("fetch", "signposts"),
    ("readings", "signposts"),
    ("fetch", "forecasts"),
    ("signposts", "world"),
    ("fetch", "world"),
    ("world", "scenarios"),
    ("research", "scenarios"),
    ("fetch", "scenarios"),
    ("world", "final"),
    ("scenarios", "final"),
    ("forecasts", "final"),
    ("world", "distribution"),
    ("scenarios", "distribution"),
    ("you", "distribution"),
    ("distribution", "q_proposals"),
    ("scenarios", "q_proposals"),
    ("distribution", "decide"),
    ("you", "decide"),
    ("backtest", "decide"),
    ("signposts", "rules"),
    ("you", "rules"),
    ("world", "backtest"),
    ("you", "backtest"),
    ("final", "research"),
    ("final", "texts"),
    ("scenarios", "texts"),
    ("forecasts", "scores"),
    ("final", "scores"),
    ("distribution", "scores"),
    ("scores", "scenarios"),
    ("q_proposals", "votes"),
    ("changes", "votes"),
]


def notes(d):
    """{stage: what it made today, in a few words}, from the day's state (pipeline.py Day)."""
    out = {}
    if d.fetch:
        out["fetch"] = f"{len(d.raw_nodes)} series fetched"
    else:
        out["fetch"] = "not fetched in this run: the data already kept"
    states = [t.get("state") for t in d.sp_today.values()]
    out["signposts"] = f"{states.count('red')} red, {states.count('yellow')} yellow of {len(states)}"
    out["forecasts"] = f"{len(d.open_qs)} open, {len(d.resolved)} resolved today"
    if d.horizons:
        moved = ", ".join(f"{h['label']} {h.get('llm_moved', 0):.0%}" for h in d.horizons.values())
        out["distribution"] = f"the LLM's scenarios moved: {moved}"
    out["world"] = f"{sum(s['dial'] is not None for s in d.world['data'].signals)} indicators read today"
    out["rules"] = f"{len(d.triggers_new)} fired today"
    out["decide"] = (
        f"{sum(a['kind'] in ('lean', 'guard') for a in d.acts['public'])} on your targets, "
        f"{sum(a['kind'] == 'trade' for a in d.acts['account'])} trades on what you hold"
    )
    walked = d.walked or {}
    helped = "moving on the data helped" if walked.get("sense") else "moving on the data did not help"
    out["backtest"] = f"run {walked['day']}: {helped}" if walked.get("day") else "not run yet"
    out["votes"] = f"{len(d.adopted)} decided today" if d.adopted else "none to take in"
    one = lambda sql: d.db.execute(sql).fetchone()  # noqa: E731
    waiting = dict(
        d.db.execute(
            "SELECT coalesce(json_extract(spec, '$.type'), 'question'), count(*) FROM proposals WHERE status = 'open' "
            "GROUP BY 1"
        ).fetchall()
    )
    out["q_proposals"] = f"{waiting.get('question', 0)} waiting for you"
    out["changes"] = f"{waiting.get('factor', 0)} waiting"
    trial = one("SELECT count(*) FROM signpost_proposals WHERE status = 'trial'")[0]
    out["sp_proposals"] = f"{trial} on trial"
    out["scores"] = f"{one('SELECT count(*) FROM resolutions WHERE bucket IS NOT NULL')[0]} forecasts resolved"
    last = {
        "research": one("SELECT date(max(created_at), 'unixepoch') FROM artifact WHERE kind = 'research'")[0],
        "texts": one("SELECT max(day) FROM texts WHERE kind = 'brief'")[0],
        "readings": one("SELECT max(day) FROM readings")[0],
    }
    for k, day in last.items():
        if day:
            out[k] = f"last {day}"
    n, day = one("SELECT count(*), max(written) FROM scenario_sets WHERE status = 'live'")
    out["scenarios"] = f"{len(d.scenarios_page)} scenarios in {n} lengths, written {day}" if day else "none written yet"
    return out


def status(d):
    """The system for the page: {"phases", "stages" (each with "state": ran, failed, off (the LLM not asked in this
    run), "error", "note"), "edges"}."""
    said = notes(d)
    stages = []
    for s in STAGES:
        failed = [d.errors[k] for k in s.get("attempts", ()) if k in d.errors]
        state = "failed" if failed else "off" if s["side"] == "llm" and not d.llm else "ran"
        stages.append({**s, "state": state, "error": "; ".join(failed) or None, "note": said.get(s["id"])})
    if "fetch" in said and d.fetch:  # a series that could not be fetched is kept under its own id
        known = {k for s in STAGES for k in s.get("attempts", ())}
        bad = sorted(k for k in d.errors if k not in known)
        if bad:
            fetch = next(s for s in stages if s["id"] == "fetch")
            fetch["error"] = f"not fetched: {', '.join(bad)}"
    return {"phases": [{"id": p, "label": label} for p, label in PHASES], "stages": stages, "edges": EDGES}

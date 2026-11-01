"""The day, in order. The data layer brings the values in (data/), the model layer computes (model/, pure), and
each step (steps/) keeps its results in the database with what they were made from (core/lineage.py); view/
writes the page's data. This file only orders the steps, in phases that share the day's state (Day), and catches
what may fail without stopping the day (attempt); run.py is its command line."""

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from . import calibrate, flow, view
from .core import lineage, store
from .data import kalshi, llm, observations
from .data import research as research_data
from .model import prob
from .model import rules as rules_m
from .steps import (
    account,
    alerts,
    decide,
    llm_data,
    probabilities,
    questions,
    readings,
    research,
    rules,
    scenarios,
    signposts,
    texts,
    world,
    you,
)
from .steps import actions as acting
from .steps import distribution as distributions
from .steps.proposals import changes as change_proposals
from .steps.proposals import questions as question_proposals
from .steps.proposals import signposts as signpost_proposals
from .steps.proposals import votes

log = logging.getLogger("board")
NY = ZoneInfo("America/New_York")


def ny_today():
    return datetime.now(NY).date()


@dataclass(slots=True)
class Day:
    """The day's state, filled in phase by phase: what one phase leaves, the next reads (each field under the phase that
    sets it). Until its phase has run a field is None."""

    # given: the database, the day, whether the LLM is asked, whether the data is fetched, the LLM steps asked now
    # whatever their cadence (scenarios, brief, research, propose, signposts, changes), and {step: error} of the steps
    # that failed, for the page
    db: Any
    today: date
    llm: bool
    fetch: bool
    force: frozenset
    errors: dict
    # start
    adopted: Any = None  # what the owner (then the LLM) adopted today from the proposals
    cfg: dict | None = None  # the config, with what was adopted
    chash: str | None = None  # its hash
    sps: list | None = None  # the signposts
    # data
    raw_nodes: dict | None = None  # each series fetched today: its lineage node
    df: Any = None  # the point-in-time frame
    series: dict | None = None  # each series in full
    evaluated: dict | None = None  # each signpost's values
    sp_errors: dict | None = None  # the signposts that could not be read
    sp_today: dict | None = None  # each signpost's state today
    # forecasts
    open_qs: list | None = None  # the open questions
    world: Any = None  # the data's world (steps/world.py build)
    world_node: int | None = None  # its lineage node
    data: list | None = None  # the numbered data the LLMs are given (steps/llm_data.py)
    resolved: list | None = None  # the questions resolved today
    # futures
    scenarios_today: dict | None = None  # each live scenario set on today's simulation (steps/scenarios for_world)
    scenario_news: list | None = None  # the scenarios' news today, for the message
    live: list | None = None  # the live scenarios, flat
    probs: dict | None = None  # each open question's probabilities
    history: dict | None = None  # their history, for the page
    held: Any = None  # what is held today
    me: dict | None = None  # you: your style, policy, rules and holdings (steps/you.py)
    scenarios_page: list | None = None  # the live scenarios as the page shows them
    horizons: dict | None = None  # the futures at each length (steps/distribution.py)
    # decisions
    lines: dict | None = None  # the rules' lines in history
    behaved: dict | None = None  # the owner's marks on the rules
    rule_rows: list | None = None  # each rule today
    triggers_new: list | None = None  # the rules that fired today
    walked: dict | None = None  # the decision step walked through history (calibrate.py)
    decided: dict | None = None  # the decision step (steps/decide.py)
    acts: dict | None = None  # the actions: public and on what you hold
    acc: dict | None = None  # the page's account part

    def attempt(self, key, fn, *args, default=None, **kwargs):
        """fn(*args, **kwargs); a step that may fail without stopping the day: its error is kept for the page and
        `default` stands in for its result."""
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            self.errors[key] = f"{type(e).__name__}: {e}"[:300]
            log.warning("%s: %s", key, self.errors[key])
            return default


def start(d):
    """The owner's votes (before the config: what was adopted counts today), the config, the proposed signposts
    joined, the versions of the rules and the LLM's methods."""
    d.adopted = votes.adopt(d.db, d.today, d.llm)  # yours first; the LLM's on what you left
    d.cfg, d.chash = store.config()
    readings.register_signposts(d.db, d.cfg, d.today)  # the signposts on the texts the LLM reads, accumulating at first
    if d.llm:  # what the LLM expects each of them to say, until they can be measured
        d.attempt("reading_expect", readings.expectations, d.db, d.cfg, d.today)
    signpost_proposals.with_candidates(d.db, d.cfg)  # those and the LLM's, accumulating, on trial, active
    log.info("start: config %s, code %s%s", d.chash, lineage.CODE, f", adopted {d.adopted}" if d.adopted else "")
    d.sps = d.cfg["signposts"]["signpost"]
    lineage.version(d.db, "rules", rules_m.version(d.cfg), d.today)
    lineage.version(d.db, "scenarios", llm.method(), d.today)
    lineage.version(d.db, "research", research_data.method(), d.today)


def data(d):
    """The data, the texts read into numbers, and the signposts on them."""
    db, cfg, today = d.db, d.cfg, d.today
    d.raw_nodes, d.errors = observations.fetch(db, cfg["series"]) if d.fetch else ({}, {})
    if d.llm:  # the texts published since the last run, read into numbers (from 2026-10-30 on, never history)
        d.attempt("readings", readings.run, db, cfg, today)
    d.df, d.series, _ = observations.load(db, cfg["series"], today)
    d.evaluated, d.sp_errors = prob.evaluate(d.sps, d.df)
    d.sp_today = signposts.signposts_today(db, d.sps, d.evaluated, today, cfg["series"], d.series)
    db.commit()


def forecasts(d):
    """The questions registered, their markets; the world (each signpost on its dial, how long each one's news
    lasts); the numbered data the LLM reads; research's notes from the news when due; what resolved."""
    db, cfg, today = d.db, d.cfg, d.today
    questions.register(db, cfg, d.chash, d.series, today)
    d.open_qs = [q for q in questions.open_questions(db, cfg["questions"]) if q["kind"] != "factor"]
    kalshi_results = questions.kalshi_markets(db, d.open_qs, d.fetch)
    d.world = d.attempt("world", world.build, cfg, d.series, d.evaluated, today)
    if d.world is None:  # it failed: history as it was, so what reads it still runs
        d.world = world.build(cfg, d.series, {}, today, {})
    d.world_node = world.record(db, d.world, d.sp_today, today)
    d.data = llm_data.snapshot_data(d.df, d.sp_today, d.sps, cfg["series"])
    if d.llm:
        llm_data.record_snapshot(db, d.data, today)
        key = research.key_now(db, d.open_qs)  # the questions furthest from usual on the last day recorded
        d.attempt("research", research.run_research, db, d.open_qs, d.data, today, key, force="research" in d.force)
    d.resolved = questions.resolve(db, d.open_qs, d.series, kalshi_results, today)


def futures(d):
    """The world's last layer: the LLM's scenarios at each length (written weekly, again when something happens;
    scored once over), on today's simulation; the questions read off the final world; you; the futures at each length
    and their limits; the outlook kept and resolved; new questions proposed."""
    db, cfg, today = d.db, d.cfg, d.today
    scenarios.scored(db, d.series, today)  # those past their end, once their factors have values
    d.scenarios_today = scenarios.for_world(d.world["data"], scenarios.live(db))
    happened = scenarios.events(cfg["factors"], d.series, today)  # news the day it happens
    d.scenario_news = [f"Unusual move: {x}" for x in happened]
    if d.llm:
        # research's notes on the week's key questions, for the writer
        notes = [n for q in research.key_now(db, d.open_qs) for n in llm_data.notes(db, q, 1)[0]]

        def written():
            _, news = scenarios.write(
                db,
                cfg,
                d.world["data"],
                d.data,
                notes,
                today,
                d.series,
                force="scenarios" in d.force,
                happened=happened,
            )
            d.scenario_news += news
            d.scenarios_today = scenarios.for_world(d.world["data"], scenarios.live(db))

        d.attempt("scenarios", written)

    world.resolve_outlook(db, cfg["factors"], d.series, today)
    d.live = d.scenarios_today["live"]
    d.probs, d.history = probabilities.read(
        db, cfg, d.chash, d.open_qs, d.series, d.world, d.scenarios_today, today, d.world_node
    )
    d.held = account.holdings_today(db, cfg, d.df)
    d.me = you.you(db, cfg, d.held)
    d.scenarios_page = scenarios.for_page(db, d.scenarios_today, today)
    d.horizons = d.attempt(
        "distribution",
        distributions.build,
        *(db, cfg, d.series, d.df, d.world, d.scenarios_today, d.me, d.me["now"], today, d.world_node),
        default={},
    )
    world.outlook(db, d.horizons, today)
    if d.llm:  # new questions, for the owner to adopt or not
        d.attempt(
            "llm_propose",
            question_proposals.run,
            *(db, cfg, d.me, d.horizons, d.scenarios_page, d.history, d.series, d.df, d.data, today),
            force="propose" in d.force,
        )


def decisions(d):
    """What to do: one decision step (your targets leaned by the world within the guardrails, the trades on what you
    hold), your rules left beside it, new factors (every four weeks), the actions, the account's view; the brief and
    the red team on Saturdays."""
    db, cfg, today, held, me = d.db, d.cfg, d.today, d.held, d.me
    d.lines = rules.rule_lines(cfg, d.df)
    d.behaved = rules.marks(db, cfg)
    names = account.names_today(db, cfg, held, d.df, cfg["factors"]["horizon"]["long"]["days"], today)
    d.rule_rows, d.triggers_new, standing = rules.rules_today(db, cfg, d.chash, d.df, d.sp_today, held, today)
    if d.llm:
        force = "changes" in d.force
        d.attempt("llm_changes", change_proposals.run, db, cfg, d.df, d.series, d.data, today, force=force)
    d.walked = store.read_json(store.DATA / calibrate.WALK_FILE)
    d.decided = d.attempt("decide", decide.run, db, cfg, me, d.horizons, held, d.live, today, d.world)
    d.acts = acting.actions(db, cfg, me, d.horizons, d.rule_rows, held, d.decided, names)
    decision = (d.decided or {}).get("decision")
    d.acc = (
        account.account_view(cfg, held, standing, today, me, d.horizons, decision, d.acts["account"]) if held else None
    )
    if d.acc:
        d.acc["actions"], d.acc["you_now"] = d.acts["account"], me["now"]
        d.acc["names"] = [{k: v for k, v in n.items() if k != "node"} for n in names]
    # each scenario as a view: what it rests on, what it means for your targets, what speaks for and against it,
    # and the actions it drives (for the brief, the red team and the page)
    read_now = scenarios.readings_now(d.series, cfg.get("readings"), cfg["series"], today)
    scenarios.explained(d.scenarios_page, d.horizons, read_now, d.acts["public"])

    if d.llm and ("brief" in d.force or today.weekday() == 5):

        def written():
            labels = {h: x["label"] for h, x in cfg["factors"]["horizon"].items()}
            brief_data = texts.brief(db, today, d.data, d.probs, d.scenarios_page, labels, cfg)
            if d.acc and d.acc["against"]:
                texts.redteam(db, today, brief_data, d.acc, d.scenarios_page)

        d.attempt("llm_brief", written)
    db.commit()


def publish(d):
    """The page's data, the day's news, and on Saturdays how long each signpost's news lasts measured again (for
    next week's world); on the first Saturday of a month the decision step walked through history again (its sense
    and lengths' weights for the month)."""
    db, cfg, today = d.db, d.cfg, d.today
    try:
        calendar = [{"day": x, "what": w} for x, w in kalshi.calendar() if x >= today.isoformat()][:12]
    except Exception:
        calendar = []
    before = (store.read_json(store.VIEW) or {}).get("actions")  # the last run's, to tell what changed
    before_acc = (store.read_json(store.ACCOUNT_VIEW) or {}).get("actions") if d.acc else None
    view.write(
        db,
        cfg,
        today,
        d.df,
        chash=d.chash,
        sp_today=d.sp_today,
        sp_errors=d.sp_errors,
        probs=d.probs,
        rule_rows=d.rule_rows,
        account=d.acc,
        errors=d.errors,
        score=questions.scores(db),
        resolved=d.resolved,
        calendar=calendar,
        lines=d.lines,
        behaved=d.behaved,
        proposed=votes.rows(db),
        you={k: v for k, v in d.me.items() if k != "now"},
        actions=d.acts["public"],
        backtest=d.walked,
        horizons=d.horizons,
        scenarios=d.scenarios_page,
        scenario_sets=scenarios.notes_for_page(scenarios.live(db)),
        decision=(d.decided or {}).get("decision"),
        signposts_proposed=signpost_proposals.rows(db),
        readings=readings.rows(db),
        system=flow.status(d),
        world=d.world,
    )
    news = alerts.alerts(db, cfg, d.sps, d.sp_today, d.probs, d.triggers_new, today, d.scenario_news)
    if d.decided:  # without the decision today the suggestions are unknown, not gone
        # the message has one reader, the owner: what they hold when it is known, the public page's portfolio if not
        if before_acc is not None:
            news += alerts.suggestions(before_acc, d.acts["account"])
        elif before is not None and not d.acc:
            news += alerts.suggestions(before, d.acts["public"])
    if news:
        alerts.notify("Scenarios\n" + "\n".join(news))
    if today.weekday() == 5:
        try:
            calibrate.timescales(today)
        except Exception as e:
            log.warning("timescales: %s: %s", type(e).__name__, e)
        if today.day <= 7:  # the first Saturday of a month: the decision step through history again
            try:
                calibrate.walk(today)
            except Exception as e:
                log.warning("walk: %s: %s", type(e).__name__, e)


PHASES = (start, data, forecasts, futures, decisions, publish)


def day(use_llm=True, fetch=True, force=()):
    """The day: each phase in turn on one Day. force: the LLM steps to ask now (Day)."""
    d = Day(db=store.connect(), today=ny_today(), llm=use_llm, fetch=fetch, force=frozenset(force), errors={})
    for phase in PHASES:
        phase(d)
    log.info(
        "day %s: %d series fetched, %d errors, %d questions open, %d new triggers",
        d.today,
        len(d.raw_nodes),
        len(d.errors),
        len(d.open_qs),
        len(d.triggers_new),
    )
    d.db.close()

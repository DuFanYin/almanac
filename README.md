# Almanac

[中文](README.zh-CN.md)

A personal investment system that keeps apart three things that usually blur together: what may happen, what you prefer, and what to do.

- **What may happen** is one set of probabilities, the *world*: a weight on every week of history since 1971, saying how much the days that followed it look like the days ahead. Today's indicators move it first; then scenarios written by LLMs, each given a probability, move it again.
- **What you prefer** (a risk level, targets, limits on the bad case) never changes a probability.
- **What to do** comes from one mechanical decision step that puts the two together, within your limits, and turns it into trades on what you hold.

Every forecast and every scenario resolves from data and is scored. Beside every figure the same figure without the LLM is kept. Nothing places orders: it suggests.

## Words it uses

| Word | Meaning |
|---|---|
| Indicator | a daily reading computed from the data (the VIX, credit spreads, the Fed's moves, valuations, a text the LLM reads), green, yellow or red, with what it is expected to say about the factors. The code calls them signposts |
| Factor | one of seven things that describe what happens: the S&P 500's return, the 10-year yield's and the real yield's change, inflation's yearly rate, unemployment's change, the dollar's return, financial conditions (NFCI) |
| Length | how far ahead: a week, a month, three months, a year, three years |
| The futures | every period of that length since 1971, each a possible future, with its weight |
| The simulation | the futures weighted by today's indicators alone |
| Scenario | a story an LLM writes about one length: the ranges of one or two factors. A period of history can be in several scenarios, or in none |
| The world | the simulation with the scenarios' probabilities put in: the one belief every forecast and decision reads |
| Usual | history's own periods, unweighted |
| Bad case | the average of the worst 10% of futures for your targets |

## How it works

| Part | What it does | Why it is kept separate |
|---|---|---|
| Indicators | about thirty, each with what it should say about the factors (which way, over which days, why). Each is measured on history, a week to three years ahead, and used only the way expected, shrunk by its own noise; how long its news lasts is measured, not set | evidence has its own timescale: extreme fear says something about the next weeks, a tightening Fed about the next years |
| The simulation | each indicator, read against its own history today, pushes its factors over the lengths it covers; history's weeks are reweighted as little as that takes, never onto less than a quarter of them | one set of weights, read at any length, is the only belief |
| Scenarios | weekly, at a month, three months, a year and three years, one LLM writes the scenarios worth telling apart. Four LLM readers (a macroeconomist, a market historian, an optimist, a pessimist) each read one scenario and say how many times as likely it is than the simulation says, for what the simulation leaves out. A reviewer checks their reasons against the data. Each scenario gets the simulation's probability times the readers' median factor; history's weeks are reweighted until it does | history has few separate three-year periods and none of what has not happened yet; the LLM brings that in, on purpose |
| Asset classes | how each class follows the factors, fitted per length on every window since 1971 (total returns), what the fit leaves unexplained kept window by window, so classes that fell together fall together; cash and Treasuries from today's yield | the same outcome pays differently from a 5% yield than from a 1% one |
| Preferences | five risk levels (a bad-case limit, a cash floor, a cap per class, how far a class may move, a rebalancing band, a step), your targets and caps, a cap and a regular purchase for Bitcoin | the same beliefs lead a cautious and a bold investor to different mixes, never to different probabilities |
| The decision step | at each length, each class's expected return in today's world against usual, less what chance alone would give, in two parts (the data's and the LLM's), each times its sensitivity; your targets moved by that, within the limit per class; then the guardrails (each length's bad case within its limit); then trades on what you hold, past the band or half a step, one step at a time | one path from the world to a trade: the same news acts once, good news acts as bad news does, and the step can be replayed through history |
| Forecasts | questions that resolve from data (the S&P 500 over two years, a 2027 recession, the Fed's decisions, CPI prints, fund ranges), each read off the world and pooled with Kalshi's prices where there are any | they score the world and show it; they drive nothing |

## How the LLM is used

It is brought in on purpose, for what history and the data cannot give. You set the preferences; resolution and arithmetic are mechanical.

- **Several LLMs, one job each.** The writer, the four readers, the reviewer (it gives no numbers), and a coach that reads the scores for every role. Scenarios are revised weekly, and again when a factor moves unusually. Whether a scenario is good or bad for your targets is computed, never asked.
- **Its factors go in as given.** They are not shrunk toward 1. Each set is scored when its length is over, and the score is told to the LLMs the next time.
- **It acts both ways.** Its part of a class's gap from usual moves your targets by your sensitivity for it (1.0 by default); the guardrails still win.
- **A control without it.** Every figure is also computed without anything the LLM made, and shown beside.
- **It reads texts into numbers.** The Fed's statements (how hawkish), the news and the largest AI investors' earnings releases on AI spending, the day's headlines (what mood): each text read three times onto a scale, an answer kept only if what it quotes is in the text word for word. Only texts from 30 October 2026 on, since the model knows how earlier ones turned out.
- **It proposes.** Up to three forecasts a week and two factors every four weeks, each checked on the data first. One you leave for three days the LLM decides on (asked three times, a majority adopts; at most two forecasts and one factor a month); you can withdraw anything adopted.
- **It writes.** A weekly brief and a case against your holdings, where every number must be one it was given; research notes on the week's key forecasts, every four weeks, where a point whose number is not its source's is dropped.
- **It never** sets a threshold, a step, a limit or a sensitivity; places or sizes a trade; resolves an outcome; scores itself; or changes a registered question.

## Checks before any outcome arrives

| Check | What it catches |
|---|---|
| each class's fit to the factors, out of sample on held-out stretches of history | a fit that only describes the past |
| each indicator against what it is expected to say, length by length, with a false discovery control | a signal that looks predictive by luck, or says the opposite of its mechanism (not used) |
| the decision step replayed every four weeks since 1981 on the world as it could be built then, its sensitivity chosen on the first half and judged on the second, against your targets rebalanced, your risk level's mix and 60/40 | a decision that adds nothing over holding the mix |
| each class's gap from usual at each length, beside what chance alone gives | a move that a different draw of history would reverse |
| each rule's line moved either way in history | a threshold whose exact place is luck |
| every version of the rules, of the scenario writing and of the proposals counted | a good record bought by many tries |

## Status and limits

- **No live scores yet.** The first forecasts resolve from late November 2026; the first month's scenarios a month after they were written.
- **The replay found no gain in moving on the data.** From 1981, no sensitivity added 0.2 points a year on the first half without a deeper fall, so the data's part is set to 0 and the guardrails and the band act alone. It is replayed on the first Saturday of every month. The LLM's scenarios and market prices cannot be rebuilt for the past, so the LLM's part is untested.
- **The guardrails cost.** Through history the bad-case limits cost about 0.7 points a year and did not make the 2000–02 fall shallower; they are a preference, kept as set.
- **Thin history at long lengths.** About 54 separate one-year periods since 1971, and 17 three-year ones.
- **What the factors miss.** Gold and Bitcoin are mostly unexplained by them; Bitcoin keeps its own history and has no indicator the world reads it on.
- **One model family** (DeepSeek); variety comes from viewpoints, wordings and data left out.
- **Not investment advice.**

## Quick start

```bash
uv sync --all-groups                       # Python and the dependencies (pyproject.toml, uv.lock)
cp .env.example .env && chmod 600 .env     # the model provider and its key
uv run python -m src.board.run --no-llm    # one day of the board, without LLM calls
uv run python -m src.board.run             # the whole day, the LLMs included
uv run python -m src.web                   # the page: http://127.0.0.1:8083
uv run pytest                              # the tests; none calls a model or the network
```

The other commands (the weekly brief, research, scenarios or proposals on demand, the replay through history, a number's lineage) are listed by `python -m src.board.run --help`.

To run it every day, schedule `python -m src.board.run` after the US close (with a systemd timer or cron), after whatever refreshes your positions file, and keep `python -m src.web` running. The page has no sign-in of its own: open it on the machine, or put it behind a server that signs you in.

## Your own setup

Everything about you lives in `local/`, which git ignores. Without it the board runs on its defaults.

| File | What |
|---|---|
| `local/rules.toml` | your risk level (1 to 5), your categories and their targets, each symbol's category, Bitcoin's cap and regular purchase, your rules. Start from `src/board/config/rules.example.toml` |
| `local/settings.toml` | `positions`: a JSON file of what you hold, kept in sync with your brokers by another program (its shape is in `src/board/data/account.py`; without it there is no account part); `notify`: a command that takes one message as its argument; `research_model`; `[web]`: links for the page's header, none by default |
| `local/questions.toml`, `local/factors.toml` | the forecasts and factors adopted from the LLM's proposals, written by the board |

Also out of git: `.env` (the model provider and its key) and `data/` (what the board makes, and your inputs from the page).

## Layout

Three layers in one direction: the model layer, research, the board; the page on top. Each imports only those below it (`tests/test_layers.py`).

| Path | What |
|---|---|
| `src/llm/` | the model layer: provider settings, one client with a model chain, call records and costs, and the methods built to make answers checkable: `propose`, `forecast`, `scenarios`, `write` (text whose every number must be a data item it cites), `indicators`, `readings` (a text read onto a scale); `forecast`, `write` and `indicators` also as a command line, JSON in and out (`python -m src.llm.methods`) |
| `src/research/` | research, a library: news searched for one question, read into checked notes, then the forecast method on them |
| `src/board/` | the board: `core/` (database, lineage, the indicators' expression language), `data/` (sources, checks, the account's files, the LLM's texts), `model/` (pure computations), `steps/` (one module per step of the day), `pipeline.py` (their order), `calibrate.py` (how long each indicator's news lasts, and the replay through history), `view/` (the page's data), `config/` (series, indicators, factors, asset classes, forecasts) |
| `src/web/` | the page: `app.py` (routes, API, assets), `static/` (the page, its scripts and styles), `design/` (shared styles, chart helper, fonts, icon) |
| `tests/` | one folder per layer, and `test_layers.py` |

## License

MIT: see [LICENSE](LICENSE).

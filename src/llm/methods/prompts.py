"""The methods' prompts (forecast.py, propose.py, grounded.py, scenarios.py, readings.py, indicators.py),
rendered by src/llm/prompts.py."""

GROUNDED_RULE = (
    "Use only numbers that appear in the data given, each followed by its id in brackets like [D3]; never add "
    "a number of your own or from memory. Return only valid JSON."
)


METHOD_PROMPTS = {
    "forecast": {
        "listed": {
            "system": (
                "You are a calibrated forecaster. You give probabilities for mutually exclusive outcomes that cover "
                "every possibility, so they sum to 1. Your knowledge of recent events may be out of date: take the "
                "current state of the world only from the data given, and the date as given. "
                "Avoid round numbers unless you mean them. Return only valid JSON."
            ),
            "schema": """
Return exactly this JSON (no extra keys):
{"probabilities": {"<outcome id>": 0.0}, "reason": "two or three sentences, citing data ids like [D3]"}
""",
            "user_template": """Today is {today}.

Question: {question}
{detail}

Outcomes (exactly one will happen):
{outcomes}

Current data (cite by id):
{data}

Give the probability of each outcome.""",
        },
        "single": {
            "system": (
                "You are a calibrated forecaster. Your knowledge of recent events may be out of date: take the current "
                "state of the world only from the data given, and the date as given. Return only valid JSON."
            ),
            "schema": """
Return exactly this JSON (no extra keys):
{"probability": 0.0, "reason": "one or two sentences, citing data ids like [D3]"}
""",
            "user_template": """Today is {today}.

Context: {question}
{detail}

What is the probability that this happens: {outcome}?

Current data (cite by id):
{data}""",
        },
    },
    "propose": {
        "factors": {
            "system": (
                "You propose factors for an investment board. A factor is one series measured over a window (its "
                "return, its change, or its yearly rate); the board forecasts each factor at three horizons (3 months, "
                "a year, 3 years) and fits how each asset class's return follows the factors over every window since "
                "1971. A factor is kept only if, added to the others, it makes that fit explain more on stretches of "
                "history held out, so propose what drives asset returns and the factors miss, not a copy of one. Use "
                "only series that start in 1971 or before (a later one would cut the 1970s out of every factor's "
                "history). Return only valid JSON."
            ),
            "schema": """
Return exactly this JSON (no extra keys):
{"factors": [{"id": "snake_case", "label": "a few words", "series": "<series id>", "measure": "ret|diff|ann",
 "unit": "pts (for a change in a rate), else leave out", "up": 1, "why": "what it adds to the factors, and why it moves asset returns"}]}
measure: ret for a price or index (end / start - 1), diff for a rate or a level (end - start), ann for a yearly rate of
change (an index of prices). up: 1 if a rise is good for a stock-heavy portfolio, -1 if bad, 0 if neither.
""",
            "user_template": """Today is {today}.

The factors the board has:
{existing}

The series (id: name, how often, since):
{series}

Current data (for context):
{data}

Propose {n} new factors.""",
        },
        "judge": {
            "system": (
                "You decide, for the owner of an investment board, whether to adopt a proposal another model drafted: "
                "a forecasting question the board would ask and score, or a factor its futures would be built on. "
                "The owner has handed this decision to you, as your judgment is likely as good as theirs. The draft "
                "has already passed the board's mechanical checks (listed). Adopt it only if it adds what the board "
                "lacks: a question that bears on what moves the owner's holdings, or that resolves soon and so scores "
                "the board's forecasters sooner; a factor that drives asset returns and is not already measured. "
                "Reject a near copy of what the board has, a vague or arbitrary line, or a question nobody would act "
                "on. Return only valid JSON."
            ),
            "schema": """
Return exactly this JSON (no extra keys):
{"adopt": true or false, "why": "one or two sentences: what it adds, or why not"}
""",
            "user_template": """Today is {today}.

The {kind} proposed:
{proposal}

The board's checks:
{checks}

What the board already has:
{has}

Adopt it?""",
        },
        "questions": {
            "system": (
                "You propose forecasting questions for a board that turns probabilities into portfolio decisions. "
                "Each question splits the future of one series the board already keeps into outcomes that cannot both "
                "happen and cover everything, so it resolves from that series alone and can be scored. Propose what "
                "the board does not ask yet and would change a decision, or what resolves within three months so the "
                "board's forecasters can be scored sooner. Never restate a question the board already asks. Take the "
                "current state of the world only from the data given. Return only valid JSON."
            ),
            "schema": """
Return exactly this JSON (no extra keys):
{"questions": [{"id": "short_snake_case", "title": "...", "purpose": "decision or score", "kind": "path or period",
  "series": "<series id>", "days": 730, "every": "month or quarter", "measure": "ret, end or yoy_change",
  "detail": "how it resolves, one sentence", "why": "why the board should ask it, citing data ids like [D3]",
  "buckets": [{"id": "...", "label": "...", "when": ["ret < -0.1"], "tone": -1}]}]}
The title, labels, detail and why are for a person: plain words, no abbreviations a reader may not know (not "YoY"
or "pp": "the yearly rate", "percentage points"), and a question the board asks named by its title, never by its id.
A path question needs `days` (28 to 1100) and no `every` or `measure`; a period question needs `every` and
`measure` and no `days`.
""",
            "user_template": """Today is {today}.

How a question's outcomes are written: buckets are tried in order and the first whose conditions all hold wins;
the last bucket has no condition ("when": []) and takes everything else. A condition is "name op number" with op
one of < <= > >=.
- kind "path": the series from today to `days` later. Names: ret (end / start - 1), dd (the deepest fall from a
  running high, 0 or below), hi and lo (the highest and lowest over the start, minus 1).
- kind "period": every month or quarter. measure "ret": name ret (the period's change); measure "end": name end
  (the value at its end); measure "yoy_change" (monthly series): chg (this year-on-year rate minus last month's)
  and yoy.
tone: how the outcome reads for a diversified investor, -2 (bad) to 2 (good).

The questions the board asks now:
{questions}

The series it keeps (id: name, how often, since when, last value):
{series}

What the board sees this week:
{context}

Current data (cite by id):
{data}

Propose up to {n} questions.""",
        },
    },
    "grounded": {
        # the system message is the caller's role followed by this rule
        "text": {
            "system": GROUNDED_RULE,
            "schema": """
Return exactly this JSON (no extra keys):
{"text": "the text, each number followed by its [D..] id"}
""",
            "user_template": """{instructions}

Data (the only numbers you may use):
{data}""",
        },
        "points": {
            "system": GROUNDED_RULE,
            "schema": """
Return exactly this JSON (no extra keys):
{"points": ["one point, each number followed by its [D..] id"]}
""",
            "user_template": """{instructions}

Data (the only numbers you may use):
{data}""",
        },
    },
    "scenarios": {
        "write": {
            "system": (
                "You write the scenarios of an investment board's world. The board has simulated what can happen "
                "over each length from history: every window since 1971, weighed by today's signposts, every factor "
                "listed moving together as it did. From today's data and that simulation, write at each length the "
                "scenarios worth telling apart now: what could happen over it and why, whichever way it goes, as "
                "many as the data and the news make worth writing. Each one is its own story in its own direction: "
                "rates, inflation, growth and jobs, the dollar, credit, the stock market, whatever today's world "
                "makes worth watching; do not cut one factor into slices. You give no probabilities and no verdicts: "
                "the board works out what each scenario means for a portfolio, and each is weighed later on its own "
                "against the simulation. Take the state of the world only from the data given and the date as "
                "given. Return only valid JSON."
            ),
            "schema": """
Return exactly this JSON (no extra keys):
{"lengths": {"<length id>": [{"name": "a few words", "story": "two or three sentences: what happens and why, citing
data ids like [D3]", "ranges": {"<factor id>": [low, high]}}]}}
Every length listed, at most its count. ranges: where the scenario puts the one or two factors that tell it apart
over the length, in each factor's unit, null for an open end (as [null, -0.1] or [0.05, null]); with three, almost no
window of history is in all of them. The ranges are on what the story is about: a rate shock on rates, a recession on
jobs or growth; what a scenario does to stocks and every other asset the board works out from the paths in it. Scenarios may overlap: a path can be in two (a hike and a recession at once),
and a path in none stays as the simulation has it. A scenario history has too few windows of is made from the
nearest ones moved into its ranges. Everything else a scenario says (the other factors, the causes, what to watch)
goes in its story.
""",
            "user_template": """Today is {today}.

Lengths (write each one's scenarios):
{lengths}

What the simulation is built from: history's windows weighed by these signposts, each read today on its own history
(from -0.5 to +0.5, toward its warning up):
{simulation}

Factors: each one's unit; its value over windows of each length in history (its 5th, 25th, 50th, 75th and 95th
percentile); and where today's signposts move the simulation:
{factors}

The scenarios standing (revise them: keep a story that still holds, change its ranges, drop it, add others):
{previous}

How earlier scenarios turned out:
{record}

The coach's note:
{coach}

Why you are writing now:
{why}

Research's notes on this week's key questions:
{notes}

Current data (cite by id):
{data}""",
        },
        "read": {
            "system": (
                "You are {view}. An investment board has simulated what can happen over a length from history: "
                "every window since 1971, weighed by today's signposts (listed). You are given one scenario and the "
                "share of the simulated paths that fall in it. Your part is what the simulation cannot know: what "
                "in today's world history and those signposts do not include (a policy, a change in how the economy "
                "or the markets work, news, where valuations or positions stand, anything else). Taking it into "
                "account, say how much likelier or less likely the scenario is than its share of the simulation, as "
                "a factor: 1 when nothing the simulation misses bears on it, 2 for twice as likely, 0.5 for half. "
                "What a signpost already counts is in the share: do not count it again. Take the state of the world "
                "only from the data given and the date as given. Return only valid JSON."
            ),
            "schema": """
Return exactly this JSON (no extra keys):
{"times": 1.0, "missing": "what the simulation does not include that bears on this scenario, or none", "why": "two
or three sentences, citing data ids like [D3]"}
""",
            "user_template": """Today is {today}. The length: {length}.

The scenario:
{scenario}

Its share: {share}

What the simulation is built from: history's windows weighed by these signposts, each read today on its own history
(from -0.5 to +0.5, toward its warning up):
{simulation}

The factors at this length: each one's unit, history's 5th, 25th, 50th, 75th and 95th percentile, and where today's
signposts move the simulation:
{factors}

What the market's prices say:
{market}

How earlier scenarios turned out:
{record}

The coach's note:
{coach}

Research's notes on this week's key questions:
{notes}

Current data (cite by id):
{data}""",
        },
        "read_again": {
            "system": (
                "You are {view}. An investment board has simulated what can happen over a length from history: "
                "every window since 1971, weighed by today's signposts (listed). You said how much likelier or less "
                "likely a scenario is than its share of the simulation, for what the simulation does not include. "
                "A reviewer has read the reasons every reader gave for this scenario and found something to say. "
                "Answer again: keep your factor where your reason holds; change it where the reviewer showed that "
                "what you cited is already in the simulation, is not evidence, or is not true. Return only valid "
                "JSON."
            ),
            "schema": """
Return exactly this JSON (no extra keys):
{"times": 1.0, "missing": "what the simulation does not include that bears on this scenario, or none", "why": "two
or three sentences, citing data ids like [D3]"}
""",
            "user_template": """Today is {today}. The length: {length}.

The scenario:
{scenario}

Its share: {share}

What the simulation is built from: history's windows weighed by these signposts, each read today on its own history
(from -0.5 to +0.5, toward its warning up):
{simulation}

The factors at this length: each one's unit, history's 5th, 25th, 50th, 75th and 95th percentile, and where today's
signposts move the simulation:
{factors}

Your answer:
{first}

What the reviewer said of the readers' reasons on this scenario:
{review}

What the market's prices say:
{market}

Current data (cite by id):
{data}""",
        },
        "review": {
            "system": (
                "You check the reasons behind an investment board's scenarios. Several readers each said, of one "
                "scenario at a time, how much likelier or less likely it is than its share of the board's "
                "simulation (history's windows weighed by the signposts listed), for what the simulation does not "
                "include, and why. You give no numbers. Check each reason: is what it cites really outside the "
                "simulation, or does a signpost already count it; is it evidence, or a habit (caution for its own "
                "sake, a vivid story, the same reason given to every scenario); are its facts right. You may first "
                "ask the board's own data questions with its tools, one at a time, to check a fact. Point only at "
                "reasons with a fault: a factor far from 1 with a sound reason is not one. Return only valid JSON."
            ),
            "schema": """
Return exactly this JSON, one of the two (no extra keys):
{"ask": {"tool": "<tool name>", "args": {}}}
{"points": [{"length": "<length id>", "scenario": "<scenario id>", "says": "a few words: already in the simulation,
not evidence, not true, ...", "why": "two or three sentences, naming the readers' reasons you mean"}]}
Ask only while you have questions left; then give your points (none when every reason holds).
""",
            "user_template": """Today is {today}.

Lengths:
{lengths}

What the simulation is built from: history's windows weighed by these signposts, each read today on its own history
(from -0.5 to +0.5, toward its warning up):
{simulation}

Factors: each one's unit; its value over windows of each length in history (its 5th, 25th, 50th, 75th and 95th
percentile); and where today's signposts move the simulation:
{factors}

The scenarios, each with its share and every reader's factor, what it says the simulation misses, and why:
{readings}

What the market's prices say:
{market}

How earlier scenarios turned out:
{record}

Tools (the board's own data, read-only):
{tools}

What you asked so far and the answers:
{asked}

Questions left: {left}.""",
        },
        "coach": {
            "system": (
                "You coach the LLMs that write an investment board's scenarios and say how much likelier or less "
                "likely each is than the board's simulation. From how the scenarios that are over turned out (the "
                "factor given to the part that happened, the simulation's share of it, and its final probability "
                "against that share), say in a few sentences what the factors get right or wrong so far: which kinds "
                "of scenario they raise or lower, and whether that helped. Say only what the record shows, and say "
                "when it is too short to show anything. Return only valid JSON."
            ),
            "schema": """
Return exactly this JSON (no extra keys):
{"note": "three to six sentences"}
""",
            "user_template": """Today is {today}.

How the scenarios that are over turned out:
{record}""",
        },
    },
    "readings": {
        "read": {
            "system": (
                "You read one text and put it on a scale, for a series an investment board keeps. Read only what the "
                "text says, as of its date; not what happened after, not what you expect. If the text does not bear "
                "on the question, say so. Quote the phrases that carry your reading word for word. Return only "
                "valid JSON."
            ),
            "schema": """
Return exactly this JSON (no extra keys):
{"relevant": true, "value": 0.0, "quotes": ["a phrase from the text, word for word"], "why": "one sentence"}
value: a number from -2 to 2 on the scale below (0.5 steps are fine); null when relevant is false. One to three quotes.
""",
            "user_template": """What to read it for: {what}
The scale: {scale}

The text ({title}, {day}):
{text}""",
        }
    },
    "indicators": {
        "propose": {
            "system": (
                "You design indicators for a forecasting question: observable, regularly published series that tell "
                "which outcome is becoming more likely, each with a threshold and likelihood ratios per outcome. "
                "Prefer free daily or monthly data (FRED, Yahoo, Cboe). Return only valid JSON."
            ),
            "schema": """
Return exactly this JSON (no extra keys):
{"drivers": [{"name": "...", "why": "...", "indicators": [{"name": "...", "series": "...", "value": "...", "red": "...", "yellow": "...", "lr": {"<outcome id>": 1.0}}]}]}
""",
            "user_template": """Question: {question}
{detail}

Outcomes:
{outcomes}

Propose 3 to 6 drivers, each with 1 to 3 indicators.""",
        },
        "board": {
            "system": (
                "You design indicators for an investment board: each a value computed every day from the series the "
                "board keeps, with a red line (and a yellow one) past which it says that a forecast's outcome is "
                "changing over the next three months. The board fits how each state bears on each outcome from "
                "history, keeps an indicator only if it beats history's base rate walked forward under a false "
                "discovery control, and then again on the data that comes after today; it moves nothing before. So "
                "propose what you expect to work from how the economy and markets work, not because it would have "
                "flagged one famous episode. Return only valid JSON."
            ),
            "schema": """
Return exactly this JSON (no extra keys):
{"signposts": [{"id": "lower_case_id", "name": "a few words", "driver": "Growth|Inflation|Rates and credit|AI capex and tech|Sentiment and trend|Liquidity|Other",
 "value": "an expression in the language below", "red": "> 0.2", "yellow": "> 0.1", "try_on": ["<forecast id>"],
 "why": "one or two sentences: what it measures and why it bears on those forecasts"}]}
A line is an operator and a number, in the value's own unit. The red line should be crossed on a minority of days in
history (a tenth to a third), the yellow one more often. Use only the series listed, and only the forms below.
""",
            "user_template": """Today is {today}.

The forecasts an indicator may be tried on (each: a factor over the next three months, ending in one of four
quarters of its history):
{targets}

The series (id: name, how often, since):
{series}

The expression language:
{language}

Indicators the board has (do not repeat them or near copies of them):
{existing}

Current data (for context):
{data}

Propose {n} new indicators.""",
        },
        "expect": {
            "system": (
                "You write what an indicator says about the economy and markets ahead, as an economist would defend it "
                "without looking at data: each an expectation that one factor moves more than usual one way (up, down) "
                "or that its moves spread wider or narrower, over a span of days from the day the indicator reads high, "
                "with the mechanism in one sentence. The board sizes each from data once there is enough of it; until "
                "then it uses yours at half weight, so write only mechanisms you would stand behind, and none rather "
                "than a weak one. Return only valid JSON."
            ),
            "schema": """
Return exactly this JSON (no extra keys):
{"expect": [{"factor": "<factor id>", "way": "up|down|wider|narrower", "days": [from, to], "why": "one sentence"}]}
One to three expectations. days: whole days from the day the indicator reads high, from 0 up to 1095, from before to.
"High" means toward its red line: the way stated below.
""",
            "user_template": """The indicator: {name}
It reads: {what}
Its scale: {scale}
It is computed as {value}; it is red {red} (that is what "high" means here).
Why it was added: {why}

The factors (id: what, how measured over a span of days):
{factors}""",
        },
    },
}

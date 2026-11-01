"""The LLM steps, through the model layer's methods (src/llm/methods): the world's scenarios (the method runs its
roles; this module gives it the models), the proposals of new questions, factors and signposts and the judgment on
them, the texts read into numbers, grounded writing for the board's texts. src/llm owns how a model is asked (the
shape of an answer, the check that a text's numbers are the data's); this board owns what is asked: the data and
what it computed, its texts' instructions (TEXTS). The provider and its key: src/llm/config.py (the repository's
.env)."""

import asyncio

from src.llm import config
from src.llm.calls import LLMCallCollector
from src.llm.client import LLMClient
from src.llm.methods import grounded
from src.llm.methods import indicators as indicators_method
from src.llm.methods import propose as propose_method
from src.llm.methods import readings as readings_method
from src.llm.methods import scenarios as scenarios_method
from src.llm.prompts import quoted  # noqa: F401  (a text the LLM reads, fenced as a quote: for the steps)

# flash on purpose (the owner, 2026-10-29): cheap, so the board can ask many times; one model family for now
MODELS = ["deepseek-v4-flash"]
TEMPERATURE = 0.7
# The board's own texts: role, who writes and for whom; instructions, what to write, with {today} and the parts
# the pipeline fills in. Every number in them must be one of the data's values, cited by its id, or it is asked
# again (grounded writing).
PLAIN = (
    "Write for a person, in plain words. Probabilities and shares as whole percentages (24%, not 0.242911); other "
    "values with only the digits that matter (4.98%, 77,198). Name rules, signposts, questions and scenarios in "
    "words, as the data names them, never by an id or a code (no btc_cap, no 2026-W44, no spx_2y). An indicator a "
    "reader may not know gets a few words the first time (AHR999, a gauge of how cheap Bitcoin is). Each number "
    "keeps its [D..] id right after it."
)
TEXTS = {
    "brief": {
        "kind": "text",
        "role": (
            "You write a short weekly note for one investor about what changed in their scenario board and why. "
            "Plain Chinese, no headings, no advice beyond what the rules say. " + PLAIN
        ),
        "instructions": """Today is {today}. Write 4 to 8 sentences in Chinese, each number followed by its [D..] id.

This week's changes:
{changes}

The scenarios LLMs wrote for the board at each length, the last layer of its world: each one's probability (the
simulation's share times the factor the LLMs gave it for what the simulation does not include), the share and the
factor (each the data item named):
{futures}""",
    },
    "redteam": {
        "kind": "points",
        "role": (
            "You argue against an investor's current positioning, as a sceptical risk manager. Be specific: which "
            "outcomes hurt this portfolio most, which assumption behind it is weakest, what would show it is "
            "wrong. Plain Chinese. " + PLAIN
        ),
        "instructions": "Today is {today}. Give 3 to 5 points in Chinese, each one or two sentences, numbers followed "
        """by their [D..] id.

The portfolio by share (no amounts; each value is the data item named):
{weights}

The portfolio on the board's futures at each horizon, and the scenarios most of its worst 10% comes from (each value
is the data item named):
{scenarios}

Argue against this positioning.""",
    },
}


def usage(collector):
    cost, prompt, completion = collector.totals()
    done = [c for c in collector.all() if c.status == "success"]
    return {
        "calls": len(done),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cost_usd": cost,
        "models": sorted({c.model for c in done}),
    }


def ready():
    if not config.LLM_API_KEY:
        raise RuntimeError("no LLM_API_KEY (the repository's .env)")


def propose_factors(**kw):
    """Drafts of new factors (src/llm/methods/propose.py factors): {"method", "factors", "rejected", "usage"}."""
    ready()
    calls = LLMCallCollector()
    client = LLMClient(model=MODELS[0], temperature=0.8, max_tokens=3000, collector=calls)
    out = asyncio.run(propose_method.factors(**kw, client=client))
    return {**out, "usage": usage(calls)}


JUDGES = 3  # independent readings of whether to adopt a proposal: a majority adopts


def judge_proposal(today, kind, proposal, checks, has, n=JUDGES):
    """Whether to adopt a proposal that passed its checks, read n times independently (propose.py judge): {"method",
    "votes" [{"adopt", "why"}], "adopt" (a majority said so), "why" (the first of the majority's), "usage"}."""
    ready()
    calls = LLMCallCollector()
    client = LLMClient(model=MODELS[0], temperature=TEMPERATURE, max_tokens=800, collector=calls)

    async def all_of():
        return await asyncio.gather(
            *(
                propose_method.judge(today=today, kind=kind, proposal=proposal, checks=checks, has=has, client=client)
                for _ in range(n)
            )
        )

    votes = asyncio.run(all_of())
    yes = [v for v in votes if v["adopt"]]
    adopt = len(yes) * 2 > n
    why = next((v["why"] for v in (yes if adopt else [v for v in votes if not v["adopt"]])), "")
    return {
        "method": votes[0]["method"] if votes else None,
        "votes": [{"adopt": v["adopt"], "why": v["why"]} for v in votes],
        "adopt": adopt,
        "why": why,
        "usage": usage(calls),
    }


def propose(today, questions, series, data, context, n=3):
    """Drafts of new questions (src/llm/methods/propose.py): {"method", "proposals", "rejected", "usage"}."""
    ready()
    calls = LLMCallCollector()
    out = asyncio.run(
        propose_method.propose(
            questions,
            series,
            data,
            today=today,
            context=context,
            n=n,
            client=LLMClient(model=MODELS[0], temperature=0.8, max_tokens=4000, collector=calls),
        )
    )
    return {**out, "usage": usage(calls)}


def signposts(**kw):
    """New indicators in the board's form (src/llm/methods/indicators.py for_board): {"method", "signposts",
    "rejected", "usage"}."""
    ready()
    calls = LLMCallCollector()
    out = asyncio.run(
        indicators_method.for_board(
            **kw, client=LLMClient(model=MODELS[0], temperature=0.8, max_tokens=4000, collector=calls)
        )
    )
    return {**out, "usage": usage(calls)}


def expect(indicator, factors):
    """What an indicator says about the factors ahead (src/llm/methods/indicators.py expect): {"method", "expect",
    "rejected", "usage"}."""
    ready()
    calls = LLMCallCollector()
    out = asyncio.run(
        indicators_method.expect(
            indicator, factors, client=LLMClient(model=MODELS[0], temperature=0.3, max_tokens=1500, collector=calls)
        )
    )
    return {**out, "usage": usage(calls)}


def read(text, *, what, scale, title, day, samples):
    """A text read into a number (src/llm/methods/readings.py): {"method", "relevant", "value", "spread", "answers",
    "dropped", "usage"}."""
    ready()
    calls = LLMCallCollector()
    out = asyncio.run(
        readings_method.read(
            text,
            what=what,
            scale=scale,
            title=title,
            day=day,
            samples=samples,
            client_for=lambda i: LLMClient(model=MODELS[0], temperature=0.7, max_tokens=600, collector=calls),
        )
    )
    return {**out, "usage": usage(calls)}


# the scenarios' models in turn ("provider:model", local/settings.toml scenario_models): flash only, on purpose (the
# owner, 2026-10-31: the pro model costs too much); another family once its provider's key is in .env
ROSTER = ["deepseek:deepseek-v4-flash"]


def client(spec, calls, temperature, max_tokens):
    """A client for a roster entry: "provider:model", or a model of the main provider."""
    provider, sep, model = spec.partition(":")
    if not sep:
        provider, model = None, spec
    return LLMClient(model=model, provider=provider, temperature=temperature, max_tokens=max_tokens, collector=calls)


def scenarios(*, roster=None, **kw):
    """The world's scenarios by several LLMs (src/llm/methods/scenarios.py run, every argument but the models its
    own): on the roster's models (ROSTER without one). The method's answer and "usage"."""
    ready()
    calls = LLMCallCollector()
    out = asyncio.run(
        scenarios_method.run(roster=list(roster or ROSTER), client=lambda spec, t, m: client(spec, calls, t, m), **kw)
    )
    return {**out, "usage": usage(calls)}


times = scenarios_method.times  # each scenario's factor: the median of its readers' final ones
span = scenarios_method.span  # a range in words


def scenarios_coach(today, record, roster=None):
    """The coach's note on the scenarios' factors (src/llm/methods/scenarios.py coach): {"method", "note", "usage"}."""
    ready()
    calls = LLMCallCollector()
    out = asyncio.run(
        scenarios_method.coach(today=today, record=record, client=client((roster or ROSTER)[0], calls, 0.3, 1500))
    )
    return {**out, "usage": usage(calls)}


def write(text_cfg, data, **parts):
    """One of the board's texts (TEXTS): {"output", "valid", "prompt_hash", "attempts", "usage"}."""
    ready()
    calls = LLMCallCollector()
    out = asyncio.run(
        grounded.write(
            text_cfg["role"],
            text_cfg["instructions"].format(**parts),
            data,
            kind=text_cfg["kind"],
            client=LLMClient(model=MODELS[0], max_tokens=2000, collector=calls),
        )
    )
    return {**out, "prompt_hash": out["method"], "usage": usage(calls)}


def method():
    """The version of the scenarios' writing in use: its prompts' and code's hash, and the model."""
    return f"{scenarios_method.method_hash()}-{MODELS[0]}"

"""Research's prompts (research.py), rendered by src/llm/prompts.py; the notes are grounded writing
(src/llm/methods/grounded.py), so only their role and instructions are here."""

PROMPTS = {
    "plan": {
        "system": (
            "You plan the research behind a forecast. You name what would move the answer most and the news "
            "searches that would show where those stand now. Your knowledge of recent events may be out of date: "
            "take the current state of the world only from the data given, and the date as given; searches are about "
            "now. Return only valid JSON."
        ),
        "schema": """
Return exactly this JSON (no extra keys):
{"drivers": ["what would move the answer, a few words each"], "queries": ["a news search, three to six words"]}
""",
        "user_template": """Today is {today}.

Question: {question}
{detail}

Outcomes (exactly one will happen):
{outcomes}

Current data:
{data}

Name 3 to 5 drivers and 3 to 5 news searches.""",
    },
    "notes": {
        "role": (
            "You read the news for a forecaster and note what bears on one question: plain English, facts and "
            "developments only, no forecast of your own."
        ),
        "instructions": """Today is {today}.

Question: {question}
{detail}

Outcomes:
{outcomes}

What would move the answer: {drivers}

From the news items and the data below, write 4 to 10 points. Each point is one fact or development that makes an
outcome more or less likely: say which, and cite the items it comes from by id, like [D61]. Leave out what does not
bear on the question; if nothing does, say so in one point.""",
    },
    "judge": {
        "system": (
            "You decide whether research notes are enough to forecast a question, or which searches would fill "
            "what they miss. Your knowledge of recent events may be out of date: searches are about now, as dated "
            "below, never about a year you remember as recent. Return only valid JSON."
        ),
        "schema": """
Return exactly this JSON (no extra keys):
{"enough": true, "missing": "what the notes do not cover, or empty", "queries": ["a news search, three to six words"]}
""",
        "user_template": """Today is {today}.

Question: {question}

What would move the answer: {drivers}

The notes so far ({found} news items found in round {round} of {rounds}):
{notes}

Are the notes enough to judge each driver's state today? If not, give up to 3 searches for what is missing.""",
    },
}

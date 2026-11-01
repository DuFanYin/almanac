"""Text grounded in the caller's data: every number in it must be one of the data's values, cited by its id.

The caller gives its role (who writes, for whom, in what language), the instructions and the data; the answer
is checked for numbers that are not in the data, or not the item they cite, and asked again (twice at most); the
answer says whether it passed.
"""

from __future__ import annotations

import json
import re

from src.llm.client import LLMClient
from src.llm.methods.method import method_hash as _method_hash
from src.llm.methods.prompts import METHOD_PROMPTS
from src.llm.prompts import numbered_data, render

# not inside a word, an id ([D3]) or a decimal; \w would take CJK characters as part of a word; a sign, ASCII or the
# minus sign (U+2212), belongs to it
NUMBER = re.compile(r"(?<![A-Za-z0-9_\[.])[-+\u2212]?\d(?:[\d,]*\d)?(?:\.\d+)?%?")
ATTEMPTS = 3


def method_hash() -> str:
    return _method_hash("grounded", __file__)


CITED = re.compile(r"\s*(?:\[(D\d+)\])")  # an id right after a number: the number must be that item's


def _number(raw: str) -> float:
    return float(raw.replace(",", "").replace("\u2212", "-").rstrip("%"))


def _values(d: dict) -> list[float]:
    """An item's values: its value, `also`, and every number in its name, or in its value when that is a text."""
    out, texts = [], [str(d.get("name", ""))]
    for v in [d.get("value"), *(d.get("also") or [])]:
        try:
            out.append(float(str(v).replace(",", "").rstrip("%")))
        except (TypeError, ValueError):
            if isinstance(v, str):
                texts.append(v)
    for t in texts:
        out += [_number(m.group()) for m in NUMBER.finditer(t)]
    return out


def _matches(raw: str, allowed: list[float]) -> bool:
    """As given, rounded (within 0.6%, or within half a unit of the last digit written: 6.0 for 6.042), or as a
    percentage of a fraction (34% for 0.34). Written without a sign, its size is compared: the words carry the
    direction ("fell 18%", "亏损18%" for -0.18); written with one, the sign must agree too."""
    x = _number(raw)
    signed = raw[0] in "-+\u2212"
    digits = len(raw.rstrip("%").partition(".")[2])
    half = 0.5 * 10**-digits
    return any(
        abs(x - b) <= max(0.006 * max(1.0, abs(b)), half)
        or abs(x / 100 - b) <= max(0.006 * max(0.01, abs(b)), half / 100)
        for a in allowed
        for b in ((a,) if signed else (a, -a))
    )


def unknown_numbers(text: str, data: list[dict]) -> list[str]:
    """Numbers in the text that are not the data's: one followed by an id ([D3]) must be that item's value; one
    without, any item's. A number in an item's name counts as its value (a 200-day mean). Small whole numbers
    (counts of days or months) and years pass when they cite no item."""
    by_id = {str(d.get("id")): _values(d) for d in data}
    every = [a for vs in by_id.values() for a in vs]
    bad = []
    for m in NUMBER.finditer(text):
        raw = m.group()
        cited = CITED.match(text, m.end())
        if not cited and raw.isdigit() and (int(raw) <= 31 or 1990 <= int(raw) <= 2100):
            continue
        allowed = by_id.get(cited.group(1), []) if cited else every
        if not _matches(raw, allowed):
            bad.append(f"{raw} [{cited.group(1)}]" if cited else raw)
    return bad


def text_of(output: dict, kind: str) -> str:
    return output.get("text", "") if kind == "text" else " ".join(str(p) for p in output.get("points", []))


async def write(
    role: str, instructions: str, data: list[dict], *, kind: str = "text", client: LLMClient | None = None
) -> dict:
    """{"method", "output" ({"text"} or {"points"}), "valid", "attempts": [{"raw", "unknown_numbers"}]}."""
    if kind not in ("text", "points"):
        raise ValueError(f"kind must be text or points, not {kind!r}")
    client = client or LLMClient(max_tokens=2000)
    rule, user = render(METHOD_PROMPTS["grounded"][kind], instructions=instructions, data=numbered_data(data))
    system = f"{role.strip()} {rule}"
    attempts, out = [], {}
    for _ in range(ATTEMPTS):
        raw = await client.complete(user, system=system, node=f"grounded.{kind}")
        out = json.loads(raw)
        bad = unknown_numbers(text_of(out, kind), data)
        attempts.append({"raw": raw, "unknown_numbers": bad})
        if not bad:
            return {"method": method_hash(), "output": out, "valid": True, "attempts": attempts}
        user += (
            f"\n\nYour last answer used numbers not in the data, or not the value of the item cited after them: "
            f"{', '.join(bad[:10])}. Use only the data's numbers, each with the id of the item it comes from."
        )
    return {"method": method_hash(), "output": out, "valid": False, "attempts": attempts}

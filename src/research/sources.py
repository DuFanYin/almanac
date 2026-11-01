"""Where research finds its evidence: Google News's search feed (free, no key: titles, the outlet and the day) and,
when TAVILY_API_KEY is set (the repository's .env), Tavily's search (the pages' own text). Each item:
{"kind": "news" | "web", "title", "source", "day", "url", "text"}."""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

import httpx
import src.llm.config  # noqa: F401 — the repository's .env, for TAVILY_API_KEY

log = logging.getLogger("research")

TIMEOUT = 15.0
RECENT = "60d"  # Google News: only what was published in this many days


def google_news(query: str, limit: int = 8) -> list[dict]:
    r = httpx.get(
        "https://news.google.com/rss/search",
        params={"q": f"{query} when:{RECENT}", "hl": "en-US", "gl": "US", "ceid": "US:en"},
        timeout=TIMEOUT,
        follow_redirects=True,
    )
    r.raise_for_status()
    out = []
    for it in ET.fromstring(r.text).iter("item"):
        title, source = it.findtext("title") or "", it.findtext("source") or ""
        if source and title.endswith(f" - {source}"):  # the feed appends the outlet to the title
            title = title[: -len(source) - 3]
        try:
            day = parsedate_to_datetime(it.findtext("pubDate") or "").date().isoformat()
        except (TypeError, ValueError):
            day = None
        out.append(
            {"kind": "news", "title": title, "source": source, "day": day, "url": it.findtext("link"), "text": ""}
        )
        if len(out) >= limit:
            break
    return out


def tavily(query: str, limit: int = 5) -> list[dict]:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return []
    r = httpx.post(
        "https://api.tavily.com/search",
        json={"query": query, "max_results": limit},
        headers={"Authorization": f"Bearer {key}"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return [
        {
            "kind": "web",
            "title": x.get("title", ""),
            "source": urlparse(x.get("url", "")).netloc,
            "day": (x.get("published_date") or "")[:10] or None,
            "url": x.get("url"),
            "text": x.get("content", ""),
        }
        for x in r.json().get("results", [])
    ]


def search(query: str) -> list[dict]:
    """Every source for one query; a source that fails gives nothing (logged), the others still count."""
    out = []
    for source in (google_news, tavily):
        try:
            out += source(query)
        except Exception as e:  # a feed down, a changed format, a bad key
            log.warning("%s %r: %s: %s", source.__name__, query, type(e).__name__, e)
    return out

"""Texts for the readings (steps/readings.py, config/readings.toml): what the Federal Reserve publishes on monetary
policy, news headlines, and companies' earnings releases from SEC EDGAR. Each text: {"id", "day" (published, ISO),
"title", "url", "text"}. Free, no key; under the board's own name (sources.py), EDGAR's under the owner's contact
(local/settings.toml sec_contact), as EDGAR asks. Nothing here reads them."""

import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from zoneinfo import ZoneInfo

from . import sources

NY = ZoneInfo("America/New_York")
FED_FEEDS = (
    "https://www.federalreserve.gov/feeds/press_monetary.xml",  # statements, minutes, other monetary releases
    "https://www.federalreserve.gov/feeds/speeches.xml",  # the Board's members
)


class _Article(HTMLParser):
    """The text of the element with id="article" (the Fed's pages), scripts and styles left out."""

    def __init__(self):
        super().__init__()
        self.depth, self.skip, self.parts = 0, 0, []

    def handle_starttag(self, tag, attrs):
        if self.depth:
            self.depth += tag not in ("br", "img", "hr", "meta", "link", "input")
        elif dict(attrs).get("id") == "article":
            self.depth = 1
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip:
            self.skip -= 1
        if self.depth:
            self.depth -= 1
            if tag in ("p", "div", "li", "h1", "h2", "h3", "h4"):
                self.parts.append("\n")

    def handle_data(self, data):
        if self.depth and not self.skip:
            self.parts.append(data)


def article_text(html):
    p = _Article()
    p.feed(html)
    return re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", "".join(p.parts))).strip()


class _Text(HTMLParser):
    """A whole document's text: a line a block, table cells spaced, scripts and styles left out."""

    def __init__(self):
        super().__init__()
        self.skip, self.parts = 0, []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip:
            self.skip -= 1
        if tag in ("p", "div", "tr", "li", "h1", "h2", "h3", "h4", "table"):
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def document_text(html):
    p = _Text()
    p.feed(html)
    return re.sub(r"\n\s*\n+", "\n", re.sub(r"[ \t\xa0]+", " ", "".join(p.parts))).strip()


def day_of(pub):
    """A feed's pubDate as the day in New York."""
    try:
        return parsedate_to_datetime(pub).astimezone(NY).date().isoformat()
    except (TypeError, ValueError):
        return None


def feed_items(xml):
    """[(title, link, day)] of an RSS feed."""
    out = []
    for it in ET.fromstring(xml.lstrip("﻿")).iter("item"):
        out.append(
            ((it.findtext("title") or "").strip(), (it.findtext("link") or "").strip(), day_of(it.findtext("pubDate")))
        )
    return out


def fed(since):
    """The Fed's monetary policy releases and Board speeches published on or after `since` (a date), with their text."""
    out = []
    for url in FED_FEEDS:
        for title, link, day in feed_items(sources.get_text(url)):
            if not day or date.fromisoformat(day) < since or not link:
                continue
            text = article_text(sources.get_text(link))
            if text:
                out.append({"id": link, "day": day, "title": title, "url": link, "text": text})
    return out


def news(queries, days, today, most=40):
    """One text of the headlines of the last `days` days on each query (Google News), each once, newest first:
    {"id": "<today>", "day": today, "title", "url": None, "text": one headline a line, with its outlet and day}."""
    seen, lines = set(), []
    for q in queries:
        url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
            {"q": f"{q} when:{days}d", "hl": "en-US", "gl": "US", "ceid": "US:en"}
        )
        root = ET.fromstring(sources.get_text(url))
        for it in root.iter("item"):
            title, outlet = it.findtext("title") or "", it.findtext("source") or ""
            if outlet and title.endswith(f" - {outlet}"):
                title = title[: -len(outlet) - 3]
            key = title.lower().strip()
            if key and key not in seen:
                seen.add(key)
                lines.append(
                    (day_of(it.findtext("pubDate")) or "", f"{title} ({outlet}, {day_of(it.findtext('pubDate'))})")
                )
    lines = [x for _, x in sorted(lines, reverse=True)[:most]]
    if not lines:
        return []
    stamp = datetime.now(NY).isoformat(timespec="minutes")
    return [
        {
            "id": today.isoformat(),
            "day": today.isoformat(),
            "title": f"{len(lines)} headlines, {stamp}",
            "url": None,
            "text": "\n".join(lines),
        }
    ]


EDGAR = "https://www.sec.gov"
EDGAR_DATA = "https://data.sec.gov/submissions/CIK{:010d}.json"


def press_release(index_html):
    """The path of a filing's press release (its EX-99.1) on its EDGAR index page; None without one."""
    found = re.findall(r'href="([^"]+)"[^>]*>[^<]*</a></td>\s*<td[^>]*>EX-99\.1<', index_html)
    return found[0] if found else None


def releases(companies, since, contact):
    """The earnings releases (an 8-K with item 2.02, its exhibit 99.1) filed on or after `since` by `companies`
    ({name: SEC CIK}), each with its text; asked under `contact`."""
    out = []
    for name, cik in companies.items():
        r = json.loads(sources.get_text(EDGAR_DATA.format(cik), contact))["filings"]["recent"]
        for form, day, acc, items in zip(r["form"], r["filingDate"], r["accessionNumber"], r["items"]):
            if form != "8-K" or "2.02" not in items or date.fromisoformat(day) < since:
                continue
            folder = f"{EDGAR}/Archives/edgar/data/{cik}/{acc.replace('-', '')}"
            path = press_release(sources.get_text(f"{folder}/{acc}-index.html", contact))
            if not path:
                continue
            text = document_text(sources.get_text(EDGAR + path, contact))
            if text:
                out.append(
                    {"id": acc, "day": day, "title": f"{name}: earnings release", "url": EDGAR + path, "text": text}
                )
    return out

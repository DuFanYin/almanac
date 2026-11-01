"""Market and economic data sources: each returns {series id: [(day, value), ...]} (Kalshi is in kalshi.py).

All free, no keys: Yahoo through yfinance, Cboe's index histories, FRED's CSVs, CNN's Fear & Greed, DefiLlama,
multpl.com, and the one-time archives below."""

import csv
import gzip
import io
import json
import math
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
BROWSER = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/141.0.0.0 Safari/537.36"
)
CNN = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{}"
CNN_HEADERS = {"Accept": "application/json, text/plain, */*", "Referer": "https://edition.cnn.com/"}
CBOE = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}&cosd={}"
LLAMA = "https://stablecoins.llama.fi/stablecoincharts/all"
AGENT = "scenario-board/1.0 (a personal dashboard; daily, one request a series)"


def get_bytes(url, agent=AGENT):
    """A plain GET under our own name (or `agent`: SEC EDGAR asks for a contact). Not the Market page's browser
    User-Agent: FRED's front (Akamai) holds a request that claims to be Chrome without being one open with no answer
    (2026-10-29: 15 s timeouts with it, 0.4 s without)."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": agent})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def get_text(url, agent=AGENT):
    return get_bytes(url, agent).decode("utf-8-sig", "replace")


def get_json(url, headers=None):
    """JSON of a GET that sends a browser's User-Agent (CNN and Kalshi answer only those) plus `headers`."""
    req = urllib.request.Request(url, headers={"User-Agent": BROWSER, "Accept-Encoding": "gzip", **(headers or {})})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read()
    return json.loads(gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body)


def open_day(symbol):
    """The day whose bar is still forming, left out: a coin's current UTC day (Yahoo dates crypto in UTC); for
    the rest, today in New York until a little after the close."""
    if symbol.endswith("-USD"):
        return datetime.now(timezone.utc).date()
    now = datetime.now(NY)
    return now.date() if now.hour * 60 + now.minute < 16 * 60 + 30 else now.date() + timedelta(days=1)


def yahoo(series, since):
    """Daily closes (split-adjusted, not dividend-adjusted: Yahoo's Close; with adjusted = true, its Adj Close:
    dividends reinvested) from `since`, without a bar still forming."""
    import yfinance as yf

    syms = sorted({s["symbol"] for s in series.values()})
    df = yf.download(syms, start=since.isoformat(), progress=False, auto_adjust=False, threads=False)
    out = {}
    for key, s in series.items():
        field, sym = ("Adj Close" if s.get("adjusted") else "Close"), s["symbol"]
        col = df[field][sym].dropna() if sym in df[field] else []
        cut = open_day(sym)
        out[key] = [(d.date().isoformat(), float(v)) for d, v in col.items() if d.date() < cut]
    return out


def cboe(series, since):
    out = {}
    for key, s in series.items():
        rows = list(csv.reader(io.StringIO(get_text(CBOE.format(s["symbol"])))))
        head = [h.strip().upper() for h in rows[0]]
        col = head.index("CLOSE") if "CLOSE" in head else len(head) - 1
        vals = []
        for r in rows[1:]:
            try:
                d = datetime.strptime(r[0].strip(), "%m/%d/%Y").date()
                if d >= since:
                    vals.append((d.isoformat(), float(r[col])))
            except (ValueError, IndexError):
                continue
        out[key] = vals
    return out


def fred(series, since):
    """One CSV per series, each tried three times; the last failure is raised (the caller fetches one series
    at a time, so one failing series loses only itself)."""
    out = {}
    for key, s in series.items():
        for attempt in range(3):
            try:
                rows = list(csv.reader(io.StringIO(get_text(FRED.format(s["symbol"], since.isoformat())))))
                if not rows or not rows[0] or rows[0][0] != "observation_date":
                    raise ValueError(f"{s['symbol']}: not a FRED CSV")
                out[key] = [(r[0], float(r[1])) for r in rows[1:] if len(r) > 1 and r[1] not in ("", ".")]
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))
    return out


def cnn(series, since):
    """CNN's Fear & Greed by day, and its seven parts (a series with `part`: the key CNN gives it, such as
    market_momentum_sp500); CNN keeps about two years (an older start answers 500)."""
    start = max(since, date.today() - timedelta(days=700))
    j = get_json(CNN.format(start.isoformat()), CNN_HEADERS)
    out = {}
    for key, s in series.items():
        block = j[s["part"]] if s.get("part") else j["fear_and_greed_historical"]
        by_day = {
            datetime.fromtimestamp(p["x"] / 1000, timezone.utc).date().isoformat(): round(float(p["y"]), 2)
            for p in block.get("data", [])
        }
        out[key] = sorted(by_day.items())
    return out


def llama(series, since):
    import json

    rows = json.loads(get_text(LLAMA))
    vals = []
    for r in rows:
        d = datetime.fromtimestamp(int(r["date"]), timezone.utc).date()
        if d >= since:
            vals.append((d.isoformat(), float(sum((r.get("totalCirculatingUSD") or {}).values()))))
    return {key: vals for key in series}


def multpl(series, since):
    """A monthly table from multpl.com (the Shiller CAPE: 1871 on, monthly average prices over ten years of real
    earnings), one row per month dated its first day (or, with dated = "end", its last: the dividend yield);
    today's running value is left out (a month not over)."""
    import html
    import re

    out = {}
    for key, s in series.items():
        rows = re.findall(r"<tr[^>]*>\s*<td>([^<]+)</td>\s*<td>(.*?)</td>", get_text(s["url"]), re.S)
        vals = []
        for d, v in rows:
            day = datetime.strptime(d.strip(), "%b %d, %Y").date()
            num = re.sub(r"[^0-9.\-]", "", html.unescape(v))
            month_end = (day + timedelta(days=1)).day == 1
            if (month_end if s.get("dated") == "end" else day.day == 1) and since <= day <= date.today() and num:
                vals.append((day.isoformat(), float(num)))
        out[key] = sorted(vals)
    return out


# ---- archives: one-time histories that carry a series back (series.toml: source = "archive", a format, and
# backfill on the series they extend); fetched while the series has no values, never again


def coinmetrics_prices(s):
    """Coin Metrics' free community API: daily reference prices of `symbol` up to `until`."""
    import json

    url = (
        "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets={}&metrics=PriceUSD&frequency=1d"
        "&start_time=2009-01-01&end_time={}&page_size=10000"
    ).format(s["symbol"], s["until"])
    return [(r["time"][:10], float(r["PriceUSD"])) for r in json.loads(get_text(url))["data"]]


def csv_column(s):
    """A CSV at `url`: the `column` by the `date` column, from the header row on (notes above it are skipped)."""
    lines = get_text(s["url"]).splitlines()
    head = next(i for i, x in enumerate(lines) if x.startswith(s["date"] + ","))
    out = []
    for r in csv.DictReader(lines[head:]):
        try:
            out.append((r[s["date"]][:10], float(r[s["column"]])))
        except (TypeError, ValueError):
            pass
    return out


def msci_levels(s):
    """MSCI's index levels (the end-of-day service behind its index pages; it starts on 1997-01-01)."""
    import json

    url = (
        "https://app2.msci.com/products/service/index/indexmaster/getLevelDataForGraph?currency_symbol=USD"
        "&index_variant=STRD&start_date=19970101&end_date={}&data_frequency=DAILY&index_codes={}"
    ).format(s["until"].replace("-", ""), s["symbol"])
    rows = json.loads(get_text(url))["indexes"]["INDEX_LEVELS"]
    return [
        (
            f"{r['calc_date'] // 10000}-{r['calc_date'] // 100 % 100:02d}-{r['calc_date'] % 100:02d}",
            float(r["level_eod"]),
        )
        for r in rows
    ]


def worldbank_monthly(s):
    """The World Bank's commodity prices (the Pink Sheet workbook, monthly averages): the `column` of its
    Monthly Prices sheet, each month dated on its last day (the average is known once the month is over).
    Read with the standard library: an xlsx is zipped XML."""
    import calendar
    import re
    import xml.etree.ElementTree as ET
    import zipfile

    m = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    z = zipfile.ZipFile(io.BytesIO(get_bytes(s["url"])))
    strings = ["".join(t.text or "" for t in si.iter(m + "t")) for si in ET.fromstring(z.read("xl/sharedStrings.xml"))]
    rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    rid = next(
        x.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        for x in ET.fromstring(z.read("xl/workbook.xml")).iter(m + "sheet")
        if x.get("name") == "Monthly Prices"
    )
    col, out = None, []
    for row in ET.fromstring(z.read("xl/" + rels[rid])).iter(m + "row"):
        cells = {}
        for c in row.iter(m + "c"):
            v = c.find(m + "v")
            if v is not None:
                cells[re.match(r"[A-Z]+", c.get("r")).group()] = strings[int(v.text)] if c.get("t") == "s" else v.text
        if col is None:
            col = next((k for k, x in cells.items() if x.strip() == s["column"]), None)
            continue
        month = re.fullmatch(r"(\d{4})M(\d{2})", cells.get("A", ""))
        if month and col in cells:
            y, mo = int(month[1]), int(month[2])
            try:
                out.append((f"{y}-{mo:02d}-{calendar.monthrange(y, mo)[1]:02d}", float(cells[col])))
            except ValueError:
                pass
    return out


ARCHIVE = {"coinmetrics": coinmetrics_prices, "csv": csv_column, "msci": msci_levels, "worldbank": worldbank_monthly}


def archive(series, since):
    return {key: ARCHIVE[s["format"]](s) for key, s in series.items()}


FETCH = {"yahoo": yahoo, "cboe": cboe, "fred": fred, "cnn": cnn, "llama": llama, "multpl": multpl, "archive": archive}


def ahr999(btc, cfg=None, known=()):
    """AHR999 for every day with 200 closes before it: (price / 200-day mean) x (price / growth valuation) on
    daily closes; once kept, only the last 400 days again."""
    if len(btc) <= 200:
        return []
    if known and known[0][0] <= btc[199][0]:
        btc = btc[-400:]
    out, closes = [], [v for _, v in btc]
    for i in range(199, len(btc)):
        day, px = btc[i]
        sma = sum(closes[i - 199 : i + 1]) / 200
        at = datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp()
        out.append((day, round(px / sma * px / growth_valuation(at), 4)))
    return out


def par_bond(yields, cfg, known=()):
    """A total-return index of a Treasury held at constant maturity (cfg `years`), from its yield in percent:
    each day a bond at par, coupon at that day's yield (semiannual), sold the next day at the next yield, plus
    the coupon earned between; 100 on the first day. Recomputed whole: each value rests on every one before."""
    n = 2 * cfg["years"]
    out, level = [], 100.0
    for i, (day, y) in enumerate(yields):
        if i:
            (d0, y0), y1 = yields[i - 1], max(y, 0.01) / 100
            y0 /= 100
            v = (1 + y1 / 2) ** -n
            years = (date.fromisoformat(day) - date.fromisoformat(d0)).days / 365
            level *= (y0 / y1) * (1 - v) + v + y0 * years
        out.append((day, round(level, 6)))
    return out


DERIVED = {"ahr999": ahr999, "par_bond": par_bond}  # series.toml `formula`: computed here from `from`


def growth_valuation(at):
    """AHR999's growth valuation of BTC at epoch `at`: 10^(5.84 log10(days since the genesis block) - 17.01)."""
    age = (at - datetime(2009, 1, 3, tzinfo=timezone.utc).timestamp()) / 86400
    return 10 ** (5.84 * math.log10(age) - 17.01)

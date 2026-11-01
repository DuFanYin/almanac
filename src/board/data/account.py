"""The account's inputs, from outside this repository: the positions (a JSON file another program keeps in sync
with the brokers, named `positions` in local/settings.toml) and what the owner enters on the page (the page
server's database, `inputs` in local/settings.toml): their own probabilities, the firings marked done or skipped,
the proposed questions adopted or rejected. Without a positions file the board runs without holdings.

The positions file: {"date", "csv_end", "missing", "ibkr": {"nav", "cash", "positions": [{"symbol",
"underlying", "asset", "qty", "mv"}]}, "btc": {"qty", "cost", "price", "buys": [{"date", "usd"}]}}."""

import json
import sqlite3

from ..core import lineage, store


def exposure(db):
    """(the positions, their lineage node): each version kept once under data/account/exposures/ by its hash, so
    a rule firing can be traced to the positions it saw. (None, None) without a positions file."""
    path = store.settings().get("positions")
    try:
        exp = json.loads(path.read_text()) if path else None
    except OSError:
        exp = None
    if not exp:
        return None, None
    ehash = lineage.digest(exp)
    snap = store.ACCOUNT / "exposures" / f"{ehash}.json"
    if not snap.exists():
        store.write_json(snap, exp)
    node = lineage.record(
        db,
        "exposure",
        "exposure",
        {"hash": ehash},
        {"file": f"data/account/exposures/{ehash}.json", "date": exp.get("date")},
        step="read",
    )
    return exp, node


def judgment(qid):
    """(probabilities, note, when) of the owner's latest own view of a question, from the page's inputs; None
    without one."""
    path = store.INPUTS
    if not path.exists():
        return None
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        row = con.execute(
            "SELECT probs, note, at FROM judgments WHERE question = ? ORDER BY at DESC LIMIT 1", (qid,)
        ).fetchone()
    except sqlite3.OperationalError:  # no judgments table yet
        row = None
    finally:
        con.close()
    return (json.loads(row[0]), row[1], row[2]) if row else None


def decisions():
    """{trigger id: "done" | "skipped"} as the owner marked the firings on the page; empty without inputs."""
    path = store.INPUTS
    if not path.exists():
        return {}
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return dict(con.execute("SELECT trigger, status FROM decisions").fetchall())
    except sqlite3.OperationalError:  # no decisions table yet
        return {}
    finally:
        con.close()


def votes():
    """{proposal id: (status "adopted" | "rejected", note)} as the owner decided on the page; empty without inputs."""
    path = store.INPUTS
    if not path.exists():
        return {}
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {i: (s, n) for i, s, n in con.execute("SELECT id, status, note FROM proposals")}
    except sqlite3.OperationalError:  # no proposals table yet
        return {}
    finally:
        con.close()


NAMES = store.ACCOUNT / "names.json"  # single names' price histories: with the account data, they say what is held


def name_prices(db, symbols, today):
    """{symbol: (pd.Series of daily closes, lineage node)} for the single names held in size: Yahoo's closes (split-
    adjusted, not for dividends, as the class indexes they are set against), fetched once a day and kept under
    data/account/; a symbol that cannot be fetched is left out."""
    import pandas as pd

    try:
        kept = json.loads(NAMES.read_text())
    except (OSError, ValueError):
        kept = {}
    stale = [s for s in symbols if (kept.get(s) or {}).get("fetched") != today.isoformat()]
    if stale:
        try:
            import yfinance as yf

            df = yf.download(stale, start="1990-01-01", progress=False, auto_adjust=False, threads=False)
            for s in stale:
                col = df["Close"][s].dropna() if s in df["Close"] else None
                if col is not None and len(col):
                    rows = [[d.date().isoformat(), float(v)] for d, v in col.items()]
                    kept[s] = {"fetched": today.isoformat(), "rows": rows}
            store.write_json(NAMES, kept)
        except Exception:  # no network: yesterday's, if any
            pass
    out = {}
    for s in symbols:
        rows = (kept.get(s) or {}).get("rows")
        if not rows:
            continue
        series = pd.Series([v for _, v in rows], index=pd.to_datetime([d for d, _ in rows]))
        node = lineage.record(
            db,
            "raw_snapshot",
            f"name:{s}",
            {"last": rows[-1][1], "last_day": rows[-1][0], "first_day": rows[0][0], "days": len(rows)},
            {"source": "yahoo", "symbol": s, "file": "data/account/names.json"},
            step="fetch",
        )
        out[s] = (series, node)
    return out

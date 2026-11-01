"""What the LLM is given to read: the numbered data (key series today and every signpost, each with its source, so
what it cites can be traced), kept once a day; research's notes from the news on a question; and what its answers
cited, counted by source."""

import json
import logging
import re
from datetime import date

import pandas as pd

from ..core import lineage
from ..data.llm import quoted
from ..model import describe

log = logging.getLogger("board")


SNAPSHOT = [
    "GSPC",
    "IXIC",
    "SOX",
    "BTC",
    "GOLD",
    "VIX",
    "VIX3M",
    "DGS10",
    "DGS3MO",
    "T10Y3M",
    "HY",
    "DFF",
    "DFII10",
    "DXY",
    "OIL",
    "FG",
    "AHR999",
    "UNRATE",
    "SAHM",
    "STABLES",
]


def snapshot_data(df, sp_today, sps, series_cfg):
    """The numbered data an LLM may use: key series today and every signpost's value and state, each with its source
    ("series:GSPC", "signpost:vix_high"), so what a model cites can be traced to it. Not the signposts with a model
    upstream (one the LLM proposed, one on a text it read): the LLM would be told its own work back as evidence."""
    data, day = [], df.index[-1]
    for sid in SNAPSHOT:
        if sid in df and pd.notna(df[sid].get(day)):
            data.append(
                {
                    "name": series_cfg[sid]["name"],
                    "value": describe.figure(float(df[sid][day])),
                    "source": f"series:{sid}",
                }
            )
    for sp in sps:
        t = sp_today.get(sp["id"])
        if t and t["value"] is not None and "proposed" not in sp:
            data.append(
                {
                    "name": f"{sp['name']} (signpost, {t['state']})",
                    "value": describe.figure(t["value"]),
                    "source": f"signpost:{sp['id']}",
                }
            )
    return [{"id": f"D{i + 1}", **d} for i, d in enumerate(data)]


CITE = re.compile(r"\[(D\d+)\]")


def cited(answers, items):
    """{source: how many times the answers' reasons cited it} over every round: the snapshot's items by their source,
    research's notes as "news"; ids the data does not hold are left out."""
    by_id = {
        d["id"]: d.get("source") or ("news" if str(d.get("name", "")).startswith("News note") else None) for d in items
    }
    out = {}
    for a in answers:
        for i in CITE.findall(a.get("reason") or ""):
            if by_id.get(i):
                out[by_id[i]] = out.get(by_id[i], 0) + 1
    return dict(sorted(out.items(), key=lambda x: -x[1]))


def notes(db, qid, first_id):
    """Research's notes from the news on a question, as data items the panel may cite ([D41]), without their
    citations of news items the panel does not see, each fenced as a quote; and their node."""
    row = db.execute(
        "SELECT id, meta, created_at FROM artifact WHERE name = ? ORDER BY id DESC LIMIT 1", (f"research:{qid}",)
    ).fetchone()
    if not row:
        return [], None
    when = date.fromtimestamp(row[2]).isoformat()
    items = [
        {
            "id": f"D{first_id + i}",
            "name": f"News note, research of {when}",
            "value": quoted(re.sub(r"\s*\[D\d+\]", "", n["value"]), 600),
        }
        for i, n in enumerate(json.loads(row[1] or "{}").get("notes") or [])
    ]
    return items, row[0]


def record_snapshot(db, data, today):
    """The numbered data the LLM steps read today, recorded once (its node)."""
    return lineage.record(
        db,
        "data_snapshot",
        "snapshot",
        data,
        {"day": today.isoformat()},
        [lineage.latest(db, f"raw:{s}") for s in SNAPSHOT],
        step="snapshot",
    )

"""The day's news in one message: rules that fired, signposts turned red, headline probabilities that moved, the
LLM's scenarios (a factor's unusual move, a scenario's probability moved ten points), and the
decision step's suggestions new, changed or gone (a lean, a guardrail's move, a trade on what you hold)."""

import json
import logging
import subprocess
from datetime import timedelta
from pathlib import Path

from ..core import store

log = logging.getLogger("board")


ALERT_PTS = 0.10  # a headline probability moving this much in a week is sent once, the day it gets there


def notify(text):
    """One message through the command `notify` in local/settings.toml (it takes the text as its argument);
    nothing without one."""
    cmd = store.settings().get("notify")
    if cmd and Path(cmd).exists():
        subprocess.run([str(cmd), "--quiet", text], check=False)


SUGGESTED = ("lean", "guard", "trade")  # the decision step's suggestions (steps/decide.py)
ALERT_SIZE = 0.01  # a suggestion whose size moves this much since the last run is news again


def suggestions(before, now):
    """The decision step's suggestions that are new, changed by ALERT_SIZE or more, or gone since the page's last
    data: lines. before, now: the actions, the last run's and today's."""
    was = {a["id"]: a for a in before or () if a.get("kind") in SUGGESTED}
    is_ = {a["id"]: a for a in now or () if a.get("kind") in SUGGESTED}
    lines = []
    for i, a in is_.items():
        if i not in was:
            lines.append(f"Suggested: {a['title']}")
        elif abs((a.get("size") or 0) - (was[i].get("size") or 0)) >= ALERT_SIZE:
            lines.append(f"Suggested now: {a['title']} (was {abs(was[i].get('size') or 0):.1%})")
    lines += [f"No longer suggested: {a['title']}" for i, a in was.items() if i not in is_ and a["kind"] != "trade"]
    return lines


def alerts(db, cfg, sps, sp_today, probs, triggers_new, today, futures_news=()):
    """Today's news, one line each: rules that fired, signposts that turned red today, each question's headline
    probability that has moved ALERT_PTS or more in a week (sent on the first day it has), and the scenarios' news
    (futures_news, lines already written)."""
    lines = [f"Rule: {r['title']}" for r in triggers_new] + list(futures_news)
    names = {sp["id"]: sp["name"] for sp in sps}
    lines += [
        f"Signpost red: {names[i]}"
        for i, t in sp_today.items()
        if t["state"] == "red" and t["since"] == today.isoformat()
    ]
    for qid, v in probs.items():
        tmpl = json.loads(db.execute("SELECT spec FROM questions WHERE id = ?", (qid,)).fetchone()[0])["template"]
        head = cfg["questions"].get(tmpl, {}).get("headline")
        if not head:
            continue

        def headline_on(d):
            row = db.execute(
                "SELECT MAX(day) FROM probs WHERE question = ? AND source = 'model' AND day <= ?", (qid, d.isoformat())
            ).fetchone()
            if not row[0]:
                return None
            ps = dict(
                db.execute(
                    "SELECT bucket, p FROM probs WHERE question = ? AND source = 'model' AND day = ?", (qid, row[0])
                ).fetchall()
            )
            return sum(ps.get(b, 0) for b in head["buckets"])

        now, week, yday, yweek = (
            sum(v["model"].get(b, 0) for b in head["buckets"]),
            headline_on(today - timedelta(days=7)),
            headline_on(today - timedelta(days=1)),
            headline_on(today - timedelta(days=8)),
        )
        if week is None or abs(now - week) < ALERT_PTS:
            continue
        if (
            yday is not None
            and yweek is not None
            and abs(yday - yweek) >= ALERT_PTS
            and (yday - yweek) * (now - week) > 0
        ):
            continue  # already sent: the move was there yesterday
        lines.append(f"{head['short']}: {week:.0%} → {now:.0%} in a week")
    return lines

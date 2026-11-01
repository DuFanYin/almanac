"""The board's page: what the daily job writes in data/, shown, and the owner's inputs from it
taken. A small FastAPI app on 127.0.0.1 only, with no sign-in of its own: open it on the machine
(http://127.0.0.1:8083), or put it behind a server that signs you in and forwards a path here unchanged (every link in
the page is relative, so it works at any prefix).

It reads data/scenarios.json (no account numbers), data/account/scenarios-account.json and data/scenarios.db (the
questions, firings and lineage), read only, and writes only data/inputs.db: a rule firing marked done or skipped
(`decisions`), the owner's own probabilities for a question (`judgments`) and a question or factor the LLM proposed,
adopted, rejected or withdrawn (`proposals`); the board reads the judgments and the votes from there (data/account.py).

The page's header (its links to the pages beside it, and the dot home) is [web] in local/settings.toml."""

import asyncio
import hashlib
import html
import json
import math
import sqlite3
import time
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from src.board.core import store

HERE = Path(__file__).parent
STATIC = HERE / "static"
# design/: tokens, components, the chart helper, fonts and icon
DESIGN = HERE / "design"
DESIGN_FILES = ("ui.css", "chart.js", "favicon.png", "fonts/Geist-Variable.woff2", "fonts/GeistMono-Variable.woff2")
# no inline script or style: a strict CSP of its own
CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; font-src 'self'; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)
NO_STORE = {"Cache-Control": "no-store"}
IMMUTABLE = {"Cache-Control": "public, max-age=31536000, immutable"}
# Chrome renders a page in the background while the pointer rests on a link in the header (about 200 ms), so the
# switch is immediate; the rules are a file of their own, named in a header, so the page needs no inline script
SPECULATION = {
    "prerender": [{"source": "document", "where": {"selector_matches": ".ui-tabs a"}, "eagerness": "moderate"}]
}
OWNER_TABLES = """
CREATE TABLE IF NOT EXISTS decisions (trigger TEXT PRIMARY KEY, status TEXT, note TEXT, at INTEGER);
CREATE TABLE IF NOT EXISTS judgments (at INTEGER, question TEXT, probs TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS proposals (id TEXT PRIMARY KEY, status TEXT, note TEXT, at INTEGER);
"""
UPSTREAM = """WITH RECURSIVE up(id, depth) AS (SELECT ?, 0 UNION SELECT e.parent, up.depth + 1 FROM lineage_edge e
              JOIN up ON e.child = up.id WHERE up.depth < 12) """
NOTE_MAX = 300


def board():
    """The board's database, read only."""
    return sqlite3.connect(f"file:{store.DB}?mode=ro", uri=True, timeout=10)


def inputs():
    db = sqlite3.connect(store.INPUTS, timeout=10)
    db.executescript(OWNER_TABLES)
    return db


def load(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def snapshot():
    view = load(store.VIEW)
    if view is None:
        return {"view": None}
    out = {"view": view, "account": load(store.ACCOUNT_VIEW), "decisions": {}, "judgments": {}, "votes": {}}
    if store.INPUTS.exists():
        db = inputs()
        try:
            out["decisions"] = {
                t: {"status": s, "note": n, "at": a} for t, s, n, a in db.execute("SELECT * FROM decisions")
            }
            for at, q, probs, note in db.execute("SELECT at, question, probs, note FROM judgments ORDER BY at"):
                out["judgments"][q] = {"at": at, "probs": json.loads(probs), "note": note}
            out["votes"] = {
                i: {"status": s, "note": n, "at": a} for i, s, n, a in db.execute("SELECT * FROM proposals")
            }
        finally:
            db.close()
    return out


def lineage(node):
    """The node and everything upstream: [{id, kind, name, value, meta, created_at, excluded, parents}]."""
    db = board()
    try:
        rows = db.execute(
            UPSTREAM
            + """SELECT id, kind, name, value, meta, created_at, excluded FROM artifact
                          WHERE id IN (SELECT id FROM up)""",
            (node,),
        ).fetchall()
        ids = [r[0] for r in rows]
        edges = (
            db.execute(
                f"SELECT parent, child, step, code FROM lineage_edge WHERE child IN ({','.join('?' * len(ids))})", ids
            ).fetchall()
            if ids
            else []
        )
    finally:
        db.close()
    return [
        {
            "id": i,
            "kind": k,
            "name": n,
            "value": json.loads(v),
            "meta": json.loads(m),
            "created_at": at,
            "excluded": bool(x),
            "parents": [{"id": p, "step": s, "code": c} for p, ch, s, c in edges if ch == i],
        }
        for i, k, n, v, m, at, x in sorted(rows)
    ]


def decide(body):
    trigger, status, note = body.get("trigger"), body.get("status"), str(body.get("note") or "")[:NOTE_MAX]
    if not isinstance(trigger, str) or len(trigger) > 80 or status not in ("done", "skipped", "open"):
        raise HTTPException(400)
    known = board()
    try:
        if not known.execute("SELECT 1 FROM triggers WHERE id = ?", (trigger,)).fetchone():
            raise HTTPException(404)
    finally:
        known.close()
    db = inputs()
    try:
        if status == "open":
            db.execute("DELETE FROM decisions WHERE trigger = ?", (trigger,))
        else:
            db.execute("INSERT OR REPLACE INTO decisions VALUES (?,?,?,?)", (trigger, status, note, int(time.time())))
        db.commit()
    finally:
        db.close()


def judge(body):
    """The owner's probabilities for an open question: one number per bucket (percent or share), normalised."""
    qid, probs, note = body.get("question"), body.get("probs"), str(body.get("note") or "")[:NOTE_MAX]
    if not isinstance(qid, str) or len(qid) > 80 or not isinstance(probs, dict):
        raise HTTPException(400)
    known = board()
    try:
        row = known.execute(
            "SELECT q.buckets FROM questions q LEFT JOIN resolutions r ON r.question = q.id "
            "WHERE q.id = ? AND r.question IS NULL",
            (qid,),
        ).fetchone()
    finally:
        known.close()
    if not row:
        raise HTTPException(404)
    ids = [b["id"] for b in json.loads(row[0])]
    try:
        vals = {k: float(probs[k]) for k in ids}
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400)
    total = sum(vals.values())
    if set(probs) != set(ids) or any(not math.isfinite(v) or v < 0 for v in vals.values()) or not total:
        raise HTTPException(400)
    db = inputs()
    try:
        db.execute(
            "INSERT INTO judgments VALUES (?,?,?,?)",
            (int(time.time()), qid, json.dumps({k: v / total for k, v in vals.items()}), note),
        )
        db.commit()
    finally:
        db.close()


def vote(body):
    """A question or factor the LLM proposed and the board's checks passed: adopted or rejected (or "open" again,
    until the board's next run takes the vote), one the LLM declined adopted after all, or one adopted (by the owner
    or the LLM) withdrawn."""
    pid, status, note = body.get("id"), body.get("status"), str(body.get("note") or "")[:NOTE_MAX]
    if not isinstance(pid, str) or len(pid) > 80 or status not in ("adopted", "rejected", "open", "withdrawn"):
        raise HTTPException(400)
    known = board()
    try:
        row = known.execute("SELECT status FROM proposals WHERE id = ?", (pid,)).fetchone()
    except sqlite3.OperationalError:  # a board from before proposals
        row = None
    finally:
        known.close()
    waiting = ("open", "declined by the LLM")  # the board's statuses a vote may be cast on
    allowed = {"withdrawn": ("adopted",), "open": (*waiting, "adopted")}.get(status, waiting)
    if not row or row[0] not in allowed:
        raise HTTPException(404)
    db = inputs()
    try:
        if status == "open":
            db.execute("DELETE FROM proposals WHERE id = ?", (pid,))
        else:
            db.execute("INSERT OR REPLACE INTO proposals VALUES (?,?,?,?)", (pid, status, note, int(time.time())))
        db.commit()
    finally:
        db.close()


async def json_body(request):
    """A change's JSON body. Only a script on this site can send one: a form on another site cannot send JSON, and a
    cross-site fetch of it would need CORS, which this app never allows."""
    if not request.headers.get("content-type", "").startswith("application/json") or (
        request.headers.get("sec-fetch-site") == "cross-site"
    ):
        raise HTTPException(403)
    try:
        data = await request.json()
    except ValueError:
        raise HTTPException(400)
    if not isinstance(data, dict):
        raise HTTPException(400)
    return data


def static_version():
    """Hash of every static file's name, size and mtime; changes whenever the front end is edited."""
    h = hashlib.sha1()
    for f in [*(DESIGN / n for n in DESIGN_FILES), *sorted(STATIC.rglob("*"))]:
        if f.is_file():
            st = f.stat()
            h.update(f"{f.name} {st.st_size} {st.st_mtime_ns}".encode())
    return h.hexdigest()[:10]


def header():
    """The header's links ([web] nav in local/settings.toml: [label, address] each, "./" the page itself) and the dot
    home ([web] home), in HTML; none without them."""
    web = store.settings().get("web") or {}
    current = ' aria-current="page"'
    links = "".join(
        f'<a href="{html.escape(href)}"{current if href == "./" else ""}>{html.escape(label)}</a>'
        for label, href in web.get("nav") or []
    )
    out = f'<nav class="ui-tabs" aria-label="Pages">{links}</nav>' if links else ""
    if home := web.get("home"):
        out += (
            f'<a class="ui-home" href="{html.escape(home)}" aria-label="Home" title="{html.escape(home)}">'
            '<svg width="11" height="11" viewBox="0 0 10 10" aria-hidden="true">'
            '<circle cx="5" cy="5" r="5" fill="currentColor"/></svg></a>'
        )
    return out


def page(initial):
    """The page with versioned asset URLs (s/<version>/..., so no cache serves an old stylesheet or script; module
    imports inherit it by their relative paths), its header and its first data embedded (the first render waits for
    no request)."""
    text = (STATIC / "index.html").read_text().replace('="static/', f'="s/{static_version()}/')
    text = text.replace("<!--header-->", header())
    data = json.dumps(initial, separators=(",", ":")).replace("<", "\\u003c")
    text = text.replace(
        '<script id="initial" type="application/json">null</script>',
        f'<script id="initial" type="application/json">{data}</script>',
    )
    return HTMLResponse(
        text,
        headers={
            "Cache-Control": "no-cache",
            "Speculation-Rules": '"speculation.json"',
            "Content-Security-Policy": CSP,
        },
    )


def router():
    r = APIRouter()

    @r.get("/")
    async def index():
        return page(await asyncio.to_thread(snapshot))

    @r.get("/api/scenarios")
    async def scenarios_data():
        return JSONResponse(await asyncio.to_thread(snapshot), headers=NO_STORE)

    @r.get("/api/scenarios/lineage/{node}")
    async def scenarios_lineage(node: int):
        if not store.DB.exists():
            raise HTTPException(404)
        return JSONResponse(await asyncio.to_thread(lineage, node), headers=NO_STORE)

    @r.get("/api/scenarios/lineage-by-name")
    async def scenarios_lineage_by_name(name: str):
        """The latest node of that name (a rule firing: trigger:<rule>:<period>)."""
        if not store.DB.exists() or len(name) > 120:
            raise HTTPException(404)
        db = board()
        try:
            row = db.execute(
                "SELECT id FROM artifact WHERE name = ? ORDER BY seen DESC, id DESC LIMIT 1", (name,)
            ).fetchone()
        finally:
            db.close()
        if not row:
            raise HTTPException(404)
        return JSONResponse(await asyncio.to_thread(lineage, row[0]), headers=NO_STORE)

    @r.post("/api/scenarios/decision")
    async def scenarios_decision(request: Request):
        await asyncio.to_thread(decide, await json_body(request))
        return JSONResponse({"ok": True}, headers=NO_STORE)

    @r.post("/api/scenarios/judgment")
    async def scenarios_judgment(request: Request):
        await asyncio.to_thread(judge, await json_body(request))
        return JSONResponse({"ok": True}, headers=NO_STORE)

    @r.post("/api/scenarios/proposal")
    async def scenarios_proposal(request: Request):
        await asyncio.to_thread(vote, await json_body(request))
        return JSONResponse({"ok": True}, headers=NO_STORE)

    @r.get("/sw.js")
    def service_worker():
        """The page's service worker (static/sw.js): at the root, so its scope is the page."""
        return FileResponse(STATIC / "sw.js", media_type="text/javascript", headers={"Cache-Control": "no-cache"})

    @r.get("/speculation.json")
    def speculation():
        return JSONResponse(
            SPECULATION,
            media_type="application/speculationrules+json",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @r.get("/favicon.ico")
    def favicon():
        return FileResponse(
            DESIGN / "favicon.png", media_type="image/png", headers={"Cache-Control": "public, max-age=86400"}
        )

    @r.get("/s/{version}/{path:path}")
    def versioned_static(version: str, path: str):
        if path in DESIGN_FILES:
            return FileResponse(DESIGN / path, headers=IMMUTABLE)
        f = (STATIC / path).resolve()
        if not f.is_file() or not f.is_relative_to(STATIC.resolve()):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        return FileResponse(f, headers=IMMUTABLE)

    return r


app = FastAPI(title="almanac", docs_url=None, redoc_url=None, openapi_url=None, redirect_slashes=False)
app.include_router(router())

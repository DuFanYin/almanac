"""The page (src/web/app.py): it reads the board's data folder, read only, and writes only the owner's inputs
(data/inputs.db); the page itself carries its header from local/settings.toml, its first data, versioned assets and a
strict CSP. A throwaway board folder; no network, no account data."""

import json
import sqlite3

from src.board.core import store
from src.web import app as web
from starlette.testclient import TestClient
from tests.board.support import check


def board():
    store.DATA.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(store.DB)
    db.executescript("""
    DROP TABLE IF EXISTS questions; DROP TABLE IF EXISTS resolutions; DROP TABLE IF EXISTS triggers;
    DROP TABLE IF EXISTS artifact; DROP TABLE IF EXISTS lineage_edge; DROP TABLE IF EXISTS proposals;
    CREATE TABLE questions (id TEXT PRIMARY KEY, title TEXT, kind TEXT, opened TEXT, closes TEXT, buckets TEXT,
        spec TEXT, config TEXT);
    CREATE TABLE resolutions (question TEXT PRIMARY KEY, day TEXT, bucket TEXT, how TEXT);
    CREATE TABLE triggers (id TEXT PRIMARY KEY, rule TEXT, period TEXT, day TEXT);
    CREATE TABLE artifact (id INTEGER PRIMARY KEY, kind TEXT, name TEXT, value TEXT, content_hash TEXT,
        created_at INTEGER, meta TEXT, excluded INTEGER DEFAULT 0, seen INTEGER);
    CREATE TABLE lineage_edge (parent INTEGER, child INTEGER, step TEXT, code TEXT, PRIMARY KEY (parent, child));
    CREATE TABLE proposals (id TEXT PRIMARY KEY, day TEXT, spec TEXT, checks TEXT, passed INTEGER, status TEXT,
        node INTEGER, decided TEXT);
    """)
    for row in (
        ("cpi_2y", "open", None),
        ("bad_q", "failed", None),
        ("llm_q", "adopted", "2026-11-01"),
        ("no_q", "declined by the LLM", "2026-11-01"),
    ):
        db.execute("INSERT INTO proposals VALUES (?,'2026-10-29','{}','{}',1,?,NULL,?)", row)
    db.execute(
        "INSERT INTO questions VALUES ('q1','Q','path','2026-10-29','2028-10-31',?, '{}', 'x')",
        (json.dumps([{"id": "a", "label": "A"}, {"id": "b", "label": "B"}]),),
    )
    db.execute("INSERT INTO triggers VALUES ('r1:2026-11','r1','2026-11','2026-11-02')")
    db.execute("INSERT INTO artifact VALUES (1,'raw_snapshot','raw:X','{}','h1',0,'{}',0,1)")
    db.execute("INSERT INTO artifact VALUES (2,'signpost','signpost:s','{}','h2',0,'{}',0,2)")
    db.execute("INSERT INTO lineage_edge VALUES (1,2,'signpost','c')")
    db.commit()
    db.close()
    store.VIEW.write_text(json.dumps({"questions": [], "triggers": []}))
    store.INPUTS.unlink(missing_ok=True)


def test_the_owner_s_inputs_checked_and_kept_apart_from_the_board():
    """A judgment normalised and on a known open question only; a firing marked on a known firing only; a vote cast
    only on a proposal the board's checks passed and its status allows; JSON only; the inputs in data/inputs.db,
    never in the board's database; lineage served."""
    board()
    cl = TestClient(web.app)
    get = lambda: cl.get("/api/scenarios").json()  # noqa: E731
    post = lambda path, **kw: cl.post(f"/api/scenarios/{path}", **kw).status_code  # noqa: E731
    check("the page's data comes from the board's folder", get()["view"] == {"questions": [], "triggers": []})
    check(
        "judgment saved, normalised",
        post("judgment", json={"question": "q1", "probs": {"a": 30, "b": 10}}) == 200
        and abs(get()["judgments"]["q1"]["probs"]["a"] - 0.75) < 1e-9,
    )
    check("judgment with wrong buckets refused", post("judgment", json={"question": "q1", "probs": {"a": 1}}) == 400)
    check(
        "judgment on an unknown question refused", post("judgment", json={"question": "zz", "probs": {"a": 1}}) == 404
    )
    check(
        "decision saved, its note cut",
        post("decision", json={"trigger": "r1:2026-11", "status": "skipped", "note": "x" * 999}) == 200
        and len(get()["decisions"]["r1:2026-11"]["note"]) == web.NOTE_MAX,
    )
    check("decision needs a known firing", post("decision", json={"trigger": "nope", "status": "done"}) == 404)
    check("form posts refused", post("decision", data={"trigger": "r1:2026-11"}) == 403)
    check(
        "a cross-site post refused",
        post("decision", json={"trigger": "r1:2026-11", "status": "done"}, headers={"sec-fetch-site": "cross-site"})
        == 403,
    )
    check(
        "a proposal adopted, then open again",
        post("proposal", json={"id": "cpi_2y", "status": "adopted"}) == 200
        and get()["votes"]["cpi_2y"]["status"] == "adopted"
        and post("proposal", json={"id": "cpi_2y", "status": "open"}) == 200
        and "cpi_2y" not in get()["votes"],
    )
    check(
        "one adopted withdrawn; one open cannot be",
        post("proposal", json={"id": "llm_q", "status": "withdrawn"}) == 200
        and post("proposal", json={"id": "cpi_2y", "status": "withdrawn"}) == 404,
    )
    check("one the LLM declined adopted after all", post("proposal", json={"id": "no_q", "status": "adopted"}) == 200)
    check(
        "one that failed its checks, an unknown one, another status: refused",
        post("proposal", json={"id": "bad_q", "status": "adopted"}) == 404
        and post("proposal", json={"id": "zz", "status": "adopted"}) == 404
        and post("proposal", json={"id": "cpi_2y", "status": "maybe"}) == 400,
    )
    check("lineage served", [n["name"] for n in cl.get("/api/scenarios/lineage/2").json()] == ["raw:X", "signpost:s"])
    check(
        "the owner's inputs in data/inputs.db, not the board's database",
        sqlite3.connect(store.INPUTS).execute("SELECT COUNT(*) FROM judgments").fetchone()[0] == 1
        and "judgments" not in {r[0] for r in sqlite3.connect(store.DB).execute("SELECT name FROM sqlite_master")},
    )


def test_the_page_its_header_assets_and_csp(monkeypatch):
    """The page: its first data in it, its assets at versioned addresses that exist, its header from [web] in the
    settings (the page itself current, labels escaped), a strict CSP; nothing outside its folders served."""
    board()
    monkeypatch.setattr(
        store,
        "settings",
        lambda: {"web": {"nav": [["Status", "../"], ["Scenarios", "./"], ["<b>", "x"]], "home": "https://h/"}},
    )
    cl = TestClient(web.app)
    r = cl.get("/")
    text = r.text
    check("the first data in the page", '"view":{"questions":[],"triggers":[]}' in text)
    check(
        "the header from the settings, the page itself current, labels escaped",
        '<a href="../">Status</a>' in text
        and '<a href="./" aria-current="page">Scenarios</a>' in text
        and "&lt;b&gt;" in text
        and 'class="ui-home" href="https://h/"' in text,
    )
    check("a strict CSP, no inline script", "script-src 'self'" in r.headers["content-security-policy"])
    v = web.static_version()
    assets = [
        a
        for a in (
            "ui.css",
            "chart.js",
            "scenarios.css",
            "js/scenarios/main.js",
            "js/dom.js",
            "fonts/Geist-Variable.woff2",
        )
    ]
    check(
        "its assets at versioned addresses, each there",
        f'href="s/{v}/ui.css"' in text and all(cl.get(f"/s/{v}/{a}").status_code == 200 for a in assets),
    )
    check("nothing outside its folders", cl.get(f"/s/{v}/../app.py").status_code == 404)
    monkeypatch.setattr(store, "settings", lambda: {})
    check("no header links without the settings", 'class="ui-tabs"' not in cl.get("/").text)
    check("its service worker at its root", cl.get("/sw.js").status_code == 200)

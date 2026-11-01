"""Where the scenario board keeps things, and its database.

The code and the generic definitions (src/board/config/) are in git. Two folders at the repository's root stay
out of it:
  local/   the owner's own: rules.toml (the style, the categories and their targets, each symbol's category,
           the rules and their pace; without it src/board/config/rules.example.toml), settings.toml (where the
           positions and the owner's page inputs are, outside this repository), questions.toml (the questions the
           LLM proposed and the owner adopted on the page, written by the job; read beside config/questions.toml)
  data/    what the job makes (mode 700): scenarios.db, scenarios.json (the page's model), timescales.json;
           data/account/ the account part (holdings against targets, amounts, the holding under each outcome,
           the red team, the positions each firing saw)
Without a positions file named in local/settings.toml the board runs without holdings: questions, signposts,
the world, and each asset class at each length.

data/scenarios.db (SQLite, a rollback journal), written by the daily job (run.py) only:
  observations   every value of every series as it was seen: (series, day, value, seen_at); a revised
                 value is a new row, so a day can be computed from what was known then
  questions      each question as registered: its buckets and spec frozen at registration
  probs          each source's probability of each bucket, every day, with the config's hash and its lineage node
                 (also the pool, the model, and the model without the LLM; a factor forecast's: math, the data's;
                 futures, what the LLM's futures imply; no_llm, what history's futures imply; model, the mix)
  rounds         each round of drafts the LLM was asked for (questions, signposts, changes), kept or not: their cadence
                 (the share they took is min(0.95, weight x sum)), for the weight's fit on what resolved
  outlook        each day, each length the world is read at and each factor: history's quartiles over that length
                 (edges), the world's probability of each ({source: [p]}: with the LLM, without it, the data's,
                 history's, and with the LLM at each weight tried), and once the window is over its outcome
  trades         each trade the decision step suggested on what you hold: the category, its size, the day and the
                 share held then (done once the share has moved half of it); no amounts
  anchors        each category's decided share the last time you held within half a step of it (model/decide.py)
  signposts      each signpost's state every day
  resolutions    a question's outcome once it is known
  triggers       each rule firing (rule, period); no amounts: those are account data (data/account/)
  rule_days      whether each rule held, each day (an event rule fires again only after a run of days without)
  versions       each version of the rules, of the scenarios' writing and of research, and each question proposed,
                 the day it was first used: how many were tried
  proposals      each question the LLM proposed: its spec, its checks, whether it passed, and what the owner
                 decided (open, adopted, rejected)
  scenario_sets  each length's scenarios as the LLM wrote them (the world's last layer): their stories, ranges and
                 weights (each reading's and the median kept), the simulation's share of each when written, live or
                 replaced, and how the set turned out (which scenario happened)
  signpost_proposals   each signpost the LLM proposed, and each on a reading: its spec, its checks, where it stands
                 on history and on what came after (accumulating, failed, trial, active, dropped)
  readings       each text the LLM read into a number: the series, the day published, its title and link, the number
                 (None: not relevant), the spread of the answers, their quotes and reasons
  texts          the weekly brief; the red team's check only (its text names account shares: it is in data/account/)
  artifact, lineage_edge   what each product was made from (lineage.py)
The owner's inputs from the page (a firing marked done or skipped, their own probabilities, a proposed question
adopted or rejected) are written by the page's server into its own database (settings.toml `inputs`); the job only
reads them."""

import hashlib
import json
import os
import sqlite3
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repository
PACKAGE = Path(__file__).resolve().parents[1]  # src/board
CONFIG = Path(os.environ.get("BOARD_CONFIG", PACKAGE / "config"))
LOCAL = Path(os.environ.get("BOARD_LOCAL", ROOT / "local"))
DATA = Path(os.environ.get("BOARD_DATA", ROOT / "data"))
ACCOUNT = DATA / "account"
DB = DATA / "scenarios.db"
VIEW = DATA / "scenarios.json"  # the page's model, no account numbers
ACCOUNT_VIEW = ACCOUNT / "scenarios-account.json"  # the page's account part
INPUTS = DATA / "inputs.db"  # the owner's inputs from the page (src/web): marks on firings, own views, votes


def settings():
    """local/settings.toml, its paths expanded; {} without it."""
    try:
        raw = tomllib.loads((LOCAL / "settings.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return {k: Path(v).expanduser() if isinstance(v, str) and v.startswith(("/", "~")) else v for k, v in raw.items()}


SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (series TEXT, day TEXT, value REAL, seen_at INTEGER,
    PRIMARY KEY (series, day, seen_at));
CREATE TABLE IF NOT EXISTS questions (id TEXT PRIMARY KEY, title TEXT, kind TEXT, opened TEXT, closes TEXT,
    buckets TEXT, spec TEXT, config TEXT);
CREATE TABLE IF NOT EXISTS probs (day TEXT, question TEXT, bucket TEXT, source TEXT, p REAL, config TEXT,
    artifact INTEGER,
    PRIMARY KEY (day, question, bucket, source));
CREATE TABLE IF NOT EXISTS signposts (day TEXT, id TEXT, value REAL, state TEXT, PRIMARY KEY (day, id));
CREATE TABLE IF NOT EXISTS resolutions (question TEXT PRIMARY KEY, day TEXT, bucket TEXT, how TEXT);
CREATE TABLE IF NOT EXISTS triggers (id TEXT PRIMARY KEY, rule TEXT, period TEXT, day TEXT);
CREATE TABLE IF NOT EXISTS rule_days (day TEXT, rule TEXT, holds INTEGER, PRIMARY KEY (day, rule));
CREATE TABLE IF NOT EXISTS versions (kind TEXT, hash TEXT, first_day TEXT, PRIMARY KEY (kind, hash));
CREATE TABLE IF NOT EXISTS texts (day TEXT, kind TEXT, body TEXT, meta TEXT, PRIMARY KEY (day, kind));
CREATE TABLE IF NOT EXISTS artifact (id INTEGER PRIMARY KEY, kind TEXT, name TEXT, value TEXT, content_hash TEXT,
    created_at INTEGER, meta TEXT, excluded INTEGER DEFAULT 0, seen INTEGER);
CREATE INDEX IF NOT EXISTS artifact_name ON artifact (name, content_hash);
CREATE TABLE IF NOT EXISTS lineage_edge (parent INTEGER, child INTEGER, step TEXT, code TEXT,
    PRIMARY KEY (parent, child));
CREATE INDEX IF NOT EXISTS lineage_child ON lineage_edge (child);
CREATE TABLE IF NOT EXISTS proposals (id TEXT PRIMARY KEY, day TEXT, spec TEXT, checks TEXT, passed INTEGER,
    status TEXT, node INTEGER, decided TEXT);
CREATE TABLE IF NOT EXISTS scenario_sets (id TEXT PRIMARY KEY, horizon TEXT, written TEXT, expires TEXT, spec TEXT,
    status TEXT, node INTEGER, how TEXT);
CREATE TABLE IF NOT EXISTS signpost_proposals (id TEXT PRIMARY KEY, day TEXT, spec TEXT, checks TEXT, status TEXT,
    node INTEGER, judged TEXT);
CREATE TABLE IF NOT EXISTS readings (id TEXT, series TEXT, day TEXT, read_at INTEGER, title TEXT, url TEXT,
    value REAL, spread REAL, detail TEXT, node INTEGER, PRIMARY KEY (series, id));
CREATE TABLE IF NOT EXISTS rounds (kind TEXT, day TEXT, PRIMARY KEY (kind, day));
CREATE TABLE IF NOT EXISTS outlook (day TEXT, days INTEGER, factor TEXT, edges TEXT, probs TEXT, outcome INTEGER,
    value REAL, PRIMARY KEY (day, days, factor));
CREATE TABLE IF NOT EXISTS trades (id TEXT PRIMARY KEY, category TEXT, size REAL, day TEXT, held REAL, node INTEGER);
CREATE TABLE IF NOT EXISTS anchors (category TEXT PRIMARY KEY, share REAL, day TEXT);
"""


def connect(path=None):
    DATA.mkdir(mode=0o700, parents=True, exist_ok=True)
    db = sqlite3.connect(path or DB, timeout=30)
    # a rollback journal, not WAL: the page's server opens the database read only from a read-only home
    # (systemd ProtectHome), and a WAL database cannot be opened without writing its -shm file
    db.execute("PRAGMA journal_mode=DELETE")
    db.executescript(SCHEMA)
    if "seen" not in {r[1] for r in db.execute("PRAGMA table_info(artifact)")}:  # databases made before it
        db.execute("ALTER TABLE artifact ADD COLUMN seen INTEGER")
        db.execute("UPDATE artifact SET seen = id")
        db.commit()
    # a factor forecast's probability from the LLM's futures was kept as "llm", the panel's name, before 2026-10-30
    if db.execute(
        "SELECT 1 FROM probs WHERE source = 'llm' AND question IN (SELECT id FROM questions WHERE kind = 'factor') "
        "LIMIT 1"
    ).fetchone():
        db.execute(
            "UPDATE probs SET source = 'futures' WHERE source = 'llm' "
            "AND question IN (SELECT id FROM questions WHERE kind = 'factor')"
        )
        db.commit()
    return db


def rules_file():
    """The owner's rules (local/rules.toml), else the example."""
    own = LOCAL / "rules.toml"
    return own if own.exists() else CONFIG / "rules.example.toml"


ADOPTED = LOCAL / "questions.toml"  # the questions the owner adopted from the LLM's proposals
ADOPTED_FACTORS = LOCAL / "factors.toml"  # the factors the owner adopted from the LLM's proposals


def config():
    """Every config file by name (series, questions, signposts, asset_classes, rules), and their hash: a
    probability or a firing records the hash, so it is known which version of the definitions made it. The
    questions the owner adopted (local/questions.toml) join config/questions.toml."""
    out, h = {}, hashlib.sha1()
    for f in sorted([*(x for x in CONFIG.glob("*.toml") if x.name != "rules.example.toml"), rules_file()]):
        raw = f.read_bytes()
        h.update(f.name.encode() + raw)
        out["rules" if f == rules_file() else f.stem] = tomllib.loads(raw.decode())
    if ADOPTED.exists():
        raw = ADOPTED.read_bytes()
        h.update(b"adopted" + raw)
        adopted = tomllib.loads(raw.decode())
        if set(adopted) & set(out["questions"]):
            raise ValueError(f"local/questions.toml: {sorted(set(adopted) & set(out['questions']))} already asked")
        out["questions"] |= adopted
    if ADOPTED_FACTORS.exists():  # [factor.<id>] tables, joined to config/factors.toml's
        raw = ADOPTED_FACTORS.read_bytes()
        h.update(b"adopted factors" + raw)
        more = tomllib.loads(raw.decode()).get("factor", {})
        if set(more) & set(out["factors"]["factor"]):
            raise ValueError(f"local/factors.toml: {sorted(set(more) & set(out['factors']['factor']))} already factors")
        out["factors"]["factor"] |= more
    profile = out["rules"].get("profile")
    if profile is not None and str(profile) not in out.get("asset_classes", {}).get("profile", {}):
        raise ValueError(f"rules: profile {profile!r} is not one of the styles in config/asset_classes.toml")
    classes = out.get("asset_classes", {}).get("class", {})
    for k, c in out["rules"].get("categories", {}).items():
        if c.get("class") not in classes:
            raise ValueError(
                f"rules: category {k} names no asset class of config/asset_classes.toml ({c.get('class')!r})"
            )
    for name, w in out.get("asset_classes", {}).get("baseline", {}).items():
        if set(w) - set(classes):
            raise ValueError(f"asset_classes: baseline {name} names unknown classes {sorted(set(w) - set(classes))}")
    for n, pr in out.get("asset_classes", {}).get("profile", {}).items():
        w = pr.get("policy") or {}
        if set(w) - set(classes) or abs(sum(w.values()) - 1) > 1e-6 or w.get("cash", 0) < pr["cash_min"] - 1e-9:
            raise ValueError(f"asset_classes: style {n}'s policy must name known classes, sum to 1 and hold cash_min")
    return out, h.hexdigest()[:10]


def read_json(path):
    """A JSON file this job wrote; None without it."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def timescales():
    """The last measures of how long each signpost's news lasts (data/timescales.json, written by run.py
    timescales); {} without them."""
    return read_json(DATA / "timescales.json") or {}


def write_json(path, obj):
    """Atomically (temp file, rename); mode 600 under data/account/ (account data), its folders 700."""
    private = path.resolve().is_relative_to(ACCOUNT.resolve())
    path.parent.mkdir(mode=0o700 if private else 0o755, parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600 if private else 0o644)
    with open(fd, "w") as f:
        json.dump(obj, f, separators=(",", ":"), ensure_ascii=False)
    tmp.replace(path)

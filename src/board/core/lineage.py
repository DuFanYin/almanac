"""Lineage: every product (a fetched snapshot, a signpost, a base rate, a market price, an LLM answer, a
judgment, a probability, a rule firing) is a node; an edge says what it was made from.

A node's content hash covers its value and its parents, so a product that did not change keeps its node
(one node per change, not per run). Each edge records the code that made it: the hash of the board's source,
its config, the owner's rules and the model layer it asks (src/llm), so uncommitted edits count.
Append only. A node marked excluded (bad data) taints everything downstream: scoring leaves those out. A node made
by a model (LLM_KINDS) marks everything downstream as having the LLM upstream (llm_upstream): the computed side keeps
such inputs out of its control without the LLM, so `no_llm` stays clean wherever the LLM enters."""

import hashlib
import json
import time

from .store import CONFIG, PACKAGE, ROOT, rules_file

LLM = ROOT / "src" / "llm"


def code_hash():
    h = hashlib.sha1()
    for f in sorted({*PACKAGE.rglob("*.py"), *CONFIG.glob("*.toml"), rules_file(), *LLM.rglob("*.py")}):
        if f.exists():
            h.update(str(f.relative_to(ROOT) if f.is_relative_to(ROOT) else f.name).encode() + f.read_bytes())
    return h.hexdigest()[:10]


CODE = code_hash()


def digest(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def record(db, kind, name, value, meta=None, parents=(), step=None):
    """The node's id: the existing one when this name already has this content, else a new node with its
    edges. `parents` are node ids (None skipped)."""
    parents = sorted({p for p in parents if p})
    h = digest([value, parents])
    row = db.execute(
        "SELECT id FROM artifact WHERE name = ? AND content_hash = ? ORDER BY id DESC LIMIT 1", (name, h)
    ).fetchone()
    if row:  # the same content again (after A, B, A: A's node), now the latest under its name
        db.execute("UPDATE artifact SET seen = ? WHERE id = ?", (time.time_ns(), row[0]))
        return row[0]
    cur = db.execute(
        "INSERT INTO artifact (kind, name, value, content_hash, created_at, meta, seen) VALUES (?,?,?,?,?,?,?)",
        (
            kind,
            name,
            json.dumps(value, default=str),
            h,
            int(time.time()),
            json.dumps(meta or {}, default=str, ensure_ascii=False),
            time.time_ns(),
        ),
    )
    child = cur.lastrowid
    db.executemany(
        "INSERT OR IGNORE INTO lineage_edge (parent, child, step, code) VALUES (?,?,?,?)",
        [(p, child, step or kind, CODE) for p in parents],
    )
    return child


def latest(db, name):
    """The node last recorded under a name (by when it was last recorded, not when it was first made)."""
    row = db.execute("SELECT id FROM artifact WHERE name = ? ORDER BY seen DESC, id DESC LIMIT 1", (name,)).fetchone()
    return row[0] if row else None


def value(db, node):
    """A node's value; None without the node."""
    row = db.execute("SELECT value FROM artifact WHERE id = ?", (node,)).fetchone() if node else None
    return json.loads(row[0]) if row else None


UPSTREAM = """WITH RECURSIVE up(id, depth) AS (SELECT ?, 0 UNION SELECT e.parent, up.depth + 1 FROM lineage_edge e
              JOIN up ON e.child = up.id WHERE up.depth < 12) """


def tree(db, node):
    """The node and everything upstream of it: [{id, kind, name, value, meta, created_at, excluded, parents}]."""
    rows = db.execute(
        UPSTREAM
        + """SELECT a.id, a.kind, a.name, a.value, a.meta, a.created_at, a.excluded
                      FROM artifact a WHERE a.id IN (SELECT id FROM up)""",
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
    out = []
    for i, kind, name, value, meta, at, excluded in rows:
        out.append(
            {
                "id": i,
                "kind": kind,
                "name": name,
                "value": json.loads(value),
                "meta": json.loads(meta),
                "created_at": at,
                "excluded": bool(excluded),
                "parents": [{"id": p, "step": s, "code": c} for p, ch, s, c in edges if ch == i],
            }
        )
    return out


def tainted(db, node):
    """True when the node or anything upstream of it is marked excluded."""
    return bool(
        db.execute(
            UPSTREAM + "SELECT 1 FROM artifact WHERE id IN (SELECT id FROM up) AND excluded = 1 LIMIT 1", (node,)
        ).fetchone()
    )


def exclude(db, name, since=None, until=None):
    """Marks the nodes of `name` (created between since and until, epoch seconds) as bad data."""
    q, args = "UPDATE artifact SET excluded = 1 WHERE name = ?", [name]
    if since:
        q, args = q + " AND created_at >= ?", args + [since]
    if until:
        q, args = q + " AND created_at <= ?", args + [until]
    return db.execute(q, args).rowcount


def version(db, kind, h, day):
    """Notes the version `h` of `kind` (the rules, the forecast method) as in use on `day`, once: each one tried
    is counted, so a result can be read against how many versions it took."""
    db.execute("INSERT OR IGNORE INTO versions VALUES (?,?,?)", (kind, h, day.isoformat()))


def versions(db):
    """{kind: {"n": versions tried, "first": the first's day, "last": the newest's day}}."""
    return {
        k: {"n": n, "first": a, "last": b}
        for k, n, a, b in db.execute(
            "SELECT kind, COUNT(*), MIN(first_day), MAX(first_day) FROM versions GROUP BY kind"
        )
    }


# what a model made: the panel, the old priors, research, the old stress scenarios, proposed questions, the named
# futures, proposed signposts, each text it read into a number (llm_reading) and the series made of them (llm_series)
LLM_KINDS = (
    "llm_panel",
    "llm_prior",
    "research",
    "stress",
    "proposal",
    "llm_series",
    "llm_scenario",
    "llm_signpost",
    "llm_reading",
)


def llm_upstream(db, node, cache=None):
    """The names of the model-made nodes at or upstream of a node ([] when there are none), memoised in `cache`."""
    if not node:
        return []
    if cache is not None and node in cache:
        return cache[node]
    marks = ",".join("?" * len(LLM_KINDS))
    names = [
        r[0]
        for r in db.execute(
            UPSTREAM + f"SELECT name FROM artifact WHERE id IN (SELECT id FROM up) AND kind IN ({marks})",
            (node, *LLM_KINDS),
        )
    ]
    if cache is not None:
        cache[node] = names
    return names


def model_answer(db, name, kind=None):
    """(summary, node, when) of the latest model-written answer kept under a name (llm:<question>,
    research:<question>), of that kind when given (the panel's llm_panel, not the monthly priors that came before it
    under the same name); Nones without one."""
    row = db.execute(
        "SELECT id, value, created_at FROM artifact WHERE name = ? AND kind = coalesce(?, kind) "
        "ORDER BY id DESC LIMIT 1",
        (name, kind),
    ).fetchone()
    return (json.loads(row[1]), row[0], row[2]) if row else (None, None, None)

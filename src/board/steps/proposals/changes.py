"""New factors the LLM proposes for the futures (config/factors.toml), kept beside the proposed questions (the
proposals table, spec "type" factor; adopted the same way: votes.py): checked on the data (model/proposals.py
check_factor: a history from 1971, not a copy of one, the asset classes' fit explains more out of sample with it);
adopted, it is written to local/factors.toml and fitted and given to the LLM's futures from the next run. (Changes to
the rule conditions were proposed here too until 10-31: the rules that read forecasts retired.)

Every EVERY days, up to PER_ROUND. A draft with an id proposed before is not checked again."""

import json
import logging

from ...core import lineage, store
from ...data import llm
from ...model import describe
from ...model import proposals as checks
from . import common
from .common import keep

log = logging.getLogger("board")

EVERY = 28  # days between drafts
PER_ROUND = 2  # of each kind


def due(db, today):
    row = db.execute("SELECT max(day) FROM proposals WHERE json_extract(spec, '$.type') = 'factor'").fetchone()
    return common.due(db, "changes", EVERY, today, row[0])


def run(db, cfg, df, series, data, today, force=False):
    """Drafts up to PER_ROUND new factors when EVERY days have passed (or force), checks each and keeps it. How many
    passed."""
    if not force and not due(db, today):
        return 0
    tried = {r[0] for r in db.execute("SELECT id FROM proposals")}
    sers = describe.series_list(cfg, series)
    parents = [lineage.latest(db, "snapshot")]
    fcfg, passed = cfg["factors"], 0
    first = int(fcfg["history_from"][:4]) + 1  # a factor's series must go back this far: only those are offered
    out = llm.propose_factors(
        today=today.isoformat(),
        existing=[f"{k}: {f['label']}, {f['series']}, {f['measure']}" for k, f in fcfg["factor"].items()],
        series=[s for s in sers if s["since"] <= first],
        data=data,
        n=PER_ROUND,
    )
    common.asked(db, "changes", today)
    for f in out["factors"]:
        pid = f"factor_{f['id']}"
        if pid in tried:
            continue
        res = checks.check_factor(f, fcfg, series, df, cfg["asset_classes"]["class"])
        keep(db, pid, {**f, "type": "factor", "title": f"New factor: {f['label']}"}, res, out, parents, today)
        passed += res["passed"]
    db.commit()
    log.info("changes proposed: %d passed", passed)
    return passed


# ---- adopting


def toml_value(v):
    """A value in TOML: strings and lists as JSON writes them (TOML reads those), booleans in lower case."""
    if isinstance(v, bool):
        return "true" if v else "false"
    return json.dumps(v, ensure_ascii=True)


def adopt_factor(spec, today=None, by=None):
    """The factor appended to local/factors.toml (adopted_by = "llm" when the LLM adopted it)."""
    path = store.ADOPTED_FACTORS
    text = (
        path.read_text()
        if path.exists()
        else (
            "# The factors the owner adopted from the LLM's proposals (written by the board's job,\n"
            "# src/board/steps/proposals/changes.py), joined to src/board/config/factors.toml's.\n"
        )
    )
    lines = (
        [f"\n[factor.{spec['id']}]"]
        + [f"{k} = {toml_value(spec[k])}" for k in ("label", "series", "measure", "unit", "up") if k in spec]
        + ([f"adopted_by = {toml_value(by)}"] if by else [])
    )
    path.write_text(text + "\n".join(lines) + "\n")
    return {"file": str(path.name)}

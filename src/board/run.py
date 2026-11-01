"""The scenario board's daily job and its by-hand commands.

usage (from the repository's root; `uv run` or .venv/bin/python):
  python -m src.board.run                 the day: fetch, signposts, the world, questions, the decision,
                                                    scores, the page's data; the LLM's scenarios and proposed
                                                    questions weekly, the brief on Saturdays
  python -m src.board.run --no-llm        the same without any LLM call
  python -m src.board.run --no-fetch      the same on the data already kept (the model alone)
  python -m src.board.run --brief         write the weekly brief and the red team now
  python -m src.board.run --research      research the week's key questions now (else every 28 days)
  python -m src.board.run --propose       ask the LLM to propose new questions now (else every week)
  python -m src.board.run --scenarios     ask the LLM for the scenarios of every length now (else every week)
  python -m src.board.run --signposts     ask the LLM to propose signposts now (else every four weeks)
  python -m src.board.run --changes       ask the LLM to propose factors now
  python -m src.board.run exclude NAME [SINCE [UNTIL]]   mark a node as bad data (ISO days)
  python -m src.board.run withdraw ID WHY  take an open question out: no longer computed or scored
  python -m src.board.run lineage ID      print a node's lineage tree
  python -m src.board.run timescales      how long each signpost's news lasts, a week to three years, on
                                                    every factor (data/timescales.json: the world reads it); also
                                                    every Saturday
  python -m src.board.run walk            the decision step walked through history since 1981 (data/walk.json:
                                                    its sense and lengths' weights, chosen on the first half); also
                                                    the first Saturday of a month (minutes; a few more the first
                                                    time, measuring the signposts year by year)

The package is in layers, each reviewable on its own (tests/board/test_board_layers.py checks the imports):
  core/    the database and config (store), lineage, the expression language
  data/    getting values in and keeping them: sources, Kalshi, the observations and their backfills, the
           account's files, the LLM (src/llm's methods) and research (src/research); no model logic
  model/   pure functions from data to results: questions, probabilities, the portfolio, the rules; no
           database, network or files
  steps/   one module per step of the day (steps/proposals/ a package: what the LLM proposes): it reads what it
           needs (data/), computes (model/) and keeps each result in the database with its lineage; no step
           imports another
  view/    the page's data, a module per part of the page
  pipeline.py orders the steps for the day, in phases; flow.py describes them for the page (the System tab);
  calibrate.py measures how long each signpost's news lasts and walks the decision step through history.
Account numbers stay in data/account/ (out of git)."""

import json
import logging
import sys
from datetime import datetime

from src.board import calibrate, pipeline
from src.board.core import lineage, store
from src.board.steps import questions


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = sys.argv[1:]
    if args and args[0] == "exclude":
        db = store.connect()
        since = int(datetime.fromisoformat(args[2]).timestamp()) if len(args) > 2 else None
        until = int(datetime.fromisoformat(args[3]).timestamp()) if len(args) > 3 else None
        print(f"{lineage.exclude(db, args[1], since, until)} nodes marked excluded")
        db.commit()
        return
    if args and args[0] == "withdraw":
        db = store.connect()
        print("withdrawn" if questions.withdraw(db, args[1], " ".join(args[2:]), pipeline.ny_today()) else "not open")
        return
    if args and args[0] == "timescales":
        calibrate.timescales(pipeline.ny_today())
        return
    if args and args[0] == "walk":
        out = calibrate.walk(pipeline.ny_today())
        print(f"sense {out['sense']}, weights {out['weights_name']}")
        return
    if args and args[0] == "lineage":
        db = store.connect()
        for n in lineage.tree(db, int(args[1])):
            print(n["id"], n["kind"], n["name"], json.dumps(n["value"])[:120], "<-", [p["id"] for p in n["parents"]])
        return
    flags = {"--brief": "brief", "--research": "research", "--propose": "propose"}
    flags |= {"--scenarios": "scenarios", "--signposts": "signposts", "--changes": "changes"}
    unknown = [a for a in args if a not in flags and a not in ("--no-llm", "--no-fetch")]
    if unknown:  # --help among them: the usage, and no day run by mistake
        print(__doc__)
        sys.exit(0 if set(unknown) <= {"-h", "--help"} else 2)
    pipeline.day(
        use_llm="--no-llm" not in args,
        fetch="--no-fetch" not in args,
        force=[step for flag, step in flags.items() if flag in args],
    )


if __name__ == "__main__":
    main()

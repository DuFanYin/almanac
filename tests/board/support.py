"""Shared by the board's tests: a throwaway data and local directory for the board, set before it is imported
(conftest.py imports this first), and check(name, cond), an assertion that says what failed."""

import os
import tempfile
from pathlib import Path

tmp = Path(tempfile.mkdtemp())
for d in ("data", "local"):
    (tmp / d).mkdir()
os.environ.update(BOARD_DATA=str(tmp / "data"), BOARD_LOCAL=str(tmp / "local"))
HERE = Path(__file__).resolve()


def check(name, cond):
    assert cond, name

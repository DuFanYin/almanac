"""The board's core: the signpost and rule expression language, the data's backfills, lineage. (No network, no account
data.)"""

import numpy as np
import pandas as pd
from src.board.core import expr, lineage, store
from src.board.data import observations
from tests.board.support import check


def test_the_signpost_language():
    """The signpost language."""
    days = pd.date_range("2020-01-01", periods=400, freq="D")
    df = pd.DataFrame({"A": np.linspace(100, 200, 400), "B": np.linspace(10, 5, 400)}, index=days)
    check("ret over calendar days", abs(expr.evaluate("ret(A, 10)", df).iloc[-1] - (200 / df.A.iloc[-11] - 1)) < 1e-12)
    check(
        "names are the series, not the functions",
        expr.names("ret(SOX, 91) - mean(GSPC, 5) / B") == ["B", "GSPC", "SOX"],
    )
    bf_cfg = {"A": {"backfill": [{"series": "B"}, {"expr": "C * 2", "join": "level"}]}, "B": {}, "C": {}}
    bf = observations.backfill(
        {"A": [("2020-01-03", 20.0)], "B": [("2020-01-02", 5.0), ("2020-01-03", 10.0)], "C": [("2020-01-01", 7.0)]},
        bf_cfg,
    )
    check(
        "backfill: scaled to meet at the join, then an expression as it is",
        bf["A"] == [("2020-01-01", 14.0), ("2020-01-02", 10.0), ("2020-01-03", 20.0)],
    )
    check(
        "backfill: a series' sources are itself and what carries it back",
        observations.sources_of("A", bf_cfg) == ["A", "B", "C"],
    )
    check("arithmetic on series", abs(expr.evaluate("A / B - 1", df).iloc[-1] - 39) < 1e-9)
    for bad in ("__import__('os')", "A.__class__", "open('x')", "lambda: 1", "3"):
        try:
            expr.evaluate(bad, df)
            check(f"refuses {bad!r}", False)
        except (ValueError, SyntaxError):
            check(f"refuses {bad!r}", True)
    s = pd.Series([0.30, 0.44, 0.47, 0.60, 0.70, 0.90, 0.40], index=pd.date_range("2024-01-01", periods=7))
    st = list(expr.states(s, "< 0.45", "< 0.8"))
    check(
        "red on the line, held near it, released a quarter back",
        st == ["red", "red", "red", "yellow", "yellow", "green", "red"],
    )
    check("missing value is grey", list(expr.states(pd.Series([np.nan]), "> 1"))[0] == "grey")
    f = observations.frame(
        {"M": [("2024-01-01", 1.0), ("2024-02-01", 2.0)]}, {"M": {"lag": 40, "max_age": 80}}, until="2024-03-20"
    )
    check(
        "monthly value used only after its lag",
        pd.isna(f.M.get(pd.Timestamp("2024-02-09"))) and f.M["2024-02-10"] == 1.0 and f.M["2024-03-12"] == 2.0,
    )


def test_lineage():
    """Lineage."""
    db = store.connect()
    a = lineage.record(db, "raw_snapshot", "raw:X", {"last": 1})
    b = lineage.record(db, "raw_snapshot", "raw:X", {"last": 1})
    c = lineage.record(db, "signpost", "signpost:s", {"v": 2}, parents=[a])
    check("unchanged content keeps its node", a == b)
    check(
        "edge recorded with the code hash",
        db.execute("SELECT code FROM lineage_edge WHERE child = ?", (c,)).fetchone()[0] == lineage.CODE,
    )
    check("tree reaches the raw data", {n["name"] for n in lineage.tree(db, c)} == {"raw:X", "signpost:s"})
    check("clean before exclusion", not lineage.tainted(db, c))
    lineage.exclude(db, "raw:X")
    check("excluded raw data taints what was made from it", lineage.tainted(db, c))

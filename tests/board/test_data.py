"""The data layer: the checks on values, a Treasury's total return from its yield. (No network, no account data.)"""

import pandas as pd
from src.board.data import observations, sources
from tests.board.support import check


def test_data_checks():
    """Data checks."""
    price_cfg = {"source": "yahoo"}
    kept, out = observations.checked(
        [("2026-01-02", 100.0), ("2026-01-03", 250.0), ("2026-01-04", -1.0), ("2099-01-01", 101.0)],
        [("2026-01-01", 99.0)],
        price_cfg,
    )
    check(
        "checks hold out a jump, a price at or below 0, a day in the future",
        kept == [("2026-01-02", 100.0)]
        and [w for _, _, w in out] == ["jump x2.50", "not above 0", "dated in the future"],
    )
    check(
        "a split: every shared day moved by one factor",
        abs(
            observations.moved_by_one_factor(
                [("d1", 25.0), ("d2", 26.0), ("d3", 27.0)], [("d1", 100.0), ("d2", 104.0), ("d3", 108.0)]
            )
            - 0.25
        )
        < 1e-9,
    )
    check(
        "an ordinary revision is not a split",
        observations.moved_by_one_factor(
            [("d1", 100.0), ("d2", 104.5), ("d3", 108.0)], [("d1", 100.0), ("d2", 104.0), ("d3", 108.0)]
        )
        is None,
    )
    k = observations.join_ratio(
        [("2020-01-01", 10.0), ("2020-01-02", 30.0), ("2020-01-03", 20.0)],
        [("2019-12-31", 5.0), ("2020-01-01", 5.0), ("2020-01-02", 10.0), ("2020-01-03", 10.0)],
    )
    check("a join's ratio is the median over the overlap, not one day's", k == 2.0)
    f = observations.frame(
        {"Y": [("2026-01-01", 1.0)], "M": [("2026-01-01", 2.0)]}, {"M": {"lag": 365}}, until="2026-03-01"
    )
    check(
        "the frame ends today: a value not out yet is not in it",
        f.index[-1] == pd.Timestamp("2026-03-01") and ("M" not in f or f["M"].isna().all()),
    )


def test_a_treasury_s_total_return_from_its():
    """A Treasury's total return from its yield."""
    flat_y = [(f"2001-01-{d:02d}", 5.0) for d in range(1, 31)]
    bond = sources.par_bond(flat_y, {"years": 10})
    check(
        "a bond at an unchanged yield earns its coupon, day by day",
        abs(bond[-1][1] / 100 - (1 + 0.05 / 365) ** 29) < 1e-9,
    )
    up = sources.par_bond([("2001-01-01", 5.0), ("2001-01-02", 6.0)], {"years": 10})
    check("a yield up 1 point loses about the duration (7.8 years at 5%)", -0.08 < up[1][1] / 100 - 1 < -0.07)

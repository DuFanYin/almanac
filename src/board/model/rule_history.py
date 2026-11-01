"""A rule's line walked through history: for a condition on a value (`dd(NDX, 365) <= -0.20`), how often it
would have started to hold, and what its subject did over the next year, at the line and at lines moved either
side. A line where the neighbours tell the same story stands on flat ground; one where a small move changes the
story stands on a cliff, and its exact place was luck. Pure: frames in, numbers out."""

from .questions import OPS

SHIFTS = (-0.2, -0.1, 0.0, 0.1, 0.2)  # the line moved by these shares of its size
HORIZON = 365  # days the subject is followed after each episode starts
MIN_EPISODES = 3
STEADY = 0.10  # every neighbour's median within this of the line's own: steady


def episodes(values, op, line, rearm_days=20):
    """The days the condition starts to hold after not holding rearm_days running (an event rule's episodes)."""
    held = OPS[op](values, line) & values.notna()
    before = held.shift(1, fill_value=False).astype(float).rolling(rearm_days, min_periods=1).max()
    return values.index[held & (before == 0)]


def after(subject, days, horizon=HORIZON):
    """The subject's return over `horizon` calendar days after each day (a daily calendar index); those that
    have not run their course yet left out."""
    s = subject.ffill()
    ahead = s.shift(-horizon) / s - 1
    return ahead.reindex(days).dropna()


def walk(values, subject, op, line, rearm_days=20):
    """[{"shift", "line", "n", "median", "se", "up"}] for the line and its neighbours (a line never reached left
    out of the verdict), and the verdict: "steady" when every neighbour's median is within STEADY of the line's
    own; "cliff" when one is further, by more than twice the standard error of the difference too; "unclear" when
    it is further but within that noise (few episodes, a wild subject); "rare" with fewer than MIN_EPISODES at
    the line itself."""
    size = abs(line) if abs(line) > 1e-3 else float(values.std())
    rows = []
    for k in SHIFTS:
        at = line + k * size
        ret = after(subject, episodes(values, op, at, rearm_days))
        n = len(ret)
        rows.append(
            {
                "shift": k,
                "line": at,
                "n": n,
                "median": float(ret.median()) if n else None,
                # a median's standard error, about 1.25 sigma / sqrt(n)
                "se": float(1.2533 * ret.std() / n**0.5) if n > 1 else None,
                "up": float((ret > 0).mean()) if n else None,
            }
        )
    centre = next(r for r in rows if r["shift"] == 0)
    if centre["n"] < MIN_EPISODES:
        return rows, "rare"
    near = [r for r in rows if r["shift"] != 0 and r["n"] >= MIN_EPISODES]
    far = [r for r in near if abs(r["median"] - centre["median"]) > STEADY]
    if not far:
        return rows, "steady"
    noise = [2 * ((r["se"] or 0) ** 2 + (centre["se"] or 0) ** 2) ** 0.5 for r in far]
    return rows, "cliff" if any(abs(r["median"] - centre["median"]) > x for r, x in zip(far, noise)) else "unclear"

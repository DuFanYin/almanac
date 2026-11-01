"""The LLM's scenarios: the world's last layer. At each length the data's world (model/world.py:
history's windows weighed by today's signposts, the simulation) holds the scenarios the LLM wrote, each in a
direction of its own: a window is in every scenario whose ranges it is in, in several or in none. Each scenario's
probability should be the simulation's share of it times the LLM's factor for it (how much likelier or less likely it
is for what the simulation does not include); the windows are reweighed so that every scenario has its own, moving
the simulation least (the weights nearest the data's in relative entropy). Inside each cell of the scenarios the
simulation's proportions stay.

  inside  windows x scenarios: whether each window is in each scenario's ranges
  made    a scenario with fewer than EPISODES windows in its ranges that do not overlap is made from the MADE nearest
          windows moved into its ranges (model/world.py made)
  share   a scenario's share of the simulation, as estimated from history's windows that do not overlap (K of them):
          (share x K + 1/2) / (K + 1), so a scenario history never had has half a window's worth, not none
  reweigh the final weights: each scenario's share times its factor is the probability it should have; the data's
          weights tilted along the scenarios until each has it, or, where they are at odds, each as near as the
          others allow
  carry   a length without scenarios of its own (a week, a question's days) takes the nearest length's: each start
          day's final weight over its data weight there, times its data weight at the other length
  happened which scenarios a path that happened is in, for the score

Pure."""

import numpy as np

from . import factors as fx
from . import world as wm

EPISODES = 3  # a scenario with fewer windows than this in its ranges, none overlapping, is made from the nearest
MADE = wm.LIKE  # windows a made scenario is made of


def clamp(ranges, X, names, span=True):
    """{factor index: (low, high)} for the ranges on factors measurable here, an open end (None) at history's span
    widened by model/world.py REACH, each kept within it (not with span False: an open end infinite)."""
    if not span:
        return {
            names.index(k): (-np.inf if lo is None else lo, np.inf if hi is None else hi)
            for k, (lo, hi) in ranges.items()
            if k in names
        }
    lo_s, hi_s = X.min(0), X.max(0)
    reach = (hi_s - lo_s) * wm.REACH
    out = {}
    for k, (lo, hi) in ranges.items():
        if k not in names:
            continue
        j = names.index(k)
        lo = lo_s[j] - reach[j] if lo is None else min(max(lo, lo_s[j] - reach[j]), hi_s[j] + reach[j])
        hi = hi_s[j] + reach[j] if hi is None else min(hi, hi_s[j] + reach[j])
        out[j] = (lo, max(hi, lo))
    return out


def distances(X, sd, cut):
    """(how far outside a scenario's ranges, how far from their middles, an open range's middle its end): each per
    window, in standard deviations; cut: clamp()'s."""
    if not cut:  # no range on a factor measurable here: it holds no window
        return np.full(len(X), np.inf), np.zeros(len(X))
    gap, mid = np.zeros(len(X)), np.zeros(len(X))
    for j, (lo, hi) in cut.items():
        gap += (np.maximum(lo - X[:, j], 0) + np.maximum(X[:, j] - hi, 0)) / sd[j]
        centre = (lo + hi) / 2 if np.isfinite(lo) and np.isfinite(hi) else lo if np.isfinite(lo) else hi
        mid += ((X[:, j] - centre) / sd[j]) ** 2
    return gap, mid / max(len(cut), 1)


def scale(X):
    """Each factor's standard deviation over the windows (1 where it does not move)."""
    sd = X.std(0) if len(X) else np.ones(X.shape[1])
    return np.where(sd > 0, sd, 1.0)


def inside(X, names, scenarios, span=True):
    """windows x scenarios: whether each window is in each scenario's ranges (none for a scenario on no factor
    measurable here). scenarios: [{"ranges"}]; span: clamp()'s."""
    out = np.zeros((len(X), len(scenarios)), dtype=bool)
    for s, sc in enumerate(scenarios):
        cut = clamp(sc["ranges"], X, names, span)
        if not cut:
            continue
        m = np.ones(len(X), dtype=bool)
        for j, (lo, hi) in cut.items():
            m &= (X[:, j] >= lo) & (X[:, j] <= hi)
        out[:, s] = m
    return out


def episodes(starts, days):
    """How many windows of `days` among these start days (int64 ns) do not overlap: counted from the first, each the
    first at least `days` after the one before."""
    n, last = 0, None
    for t in np.sort(starts):
        if last is None or t - last >= days * fx.DAY:
            n, last = n + 1, t
    return n


def shares(share, periods):
    """A scenario's share of the simulation as estimated from `periods` windows that do not overlap (share: the data
    weight of its windows; it happens or it does not): (share x periods + 1/2) / (periods + 1)."""
    return (np.asarray(share, dtype=float) * periods + 0.5) / (periods + 1)


ROUNDS = 200  # the reweighing's steps at most
TOL = 1e-6  # it stops when every scenario's probability is this near the one it should have
EDGE = 1e-4  # a probability a scenario should have is kept this far from 0 and 1


def gap(P, q):
    """How far probabilities P are from q, scenario by scenario, in information: sum of KL(q || P), each a yes or no."""
    P = np.clip(P, 1e-12, 1 - 1e-12)
    return float(np.sum(q * np.log(q / P) + (1 - q) * np.log((1 - q) / (1 - P))))


def reweigh(w, held, q, rounds=ROUNDS, tol=TOL):
    """The weights nearest w under which each scenario s (the windows held[:, s]) has the probability q[s]: w tilted
    along the scenarios only (each window's weight times exp of the sum of its scenarios' terms), the terms found by
    Gauss-Newton steps on gap(). When the scenarios can all have theirs this is the weights nearest w in relative
    entropy that give it; when they are at odds (no weights give them all), each as near as the others allow, the
    same whatever their order. A scenario with no window or every window is left out. Normalised."""
    w0 = np.asarray(w, dtype=float) / np.sum(w)
    use = held.any(0) & ~held.all(0)
    H, q = held[:, use].astype(float), np.asarray(q, dtype=float)[use]
    if not H.shape[1]:
        return w0
    lam = np.zeros(H.shape[1])

    def at(lam):
        z = H @ lam
        x = w0 * np.exp(z - z.max())
        return x / x.sum()

    w = at(lam)
    for _ in range(rounds):
        P = H.T @ w
        if np.max(np.abs(P - q)) < tol:
            break
        d = 1 / np.clip(P * (1 - P), 1e-12, None)
        C = (H * w[:, None]).T @ H - np.outer(P, P)  # how each scenario's probability moves with each term
        g = C @ (d * (P - q))
        step = np.linalg.solve(C @ (d[:, None] * C) + 1e-9 * np.eye(len(q)), g)
        now, t = gap(P, q), 1.0
        while t > 1e-6:  # halved until it closes the gap
            nxt = at(lam - t * step)
            if gap(H.T @ nxt, q) < now:
                lam, w = lam - t * step, nxt
                break
            t /= 2
        else:
            break
    return w


def at_length(a, scenarios, make=True):
    """The world at one length with the scenarios. a: World.weights at it; scenarios: [{"id", "ranges", "times" (the
    LLM's factor, 1 without)}]; make: a scenario history has too few windows of is made. {"ext" (a with the made
    windows, model/world.py extended), "w" (the final weights over ext's rows), "inside" (ext's rows x scenarios),
    "periods" (history's windows that do not overlap), "each" [{"id", "rows" (ext's start ids of its rows), "share"
    (its data weight: the simulation's share), "base" (that share as estimated from the periods: shares()), "history"
    (its share of history unweighted), "episodes", "made" (how many windows made), "times", "target" (base times the
    factor: the probability it should have), "weight" (the probability it has: the final weights in it)}]}."""
    X, names, n = a["X"], a["names"], len(a["starts"])
    if not n or not scenarios:
        return None
    sd = scale(X)
    real = inside(X, names, scenarios)
    made = []
    for s, sc in enumerate(scenarios):
        if not make or episodes(a["starts"][real[:, s]], a["days"]) >= EPISODES:
            continue
        cut = clamp(sc["ranges"], X, names)
        if not cut:
            continue
        gap, _ = distances(X, sd, cut)
        pick = np.argsort(gap, kind="stable")[:MADE]
        # made into its ranges as written, an open end at history's span widened (clamp)
        finite = {
            k: [cut[names.index(k)][0] if lo is None else lo, cut[names.index(k)][1] if hi is None else hi]
            for k, (lo, hi) in sc["ranges"].items()
            if k in names
        }
        made.append(
            {
                "s": s,
                "starts": wm.made_ids(sc["id"], len(pick)),
                "from": a["starts"][pick],
                "shift": wm.made(a, names, finite, pick),
            }
        )
    ext = wm.extended(a, made)
    held = inside(ext["X"], names, scenarios)
    of = np.concatenate([np.full(n, -1), np.repeat([m["s"] for m in made], [len(m["starts"]) for m in made])])
    for s in range(len(scenarios)):
        held[of == s, s] = True  # a made window is in the scenario it was made for, its edges as written
    periods = episodes(a["starts"], a["days"])
    share = real.T.astype(float) @ a["w"]
    base = shares(share, periods)
    times = np.array([float(sc.get("times", 1.0)) for sc in scenarios])
    target = np.clip(base * times, EDGE, 1 - EDGE)
    # the start: the data's weights, a scenario's made windows holding what its estimated share adds to its real ones
    w0 = np.concatenate([a["w"], np.zeros(len(ext["starts"]) - n)])
    for s in range(len(scenarios)):
        m = of == s
        if m.any():
            src = a["w"][ext["src"][of[n:] == s]]
            w0[m] = max(base[s] - share[s], 0.5 / (periods + 1)) * (src / src.sum() if src.sum() > 0 else 1 / m.sum())
    w = reweigh(w0, held, target)
    made_n = {m["s"]: len(m["starts"]) for m in made}
    each = [
        {
            "id": sc["id"],
            "rows": ext["starts"][held[:, s]],
            "share": float(share[s]),
            "base": float(base[s]),
            "history": float(real[:, s].mean()),
            "episodes": episodes(a["starts"][real[:, s]], a["days"]),
            "made": made_n.get(s, 0),
            "times": float(times[s]),
            "target": float(target[s]),
            "weight": float(w[held[:, s]].sum()),
        }
        for s, sc in enumerate(scenarios)
    ]
    return {"ext": ext, "w": w, "inside": held, "periods": periods, "each": each}


def on_history(ext, w, n):
    """The final weights over the real windows only: a made window's weight given to the window it was made from
    (for a question or another length, which only history's windows answer)."""
    out = np.asarray(w[:n], dtype=float).copy()
    if len(w) > n:
        np.add.at(out, ext["src"], w[n:])
    return out


def carry(a_from, w_from, a_to):
    """Weights at another length: a_to's data weights times, at each of its start days, the final weight over the
    data weight at a_from's length (w_from: on_history's) on the same start day (1 where a_from has none).
    Normalised."""
    ratio = np.ones(len(a_to["starts"]))
    pos = np.searchsorted(a_from["starts"], a_to["starts"])
    pos = np.clip(pos, 0, max(len(a_from["starts"]) - 1, 0))
    same = (a_from["starts"][pos] == a_to["starts"]) if len(a_from["starts"]) else np.zeros(len(ratio), bool)
    base = a_from["w"][pos]
    ok = same & (base > 0)
    ratio[ok] = w_from[pos[ok]] / base[ok]
    w = a_to["w"] * ratio
    return w / w.sum() if w.sum() > 0 else a_to["w"]


def nearest(lengths, days):
    """Of the lengths with scenarios ({length: its days}), the nearest to `days` (by ratio), or None."""
    if not lengths:
        return None
    return min(lengths, key=lambda h: abs(np.log(max(days, 1) / lengths[h])))


def happened(x, scenarios):
    """Which scenarios a path that happened is in: [True, False or None (a factor of its ranges without a value)].
    x {factor: its value over the length}; the ranges as written, open ends open."""
    out = []
    for sc in scenarios:
        got = True
        for k, (lo, hi) in sc["ranges"].items():
            if x.get(k) is None:
                got = None
                break
            got = got and (lo is None or x[k] >= lo) and (hi is None or x[k] <= hi)
        out.append(got)
    return out


def on_length(sets, a, days):
    """The final weights over a's windows (World.weights at a length: a week, a question's days), history's windows
    only: carried from the nearest length with scenarios (carry; at that length itself, its final weights with a made
    window's weight on the window it was made from); the data's without any. sets: {length: {"a", "ext", "w"}}
    (steps/scenarios/world.py for_world). (weights, the length they came from or None)."""
    h = nearest({h: x["a"]["days"] for h, x in sets.items()}, days)
    if h is None:
        return a["w"], None
    x = sets[h]
    return carry(x["a"], on_history(x["ext"], x["w"], len(x["a"]["starts"])), a), h

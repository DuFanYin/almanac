"""The decision step: where the world's product, the futures at each length weighed
(model/distribution.py), meets your preferences (your targets, your risk level's limits, the guardrails) and what you
hold, once. The old layers before it (risk steps, temporary moves, the adjustment, most rules) are in it since 10-31.

1. Start from your targets: the mix you would hold in a usual world.
2. Lean each class toward what today's world says of it against usual, both ways alike (lean): at each length that
   acts, the class's expected return under the world's weights less under history's on the same windows (gap), in
   two parts: the data's (the simulation against history) and the LLM's (the final world against the simulation:
   its scenarios), each pulled toward 0 by the gap a reweighting as large would make by chance (shrunk, as the
   signposts' measures are), in the class's standard deviations over that length (score); the lengths together by
   `weights`; the data's part times `sense`, set by walking history (model/walk.py), the LLM's times `llm_sense`,
   yours (the LLM's scenarios cannot be walked through history: there were none); within your risk level's `tilt` a
   class (all together at most TOTAL_TILT times it), its cap and the cash floor; cash takes the difference. A short
   length's lean goes when its news does (the world back to usual); a long one's stays as long as its news.
3. The guardrails (guard): at every length that acts, the bad case within its limit, moved GUARD_STEP at a time from
   the class whose move lifts the length furthest past its limit most into a shelter you hold; they win over step 2.
4. Trades (trades): what you hold against the decided mix, category by category: past your band, or the decided mix
   moved half a step from where you last were at it (anchor) and you half a step away; one step at a time, and a
   category traded waits PACE days once the trade is done.

Pure."""

import numpy as np

from . import allocation

ACTS = ("month", "short", "long", "longer")  # the lengths that act: a week is shown, not acted on (the owner, 10-31)
STEP_DAYS = 7  # days between the start days of history: windows longer than this overlap
TOTAL_TILT = 2  # all the leans together at most this many times one class's
GUARD_STEP = 0.01  # a guardrail moves this much at a time
GUARD_MOST = 60  # guardrail moves at most
SHELTER = allocation.SHELTER
PACE = 28  # days: a category traded waits this long after the trade is done before the next step the same way
CASH = "cash"


def shrink(g, se):
    """A gap pulled toward 0 by the gap chance alone makes: g x g² / (g² + se²)."""
    return g**3 / (g**2 + se**2) if g or se else 0.0


def gap(fut, base, data=None, days=365):
    """Each class at one length: {"gap" (expected return under the world less under history, on the same windows),
    "se" (the gap a reweighting this large makes by chance: history's spread times the weights' distance from even,
    widened by the windows' overlap), "shrunk" (gap x gap² / (gap² + se²)), "sd" (history's spread), "score"
    (shrunk / sd), "data" and "llm" (the parts: the data's weights against history, the final against the data's),
    "score_data" and "score_llm" (each part shrunk by its own reweighting's chance gap, over sd)}. fut, base, data:
    the world's futures, the same unweighted, on the data alone (model/distribution.py Futures; without data, all of
    it the data's)."""
    data = data if data is not None else fut
    widen = max(1.0, days / STEP_DAYS)

    def noise(d):
        return float(np.sqrt((d**2).sum() * widen))

    n_all, n_data, n_llm = noise(fut.w - base.w), noise(data.w - base.w), noise(fut.w - data.w)
    out = {}
    for k, r in fut.returns.items():
        mean = float(base.w @ r)
        sd = float(np.sqrt(base.w @ (r - mean) ** 2))
        g, part = float((fut.w - base.w) @ r), float((data.w - base.w) @ r)
        s = shrink(g, sd * n_all)
        out[k] = {
            "gap": g,
            "se": sd * n_all,
            "shrunk": s,
            "sd": sd,
            "score": s / sd if sd else 0.0,
            "data": part,
            "llm": g - part,
            "score_data": shrink(part, sd * n_data) / sd if sd else 0.0,
            "score_llm": shrink(g - part, sd * n_llm) / sd if sd else 0.0,
        }
    return out


def score(gaps, weights, key="score"):
    """{class: the lengths' scores together}: gaps {length: gap()}, weights {length: weight} (renormalised over the
    lengths there are); key: which score of gap()'s."""
    have = {h: x for h, x in weights.items() if h in gaps and x}
    total = sum(have.values())
    if not total:
        return {}
    out = {}
    for h, x in have.items():
        for k, g in gaps[h].items():
            out[k] = out.get(k, 0.0) + x / total * g[key]
    return out


def lean(policy, scores, limits, sense):
    """Your targets leaned by the scores: each class but cash by sense x its score, within ±tilt, at least 0 and at
    most its cap (cash and Treasuries none), all the leans together within TOTAL_TILT x tilt; cash takes the
    difference and keeps its floor (the leans up scaled down to keep it). ({class: share}, {class: lean})."""
    tilt, cap, floor = limits["tilt"], limits["class_cap"], limits.get("cash_min", 0.0)
    raw = {}
    for k, w in policy.items():
        if k == CASH or k not in scores:
            continue
        t = float(np.clip(sense * scores[k], -tilt, tilt))
        top = 1.0 if k in ("cash", "treasuries") else cap
        raw[k] = float(min(max(t, -w), max(top - w, 0.0)))
    moved = sum(abs(t) for t in raw.values())
    if moved > TOTAL_TILT * tilt:
        raw = {k: t * TOTAL_TILT * tilt / moved for k, t in raw.items()}
    room = policy.get(CASH, 0.0) - floor
    up, down = sum(t for t in raw.values() if t > 0), -sum(t for t in raw.values() if t < 0)
    if up - down > room and up:
        f = max(0.0, (room + down) / up)
        raw = {k: t * f if t > 0 else t for k, t in raw.items()}
    mix = {k: w + raw.get(k, 0.0) for k, w in policy.items()}
    mix[CASH] = policy.get(CASH, 0.0) - sum(raw.values())
    return {k: x for k, x in mix.items() if x > 1e-9}, {k: t for k, t in raw.items() if abs(t) > 1e-9}


def bad(mix, fut):
    """The mix's bad case on the futures (the mean of its worst 10% of the weight), or None."""
    r, _ = allocation.returns(mix, fut)
    return None if r is None else float(allocation.tail(r, fut.w) @ r)


def guard(mix, lengths, policy):
    """The guardrails on the bad case: while one length's bad case is past its limit, GUARD_STEP from the class whose
    move lifts that length (the one furthest past, against its limit) most into the shelter that does it, among the
    shelters in your targets. lengths: {length: (futures, limit)}. ({class: share}, [{"from", "to", "size",
    "horizon"}] merged by pair, {length: {"before", "after", "limit"}})."""
    shelters = [k for k in SHELTER if policy.get(k)]
    before = {h: bad(mix, f) for h, (f, _) in lengths.items()}
    moves = {}
    for _ in range(GUARD_MOST):
        past = [
            ((b - lim) / abs(lim), h)
            for h, (f, lim) in lengths.items()
            if lim is not None and (b := bad(mix, f)) is not None and b < lim - 1e-9
        ]
        if not past:
            break
        h = min(past)[1]
        f, lim = lengths[h]
        now, best = bad(mix, f), None
        for k, w in sorted(mix.items()):
            if k in SHELTER or w <= 1e-9 or k not in f.returns:
                continue
            size = min(GUARD_STEP, w)
            for to in shelters:
                if to not in f.returns:
                    continue
                after = bad({**mix, k: w - size, to: mix.get(to, 0.0) + size}, f)
                if after > now + 1e-9 and (best is None or after > best[0]):
                    best = (after, k, to, size)
        if not best:
            break
        _, k, to, size = best
        mix = {**mix, k: mix[k] - size, to: mix.get(to, 0.0) + size}
        m = moves.setdefault((k, to), {"from": k, "to": to, "size": 0.0, "horizon": h})
        m["size"] += size
    after = {h: bad(mix, f) for h, (f, _) in lengths.items()}
    seen = {h: {"before": before[h], "after": after[h], "limit": lim} for h, (_, lim) in lengths.items()}
    return {k: x for k, x in mix.items() if x > 1e-9}, list(moves.values()), seen


def decide(policy, horizons, limits, sense, weights, llm_sense=0.0):
    """Steps 1-3 on the futures of each length that acts. horizons: {length: {"futures", "data_futures",
    "history_futures", "days", "limit"}} (steps/distribution.py's). {"mix" (decided), "lean" {class: change},
    "scores" {class: the data's part x sense + the LLM's x llm_sense}, "parts" {class: {"data", "llm"}} (each part's
    lean before the limits), "gaps" {length: gap()}, "guard" [move], "bad" {length: {"targets", "before", "after",
    "limit"}}}."""
    acts = {h: d for h, d in horizons.items() if h in ACTS and d.get("futures") is not None}
    gaps = {
        h: gap(d["futures"], d["history_futures"], d.get("data_futures"), d["days"])
        for h, d in acts.items()
        if d.get("history_futures") is not None
    }
    s_data, s_llm = score(gaps, weights, "score_data"), score(gaps, weights, "score_llm")
    scores = {k: sense * s_data.get(k, 0.0) + llm_sense * s_llm.get(k, 0.0) for k in s_data.keys() | s_llm.keys()}
    parts = {k: {"data": sense * s_data.get(k, 0.0), "llm": llm_sense * s_llm.get(k, 0.0)} for k in scores}
    leaned, change = lean(policy, scores, limits, 1.0)
    mix, moves, seen = guard(leaned, {h: (d["futures"], d.get("limit")) for h, d in acts.items()}, policy)
    for h, d in acts.items():
        seen[h]["targets"] = bad(policy, d["futures"])
    return {"mix": mix, "lean": change, "scores": scores, "parts": parts, "gaps": gaps, "guard": moves, "bad": seen}


def by_category(mix, cats, policy):
    """The decided mix by category: each category its share of its class's target times the class's decided share,
    at most its cap (a guardrail you set; the rest to cash). cats: {category: {"class", "target", "cap"}}."""
    out, over = {}, 0.0
    for c, x in cats.items():
        k, t = x["class"], x.get("target", 0.0)
        if policy.get(k):
            out[c] = mix.get(k, 0.0) * t / policy[k]
            if x.get("cap") is not None and out[c] > x["cap"]:
                over += out[c] - x["cap"]
                out[c] = x["cap"]
    if over:
        c = next((c for c, x in cats.items() if x["class"] == CASH), None)
        if c:
            out[c] = out.get(c, 0.0) + over
    return out


def trades(decided, held, anchor, limits, cash=CASH):
    """What to do with what you hold, by category (shares of the account): {category: {"size" (+ buy, - sell),
    "gap", "why" ("band": drifted past your band; "view": the decided mix moved)}} and the anchors after today. A
    category trades when it is past your band from the decided share, or when the decided share has moved at least
    half a step from its anchor (where it was the last time you held within half a step of it) and you are at least
    half a step from it; one step at a time. Cash takes the other side; cash past the band with no other trade goes
    to (or comes from) the category furthest from its decided share. anchor: {category: share} ({} the first time:
    the decided share); cash: the cash category."""
    band, step = limits["band"], limits["step"]
    out, after = {}, {}
    for c, x in decided.items():
        h = held.get(c, 0.0)
        g = x - h
        a = anchor.get(c, x)
        why = "band" if abs(g) > band else "view" if abs(x - a) >= step / 2 and abs(g) >= step / 2 else None
        if why and c != cash:
            out[c] = {"size": float(np.sign(g) * min(abs(g), step)), "gap": g, "why": why}
        after[c] = x if abs(g) <= step / 2 else a
    if cash in decided and not out:
        g = decided[cash] - held.get(cash, 0.0)
        if abs(g) > band:
            others = {c: decided[c] - held.get(c, 0.0) for c in decided if c != cash}
            c = max(others, key=others.get) if g < 0 else min(others, key=others.get)
            out[c] = {"size": float(np.sign(-g) * min(abs(g), step)), "gap": others[c], "why": "band"}
    # buys only from the cash there is and what the sales bring
    have = max(0.0, held.get(cash, 0.0)) - sum(x["size"] for x in out.values() if x["size"] < 0)
    want = sum(x["size"] for x in out.values() if x["size"] > 0)
    if want > have + 1e-9:
        f = max(0.0, have) / want
        out = {c: {**x, "size": x["size"] * f} if x["size"] > 0 else x for c, x in out.items()}
        out = {c: x for c, x in out.items() if abs(x["size"]) > 1e-4}
    return out, after


def new_money(decided, held, cash=CASH):
    """The category most under its decided share, cash aside: where new money goes first (None when none is under)."""
    under = {c: x - held.get(c, 0.0) for c, x in decided.items() if c != cash}
    c = max(under, key=under.get, default=None)
    return c if c is not None and under[c] > 0 else None

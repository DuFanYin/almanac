"""What the owner's marks on the firings say about their style: rules lean "safer" (a sale, cash, a pause) or
"bolder" (a purchase, a buy-back, a double); skipping the safer ones while doing the bolder ones is acting bolder
than the chosen style, and the other way round. Real choices weigh more than a questionnaire's answers."""

MIN_MARKED = 3  # marked firings on a side before its rate counts
GAP = 0.3  # the difference in skip rates that suggests another style


def lean(marks, leans):
    """marks [(rule, "done" | "skipped")], leans {rule: "safer" | "bolder"}: {"safer": {"marked", "skipped"},
    "bolder": {...}, "says": "bolder" | "more careful" | "as chosen" | None (too few marks)}."""
    out = {side: {"marked": 0, "skipped": 0} for side in ("safer", "bolder")}
    for rule, status in marks:
        side = leans.get(rule)
        if side in out:
            out[side]["marked"] += 1
            out[side]["skipped"] += status == "skipped"
    rate = {s: v["skipped"] / v["marked"] for s, v in out.items() if v["marked"] >= MIN_MARKED}
    if not rate:
        return {**out, "says": None}
    diff = rate.get("safer", 0.0) - rate.get("bolder", 0.0)
    return {**out, "says": "bolder" if diff >= GAP else "more careful" if diff <= -GAP else "as chosen"}

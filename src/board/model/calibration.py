"""Many things tested at once: which of them may be kept with at most a share FDR passing by luck (the
Benjamini-Hochberg procedure), as the measures of how long each signpost's news lasts are kept (model/timescale.py).
Pure."""

FDR = 0.10  # the false discovery rate allowed over everything one calibration lets through


def discoveries(pvalues, q=FDR):
    """The keys of {key: p-value} that pass the Benjamini-Hochberg procedure at false discovery rate q: of many
    things each tested once, those that may be kept with at most a share q of them passing by luck."""
    ranked = sorted((p, k) for k, p in pvalues.items() if p is not None)
    m, last = len(ranked), 0
    for i, (p, _) in enumerate(ranked, 1):
        if p <= q * i / m:
            last = i
    return {k for _, k in ranked[:last]}

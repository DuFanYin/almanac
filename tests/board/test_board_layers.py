"""The board's layers: core imports none of the others, data neither the model nor the steps, the model only its own
modules and core's expressions and nothing that reaches a database, the network or files, a step nothing above it and
no other step; only data/ reaches the model layer (src/llm) and research."""

from tests.board.support import HERE, check


def test_layers():
    """Each layer imports only what is below it."""
    # the layers (src/board/run.py): core imports none of the others, data neither the model nor the steps, the model
    # only core's expression language and nothing that reaches a database, the network or files, and the steps
    # nothing above them
    import ast  # noqa: E402

    PKG = HERE.parents[2] / "src" / "board"
    IO = {"sqlite3", "urllib", "subprocess", "socket", "http", "requests", "httpx", "yfinance", "tomllib", "threading"}

    def imports(f):
        """Every module a file imports; a relative import with each name it takes (`from ..core import expr` is
        ..core.expr), so a layer is told apart from the module beside it."""
        out = set()
        for n in ast.walk(ast.parse(f.read_text())):
            if isinstance(n, ast.Import):
                out |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                base = "." * n.level + (n.module or "")
                out |= {base.rstrip(".") + "." + a.name if n.level else base for a in n.names}
        return out

    for layer in ("core", "data", "model", "steps", "view"):
        for f in sorted((PKG / layer).rglob("*.py")):
            got = imports(f)
            # a module in a package of the layer (steps/proposals/) imports with one more dot: read as from the
            # layer, its own package's modules marked ~ (part of the same step)
            deep = len(f.relative_to(PKG / layer).parts) - 1
            dots = lambda i: len(i) - len(i.lstrip("."))  # noqa: E731
            got = {i if not i.startswith(".") else i[deep:] if dots(i) > deep else "~" + i for i in got}
            rel, absolute = {i for i in got if i[0] in ".~"}, {i.split(".")[0] for i in got if i[0] not in ".~"}
            name = f.relative_to(PKG / layer)
            if layer != "data":  # the model layer and research only through data/ (data/llm.py, data/research.py)
                outside = {i for i in got if i.startswith(("src.llm", "src.research"))}
                check(
                    f"{layer}/{name} reaches src/llm and src/research only through data/: {sorted(outside)}",
                    not outside,
                )
            if layer == "core":
                check(f"core/{name} imports no other layer", not any(i.startswith("..") for i in rel))
            if layer == "data":
                check(
                    f"data/{name} does not import the model or the steps",
                    not any(i.startswith(("..model", "..steps")) for i in rel),
                )
            if layer == "steps":
                above = {"pipeline", "view", "run", "calibrate"}
                check(
                    f"steps/{name} imports nothing above it (pipeline, calibrate, view, run)",
                    not any(i.split(".")[2] in above for i in rel if i.startswith("..")),
                )
                check(f"steps/{name} imports no other step", not any(i.startswith(".") and dots(i) == 1 for i in rel))
            if layer == "view":
                check(
                    f"view/{name} imports nothing above it (pipeline, calibrate, run)",
                    not any(i.split(".")[2] in {"pipeline", "run", "calibrate"} for i in rel if i.startswith("..")),
                )
            if layer == "model":
                check(
                    f"model/{name} is pure: only core's expressions, no I/O",
                    all(i == "..core.expr" or i[0] == "~" or (i.startswith(".") and dots(i) == 1) for i in rel)
                    and not absolute & IO
                    and "open(" not in f.read_text()
                    and "read_text(" not in f.read_text(),
                )


def test_the_system_page_names_every_part():
    """flow.py, the System tab's source, stays whole: its phases are the pipeline's, every module it names exists,
    every step module is in a stage, every step that may fail is a stage's, and every edge joins two stages."""
    import re

    from src.board import flow, pipeline

    PKG = HERE.parents[2] / "src" / "board"
    ids = {s["id"] for s in flow.STAGES}
    check(
        "its phases are the pipeline's, in order", [p for p, _ in flow.PHASES] == [f.__name__ for f in pipeline.PHASES]
    )
    check("every stage in a phase", all(s["phase"] in dict(flow.PHASES) for s in flow.STAGES))
    named = {m for s in flow.STAGES for m in s["modules"]}
    check("every module named exists", all((PKG / m).exists() for m in named))
    steps = {str(f.relative_to(PKG)) for f in (PKG / "steps").rglob("*.py") if f.name != "__init__.py"}
    check(f"every step module is in a stage: missing {sorted(steps - named)}", steps <= named)
    used = set(re.findall(r'attempt\(\s*"(\w+)"', (PKG / "pipeline.py").read_text()))
    kept = {k for s in flow.STAGES for k in s.get("attempts", ())}
    check(f"every step that may fail is a stage's: {sorted(used ^ kept)}", used == kept)
    check("every edge joins two stages", all(a in ids and b in ids for a, b in flow.EDGES))
    check(
        "every LLM stage says what its work must pass", all(s.get("checks") for s in flow.STAGES if s["side"] == "llm")
    )

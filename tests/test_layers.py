"""The modules' dependencies point one way, llm -> research -> board -> web: src/llm imports none of the others,
src/research only src/llm, src/board both, and src/web (the page) only the board."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
ALLOWED = {
    "llm": {"llm"},
    "research": {"llm", "research"},
    "board": {"llm", "research", "board"},
    "web": {"board", "web"},
}


def modules_imported(f: Path) -> set[str]:
    out = set()
    for n in ast.walk(ast.parse(f.read_text())):
        if isinstance(n, ast.Import):
            out |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            out.add(n.module)
    return {m.split(".")[1] for m in out if m.startswith("src.") and m.count(".") >= 1}


def test_each_module_imports_only_what_is_below_it():
    wrong = []
    for top, allowed in ALLOWED.items():
        for f in sorted((SRC / top).rglob("*.py")):
            bad = modules_imported(f) - allowed
            if bad:
                wrong.append(f"{f.relative_to(SRC)} imports {sorted(bad)}")
    assert not wrong, "\n".join(wrong)

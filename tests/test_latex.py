"""LaTeX result tables (latex/*.tex in the data export)."""

import io
import shutil
import zipfile

import pytest

from balatro_analysis import export, latex
from balatro_analysis.loader import load_runs


def _balanced(tex: str) -> bool:
    depth = 0
    for i, ch in enumerate(tex):
        if ch in "{}" and (i == 0 or tex[i - 1] != "\\"):
            depth += 1 if ch == "{" else -1
            if depth < 0:
                return False
    return depth == 0


@pytest.mark.parametrize(
    ("text", "tex"),
    [
        ("run_a", r"run\_a"),
        ("50% & $5 #1", r"50\% \& \$5 \#1"),
        ("{x}~^\\", r"\{x\}\textasciitilde{}\textasciicircum{}\textbackslash{}"),
    ],
)
def test_escape(text, tex):
    assert latex.escape(text) == tex


def test_formatting():
    assert latex.with_ci(1.234, 1.0, 1.5) == "1.23 [1.00, 1.50]"
    assert latex.with_ci(1.234, float("nan"), float("nan")) == "1.23"
    assert latex.with_ci(0.25, 0.1, 0.4, latex.pct) == r"25.0\% [10.0\%, 40.0\%]"
    assert latex.pval(0.0004) == "$<$0.001"
    assert latex.pval(0.0187) == "0.019"
    assert latex.pval(float("nan")) == "--"


def test_run_tables(runs_dir, tmp_path):
    for name in ("run_a", "run_b"):
        shutil.copytree(runs_dir / "run_small", tmp_path / name)
    tables = latex.tables(load_runs(tmp_path))
    assert list(tables) == ["summary", "summary_ci", "run_comparison"]  # no groups
    for name, tex in tables.items():
        assert _balanced(tex), name
        assert tex.count(r"\toprule") == tex.count(r"\bottomrule") == 1
        assert f"\\label{{tab:{name}}}" in tex
    assert r"run\_a" in tables["summary_ci"]
    assert "run_a &" not in tables["summary_ci"]
    assert r"\resizebox" in tables["run_comparison"]  # wide
    assert r"\resizebox" not in tables["summary_ci"]
    assert "Red Deck" in tables["summary"]  # in-game names


def test_group_tables(group_runs_dir):
    tables = latex.tables(load_runs(group_runs_dir))
    assert {"group_summary", "group_comparison"} <= set(tables)
    rows = [line for line in tables["group_summary"].splitlines() if line.endswith(r"\\")]
    assert len(rows) == 3  # header + 2 groups
    assert rows[1].startswith("a & 3 & 9 & ")
    assert all(_balanced(t) for t in tables.values())


def test_export_zip_has_latex(group_runs_dir):
    data = export.export_zip(load_runs(group_runs_dir), {"runs": "all"})
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        assert "latex/group_comparison.tex" in names
        assert "latex/summary_ci.tex" in z.read("README.txt").decode()

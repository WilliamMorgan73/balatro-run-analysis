import io
import zipfile

import pandas as pd
import plotly.graph_objects as go
import pytest

from balatro_analysis import export, metrics


def _open(data: bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(data))


def _csv(z: zipfile.ZipFile, path: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(z.read(path)))


def test_export_zip_contents(run):
    z = _open(export.export_zip([run], {"runs": "run_small"}))
    files = set(z.namelist())
    assert {
        "README.txt",
        "runs/run_small/episodes.csv",
        "runs/run_small/steps.csv",
        "runs/run_small/rounds.csv",
        "tables/summary.csv",
        "tables/by_deck.csv",
        "tables/most_bought.csv",
        "tables/score_ratio.csv",
        "tables/sell_then_buy__run_small.csv",
        "tables/cooccurrence__run_small.csv",
        "tables/action_by_ante__run_small.csv",
        "runs/run_small/shop_offers.csv",
        "tables/shop_visits.csv",
        "tables/spend_by_category.csv",
        "tables/income_breakdown.csv",
        "tables/offer_conversion.csv",
        "tables/economy_kpis.csv",
        "tables/action_family_share.csv",
        "tables/decisions_per_visit.csv",
        "tables/action_availability.csv",
        "tables/action_transitions__run_small.csv",
        "tables/top_sequences__run_small.csv",
    } <= files
    readme = z.read("README.txt").decode()
    assert "runs: run_small" in readme
    assert "runs/run_small/steps.csv  (8 rows)" in readme


def test_unique_slugs():
    assert export.unique_slugs(["x/y", "x_y", "x y", "z"]) == ["x_y", "x_y_2", "x_y_3", "z"]
    assert export.unique_slugs(["a", "a_2", "a"]) == ["a", "a_2", "a_3"]


def test_export_runs_with_colliding_names(run):
    """Nested run names like "x/y" and "x_y" don't overwrite each other in the ZIP."""
    other = metrics.filter_episodes(run, 0, 99, decks=["b_blue"])
    run.name, other.name = "x/y", "x_y"
    files = set(_open(export.export_zip([run, other], {})).namelist())
    assert {"runs/x_y/episodes.csv", "runs/x_y_2/episodes.csv"} <= files
    assert {"tables/sell_then_buy__x_y.csv", "tables/top_sequences__x_y_2.csv"} <= files


def test_export_has_keys_and_names(run):
    z = _open(export.export_zip([run], {}))
    eps = _csv(z, "runs/run_small/episodes.csv")
    assert eps.loc[0, "deck"] == "b_red"
    assert eps.loc[0, "deck_name"] == "Red Deck"
    assert eps.loc[0, "final_jokers_name"] == "Joker;Greedy Joker"
    steps = _csv(z, "runs/run_small/steps.csv")
    assert steps.loc[3, "jokers_before"] == "j_joker;j_greedy_joker"  # lists flattened
    assert steps.loc[0, "action_name_name"] == "Buy card 1"
    assert steps.loc[0, "legal"] == "0;1;2;16"  # int lists flattened too
    avail = _csv(z, "tables/action_availability.csv")
    assert avail.loc[0, "action_name"] == "Buy card 1"
    swaps = _csv(z, "tables/sell_then_buy__run_small.csv")
    assert swaps.iloc[0][["sold_name", "bought_name"]].tolist() == ["Joker", "Banner"]


def test_export_respects_filter(run):
    blue = metrics.filter_episodes(run, 0, 99, decks=["b_blue"])
    z = _open(export.export_zip([blue], {}))
    assert _csv(z, "runs/run_small/episodes.csv")["episode"].tolist() == [1]
    assert set(_csv(z, "runs/run_small/steps.csv")["episode"]) == {1}


def test_render_and_charts_zip(monkeypatch):
    calls = []
    monkeypatch.setattr(
        go.Figure, "to_image", lambda self, format, **kw: calls.append(format) or b"img"
    )

    def fake_write_images(figs, paths, format, **kw):
        for p in paths:
            p.write_bytes(format.encode())

    monkeypatch.setattr(export.pio, "write_images", fake_write_images)
    fig = go.Figure(go.Bar(y=[1, 2]))
    assert export.render(fig, "png") == b"img"
    export.render(fig, "png")
    assert calls == ["png"]  # cached
    z = _open(export.charts_zip({"a": fig, "b": fig}, ["png", "svg"]))
    assert sorted(z.namelist()) == ["charts/a.png", "charts/a.svg", "charts/b.png", "charts/b.svg"]


def test_charts_zip_readme(monkeypatch):
    monkeypatch.setattr(
        export.pio, "write_images", lambda figs, paths, format, **kw: [p.touch() for p in paths]
    )
    fig = go.Figure(go.Bar(y=[1, 2]))
    descriptions = {"ante_reached": "Share of episodes per ante.", "price_paid": "Price per buy."}
    z = _open(export.charts_zip({"ante_reached": fig, "price_paid": fig}, ["png"], descriptions))
    readme = z.read("charts/README.md").decode()
    assert "## ante_reached\n\n`ante_reached.png`\n\nShare of episodes per ante." in readme
    assert "## price_paid" in readme and "Price per buy." in readme


def test_export_problem_reports_failure(monkeypatch):
    def boom(self, **kw):
        raise RuntimeError("no browser\nmore detail")

    monkeypatch.setattr(go.Figure, "to_image", boom)
    assert export.export_problem() == "RuntimeError: no browser"


@pytest.mark.slow
def test_real_render():
    if export.export_problem():
        pytest.skip("kaleido can't render here")
    png = export.render(go.Figure(go.Bar(y=[1, 2])), "png")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    svg = export.render(go.Figure(go.Bar(y=[1, 2])), "svg")
    assert b"<svg" in svg[:300]

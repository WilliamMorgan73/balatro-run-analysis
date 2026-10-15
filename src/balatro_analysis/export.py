"""Export the current selection: data as a ZIP of CSVs, charts as PNG/SVG.

Data exports always carry exact keys *and* in-game names (``<col>_name``),
whatever the dashboard's naming toggle says, so files work for both analysis
and presenting.
"""

from __future__ import annotations

import hashlib
import io
import re
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from balatro_analysis import economy, metrics
from balatro_analysis.loader import Run
from balatro_analysis.names import DEFAULT_KEYS, Names

CHART_SIZE = {"width": 1600, "height": 900}
PNG_SCALE = 2
_PRETTY = Names(pretty=True)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "chart"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def with_names(df: pd.DataFrame, keys: dict[str, str] | None = None) -> pd.DataFrame:
    """Add a ``<col>_name`` column after every key column; lists become ``;``-joined."""
    out = df.copy()
    for col in out.columns:
        if out[col].map(lambda v: isinstance(v, list)).any():
            out[col] = out[col].map(lambda v: ";".join(v) if isinstance(v, list) else v)
    kinds = {**DEFAULT_KEYS, **(keys or {})}
    for col, kind in kinds.items():
        if col in out and f"{col}_name" not in out:
            names = out[col].map(_PRETTY.of(kind))
            if kind == "items":
                names = names.str.replace(", ", ";", regex=False)
            out.insert(out.columns.get_loc(col) + 1, f"{col}_name", names)
    return out


def computed_tables(runs: list[Run], window: int = 50, top: int = 15) -> dict[str, pd.DataFrame]:
    """Every table the dashboard shows, keyed by file stem."""
    tables: dict[str, pd.DataFrame] = {
        "summary": with_names(metrics.summary(runs), {"decks": "items", "stakes": "stakes"}),
        "by_deck": with_names(metrics.by_deck(runs)),
        "learning_curve": metrics.learning_curve(runs, window),
        "ante_distribution": metrics.ante_distribution(runs),
        "action_frequency": with_names(metrics.action_frequency(runs), {"action": "action"}),
        "money_by_ante": metrics.money_by_ante(runs),
        "spend_per_episode": metrics.spend_per_episode(runs),
        "most_bought": with_names(metrics.most_bought(runs, top)),
        "first_bought_ante": with_names(metrics.first_bought_ante(runs)),
        "joker_stats": with_names(metrics.joker_stats(runs).reset_index()),
        "economy_kpis": economy.kpis(runs),
        "shop_visits": economy.shop_visits(runs),
        "money_by_round": economy.money_by_round(runs),
        "spend_by_category": economy.spend_by_category(runs),
        "spend_by_category_by_ante": economy.spend_by_category(runs, by_ante=True),
        "income_by_source": economy.income_by_source(runs),
        "income_breakdown": economy.income_breakdown(runs),
        "interest_efficiency": economy.interest_efficiency(runs),
        "reroll_stats": economy.reroll_stats(runs),
        "price_paid": with_names(economy.price_paid(runs)),
        "episode_economy": economy.episode_economy(runs),
        "economy_vs_outcome": economy.outcome_correlations(economy.episode_economy(runs)),
        "offer_conversion": economy.offer_conversion(runs),
        "skipped_jokers": with_names(economy.skipped_jokers(runs), {"key": "item"}),
        "score_ratio": with_names(metrics.score_ratio(runs)),
        "deaths": with_names(metrics.deaths(runs)),
    }
    for r in runs:
        tag = slug(r.name)
        tables[f"action_by_ante__{tag}"] = (
            metrics.action_by_ante(r).rename_axis("action").reset_index()
            if r.steps is not None
            else pd.DataFrame()
        )
        tables[f"cooccurrence__{tag}"] = (
            metrics.joker_cooccurrence(r, top).rename_axis("joker").reset_index()
        )
        tables[f"sell_then_buy__{tag}"] = with_names(
            metrics.sell_then_buy(r), {"sold": "item", "bought": "item"}
        )
    return {k: v for k, v in tables.items() if v is not None and not v.empty}


def _csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def export_zip(
    runs: list[Run],
    selection: dict[str, object],
    window: int = 50,
    top: int = 15,
) -> bytes:
    """ZIP with ``runs/<run>/{episodes,steps,rounds}.csv``, ``tables/*.csv`` and a README."""
    buf = io.BytesIO()
    written: list[str] = []
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:

        def put(path: str, df: pd.DataFrame) -> None:
            z.writestr(path, _csv(df))
            written.append(f"{path}  ({len(df)} rows)")

        for r in runs:
            base = f"runs/{slug(r.name)}"
            put(f"{base}/episodes.csv", with_names(r.episodes))
            if r.steps is not None:
                put(f"{base}/steps.csv", with_names(r.steps))
            if r.rounds is not None:
                put(f"{base}/rounds.csv", with_names(r.rounds))
            if r.shops is not None:  # one row per offered item
                put(f"{base}/shop_offers.csv", with_names(economy.offers(r), {"key": "item"}))
        for name, df in computed_tables(runs, window, top).items():
            put(f"tables/{name}.csv", df)

        lines = [
            "Balatro run analysis export",
            f"exported: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "Selection:",
            *(f"  {k}: {v}" for k, v in selection.items()),
            "",
            "Key columns (j_..., b_..., bl_..., buy_card_0) are exact game keys;",
            "each has a <column>_name twin with the in-game name. Lists are ';'-separated.",
            "",
            "Files:",
            *(f"  {w}" for w in written),
        ]
        z.writestr("README.txt", "\n".join(lines) + "\n")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def _for_export(fig: go.Figure) -> go.Figure:
    """Light theme, fixed slide size (the app restyles charts on screen)."""
    out = go.Figure(fig)
    out.update_layout(template="plotly_white", **CHART_SIZE)
    return out


def export_problem() -> str | None:
    """None if kaleido can render here, else a short reason."""
    try:
        go.Figure(go.Bar(y=[1])).to_image(format="svg", width=50, height=50)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}".splitlines()[0][:200]
    return None


_cache: dict[tuple[str, str], bytes] = {}


def render(fig: go.Figure, fmt: str) -> bytes:
    """PNG or SVG bytes for one figure (cached by figure content)."""
    fig = _for_export(fig)
    key = (hashlib.sha1(fig.to_json().encode()).hexdigest(), fmt)
    if key not in _cache:
        _cache[key] = fig.to_image(format=fmt, scale=PNG_SCALE if fmt == "png" else 1)
    return _cache[key]


def charts_zip(figs: dict[str, go.Figure], fmts: list[str]) -> bytes:
    """ZIP of ``charts/<slug>.<fmt>``; renders in one batch (one browser start)."""
    buf = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(buf, "w") as z:
        for fmt in fmts:
            paths = [Path(tmp) / f"{s}.{fmt}" for s in figs]
            pio.write_images(
                [_for_export(f) for f in figs.values()],
                paths,
                format=fmt,
                scale=PNG_SCALE if fmt == "png" else 1,
            )
            for p in paths:
                z.write(p, f"charts/{p.name}")
    return buf.getvalue()

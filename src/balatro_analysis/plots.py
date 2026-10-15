"""Plotly figure builders. Each takes a frame from ``metrics`` plus a ``Names``
(in-game names vs exact keys) and returns a figure."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from balatro_analysis.actions import FAMILIES
from balatro_analysis.names import Names

TEMPLATE = "plotly_white"


def _fig(fig: go.Figure, **layout) -> go.Figure:
    fig.update_layout(template=TEMPLATE, margin={"l": 10, "r": 10, "t": 40, "b": 10}, **layout)
    return fig


def _labels(names: Names, *cols: str) -> dict[str, str]:
    """Axis/legend titles for ``cols`` in the current naming mode."""
    return {c: names.column(c) for c in cols}


def _translucent(colour: str, alpha: float) -> str:
    """``#rrggbb`` -> ``rgba(r, g, b, alpha)``; other colour strings pass through."""
    if isinstance(colour, str) and colour.startswith("#") and len(colour) == 7:
        r, g, b = (int(colour[i : i + 2], 16) for i in (1, 3, 5))
        return f"rgba({r}, {g}, {b}, {alpha})"
    return colour


def learning_curve(df: pd.DataFrame, metric: str, names: Names, band: bool = True) -> go.Figure:
    fig = px.line(
        df,
        x="episode",
        y=metric,
        color="run",
        labels=_labels(names, "episode", "run"),
        title=f"Rolling {metric}" + (" (95% band)" if band else ""),
    )
    if band and f"{metric} low" in df:
        bands = []
        for line in fig.data:
            g = df[df["run"] == line.legendgroup]
            bands.append(
                go.Scatter(
                    x=[*g["episode"], *g["episode"][::-1]],
                    y=[*g[f"{metric} high"], *g[f"{metric} low"][::-1]],
                    fill="toself",
                    fillcolor=_translucent(line.line.color, 0.18),
                    line={"width": 0},
                    legendgroup=line.legendgroup,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        lines = len(fig.data)
        fig.add_traces(bands)
        fig.data = fig.data[lines:] + fig.data[:lines]  # bands under the lines
    return _fig(fig, hovermode="x unified")


def by_deck(df: pd.DataFrame, names: Names) -> go.Figure:
    df = df.assign(
        label=[
            f"{names.deck(d)} / {names.stake(s)}"
            for d, s in zip(df["deck"], df["stake"], strict=True)
        ]
    )
    fig = px.bar(
        df,
        x="label",
        y="mean_ante",
        color="run",
        barmode="group",
        hover_data=["episodes", "max_ante", "win_rate"],
        labels=_labels(names, "mean_ante", "run", "episodes", "max_ante", "win_rate"),
        title="Mean ante by deck and stake",
    )
    return _fig(fig, xaxis_title="")


def ante_distribution(df: pd.DataFrame, names: Names) -> go.Figure:
    fig = px.bar(
        df,
        x="ante",
        y="fraction",
        color="run",
        barmode="group",
        labels=_labels(names, "ante", "fraction", "run"),
        title="Ante reached",
    )
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(tickformat=".0%")
    return _fig(fig)


def action_frequency(df: pd.DataFrame, names: Names) -> go.Figure:
    df = names.frame(df, {"action": "action"}, rename=False)
    fig = px.bar(
        df,
        x="action",
        y="share",
        color="run",
        barmode="group",
        hover_data=["count"],
        labels=_labels(names, "action", "share", "run", "count"),
        title="Share of shop decisions per action",
    )
    fig.update_yaxes(tickformat=".0%")
    return _fig(fig)


def action_heatmap(table: pd.DataFrame, title: str, x_label: str, names: Names) -> go.Figure:
    """Rows: actions (keys), columns: anything; cells are shares."""
    table = table.rename(index=names.action)
    table.columns = [str(c) for c in table.columns]
    fig = px.imshow(
        table,
        text_auto=".0%",
        aspect="auto",
        color_continuous_scale="Blues",
        labels={"x": x_label, "y": names.column("action"), "color": names.column("share")},
        title=title,
    )
    fig.update_xaxes(type="category")
    return _fig(fig)


def action_by_ante(table: pd.DataFrame, run: str, names: Names) -> go.Figure:
    return action_heatmap(
        table, f"Action share within each ante: {run}", names.column("ante"), names
    )


def money_by_ante(df: pd.DataFrame, names: Names) -> go.Figure:
    fig = px.box(
        df,
        x="ante",
        y="dollars_before",
        color="run",
        labels=_labels(names, "ante", "dollars_before", "run"),
        title="Money before each shop decision",
    )
    fig.update_xaxes(dtick=1)
    return _fig(fig)


def spend_per_episode(df: pd.DataFrame, names: Names) -> go.Figure:
    fig = px.histogram(
        df,
        x="dollars_spent",
        color="run",
        barmode="overlay",
        opacity=0.6,
        histnorm="probability",
        labels=_labels(names, "dollars_spent", "run"),
        title="Money spent in the shop per episode",
    )
    return _fig(fig, yaxis_title="fraction of episodes")


def most_bought(df: pd.DataFrame, names: Names) -> go.Figure:
    df = names.frame(df, rename=False)  # joker column -> names
    fig = px.bar(
        df,
        x="bought",
        y="joker",
        color="run",
        barmode="group",
        orientation="h",
        labels=_labels(names, "bought", "run", "joker"),
        title="Most-bought jokers",
    )
    fig.update_yaxes(categoryorder="total ascending", title="")
    return _fig(fig, height=max(350, 22 * df["joker"].nunique()))


def first_bought_ante(df: pd.DataFrame, names: Names) -> go.Figure:
    fig = px.histogram(
        df,
        x="ante",
        color="run",
        barmode="group",
        histnorm="probability",
        labels=_labels(names, "ante", "run"),
        title="Ante in which each joker was first bought",
    )
    fig.update_xaxes(dtick=1)
    return _fig(fig, yaxis_title="fraction of joker purchases")


def cooccurrence(matrix: pd.DataFrame, run: str, names: Names) -> go.Figure:
    matrix = matrix.rename(index=names.item, columns=names.item)
    fig = px.imshow(
        matrix,
        text_auto=True,
        color_continuous_scale="Purples",
        title=f"Jokers held together at end of episode: {run}",
    )
    return _fig(fig, height=max(400, 30 * len(matrix)))


def score_ratio(df: pd.DataFrame, names: Names) -> go.Figure:
    fig = px.box(
        df,
        x="ante",
        y="ratio",
        color="run",
        facet_col="blind type",
        category_orders={"blind type": ["Small", "Big", "Boss"]},
        labels=_labels(names, "ante", "ratio", "run", "blind type"),
        title="Score / blind target (≥ 1 beats the blind)",
    )
    fig.add_hline(y=1, line_dash="dash", line_color="grey")
    fig.update_xaxes(dtick=1)
    return _fig(fig)


def deaths(df: pd.DataFrame, names: Names) -> go.Figure:
    counts = df.groupby(["run", "ante", "blind_key"], as_index=False).size()
    counts = names.frame(counts, rename=False)
    fig = px.bar(
        counts,
        x="blind_key",
        y="size",
        color="run",
        barmode="group",
        facet_col="ante",
        labels=_labels(names, "ante", "run", "blind_key"),
        title="Blind that ended the run",
    )
    fig.update_xaxes(categoryorder="total descending", title="")
    return _fig(fig, yaxis_title=names.column("episodes"))


def episode_money(timeline: pd.DataFrame, names: Names) -> go.Figure:
    timeline = names.frame(timeline, rename=False)
    fig = px.line(
        timeline,
        x="step",
        y="dollars_after",
        markers=True,
        hover_data=["action_name", "item_key", "ante"],
        labels=_labels(names, "step", "dollars_after", "action_name", "item_key", "ante"),
        title="Money after each shop decision",
    )
    return _fig(fig)


# ---------------------------------------------------------------------------
# Economy
# ---------------------------------------------------------------------------

CATEGORY_ORDER = {
    "category": [
        "Jokers",
        "Packs",
        "Rerolls",
        "Vouchers",
        "Tarot",
        "Planet",
        "Spectral",
        "Playing cards",
    ]
}


def money_curve(df: pd.DataFrame, names: Names) -> go.Figure:
    """Median money entering/leaving the shop by round, with an inter-quartile band."""
    fig = go.Figure()
    colours = px.colors.qualitative.Plotly
    for i, (run, g) in enumerate(df.groupby("run", sort=False)):
        colour = colours[i % len(colours)]
        for when, dash in (("entry", "solid"), ("exit", "dot")):
            w = g[g["when"] == when].sort_values("round")
            label = f"{run} · {'entering' if when == 'entry' else 'leaving'}"
            fig.add_scatter(
                x=list(w["round"]) + list(w["round"][::-1]),
                y=list(w["q75"]) + list(w["q25"][::-1]),
                fill="toself",
                fillcolor=colour,
                opacity=0.12,
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
                legendgroup=label,
            )
            fig.add_scatter(
                x=w["round"],
                y=w["median"],
                mode="lines+markers",
                name=label,
                legendgroup=label,
                line={"color": colour, "dash": dash},
                customdata=w[["ante", "visits"]],
                hovertemplate="round %{x} (ante %{customdata[0]})<br>median $%{y}"
                "<br>%{customdata[1]} visits<extra></extra>",
            )
    fig.update_layout(title="Money entering / leaving the shop (median, IQR band)")
    return _fig(fig, xaxis_title=names.column("round"), yaxis_title="$")


def entry_exit_by_ante(visits: pd.DataFrame, names: Names) -> go.Figure:
    long = visits.melt(
        id_vars=["run", "ante"], value_vars=["entry", "exit"], var_name="when", value_name="dollars"
    )
    long["when"] = long["when"].map({"entry": "entering", "exit": "leaving"})
    fig = px.box(
        long,
        x="ante",
        y="dollars",
        color="run",
        facet_col="when",
        labels=_labels(names, "ante", "run") | {"dollars": "$", "when": ""},
        title="Money entering vs leaving each shop, by ante",
    )
    fig.update_xaxes(dtick=1)
    return _fig(fig)


def final_dollars(episodes: pd.DataFrame, names: Names) -> go.Figure:
    fig = px.histogram(
        episodes,
        x="final_dollars",
        color="run",
        barmode="overlay",
        opacity=0.6,
        histnorm="probability",
        labels=_labels(names, "final_dollars", "run"),
        title="Money left when the run ended",
    )
    return _fig(fig, yaxis_title="fraction of episodes")


def spend_share(df: pd.DataFrame, names: Names) -> go.Figure:
    fig = px.bar(
        df,
        x="run",
        y="share",
        color="category",
        category_orders=CATEGORY_ORDER,
        hover_data=["spent", "per_episode"],
        labels=_labels(names, "run", "share", "category", "spent", "per_episode"),
        title="Where the money goes (share of shop spending)",
    )
    fig.update_yaxes(tickformat=".0%")
    return _fig(fig)


def spend_by_ante(df: pd.DataFrame, names: Names) -> go.Figure:
    fig = px.bar(
        df,
        x="ante",
        y="share",
        color="category",
        facet_col="run",
        category_orders=CATEGORY_ORDER,
        hover_data=["spent"],
        labels=_labels(names, "ante", "share", "category", "spent", "run"),
        title="Spending mix by ante",
    )
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(tickformat=".0%")
    return _fig(fig)


def income_sources(df: pd.DataFrame, names: Names) -> go.Figure:
    fig = px.bar(
        df,
        x="per_episode",
        y="source",
        color="run",
        barmode="group",
        orientation="h",
        hover_data=["kind"],
        labels=_labels(names, "run") | {"per_episode": "$ per episode", "source": ""},
        title="Income per episode by source",
    )
    fig.update_yaxes(categoryorder="total ascending")
    return _fig(fig)


def exit_brackets(visits: pd.DataFrame, names: Names) -> go.Figure:
    cap = int(visits["interest_cap"].mode().iloc[0]) if len(visits) else 25
    fig = px.histogram(
        visits,
        x="exit",
        color="run",
        barmode="group",
        histnorm="probability",
        labels=_labels(names, "run") | {"exit": "$ leaving the shop"},
        title="Money left after shopping ($5 brackets = $1 interest each)",
    )
    fig.update_traces(xbins={"start": 0, "size": 5})
    fig.add_vline(
        x=cap, line_dash="dash", line_color="grey", annotation_text=f"interest cap ${cap}"
    )
    return _fig(fig, yaxis_title="fraction of shop visits")


def interest_by_ante(visits: pd.DataFrame, names: Names) -> go.Figure:
    g = visits.groupby(["run", "ante"], as_index=False).agg(
        interest=("est_interest", "mean"), at_cap=("at_cap", "mean")
    )
    fig = px.line(
        g,
        x="ante",
        y="interest",
        color="run",
        markers=True,
        hover_data={"at_cap": ":.0%"},
        labels=_labels(names, "ante", "run") | {"interest": "$ interest", "at_cap": "at cap"},
        title="Interest the exit balance would earn (estimated), by ante",
    )
    fig.update_xaxes(dtick=1)
    return _fig(fig)


def interest_efficiency(df: pd.DataFrame, names: Names) -> go.Figure:
    fig = px.bar(
        df,
        x="ante",
        y="efficiency",
        color="run",
        barmode="group",
        hover_data=["interest", "max_interest"],
        labels=_labels(names, "ante", "run")
        | {"efficiency": "share of max interest", "max_interest": "max $"},
        title="Interest earned vs the most the cap allows (exact)",
    )
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(tickformat=".0%")
    return _fig(fig)


def rerolls_per_visit(visits: pd.DataFrame, names: Names) -> go.Figure:
    fig = px.histogram(
        visits,
        x="rerolls",
        color="run",
        barmode="group",
        histnorm="probability",
        labels=_labels(names, "run") | {"rerolls": "rerolls in one shop visit"},
        title="Rerolls per shop visit",
    )
    fig.update_xaxes(dtick=1)
    return _fig(fig, yaxis_title="fraction of shop visits")


def reroll_by_ante(df: pd.DataFrame, names: Names) -> go.Figure:
    fig = px.line(
        df,
        x="ante",
        y="reroll_spend_per_visit",
        color="run",
        markers=True,
        hover_data={"rerolls_per_visit": ":.2f", "any_reroll": ":.0%", "visits": True},
        labels=_labels(names, "ante", "run")
        | {
            "reroll_spend_per_visit": "$ on rerolls per visit",
            "rerolls_per_visit": "rerolls / visit",
            "any_reroll": "visits with a reroll",
        },
        title="Reroll spending by ante",
    )
    fig.update_xaxes(dtick=1)
    return _fig(fig)


def price_paid(df: pd.DataFrame, names: Names) -> go.Figure:
    fig = px.box(
        df,
        x="category",
        y="price",
        color="run",
        category_orders=CATEGORY_ORDER,
        labels=_labels(names, "category", "run") | {"price": "$ paid"},
        title="Price paid per purchase",
    )
    return _fig(fig)


def joker_price_by_ante(df: pd.DataFrame, names: Names) -> go.Figure:
    j = df[df["category"] == "Jokers"].groupby(["run", "ante"], as_index=False)["price"].mean()
    fig = px.line(
        j,
        x="ante",
        y="price",
        color="run",
        markers=True,
        labels=_labels(names, "ante", "run") | {"price": "mean $ per joker"},
        title="Mean price paid per joker, by ante",
    )
    fig.update_xaxes(dtick=1)
    return _fig(fig)


def economy_by_outcome(econ: pd.DataFrame, metric: str, label: str, names: Names) -> go.Figure:
    fig = px.box(
        econ,
        x="ante",
        y=metric,
        color="run",
        labels=_labels(names, "run") | {"ante": "ante reached", metric: label},
        title=f"{label} vs how far the run got",
    )
    fig.update_xaxes(dtick=1)
    return _fig(fig)


def _significance(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``sig``: "*" on bars whose BH-adjusted p (q) < 0.05, for the bar text."""
    if "significant" not in df:
        return df.assign(sig="")
    return df.assign(sig=df["significant"].map({True: "*", False: ""}).fillna(""))


def outcome_correlations(df: pd.DataFrame, names: Names) -> go.Figure:
    df = _significance(df)
    fig = px.bar(
        df,
        x="spearman_rho",
        y="metric",
        color="run",
        barmode="group",
        orientation="h",
        text="sig",
        hover_data={c: ":.3g" for c in ("p_value", "q_value") if c in df}
        | {"episodes": True, "sig": False},
        labels=_labels(names, "run", "episodes")
        | {
            "spearman_rho": "Spearman ρ with ante reached",
            "metric": "",
            "p_value": "p",
            "q_value": "q (BH-adjusted p)",
        },
        title="Which economy habits go with deeper runs?",
    )
    fig.add_vline(x=0, line_color="grey")
    fig.update_xaxes(range=[-1, 1])
    fig.update_traces(textposition="outside", cliponaxis=False)
    return _fig(fig, height=max(380, 34 * df["metric"].nunique()))


def income_breakdown(df: pd.DataFrame, names: Names) -> go.Figure:
    fig = px.bar(
        df,
        x="ante",
        y="value",
        color="component",
        facet_col="run",
        labels=_labels(names, "ante", "run") | {"value": "$ per beaten blind", "component": ""},
        title="Cash-out income per beaten blind (exact)",
    )
    fig.update_xaxes(dtick=1)
    return _fig(fig)


def offer_conversion(df: pd.DataFrame, names: Names) -> go.Figure:
    long = df.melt(
        id_vars=["run", "category"],
        value_vars=["affordable_share", "buy_rate", "buy_rate_when_affordable"],
        var_name="measure",
    )
    long["measure"] = long["measure"].map(
        {
            "affordable_share": "affordable when offered",
            "buy_rate": "bought (of offered)",
            "buy_rate_when_affordable": "bought (of affordable)",
        }
    )
    fig = px.bar(
        long,
        x="category",
        y="value",
        color="measure",
        barmode="group",
        facet_col="run",
        category_orders=CATEGORY_ORDER,
        labels=_labels(names, "category", "run") | {"value": "", "measure": ""},
        title="What the shop offered, what was affordable, what was bought",
    )
    fig.update_yaxes(tickformat=".0%")
    return _fig(fig)


def offered_vs_paid(conversion: pd.DataFrame, paid: pd.DataFrame, names: Names) -> go.Figure:
    p = paid.groupby(["run", "category"], as_index=False)["price"].mean()
    df = conversion[["run", "category", "mean_price"]].merge(p, on=["run", "category"], how="left")
    long = df.melt(id_vars=["run", "category"], value_vars=["mean_price", "price"], var_name="what")
    long["what"] = long["what"].map({"mean_price": "offered", "price": "paid"})
    fig = px.bar(
        long,
        x="category",
        y="value",
        color="what",
        barmode="group",
        facet_col="run",
        category_orders=CATEGORY_ORDER,
        labels=_labels(names, "category", "run") | {"value": "mean $", "what": ""},
        title="Mean price offered vs paid",
    )
    return _fig(fig)


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def family_share(df: pd.DataFrame, names: Names) -> go.Figure:
    fig = px.bar(
        df,
        x="share",
        y="run",
        color="family",
        orientation="h",
        hover_data=["count"],
        category_orders={"family": FAMILIES},
        labels=_labels(names, "share", "run", "count") | {"family": "Kind of decision"},
        title="What the shop decisions are",
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(autorange="reversed")  # runs top to bottom in selection order
    return _fig(fig, height=max(260, 70 * df["run"].nunique() + 160))


def decisions_per_visit(visits: pd.DataFrame, names: Names) -> go.Figure:
    fig = px.histogram(
        visits,
        x="decisions",
        color="run",
        barmode="group",
        histnorm="probability",
        labels=_labels(names, "run") | {"decisions": "decisions in one shop visit"},
        title="Decisions per shop visit",
    )
    fig.update_xaxes(dtick=1)
    return _fig(fig, yaxis_title="fraction of shop visits")


def _state(name: str, names: Names) -> str:
    if names.pretty and name in ("START", "END"):
        return {"START": "Enter shop", "END": "Log ends"}[name]
    return names.action(name)


def transition_matrix(matrix: pd.DataFrame, run: str, names: Names) -> go.Figure:
    m = matrix.rename(index=lambda n: _state(n, names), columns=lambda n: _state(n, names))
    fig = px.imshow(
        m,
        text_auto=".0%",
        aspect="auto",
        color_continuous_scale="Blues",
        labels={"x": "next decision", "y": "decision", "color": "rate"},
        title=f"What follows each decision (rows sum to 100%): {run}",
    )
    return _fig(fig, height=max(380, 30 * len(m) + 160))


def family_over_training(df: pd.DataFrame, names: Names) -> go.Figure:
    fig = px.area(
        df,
        x="episode",
        y="share",
        color="family",
        facet_row="run" if df["run"].nunique() > 1 else None,
        category_orders={"family": FAMILIES},
        labels=_labels(names, "episode", "share", "run")
        | {"family": "Kind of decision", "episode": "episode (bin start)"},
        title="Kinds of decision over training",
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_xaxes(matches=None, showticklabels=True)  # runs differ in length
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return _fig(fig, height=max(380, 260 * df["run"].nunique()))


def action_outcome_correlations(df: pd.DataFrame, names: Names) -> go.Figure:
    df = _significance(df)
    fig = px.bar(
        df,
        x="spearman_rho",
        y="family",
        color="run",
        barmode="group",
        orientation="h",
        text="sig",
        hover_data={c: ":.3g" for c in ("p_value", "q_value") if c in df}
        | {"episodes": True, "sig": False},
        category_orders={"family": FAMILIES},
        labels=_labels(names, "run", "episodes")
        | {
            "spearman_rho": "Spearman ρ with ante reached",
            "family": "",
            "p_value": "p",
            "q_value": "q (BH-adjusted p)",
        },
        title="Which shop habits (per visit) go with deeper runs?",
    )
    fig.add_vline(x=0, line_color="grey")
    fig.update_xaxes(range=[-1, 1])
    fig.update_traces(textposition="outside", cliponaxis=False)
    return _fig(fig, height=max(380, 34 * df["family"].nunique()))


def _availability_rows(df: pd.DataFrame, names: Names) -> pd.DataFrame:
    df = df[df["legal_count"] > 0]
    return names.frame(df, {"action": "action"}, rename=False)


def availability(df: pd.DataFrame, names: Names) -> go.Figure:
    long = _availability_rows(df, names).melt(
        id_vars=["run", "action", "legal_count", "chosen"],
        value_vars=["legal_rate", "chosen_when_legal"],
        var_name="measure",
        value_name="rate",
    )
    long["measure"] = long["measure"].map(
        {"legal_rate": "legal (share of decisions)", "chosen_when_legal": "chosen when legal"}
    )
    multi = long["run"].nunique() > 1
    fig = px.bar(
        long,
        x="action",
        y="rate",
        color="measure",
        barmode="group",
        facet_row="run" if multi else None,
        hover_data=["legal_count", "chosen"],
        labels=_labels(names, "action", "run")
        | {"rate": "", "measure": "", "legal_count": "decisions legal", "chosen": "times chosen"},
        title="How often each action was available, and taken when it was",
    )
    fig.update_yaxes(tickformat=".0%")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return _fig(fig, height=max(400, 300 * long["run"].nunique()))


def preference(df: pd.DataFrame, names: Names) -> go.Figure:
    fig = px.bar(
        _availability_rows(df, names),
        x="action",
        y="preference",
        color="run",
        barmode="group",
        hover_data=["chosen_when_legal", "uniform_expected", "legal_count"],
        labels=_labels(names, "action", "run")
        | {
            "preference": "chosen ÷ uniform-random rate",
            "chosen_when_legal": "chosen when legal",
            "uniform_expected": "uniform-random rate",
            "legal_count": "decisions legal",
        },
        title="Preference vs a uniform-random agent (1 = random, above = favoured)",
    )
    fig.add_hline(y=1, line_dash="dash", line_color="grey", annotation_text="uniform random")
    return _fig(fig)


def legal_counts(df: pd.DataFrame, names: Names) -> go.Figure:
    fig = px.bar(
        df,
        x="n_legal",
        y="fraction",
        color="run",
        barmode="group",
        hover_data=["count"],
        labels=_labels(names, "run", "fraction", "count") | {"n_legal": "legal actions"},
        title="Legal actions per decision",
    )
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(tickformat=".0%")
    return _fig(fig, yaxis_title="fraction of decisions")

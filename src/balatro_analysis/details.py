"""Click-for-details modals: tiles that open a ``<dialog>`` with quick information.

Everything is prerendered HTML for ``st.html(..., unsafe_allow_javascript=True)``.
A tile carries ``data-dialog="<id>"``; ``MODAL_JS`` opens the matching dialog on
click (or Enter/Space) and closes it on a backdrop click. Esc and the close button
close it without script. Ids are prefixed per grid so grids on one page don't clash.

Joker dialogs are built in ``collection`` (from ``JokerCard``); this module holds
the shared pieces plus the deck and boss blind dialogs.
"""

from __future__ import annotations

import html
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from balatro_analysis import catalog
from balatro_analysis.game_text import GameText
from balatro_analysis.names import Names

MODAL_CSS = """
<style>
.dt-tile{cursor:pointer;transition:transform .12s ease,box-shadow .12s ease}
.dt-tile:hover{transform:translateY(-3px)}
.dt-tile:focus-visible{outline:3px solid #ff9a00;outline-offset:2px}
.dt-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px}
.dt-mini{background:#3a4c56;border:2px solid #24323a;border-radius:12px;padding:10px 8px;
  display:flex;flex-direction:column;align-items:center;gap:4px;color:#fff;
  font-family:system-ui,sans-serif;box-shadow:0 3px 0 #1d282e;text-align:center}
.dt-mini img{image-rendering:pixelated}
.dt-mini .dt-name{font-weight:800;font-size:14px;line-height:1.15}
.dt-mini .dt-sub{font-size:12px;opacity:.85;line-height:1.35}
.dt-mini .dt-sub b{color:#ff9a00}
.dt-noart{border-radius:8px;background:#708b91;display:flex;align-items:center;
  justify-content:center;text-align:center;font-weight:700;padding:6px;color:#fff;
  box-sizing:border-box}
.dt-chip-noart{border-radius:50%}
dialog.dt-dialog{border:none;border-radius:16px;padding:0;max-width:min(560px,92vw);
  width:100%;background:#3a4c56;color:#fff;font-family:system-ui,sans-serif;
  box-shadow:0 10px 40px rgba(0,0,0,.45),0 4px 0 #1d282e}
dialog.dt-dialog::backdrop{background:rgba(10,16,20,.6)}
.dt-head{display:flex;gap:16px;align-items:center;padding:18px 18px 12px}
.dt-head img{image-rendering:pixelated;flex:none}
.dt-title{font-weight:800;font-size:22px;line-height:1.15}
.dt-key{font-family:ui-monospace,monospace;font-size:12px;opacity:.7;margin-top:2px}
.dt-tags{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px;align-items:center}
.dt-tag{border-radius:6px;padding:1px 8px;font-size:12px;font-weight:800;color:#fff;
  background:#708b91}
.dt-money{color:#f3b958;font-weight:800;font-size:13px}
.dt-close{margin:0;position:absolute;top:8px;right:10px}
.dt-close button{background:transparent;border:none;color:#fff;font-size:24px;line-height:1;
  cursor:pointer;opacity:.75;padding:4px 8px;border-radius:8px}
.dt-close button:hover{opacity:1;background:rgba(255,255,255,.12)}
.dt-body{padding:0 18px 18px;display:flex;flex-direction:column;gap:12px}
.dt-desc{background:#fff;color:#4F6367;border-radius:10px;padding:8px 12px;font-size:14px;
  line-height:1.4;text-align:center;font-weight:600}
.dt-desc.dt-nodesc{font-style:italic;font-weight:500}
.dt-section{font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:.06em;
  opacity:.7;margin-bottom:-6px}
.dt-stats{width:100%;border-collapse:collapse;font-size:13.5px}
.dt-stats td{padding:4px 2px;border-bottom:1px solid rgba(255,255,255,.1)}
.dt-stats td:last-child{text-align:right;font-weight:800;color:#ff9a00;
  font-variant-numeric:tabular-nums}
.dt-stats th{text-align:left;font-size:12px;opacity:.7;padding:2px;font-weight:700}
.dt-stats th:not(:first-child),.dt-stats td:not(:first-child){text-align:right}
.dt-stats.dt-plain td:not(:first-child){color:#fff;font-weight:600}
.dt-empty{font-size:13px;opacity:.8;font-style:italic}
.dt-jokers{display:flex;gap:10px;flex-wrap:wrap}
.dt-jokers figure{margin:0;display:flex;flex-direction:column;align-items:center;gap:2px;
  font-size:12px;width:86px;text-align:center}
.dt-jokers img{width:71px;height:95px;image-rendering:pixelated}
</style>
"""

MODAL_JS = """
<script>
(() => {
  if (window.__dtModals) return;
  window.__dtModals = true;
  const open = (tile) => {
    const d = document.getElementById(tile.dataset.dialog);
    if (d && !d.open) d.showModal();
  };
  document.addEventListener("click", (e) => {
    if (e.target instanceof HTMLDialogElement && e.target.classList.contains("dt-dialog")) {
      e.target.close();  // a click on the backdrop lands on the dialog itself
      return;
    }
    const tile = e.target.closest && e.target.closest("[data-dialog]");
    if (tile) open(tile);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" && e.key !== " ") return;
    const tile = e.target.closest && e.target.closest("[data-dialog]");
    if (tile) { e.preventDefault(); open(tile); }
  });
})();
</script>
"""


def dialog_id(prefix: str, key: str) -> str:
    return f"{prefix}-" + re.sub(r"[^\w-]", "_", key)


def tile_html(did: str, inner: str, label: str) -> str:
    """A clickable, focusable wrapper that opens dialog ``did``."""
    return (
        f'<div class="dt-tile" role="button" tabindex="0" data-dialog="{did}" '
        f'aria-haspopup="dialog" title="Details: {html.escape(label)}">{inner}</div>'
    )


def dialog_html(
    did: str,
    title: str,
    subtitle: str,
    art: str,
    body: str,
    tags: str = "",
) -> str:
    """A modal: header (art, title, subtitle, tags) and body. ``art`` is HTML."""
    tags = f'<div class="dt-tags">{tags}</div>' if tags else ""
    return (
        f'<dialog class="dt-dialog" id="{did}" aria-label="{html.escape(title)}">'
        '<form method="dialog" class="dt-close"><button aria-label="Close">&times;</button></form>'
        f'<div class="dt-head">{art}<div><div class="dt-title">{html.escape(title)}</div>'
        f'<div class="dt-key">{html.escape(subtitle)}</div>{tags}</div></div>'
        f'<div class="dt-body">{body}</div></dialog>'
    )


def art_html(uri: str | None, alt: str, width: int, height: int, chip: bool = False) -> str:
    if uri:
        return f'<img src="{uri}" alt="{html.escape(alt)}" width="{width}" height="{height}">'
    cls = "dt-noart dt-chip-noart" if chip else "dt-noart"
    return f'<div class="{cls}" style="width:{width}px;height:{height}px">{html.escape(alt)}</div>'


def desc_html(lines: list[str] | None, fallback: str) -> str:
    if lines:
        return f'<div class="dt-desc">{"<br>".join(lines)}</div>'
    return f'<div class="dt-desc dt-nodesc">{html.escape(fallback)}</div>'


def stats_html(pairs: Iterable[tuple[str, str]]) -> str:
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in pairs)
    return f'<table class="dt-stats">{rows}</table>'


def table_html(df: pd.DataFrame, plain: bool = True) -> str:
    """A small table; cells are escaped, column names are used as given."""
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
        for row in df.itertuples(index=False)
    )
    cls = "dt-stats dt-plain" if plain else "dt-stats"
    return f'<table class="{cls}"><tr>{head}</tr>{body}</table>'


def section(title: str) -> str:
    return f'<div class="dt-section">{html.escape(title)}</div>'


def pct(v) -> str:
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.0%}"


def num(v, fmt: str = "{:.2f}") -> str:
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else fmt.format(v)


def page(tiles: list[str], dialogs: list[str], grid_class: str = "dt-grid") -> str:
    return (
        MODAL_CSS
        + f'<div class="{grid_class}">'
        + "".join(tiles)
        + "</div>"
        + "".join(dialogs)
        + MODAL_JS
    )


# ---------------------------------------------------------------------------
# Decks
# ---------------------------------------------------------------------------


@dataclass
class DeckInfo:
    key: str
    lines: list[str] = field(default_factory=list)  # effect text as HTML (needs the game)
    icon: str | None = None
    rows: pd.DataFrame = field(default_factory=pd.DataFrame)  # metrics.by_deck rows
    top_jokers: list[tuple[str, int, str | None]] = field(default_factory=list)  # key, n, icon

    @property
    def episodes(self) -> int:
        return int(self.rows["episodes"].sum()) if len(self.rows) else 0

    def _weighted(self, col: str) -> float | None:
        n = self.episodes
        return float((self.rows[col] * self.rows["episodes"]).sum() / n) if n else None

    @property
    def mean_ante(self) -> float | None:
        return self._weighted("mean_ante")

    @property
    def win_rate(self) -> float | None:
        return self._weighted("win_rate")

    @property
    def max_ante(self) -> int | None:
        return int(self.rows["max_ante"].max()) if len(self.rows) else None


def deck_info(
    key: str,
    deck_table: pd.DataFrame,
    text: GameText | None,
    icon,
    top_jokers: pd.Series | None = None,
) -> DeckInfo:
    """``icon`` maps a key to a data URI (or None)."""
    dt = text.deck(key) if text else None
    rows = deck_table[deck_table["deck"] == key] if len(deck_table) else deck_table
    tops = [(j, int(n), icon(j)) for j, n in (top_jokers if top_jokers is not None else {}).items()]
    return DeckInfo(key, dt.lines if dt else [], icon(key), rows, tops)


def deck_tile(d: DeckInfo, names: Names, did: str) -> str:
    name = names.deck(d.key)
    sub = (
        f"{d.episodes} episodes<br>mean ante <b>{num(d.mean_ante)}</b>"
        if d.episodes
        else "Not played in the selection"
    )
    inner = (
        f'<div class="dt-mini">{art_html(d.icon, name, 71, 95)}'
        f'<div class="dt-name">{html.escape(name)}</div><div class="dt-sub">{sub}</div></div>'
    )
    return tile_html(did, inner, name)


def deck_dialog(d: DeckInfo, names: Names, did: str) -> str:
    name = names.deck(d.key)
    body = desc_html(d.lines, "Effect text needs the game (see Game assets)")
    if d.episodes:
        body += section("In the selected runs") + stats_html(
            [
                ("Episodes", str(d.episodes)),
                ("Mean ante reached", num(d.mean_ante)),
                ("Best ante reached", str(d.max_ante)),
                ("Win rate", pct(d.win_rate)),
            ]
        )
        if len(d.rows) > 1:
            per = pd.DataFrame(
                {
                    "Run": d.rows["run"],
                    "Stake": d.rows["stake"].map(names.stake),
                    "Episodes": d.rows["episodes"],
                    "Mean ante": d.rows["mean_ante"].map(num),
                    "Win rate": d.rows["win_rate"].map(pct),
                }
            )
            body += section("By run and stake") + table_html(per)
        if d.top_jokers:
            figs = "".join(
                f"<figure>{art_html(uri, names.item(j), 71, 95)}"
                f"<figcaption>{html.escape(names.item(j))}<br><b>{n}&times;</b></figcaption>"
                "</figure>"
                for j, n, uri in d.top_jokers
            )
            body += section("Most held at the end") + f'<div class="dt-jokers">{figs}</div>'
    else:
        body += '<div class="dt-empty">Not played in the selected runs.</div>'
    sub = d.key if names.pretty else catalog.name(d.key)
    return dialog_html(did, name, sub, art_html(d.icon, name, 107, 143), body)


def deck_grid_html(decks: list[DeckInfo], names: Names, prefix: str = "deck") -> str:
    ids = [dialog_id(prefix, d.key) for d in decks]
    return page(
        [deck_tile(d, names, i) for d, i in zip(decks, ids, strict=True)],
        [deck_dialog(d, names, i) for d, i in zip(decks, ids, strict=True)],
    )


# ---------------------------------------------------------------------------
# Boss blinds
# ---------------------------------------------------------------------------


@dataclass
class BlindInfo:
    key: str
    lines: list[str] = field(default_factory=list)
    icon: str | None = None
    faced: int = 0
    beaten: int = 0
    losses: int = 0
    beat_rate: float | None = None
    mean_ratio: float | None = None
    antes: list[int] = field(default_factory=list)

    @property
    def entry(self) -> dict:
        return catalog.blinds().get(self.key, {})


def blind_infos(stats: pd.DataFrame, text: GameText | None, icon) -> list[BlindInfo]:
    out = []
    for key, s in stats.iterrows():
        bt = text.blind(key) if text else None
        out.append(
            BlindInfo(
                key=key,
                lines=bt.lines if bt else [],
                icon=icon(key),
                faced=int(s["faced"]),
                beaten=int(s["beaten"]),
                losses=int(s["losses"]),
                beat_rate=float(s["beat_rate"]),
                mean_ratio=None if pd.isna(s["mean_ratio"]) else float(s["mean_ratio"]),
                antes=list(s["antes"]),
            )
        )
    return out


def _chip(b: BlindInfo, name: str, size: int) -> str:
    if b.icon:
        return art_html(b.icon, name, size, size)
    colour = b.entry.get("colour", "#708b91")
    return (
        f'<div class="dt-noart dt-chip-noart" style="width:{size}px;height:{size}px;'
        f'background:{colour};font-size:{max(10, size // 6)}px">{html.escape(name)}</div>'
    )


def blind_tile(b: BlindInfo, names: Names, did: str) -> str:
    name = names.item(b.key)
    inner = (
        f'<div class="dt-mini">{_chip(b, name, 68)}<div class="dt-name">{html.escape(name)}</div>'
        f'<div class="dt-sub">faced <b>{b.faced}</b> · beaten <b>{pct(b.beat_rate)}</b></div>'
        "</div>"
    )
    return tile_html(did, inner, name)


def blind_dialog(b: BlindInfo, names: Names, did: str) -> str:
    name = names.item(b.key)
    e = b.entry
    tags = []
    if e.get("showdown"):
        tags.append('<span class="dt-tag" style="background:#b26cbb">Finisher · ante 8</span>')
    elif e.get("min_ante"):
        tags.append(f'<span class="dt-tag">From ante {e["min_ante"]}</span>')
    if e.get("mult"):
        tags.append(
            f'<span class="dt-tag" style="background:#FE5F55">{e["mult"]:g}&times; base</span>'
        )
    if e.get("dollars"):
        tags.append(f'<span class="dt-money">${e["dollars"]} reward</span>')
    body = desc_html(b.lines, "Effect text needs the game (see Game assets)")
    body += section("In the selected runs") + stats_html(
        [
            ("Times faced", str(b.faced)),
            ("Beaten", f"{b.beaten} ({pct(b.beat_rate)})"),
            ("Runs it ended", str(b.losses)),
            ("Mean chips / target", num(b.mean_ratio, "{:.2f}&times;")),
            ("Antes met at", ", ".join(map(str, b.antes)) or "-"),
        ]
    )
    sub = b.key if names.pretty else catalog.name(b.key)
    return dialog_html(did, name, sub, _chip(b, name, 96), body, "".join(tags))


def blind_grid_html(blinds: list[BlindInfo], names: Names, prefix: str = "boss") -> str:
    ids = [dialog_id(prefix, b.key) for b in blinds]
    return page(
        [blind_tile(b, names, i) for b, i in zip(blinds, ids, strict=True)],
        [blind_dialog(b, names, i) for b, i in zip(blinds, ids, strict=True)],
    )

"""The Joker collection: every joker as a game-style card with run stats.

Cards are plain data (``JokerCard``) so filtering/sorting is testable; ``grid_html``
renders them. Art and effect text come from the player's game install when
available (``sprites`` / ``game_text``); otherwise cards fall back to the name,
rarity, cost and effect category from the catalog. Clicking a card opens a
modal with its details (see ``details``).
"""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass, field

import pandas as pd

from balatro_analysis import catalog, details
from balatro_analysis.game_text import RARITY, GameText
from balatro_analysis.names import Names
from balatro_analysis.sprites import Sprites

SORTS = {
    "Collection order": lambda c: c.order,
    "Name": lambda c: c.name.lower(),
    "Rarity": lambda c: (c.rarity, c.order),
    "Cost": lambda c: (c.cost, c.order),
    "Times bought": lambda c: (-c.bought, c.order),
    "Held at end": lambda c: (-c.held_at_end, c.order),
    "Mean ante when held": lambda c: (-(c.mean_ante_held or 0), c.order),
}


@dataclass
class JokerCard:
    key: str
    name: str
    order: int
    rarity: int
    cost: int
    effect: str
    lines: list[str] = field(default_factory=list)  # effect text as HTML (needs the game)
    icon: str | None = None  # data URI (needs the game)
    bought: int = 0
    episodes_bought: int = 0
    held_at_end: int = 0
    mean_ante_held: float | None = None
    win_rate_held: float | None = None
    first_ante: float | None = None
    held_by_run: list[tuple[str, int, int]] = field(default_factory=list)  # run, held, episodes

    @property
    def rarity_name(self) -> str:
        return RARITY.get(self.rarity, ("?", ""))[0]

    @property
    def plain_text(self) -> str:
        return " ".join(re.sub(r"<[^>]+>", "", line) for line in self.lines)


def _num(v) -> float | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def build_cards(
    stats: pd.DataFrame,
    text: GameText | None = None,
    sprites: Sprites | None = None,
    held_by_run: pd.DataFrame | None = None,
) -> list[JokerCard]:
    """``held_by_run`` is ``metrics.joker_held_by_run`` (for the per-run lines in the modal)."""
    per_run: dict[str, list[tuple[str, int, int]]] = {}
    if held_by_run is not None:
        for row in held_by_run.itertuples(index=False):
            per_run.setdefault(row.joker, []).append(
                (row.run, int(row.held_at_end), int(row.episodes))
            )
    cards = []
    for key, entry in catalog.centers().items():
        if entry["set"] != "Joker":
            continue
        s = stats.loc[key] if key in stats.index else None
        jt = text.joker(key) if text else None
        cards.append(
            JokerCard(
                key=key,
                name=catalog.name(key),
                order=entry.get("order") or 0,
                rarity=entry.get("rarity") or 0,
                cost=entry.get("cost") or 0,
                effect=entry.get("effect") or "",
                lines=jt.lines if jt else [],
                icon=sprites.data_uri(key) if sprites else None,
                bought=int(s["bought"]) if s is not None else 0,
                episodes_bought=int(s["episodes_bought"]) if s is not None else 0,
                held_at_end=int(s["held_at_end"]) if s is not None else 0,
                mean_ante_held=_num(s["mean_ante_held"]) if s is not None else None,
                win_rate_held=_num(s["win_rate_held"]) if s is not None else None,
                first_ante=_num(s["first_ante"]) if s is not None else None,
                held_by_run=per_run.get(key, []),
            )
        )
    return sorted(cards, key=lambda c: c.order)


def select(
    cards: list[JokerCard],
    query: str = "",
    rarities: list[int] | None = None,
    only_used: bool = False,
    sort: str = "Collection order",
) -> list[JokerCard]:
    """Search (name, key or effect text), filter by rarity / use in runs, then sort."""
    q = query.strip().lower()
    out = [
        c
        for c in cards
        if (not q or q in c.name.lower() or q in c.key or q in c.plain_text.lower()
            or q in c.effect.lower())
        and (rarities is None or c.rarity in rarities)
        and (not only_used or c.bought or c.held_at_end)
    ]  # fmt: skip
    return sorted(out, key=SORTS[sort])


CSS = """
<style>
.jk-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:14px}
.jk-card{background:#3a4c56;border:2px solid #24323a;border-radius:12px;padding:10px;
  height:100%;box-sizing:border-box;display:flex;flex-direction:column;align-items:center;gap:6px;color:#fff;
  font-family:system-ui,sans-serif;box-shadow:0 3px 0 #1d282e}
.jk-art{width:142px;height:190px;image-rendering:pixelated;border-radius:6px}
.jk-noart{width:142px;height:190px;border-radius:6px;background:#708b91;display:flex;
  align-items:center;justify-content:center;text-align:center;font-weight:700;padding:8px}
.jk-name{font-weight:800;font-size:15px;text-align:center;line-height:1.15}
.jk-key{font-family:ui-monospace,monospace;font-size:11px;opacity:.7}
.jk-desc{background:#fff;color:#4F6367;border-radius:8px;padding:6px 8px;font-size:12.5px;
  line-height:1.35;text-align:center;font-weight:600;width:100%;box-sizing:border-box}
.jk-desc.jk-nodesc{font-style:italic;font-weight:500}
.jk-tags{display:flex;gap:6px;align-items:center}
.jk-rarity{border-radius:6px;padding:1px 8px;font-size:12px;font-weight:800;color:#fff}
.jk-cost{color:#f3b958;font-weight:800;font-size:13px}
.jk-stats{font-size:11.5px;opacity:.85;text-align:center;line-height:1.4}
.jk-stats b{color:#ff9a00}
</style>
"""


def _stats_html(c: JokerCard) -> str:
    if not c.bought and not c.held_at_end:
        return '<div class="jk-stats">Not used in the selected runs</div>'
    parts = [f"Bought <b>{c.bought}&times;</b> in {c.episodes_bought} ep."]
    if c.first_ante is not None:
        parts.append(f"first at ante <b>{c.first_ante:g}</b>")
    if c.held_at_end:
        held = f"Held at end <b>{c.held_at_end}</b>"
        if c.mean_ante_held is not None:
            held += f" · mean ante <b>{c.mean_ante_held:.2f}</b>"
        parts.append(held)
    return f'<div class="jk-stats">{"<br>".join(parts)}</div>'


def card_html(c: JokerCard, names: Names) -> str:
    title = html.escape(names.item(c.key))
    sub = (
        f'<div class="jk-key">{html.escape(c.name)}</div>'
        if not names.pretty
        else f'<div class="jk-key">{html.escape(c.key)}</div>'
    )
    art = (
        f'<img class="jk-art" src="{c.icon}" alt="{html.escape(c.name)}">'
        if c.icon
        else f'<div class="jk-noart">{html.escape(c.name)}</div>'
    )
    if c.lines:
        desc = f'<div class="jk-desc">{"<br>".join(c.lines)}</div>'
    else:
        desc = f'<div class="jk-desc jk-nodesc">{html.escape(c.effect or "No effect text")}</div>'
    rarity, colour = RARITY.get(c.rarity, ("?", "#708b91"))
    tags = (
        f'<div class="jk-tags"><span class="jk-rarity" style="background:{colour}">{rarity}</span>'
        f'<span class="jk-cost">${c.cost}</span></div>'
    )
    return (
        f'<div class="jk-card">{art}<div class="jk-name">{title}</div>{sub}'
        f"{desc}{tags}{_stats_html(c)}</div>"
    )


def _tags_html(c: JokerCard) -> str:
    rarity, colour = RARITY.get(c.rarity, ("?", "#708b91"))
    return (
        f'<span class="dt-tag" style="background:{colour}">{rarity}</span>'
        f'<span class="dt-money">${c.cost}</span>'
    )


def joker_dialog(c: JokerCard, names: Names, did: str) -> str:
    """The modal for one joker: art, effect text and how it did in the selected runs."""
    title = names.item(c.key)
    body = details.desc_html(c.lines, c.effect or "No effect text")
    if c.bought or c.held_at_end:
        pairs = [
            ("Times bought", f"{c.bought} (in {c.episodes_bought} episodes)"),
            ("Median ante first bought", details.num(c.first_ante, "{:g}")),
            ("Episodes ending with it", str(c.held_at_end)),
            ("Mean ante when held at end", details.num(c.mean_ante_held)),
            ("Win rate when held at end", details.pct(c.win_rate_held)),
        ]
        body += details.section("In the selected runs") + details.stats_html(pairs)
        if len(c.held_by_run) > 1:
            per = pd.DataFrame(
                [
                    {"Run": run, "Held at end": held, "Of episodes": eps,
                     "Share": details.pct(held / eps if eps else None)}
                    for run, held, eps in c.held_by_run
                ]
            )  # fmt: skip
            body += details.section("Held at the end, by run") + details.table_html(per)
    else:
        body += '<div class="dt-empty">Not bought or held in the selected runs.</div>'
    sub = c.key if names.pretty else c.name
    art = details.art_html(c.icon, c.name, 142, 190)
    return details.dialog_html(did, title, sub, art, body, _tags_html(c))


def grid_html(cards: list[JokerCard], names: Names, prefix: str = "jk") -> str:
    """The collection grid; each card opens its modal."""
    ids = [details.dialog_id(prefix, c.key) for c in cards]
    return CSS + details.page(
        [
            details.tile_html(i, card_html(c, names), names.item(c.key))
            for c, i in zip(cards, ids, strict=True)
        ],
        [joker_dialog(c, names, i) for c, i in zip(cards, ids, strict=True)],
        grid_class="jk-grid",
    )


def mini_grid_html(
    cards: list[JokerCard], names: Names, prefix: str, captions: list[str] | None = None
) -> str:
    """Compact joker tiles (art, name, optional caption HTML) that open the same modal."""
    ids = [details.dialog_id(prefix, f"{n}-{c.key}") for n, c in enumerate(cards)]
    captions = captions or [""] * len(cards)
    tiles = []
    for c, i, cap in zip(cards, ids, captions, strict=True):
        title = names.item(c.key)
        sub = f'<div class="dt-sub">{cap}</div>' if cap else ""
        inner = (
            f'<div class="dt-mini">{details.art_html(c.icon, c.name, 71, 95)}'
            f'<div class="dt-name">{html.escape(title)}</div>{sub}</div>'
        )
        tiles.append(details.tile_html(i, inner, title))
    return details.page(tiles, [joker_dialog(c, names, i) for c, i in zip(cards, ids, strict=True)])

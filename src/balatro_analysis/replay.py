"""The game-styled views, as HTML: in the Episode explorer a blind replayed hand
by hand (``blind_html``), the deck (``deck_html``) and the shop visits
(``shop_html``); on the Overview each run as a run-over screen (``run_sheet_html``).

Plain functions over frames from ``hands`` / ``economy``, like
``collection.grid_html``.  Card art comes in through ``card_uri`` and item art
through ``icon_uri``; without the game both return None and the views fall back
to text cards.
"""

from __future__ import annotations

import html
import math
from collections.abc import Callable

import pandas as pd

from balatro_analysis import catalog
from balatro_analysis.cards import RANKS, SUIT_SYMBOLS, PlayingCard, joker_state, state_label
from balatro_analysis.names import Names
from balatro_analysis.theme import CHIPS, INSET, LIP, MONEY, MULT, PANEL, PLAY, TABLE, WIN

CardUri = Callable[[PlayingCard], str | None]
IconUri = Callable[[str | None], str | None]

CSS = f"""<style>
.rp{{font-family:system-ui,sans-serif;color:#fff}}
.rp-px{{font-family:var(--bx-pixel,sans-serif);letter-spacing:.02em}}
.rp-blind{{display:grid;grid-template-columns:200px 1fr;gap:14px;align-items:start}}
@media (max-width:760px){{.rp-blind{{grid-template-columns:1fr}}}}
.rp-hud{{background:{PANEL};border:2px solid {LIP};border-radius:12px;box-shadow:0 3px 0 {LIP};
  padding:12px;display:flex;flex-direction:column;gap:10px}}
.rp-hud-head{{display:flex;gap:10px;align-items:center;border-radius:8px;padding:6px 8px}}
.rp-hud-head img{{width:44px;height:44px;image-rendering:pixelated}}
.rp-hud-name{{font-size:1.35rem;line-height:1}}
.rp-well{{background:{TABLE};border-radius:8px;padding:6px 10px}}
.rp-well small{{display:block;font-size:12px;opacity:.75}}
.rp-target{{color:{MULT};font-size:1.7rem;line-height:1.1}}
.rp-total{{font-size:1.5rem;line-height:1.1}}
.rp-pair{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
.rp-hands{{color:{CHIPS};font-size:1.5rem}}
.rp-discards{{color:{MULT};font-size:1.5rem}}
.rp-money{{color:{MONEY}}}
.rp-acts{{display:flex;flex-direction:column;gap:12px}}
.rp-act{{background:{PANEL};border:2px solid {LIP};border-radius:12px;box-shadow:0 3px 0 {LIP};
  padding:10px 12px 8px;display:flex;flex-wrap:wrap;gap:8px 18px;align-items:center}}
.rp-act-head{{flex:0 0 100%;display:flex;gap:10px;align-items:baseline;font-size:13px}}
.rp-kind{{border-radius:6px;padding:0 7px;font-weight:800;font-size:12px}}
.rp-kind.play{{background:{CHIPS}}}
.rp-kind.discard{{background:{MULT}}}
.rp-row{{display:flex;gap:4px;padding-top:14px;padding-bottom:10px}}
.rp-card{{width:62px;height:83px;border-radius:5px;position:relative;flex:none;
  transition:transform .15s}}
.rp-card img{{width:100%;height:100%;image-rendering:pixelated;display:block;border-radius:5px}}
.rp-card.picked.play{{transform:translateY(-14px)}}
.rp-card.picked.play img,.rp-card.picked.play .rp-txt{{outline:3px solid {PLAY};
  outline-offset:-1px}}
.rp-card.picked.discard{{transform:translateY(8px);opacity:.38}}
.rp-card.debuff::after{{content:"";position:absolute;inset:0;border-radius:5px;
  background:repeating-linear-gradient(45deg,rgba(254,95,85,.55) 0 4px,transparent 4px 9px)}}
.rp-txt{{width:100%;height:100%;border-radius:5px;background:#fff;color:#333;display:flex;
  align-items:center;justify-content:center;font-size:20px;font-weight:800;box-sizing:border-box;
  border:2px solid #ccd}}
.rp-txt.red{{color:#e0384c}}
.ed-foil img{{filter:hue-rotate(160deg) saturate(1.6)}}
.ed-holo img{{filter:hue-rotate(300deg) saturate(1.8) brightness(1.05)}}
.ed-polychrome img{{filter:hue-rotate(90deg) saturate(2.2) contrast(1.05)}}
.ed-negative img{{filter:invert(1) hue-rotate(180deg)}}
.rp-score{{display:flex;flex-direction:column;gap:4px;min-width:250px}}
.rp-type{{font-size:1.4rem;line-height:1;display:flex;align-items:center;gap:8px}}
.rp-lvl{{background:#fff;color:{TABLE};border-radius:6px;padding:1px 7px;font-size:1.05rem}}
.rp-lvl.est{{background:transparent;color:#fff;border:2px dashed rgba(255,255,255,.55)}}
.rp-jchange{{flex:0 0 100%;font-size:12.5px}}
.rp-jchange b{{color:{PLAY}}}
.rp-calc{{display:flex;align-items:center;gap:6px;font-size:1.45rem}}
.rp-box{{border-radius:8px;padding:2px 10px;min-width:56px;text-align:right;
  box-shadow:0 3px 0 {LIP}}}
.rp-box.chips{{background:{CHIPS}}}
.rp-box.mult{{background:{MULT};text-align:left}}
.rp-x{{color:{MULT};font-size:1.6rem}}
.rp-eq{{opacity:.8}}
.rp-bar{{flex:0 0 100%;height:6px;border-radius:3px;background:{TABLE};overflow:hidden}}
.rp-bar div{{height:100%;background:{CHIPS}}}
.rp-bar.done div{{background:{WIN}}}
.rp-note{{font-size:12.5px;opacity:.8}}
.rp-deck{{background:{PANEL};border:2px solid {LIP};border-radius:12px;padding:12px;
  display:flex;flex-direction:column;gap:6px;overflow-x:auto}}
.rp-suit{{display:flex;align-items:center;gap:10px}}
.rp-suit-label{{width:58px;font-size:1.3rem;flex:none}}
.rp-fan{{display:flex}}
.rp-fan .rp-card{{width:48px;height:64px;margin-right:-20px}}
.rp-fan .rp-card:hover{{transform:translateY(-6px);z-index:2}}
.rp-fan .rp-txt{{font-size:14px}}
.rp-shop{{display:flex;flex-direction:column;gap:12px}}
.rp-visit{{background:{PANEL};border:2px solid {LIP};border-radius:12px;box-shadow:0 3px 0 {LIP};
  padding:10px 12px;display:flex;flex-direction:column;gap:8px}}
.rp-visit-head{{display:flex;gap:14px;align-items:baseline;flex-wrap:wrap;font-size:13px}}
.rp-visit-head .rp-px{{font-size:1.35rem}}
.rp-offer{{display:flex;gap:14px;align-items:flex-end;flex-wrap:wrap;background:{TABLE};
  border-radius:10px;padding:10px}}
.rp-slot{{display:flex;flex-direction:column;align-items:center;gap:4px;width:76px;
  font-size:11.5px;text-align:center;line-height:1.15}}
.rp-slot img{{width:60px;height:80px;image-rendering:pixelated;border-radius:5px}}
.rp-slot.pack img{{height:auto}}
.rp-slot .rp-noart{{width:60px;height:80px;border-radius:5px;background:{INSET};display:flex;
  align-items:center;justify-content:center;padding:3px;box-sizing:border-box}}
.rp-price{{background:{LIP};color:{MONEY};border-radius:6px;padding:0 6px;font-size:1.05rem}}
.rp-slot.bought{{transform:translateY(-6px)}}
.rp-slot.bought img,.rp-slot.bought .rp-noart{{outline:3px solid {PLAY};outline-offset:1px}}
.rp-slot.bought .rp-price{{background:{PLAY};color:{LIP}}}
.rp-sep{{width:2px;align-self:stretch;background:{PANEL}}}
.rp-reroll{{background:{WIN};border-radius:8px;padding:4px 10px;box-shadow:0 3px 0 {LIP};
  font-size:1.05rem;align-self:center}}
.rp-did{{display:flex;flex-wrap:wrap;gap:6px;font-size:12.5px}}
.rp-did span{{background:{INSET};border-radius:6px;padding:1px 8px}}
@media (prefers-reduced-motion:reduce){{.rp-card{{transition:none}}}}
</style>"""


def _esc(v) -> str:
    return html.escape(str(v))


def _num(v) -> str:
    """A score as the game shows it: thousands separated, huge ones in e-notation."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if isinstance(v, float) and math.isinf(v):
        return "naneinf"
    v = float(v)
    return f"{v:,.0f}" if abs(v) < 1e11 else f"{v:.3e}"


def _fmt_mult(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.1f}"


def card_html(card: PlayingCard, card_uri: CardUri, cls: str = "") -> str:
    """One playing card: the game's art, or a text card without the game."""
    classes = ["rp-card", cls]
    if card.edition:
        classes.append(f"ed-{card.edition}")
    if card.debuff:
        classes.append("debuff")
    uri = card_uri(card)
    if uri:
        inner = f'<img src="{uri}" alt="{_esc(card.name)}">'
    else:
        red = " red" if card.suit in ("H", "D") else ""
        text = "Stone" if card.enhancement == "m_stone" else card.label
        inner = f'<div class="rp-txt{red}">{_esc(text)}</div>'
    cls = " ".join(c for c in classes if c)
    return f'<div class="{cls}" title="{_esc(card.name)}">{inner}</div>'


def _blind_head(blind_key: str, names: Names, icon_uri: IconUri) -> str:
    entry = catalog.blinds().get(blind_key, {})
    colour = entry.get("colour", INSET)
    uri = icon_uri(blind_key)
    art = f'<img src="{uri}" alt="">' if uri else ""
    return (
        f'<div class="rp-hud-head" style="background:{_esc(colour)}">{art}'
        f'<div class="rp-px rp-hud-name">{_esc(names.blind(blind_key))}</div></div>'
    )


def _level_badge(act) -> str:
    """``lvl.2`` next to the hand played; dashed when it's only the level at the
    start of the blind (runs that didn't log the level of each play)."""
    level = getattr(act, "level", None)
    if level is None or (isinstance(level, float) and math.isnan(level)):
        return ""
    if getattr(act, "level_logged", True):
        return f'<span class="rp-lvl" title="Hand level when played">lvl.{int(level)}</span>'
    return (
        '<span class="rp-lvl est" title="Level at the start of the blind (this run didn\'t '
        f'log the level of each play)">lvl.{int(level)}</span>'
    )


def _joker_changes(before: list[tuple], now: list[tuple], names: Names) -> str:
    """What changed in the jokers held since the last action: gained, lost,
    reordered, and scaling jokers' values (Green Joker +3 Mult -> +4 Mult;
    log_version 15).  ``before`` / ``now``: (key, state) per slot."""
    keys_before, keys_now = [k for k, _ in before], [k for k, _ in now]
    gained = [names.item(k) for k in keys_now if k not in keys_before]
    lost = [names.item(k) for k in keys_before if k not in keys_now]
    parts = [
        f"{verb} <b>{_esc(', '.join(js))}</b>"
        for verb, js in (("gained", gained), ("lost", lost))
        if js
    ]
    if not parts and keys_now != keys_before:
        parts.append("reordered")
    unused = list(before)
    for key, state in now:  # pair each joker with the first unused one of its key
        match = next((i for i, (k, _) in enumerate(unused) if k == key), None)
        if match is None:
            continue
        old = unused.pop(match)[1]
        old, state = joker_state(old), joker_state(state)
        if old and state and old != state:
            parts.append(
                f"<b>{_esc(names.item(key))}</b> {_esc(state_label(key, old))} → "
                f"{_esc(state_label(key, state))}"
            )
    if not parts:
        return ""
    return f'<div class="rp-jchange">Jokers since the last action: {"; ".join(parts)}</div>'


def blind_html(
    acts: pd.DataFrame,
    names: Names,
    card_uri: CardUri,
    icon_uri: IconUri,
    beaten: bool | None = None,
) -> str:
    """One blind replayed: the HUD (blind, target, final score, hands and discards
    left, money) and every play or discard in order, with the hand as cards (the
    picked ones raised for a play, dropped for a discard) and chips times mult."""
    if acts.empty:
        return ""
    first, last = acts.iloc[0], acts.iloc[-1]
    target = first["target"]
    total = last["chips_total"]
    outcome = "" if beaten is None else ("Beaten" if beaten else "Lost")
    outcome_html = (
        f'<div class="rp-px" style="color:{WIN if beaten else MULT};font-size:1.35rem">'
        f"{outcome}</div>"
        if outcome
        else ""
    )
    hud = (
        f'<div class="rp-hud">{_blind_head(first["blind_key"], names, icon_uri)}'
        f'<div class="rp-well"><small>Score at least</small>'
        f'<div class="rp-px rp-target">{_num(target)}</div></div>'
        f'<div class="rp-well"><small>Round score</small>'
        f'<div class="rp-px rp-total">{_num(total)}</div></div>'
        '<div class="rp-pair">'
        f'<div class="rp-well"><small>Hands left</small>'
        f'<div class="rp-px rp-hands">{_esc(last["hands_left"])}</div></div>'
        f'<div class="rp-well"><small>Discards left</small>'
        f'<div class="rp-px rp-discards">{_esc(last["discards_left"])}</div></div></div>'
        f'<div class="rp-well"><small>Money</small>'
        f'<div class="rp-px rp-total rp-money">${_esc(last["dollars"])}</div></div>'
        f"{outcome_html}</div>"
    )

    rows = []
    previous_jokers = None
    for n, a in enumerate(acts.itertuples(index=False), start=1):
        picked = set(a.picked) if isinstance(a.picked, list) else set()
        kind = a.kind
        cards = "".join(
            card_html(c, card_uri, f"picked {kind}" if i in picked else "")
            for i, c in enumerate(a.cards)
        )
        head = (
            f'<div class="rp-act-head"><span class="rp-kind {kind}">'
            f"{'Play' if kind == 'play' else 'Discard'}</span>"
            f'<span class="rp-note">{n} of {len(acts)}, {len(picked)} '
            f"card{'s' if len(picked) != 1 else ''}</span></div>"
        )
        jokers = getattr(a, "jokers_held", None)
        states = getattr(a, "joker_states", None)
        if isinstance(jokers, list):
            states = states if isinstance(states, list) else [None] * len(jokers)
            held = list(zip(jokers, states, strict=False))
            if previous_jokers is not None:
                head += _joker_changes(previous_jokers, held, names)
            previous_jokers = held
        score = ""
        if kind == "play":
            debuffed = " (debuffed by the boss)" if a.debuffed else ""
            score = (
                f'<div class="rp-score"><div class="rp-px rp-type">{_esc(a.hand_type)}'
                f"{_level_badge(a)}"
                f'<span class="rp-note" style="font-family:system-ui">{debuffed}</span></div>'
                f'<div class="rp-calc rp-px"><span class="rp-box chips">{_num(a.chips)}</span>'
                '<span class="rp-x">&times;</span>'
                f'<span class="rp-box mult">{_fmt_mult(a.mult)}</span>'
                f'<span class="rp-eq">=</span><span>{_num(a.score)}</span></div></div>'
            )
        progress = 0.0
        if target and not (isinstance(a.chips_total, float) and math.isnan(a.chips_total)):
            progress = min(float(a.chips_total) / float(target), 1.0)
        bar = (
            f'<div class="rp-bar{" done" if progress >= 1 else ""}" role="img" '
            f'aria-label="{_num(a.chips_total)} of {_num(target)}">'
            f'<div style="width:{progress:.0%}"></div></div>'
        )
        rows.append(
            f'<div class="rp-act">{head}<div class="rp-row">{cards}</div>{score}{bar}</div>'
        )
    return f'{CSS}<div class="rp rp-blind">{hud}<div class="rp-acts">{"".join(rows)}</div></div>'


SUIT_ORDER = ("S", "H", "C", "D")  # the game's deck view order


def deck_html(cards: list[PlayingCard], card_uri: CardUri) -> str:
    """The deck as the game's deck view: one fanned row per suit, high to low,
    with the count of each suit; Stone cards (no rank or suit) on their own row."""
    rows = []
    stones = [c for c in cards if c.enhancement == "m_stone"]
    for suit in SUIT_ORDER:
        suited = sorted(
            (c for c in cards if c.suit == suit and c.enhancement != "m_stone"),
            key=lambda c: -RANKS.index(c.rank) if c.rank in RANKS else 0,
        )
        if not suited:
            continue
        colour = MULT if suit in ("H", "D") else "#c8d6dc"
        rows.append(
            f'<div class="rp-suit"><div class="rp-suit-label rp-px" style="color:{colour}">'
            f"{SUIT_SYMBOLS[suit]} {len(suited)}</div>"
            f'<div class="rp-fan">{"".join(card_html(c, card_uri) for c in suited)}</div></div>'
        )
    if stones:
        rows.append(
            f'<div class="rp-suit"><div class="rp-suit-label rp-px">Stone {len(stones)}</div>'
            f'<div class="rp-fan">{"".join(card_html(c, card_uri) for c in stones)}</div></div>'
        )
    return f'{CSS}<div class="rp rp-deck">{"".join(rows)}</div>'


def _slot(item: pd.Series, names: Names, icon_uri: IconUri) -> str:
    name = names.item(item["key"])
    uri = icon_uri(item["key"])
    art = (
        f'<img src="{uri}" alt="{_esc(name)}">'
        if uri
        else f'<div class="rp-noart">{_esc(name)}</div>'
    )
    cls = " ".join(c for c in ("rp-slot", item["kind"], "bought" if item["bought"] else "") if c)
    label = f"{name} (bought)" if item["bought"] else name
    return (
        f'<div class="{cls}" title="{_esc(label)}">{art}'
        f'<span class="rp-price rp-px">${_esc(item["cost"])}</span>'
        f"<span>{_esc(name)}</span></div>"
    )


def shop_html(
    offers: pd.DataFrame,
    shops: pd.DataFrame,
    steps: pd.DataFrame | None,
    names: Names,
    icon_uri: IconUri,
) -> str:
    """Every shop visit of one episode drawn as the game's shop: each offer
    (entering, then after every reroll) with its cards, voucher and packs, the
    items bought from it raised and outlined, and the other decisions taken
    before the shop changed (sells, uses, leaving)."""
    if shops.empty:
        return ""
    shops = shops.sort_values("step").reset_index(drop=True)
    starts = [*shops["step"], math.inf]
    visits: list[list[str]] = []
    heads: list[str] = []
    for i, offer in shops.iterrows():
        items = offers[offers["step"] == offer["step"]] if not offers.empty else offers
        cards = [_slot(it, names, icon_uri) for _, it in items[items["kind"] == "card"].iterrows()]
        vouchers = [
            _slot(it, names, icon_uri) for _, it in items[items["kind"] == "voucher"].iterrows()
        ]
        packs = [_slot(it, names, icon_uri) for _, it in items[items["kind"] == "pack"].iterrows()]
        groups = [g for g in (cards, vouchers, packs) if g]
        body = '<div class="rp-sep"></div>'.join("".join(g) for g in groups)
        reroll = (
            f'<div class="rp-reroll rp-px">Reroll ${_esc(offer["reroll_cost"])}</div>'
            if pd.notna(offer.get("reroll_cost"))
            else ""
        )
        did = ""
        if steps is not None and not steps.empty:
            window = steps[(steps["step"] >= starts[i]) & (steps["step"] < starts[i + 1])]
            other = window[~window["action_name"].str.startswith("buy")]
            chips = []
            for _, s in other.iterrows():
                item = names.item(s["item_key"]) if isinstance(s["item_key"], str) else ""
                chips.append(
                    f"<span>{_esc(names.action(s['action_name']))}"
                    f"{': ' + _esc(item) if item else ''}</span>"
                )
            did = f'<div class="rp-did">{"".join(chips)}</div>' if chips else ""
        block = f'<div class="rp-offer">{body}{reroll}</div>{did}'
        if offer["reason"] == "enter" or not visits:
            visits.append([])
            heads.append(
                f'<div class="rp-visit-head"><span class="rp-px">Shop after round '
                f"{_esc(offer['round'])}</span><span>ante {_esc(offer['ante'])}</span>"
                f'<span class="rp-px rp-money" style="font-size:1.2rem">${_esc(offer["dollars"])}'
                "</span></div>"
            )
        else:
            visits[-1].append(
                f'<div class="rp-note">Rerolled, ${_esc(offer["dollars"])} left</div>'
            )
        visits[-1].append(block)
    out = "".join(
        f'<div class="rp-visit">{h}{"".join(v)}</div>' for h, v in zip(heads, visits, strict=True)
    )
    return f'{CSS}<div class="rp rp-shop">{out}</div>'


SHEET_CSS = f"""<style>
.rs{{background:{PANEL};border:2px solid {LIP};border-radius:12px;box-shadow:0 3px 0 {LIP};
  padding:12px 14px;color:#fff;font-family:system-ui,sans-serif;display:flex;
  flex-direction:column;gap:10px}}
.rs.best{{border-color:{PLAY}}}
.rs-name{{font-family:var(--bx-pixel,sans-serif);font-size:1.5rem;line-height:1.05;
  overflow-wrap:anywhere}}
.rs-who{{font-size:12.5px;opacity:.8;margin-top:2px}}
.rs-best{{color:{PLAY};font-weight:800}}
.rs-wells{{display:grid;grid-template-columns:1.3fr 1fr;gap:8px}}
.rs-well{{background:{TABLE};border-radius:8px;padding:6px 10px}}
.rs-well small{{display:block;font-size:12px;opacity:.75}}
.rs-big{{font-family:var(--bx-pixel,sans-serif);font-size:2.3rem;line-height:1}}
.rs-ante{{color:{PLAY}}}
.rs-ci{{font-size:12px;opacity:.75}}
.rs-lines{{display:grid;grid-template-columns:auto 1fr;gap:3px 12px;font-size:13.5px}}
.rs-lines span:nth-child(even){{text-align:right;font-weight:700}}
.rs-decks{{display:flex;gap:4px;flex-wrap:wrap}}
.rs-decks img{{width:34px;height:46px;image-rendering:pixelated}}
</style>"""


def _agent(name: str | None, suffix: str) -> str:
    """``QLearningShopAgent`` -> ``QLearning``."""
    return (name or "?").removesuffix(suffix) or "?"


def run_sheet_html(
    name: str,
    config: dict,
    mean_ante: float,
    ci: tuple[float, float] | None,
    win_rate: float,
    best_ante: int,
    episodes: int,
    decks: list[str],
    names: Names,
    icon_uri: IconUri,
    most_played: str | None = None,
    best: bool = False,
) -> str:
    """One run as the game's run-over screen: mean ante large, then the rest."""
    who = (
        f"{_agent(config.get('shop_agent'), 'ShopAgent')} shop, "
        f"{_agent(config.get('play_agent'), 'PlayAgent')} play"
        if config
        else "no config.json"
    )
    ci_html = f'<div class="rs-ci">95% CI {ci[0]:.2f} to {ci[1]:.2f}</div>' if ci else ""
    lines = [("Best ante", str(best_ante)), ("Episodes", str(episodes))]
    if most_played:
        lines.append(("Most played hand", most_played))
    deck_art = "".join(
        f'<img src="{u}" alt="{_esc(names.deck(d))}" title="{_esc(names.deck(d))}">'
        if (u := icon_uri(d))
        else f'<span class="rs-ci">{_esc(names.deck(d))}</span>'
        for d in decks
    )
    best_html = '<span class="rs-best"> Best mean ante</span>' if best else ""
    return (
        f'{SHEET_CSS}<div class="rs{" best" if best else ""}">'
        f'<div><div class="rs-name">{_esc(name)}</div>'
        f'<div class="rs-who">{_esc(who)}{best_html}</div></div>'
        '<div class="rs-wells">'
        f'<div class="rs-well"><small>Mean ante</small>'
        f'<div class="rs-big rs-ante">{mean_ante:.2f}</div>{ci_html}</div>'
        f'<div class="rs-well"><small>Win rate</small>'
        f'<div class="rs-big">{win_rate:.0%}</div></div></div>'
        '<div class="rs-lines">'
        + "".join(f"<span>{_esc(k)}</span><span>{_esc(v)}</span>" for k, v in lines)
        + f'</div><div class="rs-decks">{deck_art}</div></div>'
    )

"""Playing cards from ``eval/hands.jsonl`` and ``eval/decks.jsonl``.

The logs write a plain card as its ``card_key`` (``"H_A"``: suit letter, rank
letter, ``T`` for ten) and a modified one as an object with only what differs:
``{"c": "H_A", "en": "m_glass", "ed": "foil", "seal": "Red", "debuff": true}``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SUITS = {"H": "Hearts", "C": "Clubs", "D": "Diamonds", "S": "Spades"}
SUIT_SYMBOLS = {"H": "♥", "C": "♣", "D": "♦", "S": "♠"}
RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
RANK_NAMES = {"T": "10", "J": "Jack", "Q": "Queen", "K": "King", "A": "Ace"}
ENHANCEMENTS = {
    "m_bonus": "Bonus",
    "m_mult": "Mult",
    "m_wild": "Wild",
    "m_glass": "Glass",
    "m_steel": "Steel",
    "m_stone": "Stone",
    "m_gold": "Gold",
    "m_lucky": "Lucky",
}
EDITIONS = {
    "foil": "Foil",
    "holo": "Holographic",
    "polychrome": "Polychrome",
    "negative": "Negative",
}


@dataclass(frozen=True)
class PlayingCard:
    suit: str  # H C D S
    rank: str  # 2-9 T J Q K A
    enhancement: str | None = None  # m_glass, ...
    edition: str | None = None  # foil, holo, polychrome, negative
    seal: str | None = None  # Red, Blue, Gold, Purple
    debuff: bool = False

    @property
    def label(self) -> str:
        """Short text form: ``A♥``, ``10♠``."""
        return ("10" if self.rank == "T" else self.rank) + SUIT_SYMBOLS.get(self.suit, "?")

    @property
    def name(self) -> str:
        """``Ace of Hearts``, then whatever modifies it: ``Glass, Foil, Red Seal``."""
        base = f"{RANK_NAMES.get(self.rank, self.rank)} of {SUITS.get(self.suit, self.suit)}"
        extra = [
            ENHANCEMENTS.get(self.enhancement or "", self.enhancement or ""),
            EDITIONS.get(self.edition or "", self.edition or ""),
            f"{self.seal} Seal" if self.seal else "",
            "debuffed" if self.debuff else "",
        ]
        extra = [e for e in extra if e]
        return f"{base} ({', '.join(extra)})" if extra else base

    @property
    def rank_index(self) -> int:
        return RANKS.index(self.rank) if self.rank in RANKS else -1


def parse_card(value: str | dict) -> PlayingCard:
    """A card as logged (a ``card_key`` string or an object) as a ``PlayingCard``."""
    if isinstance(value, dict):
        key = value.get("c", "?_?")
        extra = {
            "enhancement": value.get("en"),
            "edition": value.get("ed"),
            "seal": value.get("seal"),
            "debuff": bool(value.get("debuff", False)),
        }
    else:
        key, extra = str(value), {}
    suit, _, rank = key.partition("_")
    return PlayingCard(suit=suit, rank=rank, **extra)


def parse_cards(values) -> list[PlayingCard]:
    return [parse_card(v) for v in values] if isinstance(values, list) else []


def parse_joker(value: str | dict) -> tuple[str, str | None]:
    """A joker as logged (its key, or ``{"k": key, "ed": edition, "s": values}``):
    (key, edition); ``joker_state`` reads the values."""
    if isinstance(value, dict):
        return str(value.get("k", "")), value.get("ed")
    return str(value), None


def joker_keys(values) -> list[str] | None:
    """The keys of a logged joker list, or None if the row has no joker list."""
    return [parse_joker(v)[0] for v in values] if isinstance(values, list) else None


# A scaling joker's current values (log_version 15), keyed by the path into its
# ability: ``{"mult": 6}`` for Spare Trousers on +6 Mult.  How each path reads in
# the game; a path only one joker uses is keyed with it.
STATE_LABELS = {
    "mult": "+{} Mult",
    "x_mult": "X{} Mult",
    "caino_xmult": "X{} Mult",  # Canio
    "extra.chips": "+{} Chips",
    "extra.dollars": "${} at end of round",  # Rocket
    "extra.h_size": "+{} hand size",  # Turtle Bean
    "extra_value": "+${} sell value",  # Egg
    ("j_selzer", "extra"): "{} hands left",  # Seltzer
}


def joker_state(value) -> dict | None:
    """A scaling joker's current values (log_version 15), from its entry in a
    hands.jsonl joker list (``{"k": key, "s": {...}}``) or a steps.jsonl
    ``joker_state`` item (``{...}``); None for a joker that doesn't scale or a
    log without values.  Arrow-read items hold every path any joker had, so
    empty ones are dropped."""
    if isinstance(value, dict) and "k" in value:
        value = value.get("s")
    if not isinstance(value, dict):
        return None
    state = {
        path: v
        for path, v in value.items()
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    }
    return state or None


def joker_states(values) -> list[dict | None] | None:
    """``joker_state`` of each joker in a logged list, or None if the row has none."""
    return [joker_state(v) for v in values] if isinstance(values, list) else None


def _value(v) -> str:
    """2 -> "2", 1.5 -> "1.5", 1.9134 -> "1.91"."""
    if isinstance(v, bool) or not isinstance(v, int | float):
        return str(v)
    return f"{v:.2f}".rstrip("0").rstrip(".") if v != int(v) else str(int(v))


def state_label(key: str, state: dict | None) -> str:
    """A scaling joker's values as the game shows them: "+6 Mult", "X1.91 Mult"."""
    if not state:
        return ""
    parts = []
    for path, v in state.items():
        fmt = STATE_LABELS.get((key, path)) or STATE_LABELS.get(path)
        parts.append(fmt.format(_value(v)) if fmt else f"{path} {_value(v)}")
    return ", ".join(parts)


def state_summary(keys, states, name) -> str:
    """The scaling jokers in a held list and their values, "Green Joker +4 Mult;
    Ice Cream +85 Chips" (``name`` turns a key into its display name); "" if none
    scale or ``states`` is missing."""
    if not isinstance(keys, list) or not isinstance(states, list):
        return ""
    parsed = [(k, joker_state(s)) for k, s in zip(keys, states, strict=False)]
    return "; ".join(f"{name(k)} {state_label(k, s)}" for k, s in parsed if s is not None)

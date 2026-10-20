"""Playing cards from ``eval/hands.jsonl`` and ``eval/decks.jsonl``.

The logs write a plain card as its ``card_key`` (``"H_A"``: suit letter, rank
letter, ``T`` for ten) and a modified one as an object with only what differs:
``{"c": "H_A", "en": "m_glass", "ed": "foil", "seal": "Red", "debuff": true}``.
"""

from __future__ import annotations

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

"""Display names and sprite positions for jokers and decks.

Data comes from ``data/centers.json`` (regenerate with ``scripts/sync_centers.py``).
"""

from __future__ import annotations

import json
from functools import cache
from importlib import resources

STAKES = {
    1: "White",
    2: "Red",
    3: "Green",
    4: "Black",
    5: "Blue",
    6: "Purple",
    7: "Orange",
    8: "Gold",
}


@cache
def centers() -> dict[str, dict]:
    return json.loads(resources.files("balatro_analysis").joinpath("data/centers.json").read_text())


# Where the game's internal name differs from what it displays
DISPLAY_OVERRIDES = {"j_caino": "Canio"}


def name(key: str | None) -> str:
    """Display name for a center key, e.g. ``j_greedy_joker`` -> ``Greedy Joker``."""
    if not isinstance(key, str) or not key:
        return ""
    if key in DISPLAY_OVERRIDES:
        return DISPLAY_OVERRIDES[key]
    entry = centers().get(key)
    if entry:
        return entry["name"]
    # Unknown key (packs, tarots, ...): strip the prefix and title-case it
    head, _, rest = key.partition("_")
    return (rest or head).replace("_", " ").title()


def stake_name(stake) -> str:
    try:
        return f"{STAKES[int(stake)]} Stake"
    except (KeyError, TypeError, ValueError):
        return "Unknown stake"


def is_joker(key: str | None) -> bool:
    return isinstance(key, str) and key.startswith("j_")

"""Regenerate src/balatro_analysis/data/{centers,blinds}.json from jackdaw's data.

Keeps only what the dashboard needs: display name and set for every center
(jokers, decks, packs, vouchers, consumables, enhancements, ...), sprite
positions for jokers, decks, consumables, vouchers and packs, deck configs, and blind names, chip positions,
colours and rewards (no game art).

    uv run python scripts/sync_centers.py [path/to/jackdaw/engine/data]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT.parent / "jackdaw-balatro" / "jackdaw" / "engine" / "data"
OUT = ROOT / "src" / "balatro_analysis" / "data"
SPRITE_SETS = {"Joker", "Back", "Tarot", "Planet", "Spectral", "Voucher", "Booster"}


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC

    centers = {}
    for key, c in sorted(json.loads((src / "centers.json").read_text()).items()):
        if not c.get("set") or not c.get("name"):
            continue
        entry = {"name": c["name"], "set": c["set"]}
        if c["set"] in SPRITE_SETS:
            entry["pos"] = c["pos"]
            if c.get("soul_pos"):
                entry["soul_pos"] = c["soul_pos"]
        if c["set"] == "Joker":
            # For the collection tab: collection order, rarity (1-4), base cost, the
            # starting config (fills the effect text's #n# values) and an effect category
            entry.update(
                order=c.get("order"),
                rarity=c.get("rarity"),
                cost=c.get("cost"),
                config=c.get("config") or {},
                effect=c.get("effect") or "",
            )
        if c["set"] == "Back":
            # Fills the deck text's #n# values (back.lua's loc_args)
            entry.update(order=c.get("order"), config=c.get("config") or {})
        centers[key] = entry

    blinds = {}
    for key, b in sorted(json.loads((src / "blinds.json").read_text()).items()):
        entry = {"name": b["name"], "boss": "boss" in b, "order": b.get("order"),
                 "pos": b["pos"], "dollars": b.get("dollars"), "mult": b.get("mult")}  # fmt: skip
        if "boss" in b:
            entry["min_ante"] = b["boss"].get("min")
            entry["showdown"] = bool(b["boss"].get("showdown"))
        if b.get("boss_colour"):
            r, g, bl = (round(v * 255) for v in b["boss_colour"][:3])
            entry["colour"] = f"#{r:02x}{g:02x}{bl:02x}"
        blinds[key] = entry

    for name, data in (("centers", centers), ("blinds", blinds)):
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(data, indent=1) + "\n")
        print(f"wrote {len(data)} entries to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

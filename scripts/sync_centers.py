"""Regenerate src/balatro_analysis/data/{centers,blinds}.json from jackdaw's data.

Keeps only what the dashboard needs: display name and set for every center
(jokers, decks, packs, vouchers, consumables, enhancements, ...), sprite
positions for jokers and decks, and blind names (no game art).

    uv run python scripts/sync_centers.py [path/to/jackdaw/engine/data]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT.parent / "jackdaw-balatro" / "jackdaw" / "engine" / "data"
OUT = ROOT / "src" / "balatro_analysis" / "data"
SPRITE_SETS = {"Joker", "Back"}


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
        centers[key] = entry

    blinds = {
        key: {"name": b["name"], "boss": "boss" in b}
        for key, b in sorted(json.loads((src / "blinds.json").read_text()).items())
    }

    for name, data in (("centers", centers), ("blinds", blinds)):
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(data, indent=1) + "\n")
        print(f"wrote {len(data)} entries to {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

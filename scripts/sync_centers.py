"""Regenerate src/balatro_analysis/data/centers.json from jackdaw's centers data.

Keeps only what the dashboard needs: display name, set and sprite position
for jokers and decks (no game art).

    uv run python scripts/sync_centers.py [path/to/jackdaw/engine/data/centers.json]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT.parent / "jackdaw-balatro" / "jackdaw" / "engine" / "data" / "centers.json"
OUT = ROOT / "src" / "balatro_analysis" / "data" / "centers.json"
SETS = {"Joker", "Back"}


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC
    centers = json.loads(src.read_text())
    out = {}
    for key, c in sorted(centers.items()):
        if c.get("set") not in SETS:
            continue
        entry = {"name": c["name"], "set": c["set"], "pos": c["pos"]}
        if c.get("soul_pos"):
            entry["soul_pos"] = c["soul_pos"]
        out[key] = entry
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {len(out)} entries to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Joker, consumable, voucher, pack, deck and blind icons cut from the player's
own copy of Balatro.

No game art is stored in this repo. ``Balatro.exe`` (or ``Balatro.love``) is a
LÖVE zip archive; its sprite atlases live at
``resources/textures/{1,2}x/{Jokers,Tarots,Vouchers,boosters,Enhancers}.png`` in
71x95 cells (x scale); blind chips are in ``BlindChips.png``, 34x34 cells, one
animation per row (the first frame is used).
"""

from __future__ import annotations

import base64
import io
import os
import re
import zipfile
from pathlib import Path

from PIL import Image

from balatro_analysis.catalog import blinds, centers

CELL = (71, 95)
CELLS = {"Blind": (34, 34)}  # sets whose cells aren't CELL
ATLASES = {
    "Joker": "Jokers.png",
    "Tarot": "Tarots.png",  # tarots, planets and spectrals share one atlas
    "Planet": "Tarots.png",
    "Spectral": "Tarots.png",
    "Voucher": "Vouchers.png",
    "Booster": "boosters.png",
    "Back": "Enhancers.png",
    "Blind": "BlindChips.png",
}
STRIP_GAP = 4  # px between icons in a strip, at scale 1
ARCHIVE_NAMES = ("Balatro.exe", "Balatro.love", "Balatro")

STEAM_ROOTS = (
    "~/.local/share/Steam",
    "~/.steam/steam",
    "~/.var/app/com.valvesoftware.Steam/.local/share/Steam",
    "~/Library/Application Support/Steam",
    "C:/Program Files (x86)/Steam",
    "C:/Program Files/Steam",
)


def _steam_libraries(root: Path) -> list[Path]:
    libs = [root]
    vdf = root / "steamapps" / "libraryfolders.vdf"
    if vdf.is_file():
        for path in re.findall(r'"path"\s+"([^"]+)"', vdf.read_text(errors="ignore")):
            libs.append(Path(path.replace("\\\\", "/")))
    return libs


def game_archive(path: str | Path) -> Path | None:
    """The zip archive inside a Balatro install dir, or ``path`` itself if it is one."""
    path = Path(path).expanduser()
    candidates = [path] if path.is_file() else [path / n for n in ARCHIVE_NAMES]
    # macOS app bundle
    candidates.append(path / "Contents" / "Resources" / "Balatro.love")
    for c in candidates:
        if c.is_file() and zipfile.is_zipfile(c):
            return c
    return None


def find_balatro(steam_roots: tuple[str, ...] = STEAM_ROOTS) -> Path | None:
    """Locate the game archive: ``$BALATRO_PATH`` first, then Steam libraries."""
    env = os.environ.get("BALATRO_PATH")
    if env:
        return game_archive(env)
    for root in steam_roots:
        root = Path(root).expanduser()
        if not root.is_dir():
            continue
        for lib in _steam_libraries(root):
            found = game_archive(lib / "steamapps" / "common" / "Balatro")
            if found:
                return found
    return None


class Sprites:
    """Crops (and caches) icons from the game's atlases."""

    def __init__(self, archive: Path, scale: int = 2) -> None:
        self.archive = Path(archive)
        self.scale = scale
        self._atlases: dict[str, Image.Image | None] = {}
        self._icons: dict[str, bytes | None] = {}
        self._images: dict[str, Image.Image | None] = {}  # decoded icons, for strips
        self._strips: dict[tuple[str, ...], bytes | None] = {}

    def _atlas(self, set_name: str) -> Image.Image | None:
        if set_name not in self._atlases:
            image = None
            filename = ATLASES.get(set_name)
            if filename:
                with zipfile.ZipFile(self.archive) as z:
                    for scale in (self.scale, 1):
                        member = f"resources/textures/{scale}x/{filename}"
                        if member in z.namelist():
                            image = Image.open(io.BytesIO(z.read(member))).convert("RGBA")
                            image.info["scale"] = scale
                            break
            self._atlases[set_name] = image
        return self._atlases[set_name]

    def _cell(self, atlas: Image.Image, pos: dict, cell: tuple[int, int] = CELL) -> Image.Image:
        s = atlas.info["scale"]
        w, h = cell[0] * s, cell[1] * s
        return atlas.crop((pos["x"] * w, pos["y"] * h, (pos["x"] + 1) * w, (pos["y"] + 1) * h))

    def png(self, key: str | None) -> bytes | None:
        """PNG bytes of a joker, deck or blind icon, or None if unknown/unavailable."""
        if not key:
            return None
        if key not in self._icons:
            self._icons[key] = self._render(key)
        return self._icons[key]

    def _render(self, key: str) -> bytes | None:
        entry = centers().get(key)
        if not entry and key in blinds() and "pos" in blinds()[key]:
            entry = {"set": "Blind", "pos": {"x": 0, "y": blinds()[key]["pos"]["y"]}}
        if not entry or "pos" not in entry:
            return None
        atlas = self._atlas(entry["set"])
        if atlas is None:
            return None
        icon = self._cell(atlas, entry["pos"], CELLS.get(entry["set"], CELL))
        if "soul_pos" in entry:  # legendaries / Hologram: floating layer on top
            icon = Image.alpha_composite(icon, self._cell(atlas, entry["soul_pos"]))
        buf = io.BytesIO()
        icon.save(buf, format="PNG")
        return buf.getvalue()

    def data_uri(self, key: str | None) -> str | None:
        return _uri(self.png(key))

    def strip_png(self, keys: list[str] | tuple[str, ...]) -> bytes | None:
        """PNG of the icons of ``keys`` side by side (keys without an icon are left
        out), or None if none has one.  For a table cell showing a whole hand of
        jokers or consumables."""
        keys = tuple(keys)
        if keys not in self._strips:
            self._strips[keys] = self._render_strip(keys)
        return self._strips[keys]

    def strip_uri(self, keys: list[str] | tuple[str, ...]) -> str | None:
        return _uri(self.strip_png(keys))

    def _render_strip(self, keys: tuple[str, ...]) -> bytes | None:
        icons = [i for i in map(self._image, keys) if i is not None]
        if not icons:
            return None
        gap = STRIP_GAP * self.scale
        height = max(i.height for i in icons)
        strip = Image.new(
            "RGBA", (sum(i.width for i in icons) + gap * (len(icons) - 1), height), (0, 0, 0, 0)
        )
        x = 0
        for icon in icons:
            strip.paste(icon, (x, height - icon.height))
            x += icon.width + gap
        buf = io.BytesIO()
        strip.save(buf, format="PNG", compress_level=1)  # hundreds per table: favour speed
        return buf.getvalue()

    def _image(self, key: str) -> Image.Image | None:
        if key not in self._images:
            data = self.png(key)
            self._images[key] = Image.open(io.BytesIO(data)).convert("RGBA") if data else None
        return self._images[key]


def _uri(data: bytes | None) -> str | None:
    return "data:image/png;base64," + base64.b64encode(data).decode() if data else None

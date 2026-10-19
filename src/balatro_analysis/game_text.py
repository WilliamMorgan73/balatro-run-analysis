"""Joker, deck, blind, consumable, voucher and pack effect text, read from the
player's own copy of Balatro.

No game text is stored in this repo. From the game archive (see ``sprites``):

- ``localization/en-us.lua`` holds each joker's description lines, e.g.
  ``"{C:mult}+#1#{} Mult"``: ``{C:...}`` codes style the text, ``#n#`` are
  placeholders for values.
- ``card.lua`` fills the placeholders per joker with ``loc_vars = {...}``
  expressions such as ``self.ability.extra.s_mult``. They are evaluated here
  against the joker's *starting* ability (what the in-game collection shows),
  with a tiny whitelisted evaluator, never ``eval``. Values that depend on a
  live run (e.g. cards left in deck) come out as "?".
- ``functions/common_events.lua`` (``generate_card_ui``) does the same for
  consumables, vouchers and packs (``_c.config.extra``, ...); planets read the
  hand's level-1 values from ``game.lua``'s hands table.
"""

from __future__ import annotations

import ast
import html
import math
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from balatro_analysis import catalog

# Colours from the game's loc_colour() / G.C (globals.lua)
COLOURS = {
    "red": "#FE5F55",
    "mult": "#FE5F55",
    "blue": "#009dff",
    "chips": "#009dff",
    "green": "#4BC292",
    "money": "#f3b958",
    "gold": "#eac058",
    "attention": "#ff9a00",
    "purple": "#8867a5",
    "white": "#ffffff",
    "inactive": "#888888",
    "spades": "#374649",
    "hearts": "#FE5F55",
    "clubs": "#424e54",
    "diamonds": "#FE5F55",
    "tarot": "#a782d1",
    "planet": "#13afce",
    "spectral": "#4584fa",
    "edition": "#4ca893",
    "dark_edition": "#5d5dff",
    "legendary": "#b26cbb",
    "enhanced": "#8389DD",
    "joker": "#708b91",
    "voucher": "#fd682b",
    "booster": "#646eb7",
    "eternal": "#c75985",
    "light_black": "#4f6367",
}
TEXT_DARK = "#4F6367"
UNKNOWN_MARK = "\x00"  # placeholder value that needs a live run
UNKNOWN_HTML = '<span title="Depends on the current run" style="opacity:.55">?</span>'
RARITY = {1: ("Common", "#009dff"), 2: ("Uncommon", "#4BC292"), 3: ("Rare", "#fe5f55"),
          4: ("Legendary", "#b26cbb")}  # fmt: skip


# ---------------------------------------------------------------------------
# Lua table/expression parsing
# ---------------------------------------------------------------------------


class _Missing(Exception):
    """An expression needs live game state; its value is unknown."""


class _Obj(dict):
    """Lua-table-like: attribute access, missing keys raise _Missing."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise _Missing(name) from None


def _wrap(v: Any) -> Any:
    if isinstance(v, dict):
        return _Obj({k: _wrap(x) for k, x in v.items()})
    if isinstance(v, list):
        return [_wrap(x) for x in v]
    return v


def starting_ability(entry: dict) -> _Obj:
    """``Card:set_ability`` for a fresh joker (card.lua), from its catalog entry."""
    cfg = entry.get("config") or {}
    if not isinstance(cfg, dict):
        cfg = {}
    ability = {
        "name": entry.get("name"),
        "effect": entry.get("effect", ""),
        "set": "Joker",
        "mult": cfg.get("mult", 0),
        "h_mult": cfg.get("h_mult", 0),
        "h_x_mult": cfg.get("h_x_mult", 0),
        "h_dollars": cfg.get("h_dollars", 0),
        "p_dollars": cfg.get("p_dollars", 0),
        "t_mult": cfg.get("t_mult", 0),
        "t_chips": cfg.get("t_chips", 0),
        "x_mult": cfg.get("Xmult", 1),
        "h_size": cfg.get("h_size", 0),
        "d_size": cfg.get("d_size", 0),
        "extra": cfg.get("extra"),
        "extra_value": 0,
        "type": cfg.get("type", ""),
        "perma_bonus": 0,
        "bonus": cfg.get("bonus", 0),
        # per-joker counters set_ability initialises
        "caino_xmult": 1,
        "invis_rounds": 0,
        "burnt_hand": 0,
    }
    extra = cfg.get("extra")
    if isinstance(extra, dict):
        if "discards" in extra:
            ability["yorick_discards"] = extra["discards"]
        if "every" in extra:
            ability["loyalty_remaining"] = extra["every"]
    return _wrap(ability)


def _lua_to_python(expr: str) -> str:
    expr = expr.replace("~=", "!=")
    expr = re.sub(r"\.\.", " @ ", expr)  # concat; @ is otherwise unused
    expr = re.sub(r"#\s*([A-Za-z_][\w.]*)", r"__len(\1)", expr)
    expr = re.sub(r"\bnil\b", "None", expr)
    expr = re.sub(r"\btrue\b", "True", expr)
    expr = re.sub(r"\bfalse\b", "False", expr)
    return expr


def _truthy(v: Any) -> bool:  # Lua: only nil and false are falsy
    return v is not None and v is not False


def _tostring(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    if isinstance(v, bool):
        return "true" if v else "false"
    return "nil" if v is None else str(v)


_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
    ast.MatMult: lambda a, b: _tostring(a) + _tostring(b),
}
_CMPOPS = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
}


class _Evaluator:
    """Evaluates a Lua expression (as Python AST) with Lua semantics."""

    def __init__(self, names: dict[str, Any]) -> None:
        self.names = names

    def run(self, expr: str) -> Any:
        return self.eval(ast.parse(_lua_to_python(expr).strip(), mode="eval").body)

    def eval(self, n: ast.AST) -> Any:
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            if n.id not in self.names:
                raise _Missing(n.id)
            return self.names[n.id]
        if isinstance(n, ast.Attribute):
            obj = self.eval(n.value)
            if obj is None or not hasattr(obj, "__getattr__"):
                raise _Missing(n.attr)
            return getattr(obj, n.attr)
        if isinstance(n, ast.Subscript):
            obj, key = self.eval(n.value), self.eval(n.slice)
            if not isinstance(obj, dict) or key not in obj:
                raise _Missing(str(key))
            return obj[key]
        if isinstance(n, ast.BoolOp):
            value = self.eval(n.values[0])
            for nxt in n.values[1:]:
                if isinstance(n.op, ast.And) and not _truthy(value):
                    return value
                if isinstance(n.op, ast.Or) and _truthy(value):
                    return value
                value = self.eval(nxt)
            return value
        if isinstance(n, ast.UnaryOp):
            v = self.eval(n.operand)
            if isinstance(n.op, ast.Not):
                return not _truthy(v)
            if isinstance(n.op, ast.USub):
                return -v
            raise _Missing("unary")
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
            return _BINOPS[type(n.op)](self.eval(n.left), self.eval(n.right))
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in _CMPOPS:
            return _CMPOPS[type(n.ops[0])](self.eval(n.left), self.eval(n.comparators[0]))
        if isinstance(n, ast.Call) and not n.keywords:
            fn = self.eval(n.func)
            if not callable(fn):
                raise _Missing("call")
            return fn(*(self.eval(a) for a in n.args))
        raise _Missing(type(n).__name__)


def _split_top_level(body: str) -> list[str]:
    """Split a Lua table body on top-level commas (ignoring nested braces/strings)."""
    parts, depth, quote, cur = [], 0, None, []
    for ch in body:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    if "".join(cur).strip():
        parts.append("".join(cur))
    return [p.strip() for p in parts]


def _balanced(src: str, start: int) -> str:
    """Text between the ``{`` at ``start`` and its matching ``}``."""
    depth, quote = 0, None
    for i in range(start, len(src)):
        ch = src[i]
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return src[start + 1 : i]
    return ""


_BRANCH = re.compile(
    r"self\.ability\.name\s*==\s*'([^']+)'((?:\s*or\s*self\.ability\.name\s*==\s*'[^']+')*)"
    r"\s*then\s+loc_vars\s*=\s*\{"
)


def parse_loc_vars(card_lua: str) -> dict[str, list[str]]:
    """``{internal joker name: [positional loc_vars expressions]}`` from card.lua."""
    out: dict[str, list[str]] = {}
    for m in _BRANCH.finditer(card_lua):
        names = [m.group(1), *re.findall(r"'([^']+)'", m.group(2))]
        body = _balanced(card_lua, m.end() - 1)
        exprs = [p for p in _split_top_level(body) if p and not re.match(r"^\w+\s*=[^=]", p)]
        for name in names:
            out.setdefault(name, exprs)
    return out


ITEM_SETS = ("Tarot", "Planet", "Spectral", "Voucher", "Booster")
_SET_SECTION = re.compile(r"(?:if|elseif)\s+_c\.set\s*==\s*'(\w+)'\s*then")
_UI_BRANCH = re.compile(
    r"_c\.name\s*==\s*(['\"])(.+?)\1((?:\s*or\s*_c\.name\s*==\s*['\"][^'\"]+['\"])*)"
    r"\s*then\s+(?:desc_override\s*=\s*'(\w+)'\s*;\s*)?loc_vars\s*=\s*\{"
)
_SET_LOC_VARS = re.compile(r"(?<![\w.])loc_vars\s*=\s*\{")


@dataclass(frozen=True)
class UiLocVars:
    """How ``generate_card_ui`` fills one item's text: ``exprs`` for its ``#n#``
    values and, for packs, the description key (``desc_override``)."""

    exprs: list[str]
    desc: str | None = None


def _exprs(body: str) -> list[str]:
    return [p for p in _split_top_level(body) if p and not re.match(r"^\w+\s*=[^=]", p)]


def parse_ui_loc_vars(common_events_lua: str) -> dict[str, UiLocVars]:
    """``{internal item name or set name: UiLocVars}`` for consumables, vouchers and
    packs, from ``generate_card_ui``'s per-set sections (other sections, e.g. the
    unlock conditions shown for locked cards, are skipped).  A set's own entry
    (e.g. ``"Planet"``) applies to every item of the set without one."""
    start = common_events_lua.find("function generate_card_ui")
    if start < 0:
        return {}
    end = common_events_lua.find("\nfunction ", start + 1)
    src = common_events_lua[start : end if end > 0 else None]
    sections = list(_SET_SECTION.finditer(src))
    out: dict[str, UiLocVars] = {}
    for sec, nxt in zip(sections, [*sections[1:], None], strict=True):
        if sec.group(1) not in ITEM_SETS:
            continue
        body = src[sec.end() : nxt.start() if nxt else None]
        branches = list(_UI_BRANCH.finditer(body))
        for m in branches:
            names = [m.group(2), *re.findall(r"['\"]([^'\"]+)['\"]", m.group(3))]
            item = UiLocVars(_exprs(_balanced(body, m.end() - 1)), m.group(4))
            for name in names:
                out.setdefault(name, item)
        first_name = branches[0].start() if branches else len(body)
        default = _SET_LOC_VARS.search(body, 0, first_name)
        if default:  # e.g. every planet: loc_vars = {G.GAME.hands[...].level, ...}
            out.setdefault(sec.group(1), UiLocVars(_exprs(_balanced(body, default.end() - 1))))
    return out


_HAND = re.compile(r"\[\s*\"([^\"]+)\"\s*\]\s*=\s*\{\s*visible\s*=[^\n]*")


def parse_hands(game_lua: str) -> dict[str, dict[str, int]]:
    """``{hand: {"level", "l_mult", "l_chips", ...}}``: game.lua's starting hands table."""
    return {
        m.group(1): {k: int(v) for k, v in re.findall(r"\b(\w+)\s*=\s*(\d+)", m.group(0))}
        for m in _HAND.finditer(game_lua)
    }


def parse_descriptions(loc_lua: str, prefix: str = "j_") -> dict[str, dict]:
    """``{key: {"name", "text": [lines]}}`` from localization/en-us.lua for keys starting
    with ``prefix`` (``j_`` jokers, ``b_`` decks, ``bl_`` blinds, ``c_`` consumables,
    ``v_`` vouchers, ``p_`` packs)."""
    out = {}
    entry = re.compile(
        rf"\b({re.escape(prefix)}\w+)\s*=\s*\{{\s*name\s*=\s*\"([^\"]*)\"\s*,\s*text\s*=\s*\{{"
    )
    for m in entry.finditer(loc_lua):
        if m.group(1) in out:
            continue
        body = _balanced(loc_lua, m.end() - 1)
        out[m.group(1)] = {"name": m.group(2), "text": re.findall(r"\"((?:[^\"\\]|\\.)*)\"", body)}
    return out


_BACK_BRANCH = re.compile(r"name_to_check\s*==\s*'([^']+)'\s*then\s+loc_args\s*=\s*\{")


def parse_back_loc_args(back_lua: str) -> dict[str, list[str]]:
    """``{deck name: [positional loc_args expressions]}`` from back.lua's generate_UI."""
    out: dict[str, list[str]] = {}
    for m in _BACK_BRANCH.finditer(back_lua):
        body = _balanced(back_lua, m.end() - 1)
        out.setdefault(m.group(1), [p for p in _split_top_level(body) if p])
    return out


def parse_misc_dict(loc_lua: str, name: str) -> dict[str, str]:
    """A flat ``name={ Key="Value", ... }`` table from en-us.lua (e.g. suits_singular)."""
    m = re.search(rf"\b{name}\s*=\s*\{{", loc_lua)
    if not m:
        return {}
    # Keys are bare (Clubs=) or bracketed strings (["High Card"]=)
    pairs = re.findall(
        r"(?:\[\"([^\"]+)\"\]|\b(\w+))\s*=\s*\"([^\"]*)\"", _balanced(loc_lua, m.end() - 1)
    )
    return {quoted or bare: value for quoted, bare, value in pairs}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_line(line: str, values: list[Any]) -> str:
    """One description line -> HTML: ``{C:x}`` colours, ``{X:x}`` badges, ``#n#`` values."""

    def value(m: re.Match) -> str:
        i = int(m.group(1)) - 1
        v = values[i] if 0 <= i < len(values) else None
        return UNKNOWN_MARK if v is None else _tostring(v)

    out, style = [], ""
    for part in re.split(r"(\{[^}]*\})", line):
        if part.startswith("{") and part.endswith("}"):
            style = ""
            codes = dict(c.split(":", 1) for c in part[1:-1].split(",") if ":" in c)
            if "C" in codes:
                style += f"color:{COLOURS.get(codes['C'], TEXT_DARK)};"
            if "X" in codes:
                bg = COLOURS.get(codes["X"], TEXT_DARK)
                style += f"background:{bg};border-radius:4px;padding:0 3px;"
            continue
        text = html.escape(re.sub(r"#(\d+)#", value, part))
        if not text:
            continue
        text = text.replace(UNKNOWN_MARK, UNKNOWN_HTML)
        out.append(f'<span style="{style}">{text}</span>' if style else text)
    return "".join(out)


@dataclass
class JokerText:
    name: str
    lines: list[str]  # HTML
    unresolved: int = 0  # placeholders that needed live game state


def _unresolved(text: list[str], values: list[Any]) -> int:
    return sum(
        1
        for line in text
        for n in re.findall(r"#(\d+)#", line)
        if int(n) > len(values) or values[int(n) - 1] is None
    )


_LOCALIZE_NAME = re.compile(r"localize\s*\{[^{}]*?key\s*=\s*'(\w+)'[^{}]*\}")
# localize{type = 'name_text', set = 'Enhanced', key = _c.config.mod_conv} -> __name(...)
_LOCALIZE_NAME_EXPR = re.compile(
    r"localize\s*\{(?=[^{}]*type\s*=\s*'name_text')[^{}]*?key\s*=\s*([\w.]+)[^{}]*\}"
)


@dataclass
class GameText:
    """Joker, deck, blind, consumable, voucher and pack names and effect text from one
    game archive (parsed once)."""

    archive: Path
    descriptions: dict[str, dict] = field(default_factory=dict)
    loc_vars: dict[str, list[str]] = field(default_factory=dict)
    misc: dict[str, dict[str, str]] = field(default_factory=dict)
    decks: dict[str, dict] = field(default_factory=dict)
    blinds: dict[str, dict] = field(default_factory=dict)
    back_args: dict[str, list[str]] = field(default_factory=dict)
    loc_names: dict[str, str] = field(default_factory=dict)  # any key -> in-game name
    items: dict[str, dict] = field(default_factory=dict)  # consumables, vouchers, packs
    ui_loc_vars: dict[str, UiLocVars] = field(default_factory=dict)
    hands: dict[str, dict[str, int]] = field(default_factory=dict)

    @classmethod
    def load(cls, archive: Path) -> GameText:
        with zipfile.ZipFile(archive) as z:
            loc = z.read("localization/en-us.lua").decode("utf-8", "replace")
            card = z.read("card.lua").decode("utf-8", "replace")
            files = set(z.namelist())

            def read(name: str) -> str:
                return z.read(name).decode("utf-8", "replace") if name in files else ""

            back = read("back.lua")
            common_events = read("functions/common_events.lua")
            game = read("game.lua")
        misc = {
            kind: parse_misc_dict(loc, kind)
            for kind in ("suits_singular", "suits_plural", "poker_hands", "ranks")
        }
        loc_names = dict(re.findall(r"\b(\w+)\s*=\s*\{\s*name\s*=\s*\"([^\"]*)\"", loc)[::-1])
        return cls(
            Path(archive),
            parse_descriptions(loc),
            parse_loc_vars(card),
            misc,
            parse_descriptions(loc, "b_"),
            parse_descriptions(loc, "bl_"),
            parse_back_loc_args(back),
            loc_names,
            {k: v for p in ("c_", "v_", "p_") for k, v in parse_descriptions(loc, p).items()},
            parse_ui_loc_vars(common_events),
            parse_hands(game),
        )

    def _localize(self, key: Any, kind: Any = None) -> Any:
        return self.misc.get(kind, {}).get(key, key)

    def values(self, key: str) -> list[Any]:
        entry = catalog.centers().get(key, {})
        exprs = self.loc_vars.get(entry.get("name", ""), [])
        names = {
            "self": _Obj(ability=starting_ability(entry)),
            "G": _Obj(GAME=None, deck=None, jokers=None, playing_cards=None,
                      C=_wrap({"SUITS": {k: COLOURS[k.lower()] for k in
                                         ("Spades", "Hearts", "Clubs", "Diamonds")}})),
            "localize": self._localize,
            "tostring": _tostring,
            "math": _Obj(floor=math.floor, ceil=math.ceil, max=max, min=min, abs=abs),
            "__len": lambda v: len(v) if v is not None else 0,
        }  # fmt: skip
        return self._run_all(exprs, names)

    @staticmethod
    def _run_all(exprs: list[str], names: dict[str, Any]) -> list[Any]:
        out: list[Any] = []
        for expr in exprs:
            try:
                out.append(_Evaluator(names).run(expr))
            except (_Missing, SyntaxError, TypeError, ZeroDivisionError, ValueError):
                out.append(None)
        return out

    def _text(self, desc: dict | None, values: list[Any]) -> JokerText | None:
        if not desc:
            return None
        lines = [format_line(line, values) for line in desc["text"] if line.strip()]
        return JokerText(desc["name"], lines, _unresolved(desc["text"], values))

    def joker(self, key: str) -> JokerText | None:
        desc = self.descriptions.get(key)
        return self._text(desc, self.values(key)) if desc else None

    def deck_values(self, key: str) -> list[Any]:
        """A deck's ``loc_args`` (back.lua), evaluated against its starting config."""
        entry = catalog.centers().get(key, {})
        exprs = [
            # localize{type = 'name_text', key = 'v_telescope', ...} -> that item's name
            _LOCALIZE_NAME.sub(
                lambda m: repr(self.loc_names.get(m.group(1), catalog.name(m.group(1)))), expr
            )
            for expr in self.back_args.get(entry.get("name", ""), [])
        ]
        return self._run_all(exprs, {"effect_config": _wrap(entry.get("config") or {})})

    def deck(self, key: str) -> JokerText | None:
        return self._text(self.decks.get(key), self.deck_values(key))

    def blind(self, key: str) -> JokerText | None:
        """A blind's text; values that depend on the run (e.g. The Ox's hand) are unknown."""
        return self._text(self.blinds.get(key), [])

    def item(self, key: str) -> JokerText | None:
        """A consumable's, voucher's or pack's text with its starting values (planets at
        level 1); values that depend on the run (e.g. The Fool's last card) are unknown."""
        entry = catalog.centers().get(key, {})
        ui = self.ui_loc_vars.get(entry.get("name", "")) or self.ui_loc_vars.get(
            entry.get("set", "")
        )
        desc = self.items.get(ui.desc) if ui and ui.desc else self.items.get(key)
        names = {
            "_c": _Obj(config=_wrap(entry.get("config") or {}), name=entry.get("name")),
            "G": _Obj(GAME=_Obj(probabilities=_Obj(normal=1), hands=_wrap(self.hands))),
            "localize": self._localize,
            "__name": lambda k: self.loc_names.get(k, catalog.name(k)),
            "math": _Obj(floor=math.floor, ceil=math.ceil, max=max, min=min, abs=abs),
        }  # fmt: skip
        exprs = [_LOCALIZE_NAME_EXPR.sub(r"__name(\1)", e) for e in ui.exprs] if ui else []
        return self._text(desc, self._run_all(exprs, names))

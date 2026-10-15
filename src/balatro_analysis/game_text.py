"""Joker effect text, read from the player's own copy of Balatro.

No game text is stored in this repo. From the game archive (see ``sprites``):

- ``localization/en-us.lua`` holds each joker's description lines, e.g.
  ``"{C:mult}+#1#{} Mult"``: ``{C:...}`` codes style the text, ``#n#`` are
  placeholders for values.
- ``card.lua`` fills the placeholders per joker with ``loc_vars = {...}``
  expressions such as ``self.ability.extra.s_mult``. They are evaluated here
  against the joker's *starting* ability (what the in-game collection shows),
  with a tiny whitelisted evaluator, never ``eval``. Values that depend on a
  live run (e.g. cards left in deck) come out as "?".
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


def parse_descriptions(loc_lua: str) -> dict[str, dict]:
    """``{joker key: {"name", "text": [lines]}}`` from localization/en-us.lua."""
    out = {}
    entry = re.compile(r"\b(j_\w+)\s*=\s*\{\s*name\s*=\s*\"([^\"]*)\"\s*,\s*text\s*=\s*\{")
    for m in entry.finditer(loc_lua):
        if m.group(1) in out:
            continue
        body = _balanced(loc_lua, m.end() - 1)
        out[m.group(1)] = {"name": m.group(2), "text": re.findall(r"\"((?:[^\"\\]|\\.)*)\"", body)}
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


@dataclass
class GameText:
    """Joker names and effect text from one game archive (parsed once)."""

    archive: Path
    descriptions: dict[str, dict] = field(default_factory=dict)
    loc_vars: dict[str, list[str]] = field(default_factory=dict)
    misc: dict[str, dict[str, str]] = field(default_factory=dict)

    @classmethod
    def load(cls, archive: Path) -> GameText:
        with zipfile.ZipFile(archive) as z:
            loc = z.read("localization/en-us.lua").decode("utf-8", "replace")
            card = z.read("card.lua").decode("utf-8", "replace")
        misc = {
            kind: parse_misc_dict(loc, kind)
            for kind in ("suits_singular", "suits_plural", "poker_hands", "ranks")
        }
        return cls(Path(archive), parse_descriptions(loc), parse_loc_vars(card), misc)

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
        out: list[Any] = []
        for expr in exprs:
            try:
                out.append(_Evaluator(names).run(expr))
            except (_Missing, SyntaxError, TypeError, ZeroDivisionError, ValueError):
                out.append(None)
        return out

    def joker(self, key: str) -> JokerText | None:
        desc = self.descriptions.get(key)
        if not desc:
            return None
        values = self.values(key)
        lines = [format_line(line, values) for line in desc["text"] if line.strip()]
        unresolved = sum(
            1
            for line in desc["text"]
            for n in re.findall(r"#(\d+)#", line)
            if int(n) > len(values) or values[int(n) - 1] is None
        )
        return JokerText(desc["name"], lines, unresolved)

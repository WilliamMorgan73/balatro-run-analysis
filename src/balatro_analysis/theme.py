"""The game look on top of ``.streamlit/config.toml``: Balatro's pixel font (from
the player's own install, never stored here) on headings, tabs and numbers, and
the game's chunky "lip" under buttons and panels.

Colours live in ``config.toml``; the same values are here for the HTML views.
Without the game the font falls back to the theme's sans and nothing else changes.
"""

from __future__ import annotations

TABLE = "#2c3a3d"  # page background
PANEL = "#3a4c56"
INSET = "#4f6367"
LIP = "#1d282e"  # the solid shadow under panels and buttons
CHIPS = "#009dff"
MULT = "#fe5f55"
MONEY = "#f3b958"
PLAY = "#fda200"
WIN = "#4bc292"

PIXEL = "'m6x11', 'Source Sans', sans-serif"


def css(font_uri: str | None) -> str:
    """One ``<style>`` block for ``st.html``."""
    face = (
        f"@font-face{{font-family:'m6x11';src:url({font_uri}) format('truetype');"
        "font-display:swap}"
        if font_uri
        else ""
    )
    return f"""<style>
{face}
:root{{--bx-pixel:{PIXEL}}}
/* Streamlit's own rules are more specific: only the family needs !important */
[data-testid="stHeadingWithActionElements"] :is(h1,h2,h3),
[data-testid="stMetricValue"],
[data-testid="stTab"] p,
[data-testid="stSidebar"] [data-testid="stExpander"] summary p{{
  font-family:var(--bx-pixel)!important;letter-spacing:.02em;font-weight:400}}
[data-testid="stHeadingWithActionElements"] h1{{font-size:2.8rem}}
[data-testid="stHeadingWithActionElements"] :is(h2,h3){{font-size:1.75rem}}
[data-testid="stMetricValue"]{{font-size:2.1rem}}
[data-testid="stTab"] p{{font-size:1.3rem}}
[data-testid="stSidebar"] [data-testid="stExpander"] summary p{{font-size:1.2rem}}
[data-testid="stBaseButton-secondary"],[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-pills"],[data-testid="stBaseButton-pillsActive"]{{
  box-shadow:0 3px 0 {LIP};transition:transform .06s,box-shadow .06s}}
[data-testid="stBaseButton-secondary"]:active,[data-testid="stBaseButton-primary"]:active{{
  transform:translateY(2px);box-shadow:0 1px 0 {LIP}}}
[data-testid="stMetric"]{{background:{PANEL};border:2px solid {LIP};border-radius:10px;
  padding:8px 12px;box-shadow:0 3px 0 {LIP}}}
@media (prefers-reduced-motion:reduce){{*{{transition:none!important;animation:none!important}}}}
</style>"""

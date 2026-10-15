# Balatro run analysis

Streamlit dashboard for the shop-only RL training runs produced by
`scripts/train_random_shop.py` in
[jackdaw-balatro](../jackdaw-balatro) (branch `test/shop-rl-feasibility`).

## Usage

```bash
uv sync
uv run streamlit run app.py                                   # reads ../jackdaw-balatro/runs
uv run streamlit run app.py -- --runs-dir /path/to/runs       # or anywhere else
```

Choose the runs folder in the sidebar with **Browse…** (opens your desktop's folder
dialog: kdialog, zenity, or Tk as a fallback) or type a path. Pick any number of runs
to overlay, filter by deck and stake, and use **Reload runs** after a training run
finishes.

## Joker and deck icons

Icons are cut at runtime from **your own copy of Balatro**; no game art is stored in
this repo. The game is found automatically in Steam libraries (Linux, Flatpak, macOS,
Windows), or set `BALATRO_PATH` / the *Game assets* sidebar field to the install folder
or `Balatro.exe`. Without the game the dashboard shows names only.

Names and sprite positions come from `src/balatro_analysis/data/centers.json`, generated
from jackdaw's data by `uv run python scripts/sync_centers.py`.

## Run format

Any directory containing `episodes.csv` is a run. The other files are optional,
so older runs (only `episodes.csv`, no deck column) still load; tabs that need missing data say so.

| File | One row per | Used for |
|---|---|---|
| `episodes.csv` | episode | ante reached, wins, deck, stake, return, action counts, $ spent, final jokers |
| `config.json` | run | agents, seed prefix, CLI args (shown under *Run configs*) |
| `steps.jsonl` | shop decision | money, item bought/sold, jokers held, reward |
| `rounds.jsonl` | blind played | score vs target, which blind ended the run |

## Tabs

- **Overview**: summary per run, rolling learning curve (ante / win rate / return), ante-reached distribution
- **Decks**: deck cards with episodes and mean ante; mean ante by deck and stake per run
- **Actions**: share of each of the 17 shop actions; action × ante heatmap
- **Economy**: money before shop decisions by ante; $ spent per episode
- **Jokers**: most-bought gallery; ante first bought; end-of-episode co-occurrence; sold → next bought pairs
- **Rounds**: score / target by ante and blind type; blinds that ended runs
- **Episode explorer**: one episode's deck, final jokers, shop decisions, money curve and blinds

## Layout

```
app.py                          Streamlit UI only
src/balatro_analysis/loader.py  find_runs / load_run -> Run
src/balatro_analysis/metrics.py pure pandas metrics (testable without Streamlit)
src/balatro_analysis/plots.py   Plotly figure builders
src/balatro_analysis/catalog.py joker/deck display names
src/balatro_analysis/sprites.py icons from the local game install
src/balatro_analysis/picker.py  native folder dialog
tests/                          pytest + Streamlit AppTest on a small fixture run
```

```bash
uv run pytest
uv run ruff check .
```

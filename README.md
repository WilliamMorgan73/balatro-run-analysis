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

## Names: presenting vs analysing

The **In-game names** toggle (top of the sidebar) switches every label between the
names shown in the game ("Greedy Joker", "Black Deck", "The Hook", "Buy card 1", "$ before")
and the exact keys (`j_greedy_joker`, `b_black`, `bl_hook`, `buy_card_0`, `dollars_before`).
Open `http://localhost:8501/?names=keys` to start in key mode.

## Export

Sidebar → **Export**, always for the current run/episode/deck/stake selection:

- **Data (ZIP of CSVs)**: `runs/<run>/{episodes,steps,rounds}.csv` plus `tables/*.csv`
  for every table and chart. Each key column has a `<column>_name` twin, so the files work for
  both analysis and slides, whatever the toggle says. A `README.txt` records the selection.
- **All charts (ZIP)**: every chart currently shown, as PNG (3200x1800) and/or SVG, in a
  light theme for slides/LaTeX. Each chart also has its own **⬇ Export chart** button.

Chart rendering uses kaleido with your local Chrome/Chromium. If none is found the chart buttons
say so; `uv run kaleido_get_chrome` downloads one.

## Joker and deck icons

Icons are cut at runtime from **your own copy of Balatro**; no game art is stored in
this repo. The game is found automatically in Steam libraries (Linux, Flatpak, macOS,
Windows), or set `BALATRO_PATH` / the *Game assets* sidebar field to the install folder
or `Balatro.exe`. Without the game the dashboard shows names only.

Names and sprite positions come from `src/balatro_analysis/data/{centers,blinds}.json`, generated
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
src/balatro_analysis/catalog.py display names for game keys (centers, blinds)
src/balatro_analysis/names.py   in-game names vs exact keys (the sidebar toggle)
src/balatro_analysis/export.py  data ZIP and PNG/SVG chart export
src/balatro_analysis/sprites.py icons from the local game install
src/balatro_analysis/picker.py  native folder dialog
tests/                          pytest + Streamlit AppTest on a small fixture run
```

```bash
uv run pytest            # fast suite
uv run pytest -m slow    # also renders real PNG/SVG with kaleido
uv run ruff check .
```

The app reloads its own package modules when they change, so a server left running
picks up `git pull`s without a restart.

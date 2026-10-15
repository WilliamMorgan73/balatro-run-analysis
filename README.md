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

The runs directory can also be changed in the sidebar. Pick any number of runs to
overlay; use **Reload runs** after a training run finishes.

## Run format

Any directory containing `episodes.csv` is a run. The other files are optional,
so older runs (only `episodes.csv`) still load; tabs that need missing data say so.

| File | One row per | Used for |
|---|---|---|
| `episodes.csv` | episode | ante reached, wins, return, action counts, $ spent, final jokers |
| `config.json` | run | agents, seed prefix, CLI args (shown under *Run configs*) |
| `steps.jsonl` | shop decision | money, item bought/sold, jokers held, reward |
| `rounds.jsonl` | blind played | score vs target, which blind ended the run |

## Tabs

- **Overview**: summary per run, rolling learning curve (ante / win rate / return), ante-reached distribution
- **Actions**: share of each of the 17 shop actions; action × ante heatmap
- **Economy**: money before shop decisions by ante; $ spent per episode
- **Jokers**: most bought; ante first bought; end-of-episode co-occurrence; sold → next bought pairs
- **Rounds**: score / target by ante and blind type; blinds that ended runs
- **Episode explorer**: one episode's shop decisions, money curve and blinds

## Layout

```
app.py                          Streamlit UI only
src/balatro_analysis/loader.py  find_runs / load_run -> Run
src/balatro_analysis/metrics.py pure pandas metrics (testable without Streamlit)
src/balatro_analysis/plots.py   Plotly figure builders
tests/                          pytest + Streamlit AppTest on a small fixture run
```

```bash
uv run pytest
uv run ruff check .
```

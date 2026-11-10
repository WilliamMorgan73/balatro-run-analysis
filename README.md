# Balatro run analysis

Streamlit dashboard for shop-only RL training runs on
[Jackdaw](https://github.com/TylerFlar/jackdaw-balatro), a Python Balatro simulator.
The runs come from `balatro-shop-rl` (see [Producing runs](#producing-runs)); this repo
only reads them. The format is in [docs/run-format.md](docs/run-format.md).

## Usage

```bash
uv sync
uv run streamlit run app.py                                   # reads ../balatro-shop-rl/runs
uv run streamlit run app.py -- --runs-dir /path/to/runs       # or anywhere else
```

The sidebar holds only what you are looking at: **Runs** (pick any number to overlay; *Find
runs* searches and filters them), **Reload runs** to pick up new runs and new episodes, the
*Test* / *Evaluation* / *Training* switch, and **Filter** (deck, stake, episodes or eval round, rolling
window), with a one-line summary of the filter above it. **Settings** holds the things set
once: in-game names, the runs folder and the Balatro install (**Browse…** opens your
desktop's folder dialog: kdialog, zenity, or Tk as a fallback).

The Overview opens with a **run sheet** per run, styled like the game's run-over screen:
mean ante with its CI, win rate, best ante, most played hand (eval hand logs), and buttons
that open the run's best and worst episode in the Episode explorer. The six best runs are
shown; the rest fold into an expander.

### Runs from a manifest, and studies

If the runs folder has a `manifest.json` (written by balatro-shop-rl's
`uv run python scripts/run_manifest.py`), or one is given with `--manifest PATH` or under
**Settings**, the runs come from it rather than a scan of the folder: every run it names,
wherever it is (`runs/` and archive folders), so one group can span both. Regenerate it after
syncing runs to pick up new ones.

Its **studies** (from balatro-shop-rl's `analysis/studies.toml`) appear in a **Study** picker in
the sidebar. Choosing one picks its groups' runs, lists the manifest's problems for it (missing
seeds, runs without a test set, a group spread over several commits), and adds its tables to
the top of the Overview, on the episodes shown (the test games by default):

- per group: seeds, win % ± SD across seeds, final round, mean ante, and IQM ante with a
  stratified-bootstrap 95% CI;
- per pair A − B: the difference in ante and in win rate matched by training seed (same seed,
  same training games, so game luck cancels), with a Student-t 95% CI, p and the seeds A is
  ahead in, and the probability of improvement with its CI;
- the same per value of the study's `split_by` column (e.g. per deck), in expanders;
- **Study tables** buttons for CSV and LaTeX (booktabs).

Picking a run reads only its `episodes.csv` files (training, eval and test) and config.
Each JSONL log (`steps`, `rounds`, `shops`, `hands`, `decks`, `offers`) is read the first
time a view needs it and kept, so the Overview on the test games reads none of them and
picking every run is cheap; the training `steps.jsonl` (hundreds of MB) is only read for
the tabs that show shop decisions. Before a log is read, the tables of runs not shown are
dropped if needed to stay within the memory set aside for runs; a view that still wouldn't
fit says so instead of loading it.

Runs, and each log once parsed, are cached in `~/.cache/balatro-analysis` (or
`$XDG_CACHE_HOME`), so later starts skip parsing (a training `steps.jsonl`: ~8 s parsed,
~0.3 s cached). A run still training is brought up to date by reading only the lines
logged since it was last loaded (a second or so instead of a full parse); a log is parsed
again whole only when it, the run's `config.json` or the loading code (`loader.py`,
`runcache.py`; not comments or docstrings) changes, or a log was rewritten rather than
appended to. Editing the rest of the package keeps loaded runs in memory. Set
`BALATRO_ANALYSIS_CACHE` to use another folder, or to `0` to turn the cache off. Deleting
the folder is always safe.

## Producing runs

Runs are written by `balatro-shop-rl`, the agent code (shop environment, play
policies, shop agents, training), which pulls Jackdaw from a pinned commit on
[WilliamMorgan73/jackdaw-balatro](https://github.com/WilliamMorgan73/jackdaw-balatro).
From a checkout next to this one:

```bash
cd ../balatro-shop-rl
uv run python scripts/train.py --episodes 500 --play smart
uv run python scripts/train.py --episodes 500 --play smart \
    --decks b_red,b_blue,b_black --stakes 1,2 --run-name smart_mixed
uv run python scripts/train.py --episodes 500 --play smart \
    --seed 1 --group smart_random --run-name smart_random_s1   # one seed of a group
uv run python scripts/train.py --episodes 1000 --play smart \
    --shop qlearn --eval-every 100 --eval-episodes 20   # tabular Q-learning + greedy eval
uv run python scripts/train.py --episodes 1000 --play strong \
    --shop heuristic --eval-every 1000 --eval-episodes 500   # stronger play policy
```

Each run goes to `../balatro-shop-rl/runs/<run-name>/`, which is where the dashboard
looks by default. Its `config.json` records the `balatro-shop-rl` and Jackdaw commits
it ran on.

## Names: presenting vs analysing

The **In-game names** toggle (sidebar → Settings) switches every label between the
names shown in the game ("Greedy Joker", "Black Deck", "The Hook", "Buy card 1", "$ before")
and the exact keys (`j_greedy_joker`, `b_black`, `bl_hook`, `buy_card_0`, `dollars_before`).
Open `http://localhost:8501/?names=keys` to start in key mode.

## Export

Sidebar → **Export**, always for the current run/episode/deck/stake selection:

- **Data (ZIP of CSVs)**: `runs/<run>/{episodes,steps,rounds}.csv` plus `tables/*.csv`
  for every table and chart. Each key column has a `<column>_name` twin, so the files work for
  both analysis and slides, whatever the toggle says. A `README.txt` records the selection.
  `run_comparison.csv` uses the first selected run as the baseline.
  `latex/*.tex` holds the main result tables (summary, CIs, run comparison, IQM, probability
  of improvement and power, and the group versions when runs share a `--group`; the power
  tables use the difference set in Overview, 0.25 ante by default) as booktabs floats with in-game names and
  `mean [95% CI]` cells, ready for `\input`. They need `\usepackage{booktabs,graphicx}`;
  wide tables are scaled to `\linewidth`. Captions and `tab:<name>` labels are included.
- **All charts (ZIP)**: every chart currently shown, as PNG (3200x1800) and/or SVG, in a
  light theme for slides/LaTeX, plus `charts/README.md` with each chart's description (the
  caption shown under it in the dashboard), ready to adapt into figure captions. Each chart
  also has its own **⬇ Export chart** button.

Chart rendering uses kaleido with your local Chrome/Chromium. If none is found the chart
buttons say so; `uv run kaleido_get_chrome` downloads one.

## Game art, font and text

Icons, playing cards (with enhancements and seals), the pixel font (`m6x11plus`) and joker
effect text are read at runtime from **your own copy of Balatro**; no game art, font or text
is stored in this repo. The game is found automatically in Steam libraries (Linux, Flatpak, macOS,
Windows), or set `BALATRO_PATH` / the *Balatro install* field under Settings to the install
folder or `Balatro.exe`. Without the game the dashboard shows names and text cards, in the
theme's sans font.

The colours (`.streamlit/config.toml`) are the game's own: the dark slate table, chips blue,
mult red, money gold. `theme.py` adds the font and the chunky "lip" under buttons and panels.

Effect text comes from the game's `localization/en-us.lua`; its `#1#`-style values are filled
from the game's own `card.lua` expressions, evaluated against each joker's starting stats with
a small whitelisted evaluator (never `eval`). Values that only exist mid-run (this round's suit,
current money, ...) show as a faded **?**.

Names and sprite positions come from `src/balatro_analysis/data/{centers,blinds}.json`,
generated from jackdaw's data by `uv run python scripts/sync_centers.py`.

## Run format

Any directory containing `episodes.csv` is a run. The other files are optional, so older
runs (only `episodes.csv`, no deck column) still load; tabs that need missing data say so.

| File | One row per | Used for |
|---|---|---|
| `episodes.csv` | episode | ante reached, wins, deck, stake, return, action counts, $ spent, final jokers |
| `config.json` | run | agents, seed prefix, CLI args (shown under *Run configs*) |
| `steps.jsonl` | shop decision | money, item bought/sold, jokers held, reward; legal actions † |
| `rounds.jsonl` | blind played | score vs target, which blind ended the run; cash-out breakdown \* |
| `shops.jsonl` \* | shop offer (entering a shop, each reroll) | what was offered, at what price, with how much money |

\* `log_version` 2 runs. Older (v1) runs logged beaten bosses under the *next* ante; the loader
moves them back to the ante they were played in.
† `log_version` 3 runs.

From `log_version` 6, `episodes.csv` has `max_ante`, the highest ante a blind was played in. The
loader uses it as the episode's ante (the raw `ante` column, the ante the run ended on, is kept as
`final_ante`), so a win counts as ante 8, not 9, and a Hieroglyph or Petroglyph setback doesn't
lower how far the run got. In `steps.jsonl`, a sell's `item_cost` is what the card sold for (it
was the card's buy price before).

Runs made with `--eval-every` (`log_version` 4) also have an `eval/` folder in the same format:
greedy episodes on the fixed seeds `EVAL_0..K-1`, played before training and after every N
training episodes, with `eval_round` and `train_episodes` columns. Every run with the same eval
settings plays the same eval games. `eval/` is loaded with its run (it isn't listed as a run of
its own).

Runs from `log_version` 14 also end with a `test/` folder in the same format: the final policy
on one fixed set of games (`TEST_0..`, 500 by default), the same for every run and never used
in training. It is loaded with its run too.

The sidebar **Episodes** switch swaps every tab over to one kind of episode:

- *Test* (the default when runs have it): the results to report. Paired comparisons match runs
  game by game.
- *Evaluation*: the eval rounds, for learning curves; an **Eval round** picker (under Filter)
  defaults to the last round. Too few games per round to rank runs on.
- *Training*: the episodes trained on, with exploration, for diagnostics.

The Overview's eval learning curve is drawn from the eval episodes whichever is chosen. The run
list ranks runs on their test games where they have them (*Scored on* says which).

From `log_version` 7, eval episodes also log every play and discard (`eval/hands.jsonl`) and the
deck at each blind (`eval/decks.jsonl`), for the Hands tab and the explorer's Replay. Training
episodes don't: there are far more of them and the log is several times the size of the rest.

From `log_version` 15, scaling jokers (Spare Trousers, Ice Cream, Ramen, ...) log their current
values in `eval/hands.jsonl` and in `steps.jsonl` (`joker_state`): the Replay shows each one's
value under its card ("+6 Mult") and notes when it changes ("+0 Chips → +4 Chips"), and the
explorer's tables add a *Scaling jokers* column.

## Tabs

- **Overview**: summary per run; 95% CIs of mean ante, win rate (Wilson) and return; with
  two or more runs, a comparison against a chosen baseline (difference in mean ante with a
  bootstrap CI, Mann-Whitney U + rank-biserial effect size, Fisher's exact on wins, and a
  Wilcoxon signed-rank test on episodes both runs played on the same seed/deck/stake);
  rolling learning curve (ante / win rate / return) with a 95% band; ante-reached distribution;
  for runs with eval episodes, eval mean ante / win rate on the fixed seeds against training
  episodes, with a 95% CI. When runs share a `--group` (one config, several training seeds), a
  **Compare: Runs / Groups** switch averages each group's runs: summary with a Student-t CI
  across seeds, Welch's t-test against a baseline group, and group learning / eval curves with
  bands across seeds. Use groups for claims about a method, since a single training run can be lucky.
  Both views also give the **interquartile mean** (IQM, Agarwal et al. 2021) of ante and return
  with a stratified bootstrap 95% CI: for a group whose runs played the same eval games, training
  seeds are redrawn within each game (rliable's runs × tasks bootstrap), otherwise episodes within
  each run. **Can these runs tell agents apart?** is a power readout (two-sided, α = 0.05, 80%
  power): the smallest difference in mean ante the selection can detect, and the games (runs) or
  seeds per group (groups, from the game-to-game and seed-to-seed variance) needed to detect a
  difference you choose, with a warning when the current eval set is too small. With two or
  more runs (groups), a **probability of improvement** table gives, for each against the
  baseline, the chance one of its games reaches a higher ante than one of the baseline's (ties
  half), with a 95% CI: paired game by game when runs share games, rliable's per-game version
  for groups. A **performance profile** chart shows the share of games reaching each ante,
  with bootstrap bands, so you can see where each agent's runs break down.
- **Decks**: deck cards with episodes and mean ante; mean ante by deck and stake per run
- **Actions**: KPI strip (decisions per visit, visits leaving at once, visits with a buy, most
  common first action) and sub-tabs:
  - *Overview*: share by kind of decision (buy card/pack/voucher, sell, use, reroll, leave), share
    of each of the 17 actions, decisions per visit
  - *Context*: action share split by ante, money before the decision, jokers held, next blind or
    position in the visit
  - *Sequences*: what follows each decision (transition matrix), action share by position in the
    visit, the most common whole visits
  - *Over training*: kinds of decision across blocks of episodes
  - *Availability* †: how often each action was legal and chosen when legal; preference vs a
    uniform-random agent; legal actions per decision
  - *Actions vs outcome*: Spearman correlation of each kind of decision per visit with ante
    reached, with p-values Benjamini-Hochberg adjusted within each run (\* = q < 0.05)

  † needs runs from the current training script (`log_version` 3 in `config.json`)
- **Economy**: KPI strip (spend, selling, rerolls, money in/out of the shop, exits at the interest
  cap, money at death) and sub-tabs:
  - *Money flow*: money entering/leaving the shop by round (median + IQR), by ante, at run end
  - *Spending*: share by category (jokers, packs, rerolls, vouchers, consumables), mix by ante,
    income by source
  - *Interest & saving*: money left after shopping in $5 brackets vs the cap, estimated interest,
    money parked above the cap
  - *Rerolls & prices*: rerolls per visit, reroll spend by ante, price paid by category, joker
    price by ante
  - *Economy vs outcome*: Spearman correlation of each habit with ante reached (per-visit
    measures, since totals grow with run length), BH-adjusted like the actions one, and any
    habit by ante reached
  - *Cash-out income* \*: blind reward, interest, unused hands, joker $ and rent per beaten blind;
    interest earned vs the maximum
  - *Shop offers* \*: what was offered, affordable and bought per category; offered vs paid
    price; jokers most often affordable but skipped

  \* needs runs from the current training script (`log_version` 2 in `config.json`)
- **Jokers**: most-bought gallery; ante first bought; end-of-episode co-occurrence; sold → next bought pairs
- **Joker collection**: all 150 jokers as game-style cards (art, effect text, rarity, cost) with
  stats from the selected runs (times bought, first ante, held at end, mean ante); search,
  rarity filter, sort, "only used in runs"
- **Rounds**: score / target by ante and blind type; blinds that ended runs
- **Hands** (Evaluation, `log_version` 7): share of plays by poker hand, one play's score ÷
  the blind's target by hand, hands and discards used per blind by ante, enhanced /
  editioned / sealed cards in the deck by ante
- **Episode explorer**: suggested episodes as one-click pills, *Browse episodes* (filters and
  a clickable table), **Previous** / **Next**, and the episode's deck, stats and final jokers, then:
  - *Replay* (Evaluation, `log_version` 7): pick a blind and see every discard and play as
    in the game: the hand as card faces (played cards raised, discarded ones dropped), the
    hand type and chips × mult = score, the blind's progress, and the deck at its start;
    from `log_version` 15, scaling jokers' values under their cards and as they change
  - *Shop*: every shop visit drawn as the shop, each offer (entering and after each reroll)
    with prices, bought items raised, and the other decisions taken on it
  - *Money & blinds*: the money curve, shop decisions and blinds played

## Layout

```
app.py                          Streamlit UI only
src/balatro_analysis/loader.py  find_runs / load_run -> Run
src/balatro_analysis/metrics.py pure pandas metrics (testable without Streamlit)
src/balatro_analysis/economy.py economy metrics (shop visits, spending, interest, offers)
src/balatro_analysis/stats.py   confidence intervals, IQM, power, run-vs-baseline tests, p/q-values
src/balatro_analysis/plots.py   Plotly figure builders
src/balatro_analysis/catalog.py display names for game keys (centers, blinds)
src/balatro_analysis/names.py   in-game names vs exact keys (the sidebar toggle)
src/balatro_analysis/export.py  data ZIP and PNG/SVG chart export
src/balatro_analysis/sprites.py icons from the local game install
src/balatro_analysis/game_text.py joker effect text from the local game install
src/balatro_analysis/collection.py the Joker collection cards
src/balatro_analysis/picker.py  native folder dialog
src/balatro_analysis/cards.py   playing cards from the eval hand log
src/balatro_analysis/hands.py   play metrics from the eval hand log
src/balatro_analysis/replay.py  game-styled HTML: replay, deck, shop visits, run sheets
src/balatro_analysis/theme.py   the game font and panel styling on top of config.toml
scripts/sync_centers.py         regenerate the name/sprite catalog from jackdaw
docs/run-format.md              the run folder format (log_version 15)
.streamlit/config.toml          the theme (Balatro's colours)
tests/                          pytest + Streamlit AppTest on a small fixture run
```

## Development

```bash
uv run pytest            # fast suite
uv run pytest -m slow    # + real PNG/SVG render and a train -> load run through ../balatro-shop-rl
uv run ruff check .
```

The app reloads its own package modules when they change, so a server left running
picks up `git pull`s without a restart.

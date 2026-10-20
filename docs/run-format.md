# Run format (log_version 6)

The contract between `balatro-shop-rl`, which writes runs (`src/shop_rl/runlog.py`,
`shop_env.py`, `train.py`), and this dashboard, which only reads them
(`src/balatro_analysis/loader.py`). When the format changes, bump `log_version` in
`write_config`, add a line to [Version history](#version-history), and regenerate the test
fixtures.

A **run** is one folder, `runs/<run-name>/`:

| File | Rows | Always there? |
| --- | --- | --- |
| `config.json` | the run's settings | yes (a run without one is read as current) |
| `episodes.csv` | one per training episode | yes: it is what makes a folder a run |
| `steps.jsonl` | one per shop decision | unless `--no-step-log` |
| `rounds.jsonl` | one per blind played | unless `--no-step-log` |
| `shops.jsonl` | one per shop offer (entering a shop, each reroll) | unless `--no-step-log` |
| `eval/` | the same files for greedy eval episodes | with `--eval-every N` |
| `q_table.json` | the learned table | `--shop qlearn` only |
| `events.out.tfevents.*` | tensorboard scalars | with the `train` extra installed |

The `.jsonl` files and `episodes.csv` are flushed row by row, so a run still training can be
read up to its last line.

## config.json

Every command-line option of `scripts/train.py` (`episodes`, `play`, `shop`, `seed`,
`seed_prefix`, `decks`, `stakes`, `run_name`, `group`, `eval_every`, `eval_episodes`,
`eval_seed_prefix`, the Q-learning and heuristic parameters, ...), plus:

| Key | Meaning |
| --- | --- |
| `play_agent`, `shop_agent` | class names, e.g. `StrongPlayAgent`, `QLearningShopAgent` |
| `play_params`, `agent_params` | the agents' parameters (`{}` if none) |
| `n_actions`, `action_names` | the shop action space: 17 names, index = action id |
| `log_version` | this spec's version: 6 |
| `eval_set` | the eval games, `[{seed, deck, stake}, ...]` (`[]` without eval) |
| `started` | local start time, ISO 8601 |
| `shop_rl_commit` | `balatro-shop-rl` commit, `-dirty` if it had uncommitted changes; `null` outside git |
| `jackdaw_commit` | Jackdaw commit the run used (the pinned fork commit) |

`group` marks runs of one config with different `--seed`s; the dashboard aggregates them.
Runs from before the repo split (Jackdaw fork, up to `fc213af`) have no `shop_rl_commit`, and
their `jackdaw_commit` is the fork checkout's commit.

## episodes.csv

| Column | Meaning |
| --- | --- |
| `episode` | 0-based episode number |
| `ante` | ante the run ended on (lowered by Hieroglyph/Petroglyph) |
| `max_ante` | highest ante a blind was played in: 8 for a win |
| `won` | 1 if the run was won |
| `return` | sum of shop rewards |
| `shop_steps` | shop decisions taken |
| `shops_visited` | shops entered (returning from a pack doesn't count) |
| `dollars_spent` | money spent on shop actions |
| `seconds` | wall time of the episode |
| `seed`, `deck`, `stake` | the game played |
| `final_round`, `final_dollars` | state when the run ended |
| `final_jokers` | jokers held at the end, `;`-separated keys |
| `buy_card_0` … `skip` | how often each of the 17 shop actions was taken |

`eval/episodes.csv` adds `eval_round` (0 = before training) and `train_episodes` (training
episodes finished before that eval round).

## steps.jsonl

One object per shop decision:

| Key | Meaning |
| --- | --- |
| `episode`, `step` | episode and 0-based decision number within it |
| `ante`, `round`, `blind_on_deck` | where the run is |
| `jokers_before`, `consumables_before` | held keys before acting |
| `legal` | shop action ids that were legal |
| `action`, `action_name` | what was chosen |
| `item_key`, `item_cost` | the item it targets and the money it moves: a buy's price, a sell's sale value, a use's card price; `null` for reroll/skip |
| `dollars_before`, `dollars_after` | money around the action |
| `dollars_after_pack` | money once a pack opened by this action closes (picks can pay or cost), else `null` |
| `reward`, `done` | reward for this step and whether the run ended |

## rounds.jsonl

One object per blind played:

| Key | Meaning |
| --- | --- |
| `episode`, `ante`, `round` | the ante the blind was played in |
| `blind_key`, `target`, `chips` | the blind, chips needed and chips scored |
| `beaten` | whether the run went on |
| `dollars_at_end` | money when the blind ended |
| `interest_cap` | the interest cap at the time |
| `earn_blind`, `earn_hands`, `earn_discards`, `earn_interest`, `earn_jokers`, `earn_rent`, `earn_total` | the cash-out breakdown; `null` for a lost blind |
| `earn_tags` | cash-out money beyond the breakdown, from tags (e.g. Investment) |

## shops.jsonl

One object per shop offer:

| Key | Meaning |
| --- | --- |
| `episode`, `step` | `step` is the next shop decision this offer applies to |
| `ante`, `round`, `dollars` | state at the time |
| `reason` | `enter` or `reroll` |
| `reroll_cost` | current reroll price |
| `items` | `[{kind: card/voucher/pack, key, cost}, ...]` |

## Version history

| Version | Change |
| --- | --- |
| 1 | No `log_version` key. `rounds.jsonl` logged beaten bosses under the next ante |
| 2 | `rounds.jsonl` ante = ante the blind was played in; cash-out breakdown; `shops.jsonl` |
| 3 | `steps.jsonl` `legal` |
| 4 | Optional `eval/` with `eval_round` and `train_episodes` |
| 5 | `earn_tags`; `earn_rent` filled in; `dollars_after_pack`; returns from a pack aren't new shop visits |
| 6 | `episodes.csv` `max_ante`; ante reward only for a new highest ante; deck/stake from the env's own RNG; a sell's `item_cost` is its sale value. Since the repo split: `shop_rl_commit` and the pinned `jackdaw_commit` (additions only, so no version bump) |

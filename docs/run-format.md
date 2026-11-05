# Run format (log_version 15)

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
| `eval/hands.jsonl` | one per play or discard in an eval episode | with `--eval-every N`, unless `--no-step-log` |
| `eval/decks.jsonl` | one per blind of an eval episode: the deck and hand levels | with `--eval-every N`, unless `--no-step-log` |
| `test/` | the eval files for the held-out test set (see [Test set](#test-set)) | with `--test-episodes N` (log_version 14) |
| `q_table.json` | the learned table | `--shop qlearn` only |
| `events.out.tfevents.*` | tensorboard scalars | with the `train` extra installed |

The `.jsonl` files and `episodes.csv` are flushed row by row, so a run still training can be
read up to its last line. They are only ever appended to, which lets the dashboard read just
the lines added since it last loaded a run.

### Test set

With `--test-episodes N`, once training ends the final policy plays `N` greedy games on
held-out seeds `<test_seed_prefix>_0`, `_1`, ... (config `test_episodes`,
`test_seed_prefix`, default `TEST`), never used for training or eval. They are logged in
`test/` exactly like `eval/`, as a single eval round (`eval_round` 0, `train_episodes` the
training episodes played). `test/` has no `config.json` of its own: it is the run's. The
dashboard reads it as the run's test set (*Episodes* → *Test*), not as a run of its own.
These games are the ones to report final results from.

## config.json

Every command-line option of `scripts/train.py` (`episodes`, `play`, `shop`, `seed`,
`seed_prefix`, `decks`, `stakes`, `run_name`, `group`, `eval_every`, `eval_episodes`,
`eval_seed_prefix`, `test_episodes`, `test_seed_prefix`, the Q-learning and heuristic parameters, ...), plus:

| Key | Meaning |
| --- | --- |
| `play_agent`, `shop_agent` | class names, e.g. `StrongPlayAgent`, `QLearningShopAgent` |
| `play_params`, `agent_params` | the agents' parameters (`{}` if none) |
| `n_actions`, `action_names` | the shop action space: 17 names, index = action id |
| `log_version` | this spec's version: 15 |
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
| `joker_state` | one entry per `jokers_before` slot: a scaling joker's current values (see [Scaling jokers](#scaling-jokers)), `null` for a joker that doesn't scale (log_version 15) |
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

## eval/hands.jsonl

Eval episodes only: training episodes are many times more numerous, and this log is several
times the size of `steps.jsonl`. One object per play or discard, in order:

| Key | Meaning |
| --- | --- |
| `episode`, `ante`, `round`, `blind_key`, `target` | the blind (the ante it was played in) |
| `hand_no` | 0-based play/discard number within the blind |
| `kind` | `play` or `discard` |
| `hand` | the cards held before acting, in hand order (see [Cards](#cards)) |
| `picked` | indices into `hand` of the cards played or discarded (Cerulean Bell's forced card included) |
| `hand_level` | the played hand's level when played (before scoring; Space Joker levels up during it); `null` for a discard |
| `jokers` | jokers held when acting, in order: a key, or `{"k": key, "ed": edition, "s": values}` with only what applies (`s`: a scaling joker's current values, log_version 15; see [Scaling jokers](#scaling-jokers)) |
| `hand_type`, `chips`, `mult`, `score`, `debuffed` | the engine's scoring of a play (`score` = floor(chips × mult), inf past a double); `null` for a discard |
| `chips_total` | the blind's score after this action |
| `hands_left`, `discards_left`, `dollars` | after this action |

## eval/decks.jsonl

One object per blind of an eval episode, taken at its first hand:

| Key | Meaning |
| --- | --- |
| `episode`, `ante`, `round`, `blind_key` | the blind, matching `rounds.jsonl` |
| `deck` | every playing card the run owns (draw pile, hand, discards) |
| `hand_levels` | `{hand type: level}` for hands above level 1 |
| `hands_played` | `{hand type: times played}` this run, for hands played at least once |

## Cards

A plain card is its engine `card_key`: suit letter (`H C D S`) and rank (`2`-`9`, `T`, `J`, `Q`,
`K`, `A`), e.g. `"H_A"`, `"S_T"`. A modified card is an object with only what differs:
`{"c": "H_A", "en": "m_glass", "ed": "foil", "seal": "Red", "debuff": true}` (`en` an
enhancement key, `ed` one of `foil`/`holo`/`polychrome`/`negative`, `seal` one of
`Red`/`Blue`/`Gold`/`Purple`).

## Scaling jokers

From log_version 15, a joker that changes as the run goes on (Spare Trousers' +Mult, Ice
Cream's +Chips, Ramen's XMult) carries its current values as `{path: value}`, the path being
where the engine keeps the value on the joker (`ability`), e.g. `{"mult": 6}`,
`{"extra.chips": 12}`, `{"x_mult": 1.91}`. The paths are the targets of the joker's updates
in `balatro-shop-rl`'s joker algebra (`runlog.STATE_FIELDS`); today each scaling joker has
one. The dashboard labels them as the game does ("+6 Mult", "X1.91 Mult") and shows any other
path as logged.

## Version history

| Version | Change |
| --- | --- |
| 1 | No `log_version` key. `rounds.jsonl` logged beaten bosses under the next ante |
| 2 | `rounds.jsonl` ante = ante the blind was played in; cash-out breakdown; `shops.jsonl` |
| 3 | `steps.jsonl` `legal` |
| 4 | Optional `eval/` with `eval_round` and `train_episodes` |
| 5 | `earn_tags`; `earn_rent` filled in; `dollars_after_pack`; returns from a pack aren't new shop visits |
| 6 | `episodes.csv` `max_ante`; ante reward only for a new highest ante; deck/stake from the env's own RNG; a sell's `item_cost` is its sale value. Since the repo split: `shop_rl_commit` and the pinned `jackdaw_commit` (additions only, so no version bump) |
| 7 | `eval/hands.jsonl` and `eval/decks.jsonl` (additions only: older readers ignore them) |
| 8 | Upstream Jackdaw (Rust engine): `jackdaw_version` replaces `jackdaw_commit`; `rounds.jsonl` keeps `earn_total` but the other `earn_*` parts are `null`; `eval/hands.jsonl` `joker_effects` (not read here yet) |
| 9 | `episodes.csv` `heldout_offered` / `heldout_vetoed` / `heldout_held`; `shops.jsonl` items `"hidden": true`; `--shop dqn` runs add `dqn.jsonl` and `model.pt` (not read here yet) |
| 10 | `eval/offers.jsonl` and config `offer_jokers` (not read here yet) |
| 11 | `dqn.jsonl` `fn_table` and `redrawn`; `agent_params` `discount_per` |
| 12 | `--round-features` (`agent_params` `round_features`) |
| 13 | `--relative-suits` (config `relative_suits`) |
| 14 | Training seeds `TRAIN_s<seed>_<n>` (config `seed_prefix`); optional `test/` folder in the eval format (`--test-episodes`, see [Test set](#test-set)) |
| 15 | A scaling joker's current values: `s` on its entry in `hands.jsonl` `jokers`, and `steps.jsonl` `joker_state` (see [Scaling jokers](#scaling-jokers)) |

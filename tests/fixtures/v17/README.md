# v17 fixture

A real balatro-shop-rl run at log_version 17 (3 training episodes, stakes 4 and 7, endless,
2 test games), with its test games replayed as `retest`. To make it again, from
balatro-shop-rl:

    uv run python scripts/train.py --episodes 3 --shop dqn --play smart --stakes 4,7 \
        --endless --offer-jokers j_order --out-dir OUT --run-name run_v17 --group v17 \
        --seed 0 --test-episodes 2 --log-every 1
    uv run python scripts/replay_test.py OUT/run_v17 --tag retest

then copy `OUT/run_v17` here without `model.pt` and `events.out.tfevents.*`.

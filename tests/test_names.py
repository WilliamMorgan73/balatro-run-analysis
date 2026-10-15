import pandas as pd
import pytest

from balatro_analysis.names import Names

PRETTY, KEYS = Names(pretty=True), Names(pretty=False)


@pytest.mark.parametrize(
    ("method", "key", "pretty", "raw"),
    [
        ("item", "j_greedy_joker", "Greedy Joker", "j_greedy_joker"),
        ("item", "j_caino", "Canio", "j_caino"),
        ("item", "p_buffoon_normal_1", "Buffoon Pack", "p_buffoon_normal_1"),
        ("deck", "b_black", "Black Deck", "b_black"),
        ("blind", "bl_hook", "The Hook", "bl_hook"),
        ("stake", "2", "Red Stake", "2"),
        ("stakes", "1;2", "White Stake, Red Stake", "1, 2"),
        ("action", "buy_card_0", "Buy card 1", "buy_card_0"),
        ("action", "sell_joker_5", "Sell joker 6", "sell_joker_5"),
        ("action", "skip", "Leave shop", "skip"),
        ("column", "dollars_before", "$ before", "dollars_before"),
        ("column", "something_new", "Something new", "something_new"),
        ("items", "j_joker;j_banner", "Joker, Banner", "j_joker, j_banner"),
        ("items", ["j_joker"], "Joker", "j_joker"),
    ],
)
def test_modes(method, key, pretty, raw):
    assert getattr(PRETTY, method)(key) == pretty
    assert getattr(KEYS, method)(key) == raw


def test_missing_values():
    assert PRETTY.item(None) == ""
    assert PRETTY.item(float("nan")) == ""
    assert PRETTY.items(float("nan")) == ""


def test_frame():
    df = pd.DataFrame(
        {
            "deck": ["b_red"],
            "action_name": ["reroll"],
            "jokers_before": [["j_joker", "j_banner"]],
            "sold": ["j_joker"],
            "dollars_before": [5],
        }
    )
    pretty = PRETTY.frame(df, {"sold": "item"})
    assert list(pretty.columns) == ["Deck", "Action", "Jokers held", "Sold", "$ before"]
    assert pretty.iloc[0].tolist() == ["Red Deck", "Reroll", "Joker, Banner", "Joker", 5]
    raw = KEYS.frame(df, {"sold": "item"})
    assert list(raw.columns) == list(df.columns)
    assert raw.iloc[0].tolist() == ["b_red", "reroll", "j_joker, j_banner", "j_joker", 5]
    assert df.loc[0, "deck"] == "b_red"  # original untouched

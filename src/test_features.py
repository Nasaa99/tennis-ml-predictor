"""
test_features.py

Checks that the features don't leak information from the future.

Test 1: change the results of every match after a cutoff date and mix up
        the row order. The features of matches before the cutoff should
        stay exactly the same.
Test 2: change the results of every match on one day. The features of the
        matches on that same day should stay the same, because matches on
        the same day are not allowed to see each other.

How to run:  python src/test_features.py
"""

import pandas as pd

from features import FEATURE_COLUMNS, build_features, load_matches


def flip_winners(matches, rows_to_flip):
    """Return a copy of the matches where the chosen rows have the other player as winner."""
    changed = matches.copy()
    p1_won = changed["winner"] == changed["player_1"]
    changed.loc[rows_to_flip & p1_won, "winner"] = changed["player_2"]
    changed.loc[rows_to_flip & ~p1_won, "winner"] = changed["player_1"]
    return changed


def features_for(features, rows):
    """Pick out some rows of the feature table, with just the columns we compare."""
    return features.loc[rows, ["date", "player_a", "player_b"] + FEATURE_COLUMNS].reset_index(drop=True)


def test_future_does_not_change_past():
    matches = load_matches()
    cutoff = pd.Timestamp("2023-01-01")
    original, _ = build_features(matches)

    # Flip every result from the cutoff onwards, then shuffle all the rows
    changed = flip_winners(matches, matches["date"] >= cutoff)
    changed = changed.sample(frac=1, random_state=0)
    # Load it the same way features.py does (this re-sorts it by date)
    changed = changed.sort_values(["date", "player_1", "player_2"]).reset_index(drop=True)
    new, _ = build_features(changed)

    past = original["date"] < cutoff
    pd.testing.assert_frame_equal(features_for(original, past), features_for(new, past))
    print(f"Test 1 passed: {past.sum()} matches before {cutoff.date()} are unchanged")


def days_with_double_matches(matches):
    """Find the days where some player played two matches (e.g. after rain delays).

    These are the days where leakage could happen: if the code was wrong, the
    player's first match result would leak into their second match's features.
    """
    p1 = matches[["date", "player_1"]].rename(columns={"player_1": "player"})
    p2 = matches[["date", "player_2"]].rename(columns={"player_2": "player"})
    appearances = pd.concat([p1, p2])
    counts = appearances.value_counts()
    return counts[counts > 1].index.get_level_values("date").unique()


def test_same_day_matches_dont_see_each_other():
    matches = load_matches()
    original, _ = build_features(matches)
    risky_days = days_with_double_matches(matches)

    # Check one day at a time. If we flipped several days at once, the results
    # from an earlier day would (correctly) change a later day's features.
    for day in risky_days:
        on_this_day = matches["date"] == day
        changed = flip_winners(matches, on_this_day)
        new, _ = build_features(changed)

        # Flipping today's results must not change today's features
        pd.testing.assert_frame_equal(features_for(original, on_this_day),
                                      features_for(new, on_this_day))

    print(f"Test 2 passed: checked {len(risky_days)} days where someone played twice")


if __name__ == "__main__":
    test_future_does_not_change_past()
    test_same_day_matches_dont_see_each_other()
    print("All leakage tests passed")

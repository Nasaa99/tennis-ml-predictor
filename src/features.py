"""
features.py

Builds the features for every match and saves them to data/features.csv.

The main rule: the features for a match can only use matches played BEFORE
that day. Otherwise the model would be cheating by looking at the future.

It also saves models/player_state.json, which has every player's latest Elo,
form and head-to-head record. The Flask app can load this later to predict
new matches.

How to run:  python src/features.py
"""

import json
import os

import numpy as np
import pandas as pd

# File paths, built from the project folder so they work on any computer
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(PROJECT_DIR, "Dataset", "atp_matches_filtered.csv")
FEATURES_FILE = os.path.join(PROJECT_DIR, "data", "features.csv")
STATE_FILE = os.path.join(PROJECT_DIR, "models", "player_state.json")

START_ELO = 1500
RANDOM_SEED = 42

# Turn round names into numbers so the model knows the final comes after round 1
ROUND_NUMBER = {
    "1st Round": 1,
    "2nd Round": 2,
    "3rd Round": 3,
    "4th Round": 4,
    "Quarterfinals": 5,
    "Semifinals": 6,
    "The Final": 7,
}

# The only columns the model is allowed to use.
# Score, winner and betting odds are NOT in this list on purpose.
FEATURE_COLUMNS = [
    "elo_diff",
    "surface_elo_diff",
    "rank_diff",
    "log_rank_ratio",
    "pts_diff",
    "form_5_diff",
    "form_10_diff",
    "h2h_diff",
    "matches_played_diff",
    "surface_Hard",
    "surface_Clay",
    "surface_Grass",
    "round_num",
    "best_of_5",
]


def load_matches(path=DATA_FILE):
    """Read the CSV and sort the matches by date."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")

    # Sort by date, then by player names. The names part just makes sure the
    # order is always the same, even if the rows in the CSV get mixed up.
    df = df.sort_values(["date", "player_1", "player_2"]).reset_index(drop=True)
    return df


def k_factor(matches_played):
    """How much one match changes a player's Elo.

    New players get a big K so their rating moves quickly to where it should be.
    Players with lots of matches get a smaller K because we already trust
    their rating. (This formula is the one FiveThirtyEight used for tennis.)
    """
    return 250 / (matches_played + 5) ** 0.4


def expected_score(elo_a, elo_b):
    """Chance that A beats B, using the normal Elo formula.

    Example: if A is 400 points higher than B, A wins about 91% of the time.
    """
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))


def new_player():
    """Starting stats for a player we haven't seen yet."""
    return {
        "elo": START_ELO,
        "surface_elo": {"Hard": START_ELO, "Clay": START_ELO, "Grass": START_ELO},
        "matches": 0,
        "surface_matches": {"Hard": 0, "Clay": 0, "Grass": 0},
        "results": [],  # last 10 results, 1 = win and 0 = loss, oldest first
        "rank": None,   # latest known ATP rank
        "pts": None,    # latest known ATP points
    }


def win_rate(results, n):
    """Win rate over the player's last n matches.

    If the player has no matches yet we use 0.5, because we know nothing
    about them. If they have fewer than n matches we use what they have.
    """
    last_n = results[-n:]
    if len(last_n) == 0:
        return 0.5
    return sum(last_n) / len(last_n)


def h2h_wins(state, player, opponent):
    """How many times `player` has beaten `opponent` so far."""
    return state["h2h"].get(player, {}).get(opponent, 0)


def make_features(state, player_a, player_b, surface, round_name, best_of,
                  rank_a=None, rank_b=None, pts_a=None, pts_b=None):
    """Build the features for player A vs player B.

    Every feature is "A minus B", so a positive number means A is better
    at that thing. This function only reads `state` and doesn't change it.
    The Flask app can call it too, to predict a new match.

    Args:
        state: dictionary with every player's stats so far (see build_features).
        player_a, player_b: names as they appear in the data, e.g. "Sinner J.".
        surface: "Hard", "Clay" or "Grass".
        round_name: e.g. "1st Round" or "The Final".
        best_of: 3 or 5.
        rank_a, rank_b, pts_a, pts_b: ATP rank and points. If you leave them
            out, the player's latest known rank and points are used.

    Returns:
        A dictionary with one value for each name in FEATURE_COLUMNS.
    """
    # If we've never seen a player, treat them as a brand new player
    a = state["players"].get(player_a, new_player())
    b = state["players"].get(player_b, new_player())

    if rank_a is None:
        rank_a = a["rank"]
    if rank_b is None:
        rank_b = b["rank"]
    if pts_a is None:
        pts_a = a["pts"]
    if pts_b is None:
        pts_b = b["pts"]
    if rank_a is None or rank_b is None or pts_a is None or pts_b is None:
        raise ValueError("Need the rank and points of both players")

    return {
        "elo_diff": a["elo"] - b["elo"],
        "surface_elo_diff": a["surface_elo"][surface] - b["surface_elo"][surface],
        # A lower rank number is better, so a negative rank_diff means A is ranked higher
        "rank_diff": rank_a - rank_b,
        # The log makes #1 vs #10 a much bigger gap than #101 vs #110
        "log_rank_ratio": np.log(rank_a / rank_b),
        "pts_diff": pts_a - pts_b,
        "form_5_diff": win_rate(a["results"], 5) - win_rate(b["results"], 5),
        "form_10_diff": win_rate(a["results"], 10) - win_rate(b["results"], 10),
        "h2h_diff": h2h_wins(state, player_a, player_b) - h2h_wins(state, player_b, player_a),
        "matches_played_diff": a["matches"] - b["matches"],
        # One-hot encoding: one column per surface, 1 for this match's surface
        "surface_Hard": int(surface == "Hard"),
        "surface_Clay": int(surface == "Clay"),
        "surface_Grass": int(surface == "Grass"),
        "round_num": ROUND_NUMBER[round_name],
        "best_of_5": int(best_of == 5),
    }


def update_state(state, winner, loser, surface, winner_rank, loser_rank,
                 winner_pts, loser_pts):
    """Add one finished match to the state (Elo, form, head-to-head, rank)."""
    # Add the players if this is their first match
    if winner not in state["players"]:
        state["players"][winner] = new_player()
    if loser not in state["players"]:
        state["players"][loser] = new_player()
    w = state["players"][winner]
    l = state["players"][loser]

    # Overall Elo. The less the winner was expected to win, the more points
    # move. Each player uses their own K, based on how many matches they've played.
    expected = expected_score(w["elo"], l["elo"])
    w["elo"] += k_factor(w["matches"]) * (1 - expected)
    l["elo"] -= k_factor(l["matches"]) * (1 - expected)

    # Surface Elo works the same way but only counts matches on this surface
    expected = expected_score(w["surface_elo"][surface], l["surface_elo"][surface])
    w["surface_elo"][surface] += k_factor(w["surface_matches"][surface]) * (1 - expected)
    l["surface_elo"][surface] -= k_factor(l["surface_matches"][surface]) * (1 - expected)

    # Match counts
    w["matches"] += 1
    l["matches"] += 1
    w["surface_matches"][surface] += 1
    l["surface_matches"][surface] += 1

    # Recent results (we only need the last 10 for form)
    w["results"] = (w["results"] + [1])[-10:]
    l["results"] = (l["results"] + [0])[-10:]

    # Latest rank and points (int() turns numpy numbers into normal ones for JSON)
    w["rank"] = int(winner_rank)
    l["rank"] = int(loser_rank)
    w["pts"] = int(winner_pts)
    l["pts"] = int(loser_pts)

    # Head-to-head: one more win for winner against loser
    if winner not in state["h2h"]:
        state["h2h"][winner] = {}
    state["h2h"][winner][loser] = state["h2h"][winner].get(loser, 0) + 1


def build_features(matches, seed=RANDOM_SEED):
    """Go through the matches in date order and build one feature row per match.

    For each date:
      1. make the features for every match that day, using the state
         (which only has matches from earlier days)
      2. then update the state with that day's results

    Because step 1 is done before step 2, matches on the same day can't see
    each other's results.

    Returns the feature table (a DataFrame) and the final state.
    """
    state = {"players": {}, "h2h": {}}

    # Flip a coin for each match to decide who is "player A".
    # Without this, player A would always be player_1 and the label would
    # depend on the column order. The fixed seed gives the same coin flips every run.
    rng = np.random.default_rng(seed)
    swap = rng.random(len(matches)) < 0.5

    rows = []
    for date, day in matches.groupby("date", sort=True):

        # Step 1: features for today's matches
        for i, m in day.iterrows():
            if swap[i]:
                player_a, player_b = m["player_2"], m["player_1"]
                rank_a, rank_b = m["rank_2"], m["rank_1"]
                pts_a, pts_b = m["pts_2"], m["pts_1"]
                odd_a, odd_b = m["odd_2"], m["odd_1"]
            else:
                player_a, player_b = m["player_1"], m["player_2"]
                rank_a, rank_b = m["rank_1"], m["rank_2"]
                pts_a, pts_b = m["pts_1"], m["pts_2"]
                odd_a, odd_b = m["odd_1"], m["odd_2"]

            # Info columns, kept for splitting and checking but not used as features.
            # The odds are only saved so we can compare against the bookmakers later.
            row = {
                "date": date,
                "year": date.year,
                "tournament": m["tournament"],
                "player_a": player_a,
                "player_b": player_b,
                "odd_a": odd_a,
                "odd_b": odd_b,
            }
            row.update(make_features(state, player_a, player_b, m["surface"],
                                     m["round"], m["best of"],
                                     rank_a, rank_b, pts_a, pts_b))
            row["target"] = int(m["winner"] == player_a)  # 1 if A won
            rows.append(row)

        # Step 2: now that today's features are saved, add today's results
        for i, m in day.iterrows():
            if m["winner"] == m["player_1"]:
                loser = m["player_2"]
                winner_rank, loser_rank = m["rank_1"], m["rank_2"]
                winner_pts, loser_pts = m["pts_1"], m["pts_2"]
            else:
                loser = m["player_1"]
                winner_rank, loser_rank = m["rank_2"], m["rank_1"]
                winner_pts, loser_pts = m["pts_2"], m["pts_1"]
            update_state(state, m["winner"], loser, m["surface"],
                         winner_rank, loser_rank, winner_pts, loser_pts)

    return pd.DataFrame(rows), state


def save_state(state, path=STATE_FILE):
    """Save the players' latest stats to a JSON file for the Flask app."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=1)


def load_state(path=STATE_FILE):
    """Load the players' stats saved by save_state."""
    with open(path) as f:
        return json.load(f)


def main():
    matches = load_matches()
    features, state = build_features(matches)

    os.makedirs(os.path.dirname(FEATURES_FILE), exist_ok=True)
    features.to_csv(FEATURES_FILE, index=False)
    save_state(state)

    print(f"Built features for {len(features)} matches")
    print(f"Player A won {features['target'].mean():.1%} of matches (should be close to 50%)")
    print(f"Players in the state file: {len(state['players'])}")
    print(f"Saved {FEATURES_FILE}")
    print(f"Saved {STATE_FILE}")


if __name__ == "__main__":
    main()

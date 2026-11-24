"""Event-level cleaner for the raw Big Data Cup files (not the main pipeline).

The main pipeline is build_entries.py -> analyze.py. This module is kept for
two jobs:
  1. A clean one-row-per-event table with coordinates rotated to the acting
     team's attacking direction, for later work that needs event context
     (tracking covariates, xG, zone-exit outcomes, shots against).
  2. An independent second builder of the entry table. tests.py checks that it
     agrees with build_entries.py on every entry.

All functions are pure (no file writing).

Conventions
-----------
- Event order: the row order of the raw Events file. The clock only has
  whole seconds and many events share a second, so the row order is the only
  reliable sequence. It is stored as `event_idx` (0, 1, 2, ... within a game).
- Clock: `clock_sec` is seconds REMAINING in the period (the raw clock counts
  down). `t_game` is seconds ELAPSED since the start of the game.
- Coordinates: raw x, y are absolute rink coordinates and the attacking end
  switches every period. `x_att`, `y_att` rotate them so the team doing the
  event always attacks towards +x. From the attacker's point of view, +y is
  the left side of the ice.
- Teams: the Big Data Cup anonymises teams per competition, so "Team D" in
  2025 is not known to be "Team D" in 2026. `team_id` is season + letter.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

REG_PERIOD_SEC = 20 * 60  # regulation period length
OT_PERIOD_SEC = 5 * 60  # overtime length in these games (OT clock starts at 05:00)

SHOT_EVENTS = ("Shot", "Goal")  # a shot attempt = any Shot row (on net, missed, blocked) or Goal row
ENTRY_TYPES = ("Carried", "Dumped", "Played")


# ---------------------------------------------------------------------------
# Small parsers
# ---------------------------------------------------------------------------

def clock_to_sec(clock):
    """'19:52' -> 1192. Works on a string or a pandas Series of strings."""
    if isinstance(clock, pd.Series):
        parts = clock.astype(str).str.split(":", expand=True).astype(int)
        return parts[0] * 60 + parts[1]
    m, s = str(clock).split(":")
    return int(m) * 60 + int(s)


def parse_date(date):
    """Both '10/25/2024' (2025 files) and '2025-10-11' (2026 files) -> Timestamp."""
    date = str(date)
    if "/" in date:
        return pd.to_datetime(date, format="%m/%d/%Y")
    return pd.to_datetime(date, format="%Y-%m-%d")


def parse_game_file(path):
    """File name -> dict with season, date, away, home, game_id.

    Handles both naming styles:
      2025-2024-10-25.Team.H.@.Team.G.-.Events.csv
      2026-2025-10-11.Team.A.@.Team.D.Events.csv
    """
    name = Path(path).name
    m = re.match(
        r"^(\d{4})-(\d{4}-\d{2}-\d{2})\.Team\.([A-Z])\.@\.Team\.([A-Z])\.(?:-\.)?(Events|Shifts)\.csv$",
        name,
    )
    if m is None:
        raise ValueError(f"Unrecognised file name: {name}")
    season, date, away, home, kind = m.groups()
    return {
        "season": int(season),
        "date": pd.Timestamp(date),
        "away": away,
        "home": home,
        "kind": kind,
        "game_id": f"{season}_{date}_{away}@{home}",
    }


def period_length(period):
    """Length in seconds of a period (1-3 regulation, 4 = overtime)."""
    return np.where(np.asarray(period) <= 3, REG_PERIOD_SEC, OT_PERIOD_SEC)


def clean_player_id(values):
    """Jersey numbers as clean strings: 21 / 21.0 / '21' -> '21', 'Go' stays 'Go'.

    Blank ids (NaN, 'nan', '') become '' so they never match anything. This is
    done explicitly because pandas 2 and pandas 3 turn NaN into different
    things with astype(str).
    """
    def one(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        s = str(v).strip()
        if s.lower() in ("", "nan", "none", "<na>"):
            return ""
        if s.endswith(".0") and s[:-2].isdigit():
            s = s[:-2]
        return s

    return pd.Series(values).map(one).to_numpy(dtype=object)


def team_letter(team_name):
    """'Team A' -> 'A'."""
    return pd.Series(team_name).astype(str).str.replace("Team ", "", regex=False).values


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_events_file(path):
    """Read one Events CSV into a tidy frame (one row per event, raw order kept)."""
    info = parse_game_file(path)
    raw = pd.read_csv(path)
    df = pd.DataFrame({
        "game_id": info["game_id"],
        "season": info["season"],
        "date": parse_date(raw["Date"].iloc[0]),
        "event_idx": np.arange(len(raw)),
        "period": raw["Period"].astype(int),
        "clock_sec": clock_to_sec(raw["Clock"]),
        "home_team": team_letter(raw["Home_Team"]),
        "away_team": team_letter(raw["Away_Team"]),
        "team": team_letter(raw["Team"]),
        "player": clean_player_id(raw["Player_Id"]),
        "event": raw["Event"],
        "x": raw["X_Coordinate"].astype(float),
        "y": raw["Y_Coordinate"].astype(float),
        "detail_1": raw["Detail_1"],
        "detail_2": raw["Detail_2"],
        "detail_3": raw["Detail_3"],
        "detail_4": raw["Detail_4"],
        "player_2": clean_player_id(raw["Player_Id_2"]),
        "x_2": raw["X_Coordinate_2"].astype(float),
        "y_2": raw["Y_Coordinate_2"].astype(float),
        "home_skaters": raw["Home_Team_Skaters"].astype(int),
        "away_skaters": raw["Away_Team_Skaters"].astype(int),
        "home_goals": raw["Home_Team_Goals"].astype(int),
        "away_goals": raw["Away_Team_Goals"].astype(int),
    })
    # Sanity: file name and file contents must agree on who is home and away.
    assert (df["home_team"] == info["home"]).all() and (df["away_team"] == info["away"]).all(), path
    return df


def load_events(raw_dir):
    """Stack every Events CSV in raw_dir."""
    files = sorted(Path(raw_dir).glob("*Events.csv"))
    if not files:
        raise FileNotFoundError(f"No Events files in {raw_dir}")
    return pd.concat([load_events_file(f) for f in files], ignore_index=True)


def load_shifts_file(path):
    """Read one Shifts CSV. The 2025 and 2026 files use different column names."""
    info = parse_game_file(path)
    raw = pd.read_csv(path, dtype={"Player_Id": str})
    team_col = "team_name" if "team_name" in raw.columns else "Team"
    return pd.DataFrame({
        "game_id": info["game_id"],
        "team": team_letter(raw[team_col]),
        "player": clean_player_id(raw["Player_Id"]),
        "period": raw["period"].astype(int),
        "start_sec": clock_to_sec(raw["start_clock"]),  # clock remaining at shift start
        "end_sec": clock_to_sec(raw["end_clock"]),  # clock remaining at shift end
    })


def load_shifts(raw_dir):
    files = sorted(Path(raw_dir).glob("*Shifts.csv"))
    if not files:
        raise FileNotFoundError(f"No Shifts files in {raw_dir}")
    return pd.concat([load_shifts_file(f) for f in files], ignore_index=True)


# ---------------------------------------------------------------------------
# Event-level cleaning
# ---------------------------------------------------------------------------

def add_game_time(ev):
    """Add t_game = seconds elapsed since the start of the game."""
    ev = ev.copy()
    elapsed_in_period = period_length(ev["period"]) - ev["clock_sec"]
    before = np.where(ev["period"] <= 3, (ev["period"] - 1) * REG_PERIOD_SEC, 3 * REG_PERIOD_SEC)
    ev["t_game"] = before + elapsed_in_period
    return ev


def add_team_perspective(ev):
    """Skaters, goals and home flag from the point of view of the team doing the event."""
    ev = ev.copy()
    is_home = ev["team"] == ev["home_team"]
    ev["is_home"] = is_home.astype(int)
    ev["opponent"] = np.where(is_home, ev["away_team"], ev["home_team"])
    ev["skaters_for"] = np.where(is_home, ev["home_skaters"], ev["away_skaters"])
    ev["skaters_against"] = np.where(is_home, ev["away_skaters"], ev["home_skaters"])
    ev["goals_for"] = np.where(is_home, ev["home_goals"], ev["away_goals"])
    ev["goals_against"] = np.where(is_home, ev["away_goals"], ev["home_goals"])
    ev["score_diff"] = ev["goals_for"] - ev["goals_against"]
    ev["team_id"] = ev["season"].astype(str) + "_" + ev["team"]
    return ev


def attack_direction(ev):
    """Sign (+1 / -1) of the end each team attacks, per game, team and period.

    Rule: the sign of the median x of the team's zone entries in that period
    (entries sit on the attacking blue line, x = +-25). If a team has no
    entries in a period, use the opposite of its opponent's sign, then the
    sign of its shots. Returns a frame keyed by game_id, period, team.
    """
    keys = ["game_id", "period", "team"]
    entries = ev[ev["event"] == "Zone Entry"]
    sign = entries.groupby(keys)["x"].median().pipe(np.sign).rename("attack_sign")

    all_keys = ev[keys + ["opponent"]].drop_duplicates(keys)
    out = all_keys.merge(sign.reset_index(), on=keys, how="left")

    # Fallback 1: opposite of the opponent's sign.
    opp = sign.reset_index().rename(columns={"team": "opponent", "attack_sign": "opp_sign"})
    out = out.merge(opp, on=["game_id", "period", "opponent"], how="left")
    out["attack_sign"] = out["attack_sign"].fillna(-out["opp_sign"])

    # Fallback 2: sign of the median shot x.
    shots = ev[ev["event"].isin(SHOT_EVENTS)].groupby(keys)["x"].median().pipe(np.sign).rename("shot_sign")
    out = out.merge(shots.reset_index(), on=keys, how="left")
    out["attack_sign"] = out["attack_sign"].fillna(out["shot_sign"])

    if out["attack_sign"].isna().any() or (out["attack_sign"] == 0).any():
        bad = out[out["attack_sign"].isna() | (out["attack_sign"] == 0)]
        raise ValueError(f"Could not determine attacking direction for:\n{bad[keys]}")
    return out[keys + ["attack_sign"]]


def check_direction_consistency(ev, directions):
    """Each team's entries in a period must all be on one side, and the two
    teams must attack opposite ends. Raises if not."""
    keys = ["game_id", "period", "team"]
    entries = ev[ev["event"] == "Zone Entry"].merge(directions, on=keys)
    share = (np.sign(entries["x"]) == entries["attack_sign"]).groupby([entries[k] for k in keys]).mean()
    if (share < 1).any():
        raise ValueError(f"Entries on the wrong side of the rink:\n{share[share < 1]}")
    # In regulation the raw coordinates are absolute, so the two teams attack
    # opposite ends. In overtime (period 4) the raw file already records every
    # event in the acting team's own attacking frame (both teams' entries and
    # shots sit at +x), so both signs are +1 there and this check is skipped.
    reg = directions[directions["period"] <= 3]
    per_period = reg.groupby(["game_id", "period"])["attack_sign"].sum()
    if (per_period != 0).any():
        raise ValueError("Both teams appear to attack the same end in a regulation period.")


def normalise_coordinates(ev):
    """Rotate coordinates so the acting team always attacks towards +x."""
    directions = attack_direction(ev)
    check_direction_consistency(ev, directions)
    ev = ev.merge(directions, on=["game_id", "period", "team"], how="left", validate="many_to_one")
    new_names = {"x": "x_att", "y": "y_att", "x_2": "x_att_2", "y_2": "y_att_2"}
    for raw_col, new_col in new_names.items():
        ev[new_col] = ev[raw_col] * ev["attack_sign"]
    return ev.sort_values(["game_id", "event_idx"]).reset_index(drop=True)


def clean_events(raw_dir):
    """Load and clean every event. One row per event."""
    ev = load_events(raw_dir)
    ev = add_game_time(ev)
    ev = add_team_perspective(ev)
    ev = normalise_coordinates(ev)
    return ev


# ---------------------------------------------------------------------------
# Entry-level table
# ---------------------------------------------------------------------------

def strength_state(skaters_for, skaters_against):
    """Manpower from the entering team's point of view.

    '5v5'  : five skaters each
    'EV_other': other equal strength (4v4, 3v3; mostly 3v3 overtime)
    'PP'   : entering team has more skaters (includes 6v5 with the goalie pulled)
    'SH'   : entering team has fewer skaters
    """
    f = np.asarray(skaters_for)
    a = np.asarray(skaters_against)
    return np.select(
        [(f == 5) & (a == 5), f == a, f > a, f < a],
        ["5v5", "EV_other", "PP", "SH"],
        default="unknown",
    )


def lane(y_att, centre_half_width=85 / 6):
    """Lane of the entry from the attacker's point of view (rink split in thirds)."""
    y = np.asarray(y_att, dtype=float)
    return np.select([y > centre_half_width, y < -centre_half_width], ["left", "right"], default="centre")


STOP_EVENTS = ("Zone Entry", "Faceoff Win")  # a new entry (either team) or a stoppage ends the window


def window_counts(game_ev, entry, window_sec):
    """Shot attempts around one entry, within the same period.

    Main outcome (`shots_for`): the entering team's shot attempts in the rows
    AFTER the entry (file order) whose clock is within `window_sec` seconds,
    stopping at the next zone entry by either team or the next faceoff. The
    stop rule means a shot is never credited to two entries. This is the
    definition that reproduces the midway numbers (0.61 vs 0.25).

    `shots_for_nostop` drops the stop rule (plain 20 s count), for sensitivity.

    Placebo (`shots_for_before`): the entering team's shot attempts in the rows
    BEFORE the entry within `window_sec` seconds, with the mirror-image stop
    rule (counting back only to the previous entry or faceoff).
    `shots_for_before_nostop` drops it.
    """
    same_period = game_ev["period"].values == entry.period
    shot = np.isin(game_ev["event"].values, SHOT_EVENTS)
    goal = game_ev["event"].values == "Goal"
    stop = np.isin(game_ev["event"].values, STOP_EVENTS)
    idx = game_ev["event_idx"].values
    clock = game_ev["clock_sec"].values
    by_team = game_ev["team"].values == entry.team
    by_opp = ~by_team

    after = same_period & (idx > entry.event_idx) & (clock >= entry.clock_sec - window_sec)
    before = same_period & (idx < entry.event_idx) & (clock <= entry.clock_sec + window_sec)

    # Apply the stop rule: after -> keep rows before the first stop event;
    # before -> keep rows after the last stop event.
    after_stop = after.copy()
    stops_after = idx[after & stop]
    if stops_after.size:
        after_stop &= idx < stops_after.min()
    before_stop = before.copy()
    stops_before = idx[before & stop]
    if stops_before.size:
        before_stop &= idx > stops_before.max()

    return {
        "shots_for": int((after_stop & shot & by_team).sum()),
        "goals_for": int((after_stop & goal & by_team).sum()),
        "shots_against": int((after_stop & shot & by_opp).sum()),
        "shots_for_nostop": int((after & shot & by_team).sum()),
        "shots_for_before": int((before_stop & shot & by_team).sum()),
        "shots_for_before_nostop": int((before & shot & by_team).sum()),
        "window_after_sec": int(min(window_sec, entry.clock_sec)),  # < window when the period ends first
    }


def time_since_faceoff(game_ev, entry):
    """Seconds since the most recent faceoff in the same period (NaN if none)."""
    fo = game_ev[
        (game_ev["event"] == "Faceoff Win")
        & (game_ev["period"] == entry.period)
        & (game_ev["event_idx"] < entry.event_idx)
    ]
    if fo.empty:
        return np.nan
    return float(fo["clock_sec"].iloc[-1] - entry.clock_sec)


def shift_elapsed(game_shifts, team, player, period, clock_sec):
    """Seconds the player has been on the ice in the current shift (NaN if no shift matches).

    A shift covers start_sec >= clock >= end_sec (the clock counts down). If two
    shifts touch at a change, the one that started most recently is used.
    """
    if player == "":  # entry logged without a player
        return np.nan
    s = game_shifts[
        (game_shifts["team"] == team)
        & (game_shifts["player"] == player)
        & (game_shifts["period"] == period)
        & (game_shifts["start_sec"] >= clock_sec)
        & (game_shifts["end_sec"] <= clock_sec)
    ]
    if s.empty:
        return np.nan
    return float(s["start_sec"].min() - clock_sec)


def on_ice_shift_mean(game_shifts, team, period, clock_sec):
    """Mean seconds into the current shift over the team's skaters on the ice (goalies excluded)."""
    s = game_shifts[
        (game_shifts["team"] == team)
        & (game_shifts["period"] == period)
        & (game_shifts["start_sec"] >= clock_sec)
        & (game_shifts["end_sec"] <= clock_sec)
        & (game_shifts["player"] != "Go")
        & (game_shifts["player"] != "")
    ]
    if s.empty:
        return np.nan, 0
    per_player = s.groupby("player")["start_sec"].min() - clock_sec
    return float(per_player.mean()), int(len(per_player))


def build_entries(ev, shifts, window_sec=20):
    """One row per zone entry, with situation at the moment of entry and outcomes.

    Every covariate uses only information available at or before the entry.
    """
    rows = []
    for game_id, game_ev in ev.groupby("game_id", sort=True):
        game_ev = game_ev.sort_values("event_idx")
        game_shifts = shifts[shifts["game_id"] == game_id]
        for e in game_ev[game_ev["event"] == "Zone Entry"].itertuples(index=False):
            shift_mean, n_on_ice = on_ice_shift_mean(game_shifts, e.team, e.period, e.clock_sec)
            row = {
                "game_id": game_id,
                "season": e.season,
                "event_idx": e.event_idx,
                "entry_type": e.detail_1,
                "team_id": e.team_id,
                "opponent_id": f"{e.season}_{e.opponent}",
                "player": e.player,
                "period": e.period,
                "overtime": int(e.period >= 4),
                "clock_sec": e.clock_sec,
                "t_game": e.t_game,
                "is_home": e.is_home,
                "skaters_for": e.skaters_for,
                "skaters_against": e.skaters_against,
                "strength": strength_state([e.skaters_for], [e.skaters_against])[0],
                "score_diff": e.score_diff,
                "x_att": e.x_att,
                "y_att": e.y_att,
                "lane": lane([e.y_att])[0],
                "time_since_faceoff": time_since_faceoff(game_ev, e),
                "shift_elapsed": shift_elapsed(game_shifts, e.team, e.player, e.period, e.clock_sec),
                "on_ice_shift_mean": shift_mean,
                "n_skaters_matched": n_on_ice,
            }
            row.update(window_counts(game_ev, e, window_sec))
            rows.append(row)

    entries = pd.DataFrame(rows)
    entries["carry"] = entries["entry_type"].map({"Carried": 1, "Dumped": 0})  # NaN for Played
    entries["controlled"] = entries["entry_type"].map({"Carried": 1, "Played": 1, "Dumped": 0})
    return entries


def check_entries(entries, ev):
    """Loud checks on the entry table. Returns a short dict of counts."""
    n_raw = int((ev["event"] == "Zone Entry").sum())
    assert len(entries) == n_raw, "lost or duplicated entries"
    assert entries["entry_type"].isin(ENTRY_TYPES).all(), "unknown entry type"
    assert entries["strength"].ne("unknown").all(), "unknown manpower state"
    assert (entries["shots_for"] >= 0).all()
    # Entries sit on the attacking blue line after rotation (x = +25 give or take).
    return {
        "games": entries["game_id"].nunique(),
        "entries": len(entries),
        "by_type": entries["entry_type"].value_counts().to_dict(),
        "x_att_outside_20_30": int(((entries["x_att"] < 20) | (entries["x_att"] > 30)).sum()),
        "shift_elapsed_missing": int(entries["shift_elapsed"].isna().sum()),
        "entries_without_player": int((entries["player"] == "").sum()),
    }

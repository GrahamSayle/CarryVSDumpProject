"""Shared expected-goals (xG) features for MoneyPuck NHL shots and Big Data Cup shots.

Both sources are turned into the SAME columns, in the SAME coordinate frame, so a
model trained on MoneyPuck can be applied to Big Data Cup shots.

Coordinate frame (both sources)
-------------------------------
- Feet, centre ice at (0, 0), NHL rink: goal lines at x = +-89, blue lines at
  x = +-25, end-zone faceoff dots at (+-69, +-22).
- Rotated so the SHOOTING team attacks towards +x: the target net is at (89, 0).
  The rotation is 180 degrees (x and y flip together), so +y is the shooter's
  left, as in MoneyPuck's xCordAdjusted / yCordAdjusted.
- Big Data Cup: wrangle.py already gives x_att, y_att in the acting team's
  attacking frame (it also handles overtime, where the raw file is already in
  each team's own frame). For a shot, x_att, y_att ARE x_adj, y_adj.
- MoneyPuck: x_adj, y_adj = xCordAdjusted, yCordAdjusted (checked on 2,999
  2024-25 shots: the flip is 180 degrees and shotDistance = dist to (89, 0)).

Distance and angle follow MoneyPuck exactly (checked to machine precision):
  distance = sqrt((89 - x)^2 + y^2)
  angle    = sign(y) * atan(|y| / |89 - x|) in degrees (90 when x = 89),
             positive on the shooter's left.

Usage
-----
    python code/xg_features.py bdc data/raw out/xg_bdc_shots.csv
    python code/xg_features.py moneypuck data/raw/moneypuck/shots_2024.zip out/xg_mp_2024.csv

Please credit MoneyPuck.com wherever MoneyPuck data is used.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wrangle  # noqa: E402

NET_X = 89.0
REBOUND_SEC = 3  # MoneyPuck: previous event a shot (SHOT or MISS) by the same team within 3 s

# Harmonised shot type groups. Stathletes has no backhand and logs about as many
# snapshots as wristshots (NHL scorers: 2.4 wrist per snap), so wrist, snap and
# backhand are pooled. Fan and poke (36 BDC shots) have no NHL label.
SHOT_GROUP_MP = {
    "WRIST": "wrist_snap", "SNAP": "wrist_snap", "BACK": "wrist_snap", "": "wrist_snap",
    "SLAP": "slap", "TIP": "tip", "DEFL": "tip", "WRAP": "wrap",
}
SHOT_GROUP_BDC = {
    "Wristshot": "wrist_snap", "Snapshot": "wrist_snap", "Fan": "wrist_snap", "Poke": "wrist_snap",
    "Slapshot": "slap", "Deflection": "tip", "Bat": "tip", "Wrap Around": "wrap",
}

# Columns produced for BOTH sources, in this order.
COMMON = [
    "source", "game_id", "season", "period", "t_game", "is_home",
    "event_type", "goal", "unblocked",
    "x_adj", "y_adj", "distance", "angle", "abs_angle", "shot_type_raw", "shot_group",
    "skaters_for", "skaters_against", "empty_net_against", "score_diff",
    "time_since_faceoff", "shooter_toi",
    "shoot_toi_mean", "shoot_toi_min", "shoot_toi_max",
    "def_toi_mean", "def_toi_min", "def_toi_max", "toi_min_diff",
    "rebound", "rebound_angle_change",
]
BDC_ONLY = ["traffic", "one_timer"]  # not in MoneyPuck; usable only in BDC-side work


# ---------------------------------------------------------------------------
# Geometry (shared)
# ---------------------------------------------------------------------------

def shot_distance(x_adj, y_adj):
    return np.hypot(NET_X - np.asarray(x_adj, float), np.asarray(y_adj, float))


def shot_angle(x_adj, y_adj):
    """Signed angle in degrees, MoneyPuck convention (positive = shooter's left)."""
    x = np.asarray(x_adj, float)
    y = np.asarray(y_adj, float)
    dx = np.abs(NET_X - x)
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.degrees(np.arctan(np.abs(y) / dx))
    a = np.where(dx == 0, np.where(y == 0, 0.0, 90.0), a)
    return np.sign(y) * a


def add_geometry(df):
    df = df.copy()
    df["distance"] = shot_distance(df["x_adj"], df["y_adj"])
    df["angle"] = shot_angle(df["x_adj"], df["y_adj"])
    df["abs_angle"] = df["angle"].abs()
    return df


# ---------------------------------------------------------------------------
# MoneyPuck
# ---------------------------------------------------------------------------

def moneypuck_features(mp):
    """MoneyPuck shot rows (SHOT / MISS / GOAL; blocked shots are not in the files) -> COMMON columns."""
    home = mp["isHomeTeam"].astype(int) == 1
    out = pd.DataFrame({
        "source": "moneypuck",
        "game_id": mp["season"].astype(str) + "_" + mp["game_id"].astype(str),
        "season": mp["season"].astype(int),
        "period": mp["period"].astype(int),
        "t_game": mp["time"].astype(float),
        "is_home": home.astype(int),
        "event_type": mp["event"].astype(str),
        "goal": mp["goal"].astype(int),
        "unblocked": 1,
        "x_adj": mp["xCordAdjusted"].astype(float),
        "y_adj": mp["yCordAdjusted"].astype(float),
        "shot_type_raw": mp["shotType"].fillna("").astype(str),
        "skaters_for": np.where(home, mp["homeSkatersOnIce"], mp["awaySkatersOnIce"]).astype(int),
        "skaters_against": np.where(home, mp["awaySkatersOnIce"], mp["homeSkatersOnIce"]).astype(int),
        "score_diff": np.where(home, 1, -1) * (mp["homeTeamGoals"] - mp["awayTeamGoals"]),
        "time_since_faceoff": mp["timeSinceFaceoff"].astype(float),
        "shooter_toi": mp["shooterTimeOnIce"].astype(float),
        "shoot_toi_mean": mp["shootingTeamAverageTimeOnIce"].astype(float),
        "shoot_toi_min": mp["shootingTeamMinTimeOnIce"].astype(float),
        "shoot_toi_max": mp["shootingTeamMaxTimeOnIce"].astype(float),
        "def_toi_mean": mp["defendingTeamAverageTimeOnIce"].astype(float),
        "def_toi_min": mp["defendingTeamMinTimeOnIce"].astype(float),
        "def_toi_max": mp["defendingTeamMaxTimeOnIce"].astype(float),
        "rebound": mp["shotRebound"].astype(int),
        "mp_prev_shot_angle": mp["lastEventShotAngle"].astype(float),
        "mp_xGoal": mp["xGoal"].astype(float),  # MoneyPuck's own xG, kept as a benchmark
        "mp_shotOnEmptyNet": mp["shotOnEmptyNet"].astype(int),
    })
    # Missing time-on-ice is coded as 999 (minimum) and 0 (maximum) in the files.
    for c in ["shoot_toi_min", "def_toi_min"]:
        out.loc[out[c] >= 999, c] = np.nan
    out["shot_group"] = out["shot_type_raw"].map(SHOT_GROUP_MP).fillna("wrist_snap")
    out["empty_net_against"] = (out["skaters_against"] == 6).astype(int)
    out = add_geometry(out)
    # Rebound angle change computed the same way for both sources. The file's own
    # shotAnglePlusRebound is also non-zero for many non-rebounds, so it is not used.
    out["rebound_angle_change"] = np.where(
        out["rebound"] == 1, (out["angle"] - out["mp_prev_shot_angle"]).abs(), 0.0)
    out["toi_min_diff"] = out["shoot_toi_min"] - out["def_toi_min"]
    return out[COMMON + ["mp_xGoal", "mp_shotOnEmptyNet"]]


def load_moneypuck(path):
    """Read a MoneyPuck shots CSV or the zip it comes in."""
    return pd.read_csv(path, low_memory=False)


# ---------------------------------------------------------------------------
# Big Data Cup
# ---------------------------------------------------------------------------

def to_bool(values):
    """Traffic / one-timer flags: 2025 files use 't'/'f', 2026 files use 1/0."""
    m = {"t": 1, "true": 1, "1": 1, "1.0": 1, "f": 0, "false": 0, "0": 0, "0.0": 0}
    return pd.Series(values).astype(str).str.strip().str.lower().map(m).astype("Int64").to_numpy()


def on_ice_toi(game_shifts, team, period, clock_sec):
    """Seconds into the current shift for each skater of `team` on the ice (goalies excluded).

    A skater is on the ice if start_sec >= clock > end_sec (clock counts down),
    so a player whose shift ends exactly at `clock` is already off.
    """
    s = game_shifts[
        (game_shifts["team"] == team) & (game_shifts["period"] == period)
        & (game_shifts["start_sec"] >= clock_sec) & (game_shifts["end_sec"] < clock_sec)
        & ~game_shifts["player"].isin(["Go", ""])
    ]
    if s.empty:
        return np.array([])
    return (s.groupby("player")["start_sec"].min() - clock_sec).to_numpy(float)


def _toi_summary(t):
    if t.size == 0:
        return np.nan, np.nan, np.nan
    return float(t.mean()), float(t.min()), float(t.max())


NHL_LOGGED = ("Shot", "Goal", "Faceoff Win", "Takeaway", "Penalty Taken")
"""BDC events that the NHL also logs. Passes, puck recoveries, zone entries and
dump-ins are Stathletes-only; hits and giveaways are NHL-only. "Previous event"
features are computed on this common stream only (here: the rebound flag)."""


def bdc_shot_features(ev, shifts):
    """Clean BDC events (wrangle.clean_events) + shifts (wrangle.load_shifts) -> COMMON + BDC_ONLY."""
    rows = []
    for game_id, g in ev.groupby("game_id", sort=True):
        g = g.sort_values("event_idx")
        gs = shifts[shifts["game_id"] == game_id]
        common_stream = g[g["event"].isin(NHL_LOGGED)]
        for e in g[g["event"].isin(["Shot", "Goal"])].itertuples(index=False):
            if e.event == "Goal":
                event_type, unblocked = "GOAL", 1
            else:
                event_type = {"On Net": "SHOT", "Missed": "MISS", "Blocked": "BLOCK"}.get(e.detail_2, "UNKNOWN")
                unblocked = int(event_type in ("SHOT", "MISS"))

            # Time since the last faceoff in this period (else since the period started).
            fo = g[(g["event"] == "Faceoff Win") & (g["period"] == e.period) & (g["event_idx"] < e.event_idx)]
            start = wrangle.period_length(e.period).item()
            since_fo = float((fo["clock_sec"].iloc[-1] if len(fo) else start) - e.clock_sec)

            # Rebound: previous NHL-logged event is an unblocked shot (SHOT or MISS)
            # by the same team, in the same period, at most 3 s earlier.
            prev = common_stream[(common_stream["event_idx"] < e.event_idx)]
            rebound, angle_change = 0, 0.0
            if len(prev):
                p = prev.iloc[-1]
                if (p["period"] == e.period and p["team"] == e.team and p["event"] == "Shot"
                        and p["detail_2"] in ("On Net", "Missed")
                        and p["clock_sec"] - e.clock_sec <= REBOUND_SEC):
                    rebound = 1
                    angle_change = float(abs(shot_angle(e.x_att, e.y_att) - shot_angle(p["x_att"], p["y_att"])))

            opp = e.opponent
            st = _toi_summary(on_ice_toi(gs, e.team, e.period, e.clock_sec))
            dt = _toi_summary(on_ice_toi(gs, opp, e.period, e.clock_sec))
            rows.append({
                "source": "bdc", "game_id": game_id, "season": e.season, "period": e.period,
                "t_game": float(e.t_game), "is_home": e.is_home,
                "event_type": event_type, "goal": int(e.event == "Goal"), "unblocked": unblocked,
                "x_adj": e.x_att, "y_adj": e.y_att, "shot_type_raw": e.detail_1,
                "skaters_for": e.skaters_for, "skaters_against": e.skaters_against,
                "score_diff": e.score_diff,
                "time_since_faceoff": since_fo,
                "shooter_toi": wrangle.shift_elapsed(gs, e.team, e.player, e.period, e.clock_sec),
                "shoot_toi_mean": st[0], "shoot_toi_min": st[1], "shoot_toi_max": st[2],
                "def_toi_mean": dt[0], "def_toi_min": dt[1], "def_toi_max": dt[2],
                "rebound": rebound, "rebound_angle_change": angle_change,
                "traffic": e.detail_3, "one_timer": e.detail_4,
                "team_id": e.team_id, "event_idx": e.event_idx,
            })
    out = pd.DataFrame(rows)
    out["shot_group"] = out["shot_type_raw"].map(SHOT_GROUP_BDC)
    unknown = out.loc[out["shot_group"].isna(), "shot_type_raw"].unique()
    if len(unknown):
        raise ValueError(f"Unmapped BDC shot types: {unknown}")
    out["empty_net_against"] = (out["skaters_against"] == 6).astype(int)
    out["toi_min_diff"] = out["shoot_toi_min"] - out["def_toi_min"]
    out["traffic"] = to_bool(out["traffic"])
    out["one_timer"] = to_bool(out["one_timer"])
    out = add_geometry(out)
    return out[COMMON + BDC_ONLY + ["team_id", "event_idx"]]


def check_bdc_shots(shots):
    """Loud checks on the BDC shot table. Returns a short summary."""
    assert shots["shot_group"].notna().all()
    assert shots["x_adj"].between(-100, 100).all() and shots["y_adj"].between(-42.5, 42.5).all()
    assert (shots["unblocked"] == shots["event_type"].isin(["SHOT", "MISS", "GOAL"]).astype(int)).all()
    unb = shots[shots["unblocked"] == 1]
    return {
        "shots": len(shots), "unblocked": len(unb), "goals": int(unb["goal"].sum()),
        "goal_rate_unblocked": round(unb["goal"].mean(), 4),
        "rebounds": int(shots["rebound"].sum()),
        "median_distance_unblocked": round(unb["distance"].median(), 1),
        "shooter_toi_missing": int(shots["shooter_toi"].isna().sum()),
    }


def main(argv):
    if len(argv) != 4 or argv[1] not in ("bdc", "moneypuck"):
        print(__doc__)
        sys.exit(1)
    kind, src, out_csv = argv[1], argv[2], argv[3]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    if kind == "bdc":
        ev = wrangle.clean_events(src)
        shots = bdc_shot_features(ev, wrangle.load_shifts(src))
        print(check_bdc_shots(shots))
    else:
        shots = moneypuck_features(load_moneypuck(src))
        print({"shots": len(shots), "goals": int(shots["goal"].sum()),
               "goal_rate": round(shots["goal"].mean(), 4)})
    shots.to_csv(out_csv, index=False)
    print(f"Wrote {len(shots):,} rows -> {out_csv}")


if __name__ == "__main__":
    main(sys.argv)

"""Known-answer tests for build_entries.py, analyze.py and wrangle.py.

Run from the repo root:
    python code/tests.py
(also works with pytest). Most tests build tiny inputs by hand. The last one
runs both independent entry builders on data/raw and checks they agree on
every entry; it is skipped if the raw data is missing.
"""

from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parent
RAW = CODE.parent / "data" / "raw"
sys.path.insert(0, str(CODE))

import analyze  # noqa: E402
import build_entries  # noqa: E402
import wrangle  # noqa: E402


# ---------------------------------------------------------------------------
# build_entries.py (main pipeline)
# ---------------------------------------------------------------------------

EVENT_COLS = ["Date", "Home_Team", "Away_Team", "Period", "Clock", "Home_Team_Skaters",
              "Away_Team_Skaters", "Home_Team_Goals", "Away_Team_Goals", "Team", "Player_Id",
              "Event", "X_Coordinate", "Y_Coordinate", "Detail_1", "Detail_2", "Detail_3",
              "Detail_4", "Player_Id_2", "X_Coordinate_2", "Y_Coordinate_2"]


def _toy_files(folder):
    """One toy game: Team A (away) carries in at 18:20; Team B dumps in at 17:00."""
    rows = [
        # period, clock, team, event, detail_1, y
        (1, "20:00", "B", "Faceoff Win", "", 0),
        (1, "18:45", "B", "Shot", "", 0),          # 35 s before A's entry: outside placebo window
        (1, "18:35", "B", "Shot", "", 0),          # 15 s before: placebo (either team)
        (1, "18:30", "A", "Shot", "", 0),          # 10 s before: placebo (either team and own)
        (1, "18:20", "A", "Zone Entry", "Carried", -30),
        (1, "18:20", "A", "Shot", "", 0),          # same second, after the entry: counts
        (1, "18:10", "B", "Shot", "", 0),          # opponent: does not count
        (1, "18:05", "A", "Goal", "", 0),          # counts
        (1, "17:30", "A", "Shot", "", 0),          # 50 s later: outside the window
        (1, "17:00", "B", "Zone Entry", "Dumped", 10),
        (1, "16:55", "A", "Faceoff Win", "", 0),   # stoppage ends B's window
        (1, "16:50", "B", "Shot", "", 0),          # after the stop: does not count
    ]
    ev = pd.DataFrame([{
        "Date": "2025-10-11", "Home_Team": "Team B", "Away_Team": "Team A", "Period": p,
        "Clock": c, "Home_Team_Skaters": 5, "Away_Team_Skaters": 5, "Home_Team_Goals": 0,
        "Away_Team_Goals": 0, "Team": f"Team {t}", "Player_Id": 9, "Event": e,
        "X_Coordinate": 25, "Y_Coordinate": y, "Detail_1": d,
    } for p, c, t, e, d, y in rows]).reindex(columns=EVENT_COLS)
    sh = pd.DataFrame({
        "Date": "2025-10-11", "Home_Team": "Team B", "Away_Team": "Team A",
        "Team": ["Team A", "Team A", "Team B"], "Player_Id": ["9", "Go", "4"],
        "shift_number": 1, "period": 1, "start_clock": ["18:50", "20:00", "17:30"],
        "end_clock": ["18:00", "00:00", "16:00"], "shift_length": "",
    })
    ev_path = Path(folder) / "2026-2025-10-11.Team.A.@.Team.B.Events.csv"
    ev.to_csv(ev_path, index=False)
    sh.to_csv(Path(folder) / "2026-2025-10-11.Team.A.@.Team.B.Shifts.csv", index=False)
    return ev_path


def test_clock_to_sec():
    assert build_entries.clock_to_sec("19:52") == 1192
    assert build_entries.clock_to_sec("00:05") == 5


def test_build_game_toy():
    with tempfile.TemporaryDirectory() as d:
        ev_path = _toy_files(d)
        rows = build_entries.build_game(str(ev_path), str(ev_path).replace("Events", "Shifts"), "2026", 20)
    a, b = rows
    assert a["game"] == "2026:2025-10-11.Team.A.@.Team.B" and a["team"] == "2026:Team A"
    assert (a["carry"], a["y"], a["y_pre"], a["y_pre_own"]) == (1, 2, 2, 1)
    assert a["since_faceoff"] == 100 and a["lane"] == 30 and a["home"] == 0
    assert a["shift_time"] == 30.0       # A's #9 started at 18:50, entry at 18:20; goalie ignored
    assert (b["carry"], b["y"]) == (0, 0)  # the faceoff stops B's window before its shot
    assert b["home"] == 1


def test_game_label_both_name_styles():
    with tempfile.TemporaryDirectory() as d:
        ev_path = _toy_files(d)
        old = Path(d) / "2025-2025-10-11.Team.A.@.Team.B.-.Events.csv"
        ev_path.rename(old)
        (Path(d) / "2026-2025-10-11.Team.A.@.Team.B.Shifts.csv").rename(
            Path(d) / "2025-2025-10-11.Team.A.@.Team.B.-.Shifts.csv")
        rows = build_entries.build_game(str(old), str(old).replace("Events", "Shifts"), "2025", 20)
    assert rows[0]["game"] == "2025:2025-10-11.Team.A.@.Team.B"  # no stray ".-"


def test_failed_carry_labels_match_entry_labels():
    # analyze_final.py joins failed carries to entries by game label, so both must agree
    import failed_carries
    for name in ("2025-2024-10-25.Team.H.@.Team.G.-.Events.csv", "2026-2025-10-11.Team.A.@.Team.D.Events.csv"):
        rel = name[:4]
        stem = name.split(".Events")[0]
        stem = stem[:-2] if stem.endswith(".-") else stem
        assert failed_carries.game_id(name, rel) == f"{rel}:{stem[5:]}"
        assert not failed_carries.game_id(name, rel).endswith(".-")


# ---------------------------------------------------------------------------
# analyze.py
# ---------------------------------------------------------------------------

def test_lin_ate_recovers_known_ate():
    # Heterogeneous effect tau(x) = 1 + 2x with E[x] = 0.5, so ATE = 2; carry is confounded by x.
    rng = np.random.default_rng(0)
    n = 20000
    x = rng.uniform(0, 1, n)
    d = rng.binomial(1, 0.3 + 0.4 * x).astype(float)
    y = 1 + 3 * x + d * (1 + 2 * x) + rng.normal(0, 0.1, n)
    assert abs(analyze.lin_ate(y, d, x[:, None]) - 2.0) < 0.02
    assert y[d == 1].mean() - y[d == 0].mean() > 2.1  # the raw gap is biased upwards


def test_lin_ate_drops_constant_columns():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 1))
    d = rng.binomial(1, 0.5, 500).astype(float)
    y = 0.5 * d + x[:, 0]
    X = np.column_stack([x, np.ones(500)])  # constant column must not break the fit
    assert abs(analyze.lin_ate(y, d, X) - 0.5) < 1e-8


def test_ipw_smd_unweighted():
    df = pd.DataFrame({"carry": [1, 1, 0, 0], "z": [2.0, 4.0, 0.0, 2.0]})
    # means 3 vs 1, pooled sd = sqrt((1 + 1) / 2) = 1, so SMD = 2
    assert abs(analyze.ipw_smd(df, ["z"], np.ones(4))["z"] - 2.0) < 1e-12


# ---------------------------------------------------------------------------
# wrangle.py (event cleaner, kept for later tracking / xG work and as a cross-check)
# ---------------------------------------------------------------------------

def test_wrangle_parsers():
    assert wrangle.parse_date("10/25/2024") == pd.Timestamp("2024-10-25")
    assert wrangle.parse_date("2025-10-11") == pd.Timestamp("2025-10-11")
    a = wrangle.parse_game_file("2025-2024-10-25.Team.H.@.Team.G.-.Events.csv")
    assert a["game_id"] == "2025_2024-10-25_H@G" and a["home"] == "G"
    got = wrangle.clean_player_id([21, 21.0, "21", "Go", np.nan, "nan", ""])
    assert list(got) == ["21", "21", "21", "Go", "", "", ""]


def test_wrangle_strength_and_lane():
    got = wrangle.strength_state([5, 5, 4, 3, 6], [5, 4, 5, 3, 5])
    assert list(got) == ["5v5", "PP", "SH", "EV_other", "PP"]
    assert list(wrangle.lane([20.0, 0.0, -20.0])) == ["left", "centre", "right"]


def test_wrangle_window_counts():
    g = pd.DataFrame([
        (0, 1, 125, "A", "Shot"), (1, 1, 110, "A", "Shot"), (2, 1, 100, "A", "Zone Entry"),
        (3, 1, 100, "A", "Shot"), (4, 1, 95, "B", "Shot"), (5, 1, 90, "A", "Goal"),
        (6, 1, 85, "A", "Faceoff Win"), (7, 1, 84, "A", "Shot"), (8, 2, 1199, "A", "Shot"),
    ], columns=["event_idx", "period", "clock_sec", "team", "event"])
    got = wrangle.window_counts(g, g.iloc[2], window_sec=20)
    assert (got["shots_for"], got["goals_for"], got["shots_against"]) == (2, 1, 1)
    assert (got["shots_for_nostop"], got["shots_for_before_nostop"]) == (3, 1)


# ---------------------------------------------------------------------------
# Cross-check on the real data: two independent builders must agree
# ---------------------------------------------------------------------------

def test_builders_agree_on_real_data():
    if not list(RAW.glob("*Events.csv")):
        print("    (skipped: no raw data)")
        return
    g_rows = []
    for ev_path in sorted(RAW.glob("*Events.csv")):
        sh_path = str(ev_path).replace("Events", "Shifts")
        g_rows += build_entries.build_game(str(ev_path), sh_path, ev_path.name[:4], 20)
    g = pd.DataFrame(g_rows)

    ev = wrangle.clean_events(RAW)
    w = wrangle.build_entries(ev, wrangle.load_shifts(RAW))
    w = w[w["carry"].notna()].reset_index(drop=True)  # build_entries keeps carried + dumped only

    assert len(g) == len(w) == 1748
    assert (g["carry"].values == w["carry"].values).all()
    assert (g["y"].values == w["shots_for"].values).all()
    assert (g["y_pre_own"].values == w["shots_for_before_nostop"].values).all()
    assert (g["since_faceoff"].values == w["time_since_faceoff"].values).all()
    assert np.allclose(g["lane"].values, np.abs(w["y_att"].values))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"All {len(tests)} tests passed.")

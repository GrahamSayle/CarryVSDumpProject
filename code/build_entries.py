"""Build the zone-entry analysis table from Big Data Cup 2025 and 2026 event and shift files.

One row per carried or dumped zone entry, with:
  outcome   y         shot attempts (Shot or Goal events) by the entering team within WINDOW s,
                      stopping early at the next zone entry, faceoff or period end
  placebo   y_pre     shot attempts by either team in the WINDOW s before the entry (same period)
  treatment carry     1 = carried, 0 = dumped ("Played" entries are excluded)
  covariates          game situation measured at or before the entry

Usage: python build_entries.py <raw_data_dir> <out_csv> [window_seconds]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd


def clock_to_sec(c):
    """'MM:SS' remaining in period -> seconds remaining."""
    m, s = str(c).split(":")
    return int(m) * 60 + int(s)


def load_shifts(path):
    s = pd.read_csv(path)
    team_col = "Team" if "Team" in s.columns else "team_name"
    s = s.rename(columns={team_col: "team"})
    s = s[s["Player_Id"].astype(str) != "Go"].copy()          # drop goalies
    s["start"] = s["start_clock"].map(clock_to_sec)           # seconds remaining at start
    s["end"] = s["end_clock"].map(clock_to_sec)
    return s[["team", "period", "start", "end"]]


def mean_shift_time(shifts, team, period, t):
    """Mean seconds already skated in the current shift by `team`'s skaters on ice at time t."""
    on = shifts[(shifts.team == team) & (shifts.period == period)
                & (shifts.start >= t) & (shifts.end < t)]
    return float((on.start - t).mean()) if len(on) else np.nan


def build_game(ev_path, sh_path, release, window):
    ev = pd.read_csv(ev_path, encoding="utf-8-sig")
    sh = load_shifts(sh_path)
    game = f"{release}:{os.path.basename(ev_path).split('.Events')[0].split('.-.')[0][5:]}"
    ev["t"] = ev["Clock"].map(clock_to_sec)
    ev["Period"] = ev["Period"].astype(int)
    rows = []
    last_faceoff_t = {}
    recs = ev.to_dict("records")
    for i, r in enumerate(recs):
        if r["Event"] == "Faceoff Win":
            last_faceoff_t[r["Period"]] = r["t"]
        if r["Event"] != "Zone Entry" or r["Detail_1"] not in ("Carried", "Dumped"):
            continue
        team, p, t0 = r["Team"], r["Period"], r["t"]
        home = int(team == r["Home_Team"])
        sk_for = r["Home_Team_Skaters"] if home else r["Away_Team_Skaters"]
        sk_against = r["Away_Team_Skaters"] if home else r["Home_Team_Skaters"]
        g_for = r["Home_Team_Goals"] if home else r["Away_Team_Goals"]
        g_against = r["Away_Team_Goals"] if home else r["Home_Team_Goals"]

        # outcome: entering team's shot attempts in the next `window` seconds
        y = 0
        for r2 in recs[i + 1:]:
            if r2["Period"] != p or t0 - r2["t"] > window:
                break
            if r2["Event"] in ("Faceoff Win", "Zone Entry"):
                break
            if r2["Team"] == team and r2["Event"] in ("Shot", "Goal"):
                y += 1
        # placebo: shot attempts by either team in the `window` seconds before
        y_pre = 0
        for r2 in reversed(recs[:i]):
            if r2["Period"] != p or r2["t"] - t0 > window:
                break
            if r2["Event"] in ("Shot", "Goal"):
                y_pre += 1

        rows.append(dict(
            game=game, release=release, team=f"{release}:{team}",
            carry=int(r["Detail_1"] == "Carried"), y=y, y_pre=y_pre,
            period=min(p, 4), period_elapsed=1200 - t0 if p <= 3 else 300 - t0,
            home=home, skater_diff=sk_for - sk_against,
            even_strength=int(sk_for == sk_against),
            score_diff=float(np.clip(g_for - g_against, -2, 2)),
            lane=abs(r["Y_Coordinate"]),
            since_faceoff=last_faceoff_t.get(p, 1200) - t0,
            shift_time=mean_shift_time(sh, team, p, t0),
        ))
    return rows


def main(raw_dir, out_csv, window=20):
    rows = []
    for ev_path in sorted(glob.glob(os.path.join(raw_dir, "*Events.csv"))):
        release = os.path.basename(ev_path)[:4]
        sh_path = ev_path.replace("Events", "Shifts")
        rows += build_game(ev_path, sh_path, release, window)
    df = pd.DataFrame(rows)
    df["shift_time_missing"] = df.shift_time.isna().astype(int)
    df["shift_time"] = df.shift_time.fillna(df.shift_time.median())
    df.to_csv(out_csv, index=False)
    print(f"{len(df)} entries ({df.carry.sum()} carried, {(1 - df.carry).sum()} dumped) "
          f"from {df.game.nunique()} games -> {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20)

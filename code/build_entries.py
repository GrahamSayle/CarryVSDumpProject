"""Build the zone-entry analysis table from Big Data Cup 2025 and 2026 event and shift files.

One row per carried or dumped zone entry, with:
  outcome   y         shot attempts (Shot or Goal events) by the entering team within WINDOW s,
                      stopping early at the next zone entry, faceoff or period end
  placebo   y_pre     shot attempts by either team in the WINDOW s before the entry (same period)
            y_pre_own same, entering team only (sensitivity check)
  treatment carry     1 = carried, 0 = dumped ("Played" entries are excluded)
  covariates          game situation measured at or before the entry

Optional xG outcomes (only when an xG predictions file from xg_train.py is given):
  y_xg, y_pre_xg, y_pre_own_xg   same windows and stopping rules as y, y_pre, y_pre_own, but each
                                 shot counts its LightGBM xG instead of 1 (blocked shots count 0:
                                 MoneyPuck, the training data, has no blocked shots)
  y_xg_logit                     y_xg with the full logistic xG model (robustness)
  y_goals                        goals by the entering team in the same window
  n_unblocked                    unblocked shot attempts by the entering team in the same window

Usage: python build_entries.py <raw_data_dir> <out_csv> [window_seconds] [xg_predictions_csv]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wrangle  # noqa: E402  (only for the shared game_id used by the xG files)


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


def load_xg(xg_csv):
    """xG predictions -> {(game_id, event_idx): (xg_lgbm, xg_logit_full)}; blocked shots get 0."""
    p = pd.read_csv(xg_csv)
    p[["xg_lgbm", "xg_logit_full"]] = p[["xg_lgbm", "xg_logit_full"]].fillna(0.0)
    return {(g, int(i)): (a, b) for g, i, a, b in
            zip(p.game_id, p.event_idx, p.xg_lgbm, p.xg_logit_full)}


def build_game(ev_path, sh_path, release, window, xg=None):
    ev = pd.read_csv(ev_path, encoding="utf-8-sig")
    xg_gid = wrangle.parse_game_file(ev_path)["game_id"] if xg is not None else None
    sh = load_shifts(sh_path)
    # 2025 file names end in ".-.Events.csv", 2026 in ".Events.csv"; strip both forms
    stem = os.path.basename(ev_path).split(".Events")[0]
    stem = stem[:-2] if stem.endswith(".-") else stem
    game = f"{release}:{stem[5:]}"
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
        xg_post = []    # (row index, event) of the shots counted in y
        for j, r2 in enumerate(recs[i + 1:], start=i + 1):
            if r2["Period"] != p or t0 - r2["t"] > window:
                break
            if r2["Event"] in ("Faceoff Win", "Zone Entry"):
                break
            if r2["Team"] == team and r2["Event"] in ("Shot", "Goal"):
                y += 1
                xg_post.append((j, r2["Event"]))
        # placebo: shot attempts by either team in the `window` seconds before
        y_pre = 0
        y_pre_own = 0   # sensitivity: entering team's shots only, same window
        xg_pre = []     # (row index, own team?) of the shots counted in y_pre
        for j in range(i - 1, -1, -1):
            r2 = recs[j]
            if r2["Period"] != p or r2["t"] - t0 > window:
                break
            if r2["Event"] in ("Shot", "Goal"):
                y_pre += 1
                y_pre_own += int(r2["Team"] == team)
                xg_pre.append((j, r2["Team"] == team))

        rows.append(dict(
            game=game, release=release, team=f"{release}:{team}",
            carry=int(r["Detail_1"] == "Carried"), y=y, y_pre=y_pre, y_pre_own=y_pre_own,
            period=min(p, 4), period_elapsed=1200 - t0 if p <= 3 else 300 - t0,
            home=home, skater_diff=sk_for - sk_against,
            even_strength=int(sk_for == sk_against),
            score_diff=float(np.clip(g_for - g_against, -2, 2)),
            lane=abs(r["Y_Coordinate"]),
            since_faceoff=last_faceoff_t.get(p, 1200) - t0,
            shift_time=mean_shift_time(sh, team, p, t0),
        ))
        if xg is not None:
            post = [xg[(xg_gid, j)] for j, _ in xg_post]
            pre = [(xg[(xg_gid, j)][0], own) for j, own in xg_pre]
            rows[-1].update(
                y_xg=sum(a for a, _ in post), y_xg_logit=sum(b for _, b in post),
                y_pre_xg=sum(a for a, _ in pre), y_pre_own_xg=sum(a for a, own in pre if own),
                y_goals=sum(ev_ == "Goal" for _, ev_ in xg_post),
                n_unblocked=sum(a > 0 for a, _ in post))
    return rows


def main(raw_dir, out_csv, window=20, xg_csv=None):
    xg = load_xg(xg_csv) if xg_csv else None
    rows = []
    for ev_path in sorted(glob.glob(os.path.join(raw_dir, "*Events.csv"))):
        release = os.path.basename(ev_path)[:4]
        sh_path = ev_path.replace("Events", "Shifts")
        rows += build_game(ev_path, sh_path, release, window, xg)
    df = pd.DataFrame(rows)
    df["shift_time_missing"] = df.shift_time.isna().astype(int)
    df["shift_time"] = df.shift_time.fillna(df.shift_time.median())
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"{len(df)} entries ({df.carry.sum()} carried, {(1 - df.carry).sum()} dumped) "
          f"from {df.game.nunique()} games -> {out_csv}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20,
         sys.argv[4] if len(sys.argv) > 4 else None)

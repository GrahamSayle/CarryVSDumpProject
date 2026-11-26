"""Proxy for failed carries, which never appear as Zone Entry events.

A failed carry is approximated by the defending team gaining the puck in the neutral zone just
before the attacking blue line (X_rel in [LO, 25) ft, where X_rel is measured in the attacking
team's direction and the attacking blue line sits at +25). Two versions:
  narrow  a Takeaway there
  broad   a Takeaway or Puck Recovery there, straight after an event by the attacking team that
          was not a dump, shot or faceoff (the attacker had the puck and lost it) For each one we count, in the next
WINDOW seconds (stopping at a faceoff or the end of the period):
  shots_for      shot attempts by the team that lost the puck
  shots_against  shot attempts by the team that took it
  xg_for, xg_against   the same shots weighted by their LightGBM xG (blocked shots 0), only when
                       an xG predictions file from xg_train.py is given

The attacking direction of each team in each period is read from the sign of X at that team's
recorded zone entries in that period (entries sit at X = +/-25).

Usage: python failed_carries.py <raw_dir> <out_csv> [lo] [window] [xg_predictions_csv]
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

import wrangle
from build_entries import clock_to_sec, load_xg


def game_id(ev_path, release):
    """Same label as build_entries.py (must match: analyze_final.py joins on it)."""
    stem = os.path.basename(ev_path).split(".Events")[0]
    stem = stem[:-2] if stem.endswith(".-") else stem
    return f"{release}:{stem[5:]}"


def failed_carry_proxy(ev_path, release, lo=10.0, window=20, broad=False, xg=None):
    events = ("Takeaway", "Puck Recovery") if broad else ("Takeaway",)
    ev = pd.read_csv(ev_path, encoding="utf-8-sig")
    ev["t"] = ev["Clock"].map(clock_to_sec)
    ev["Period"] = ev["Period"].astype(int)
    ze = ev[ev.Event == "Zone Entry"]
    direction = np.sign(ze.groupby(["Team", "Period"]).X_Coordinate.median()).to_dict()
    home, away = ev.Home_Team.iloc[0], ev.Away_Team.iloc[0]
    recs = ev.to_dict("records")
    xg_gid = wrangle.parse_game_file(ev_path)["game_id"] if xg is not None else None
    rows = []
    for i, r in enumerate(recs):
        if r["Event"] not in events:
            continue
        if broad:
            prev = recs[i - 1] if i > 0 else None
            if (prev is None or prev["Period"] != r["Period"] or prev["Team"] == r["Team"]
                    or prev["Event"] in ("Dump In/Out", "Shot", "Goal", "Faceoff Win")):
                continue
        taker, p = r["Team"], r["Period"]
        loser = away if taker == home else home
        s = direction.get((loser, p))
        if s is None or s == 0:
            continue
        x_rel = s * r["X_Coordinate"]
        if not (lo <= x_rel < 25):
            continue
        f = a = 0
        xf = xa = 0.0
        for j in range(i + 1, len(recs)):
            r2 = recs[j]
            if r2["Period"] != p or r["t"] - r2["t"] > window or r2["Event"] == "Faceoff Win":
                break
            if r2["Event"] in ("Shot", "Goal"):
                f += r2["Team"] == loser
                a += r2["Team"] == taker
                if xg is not None:
                    v = xg[(xg_gid, j)][0]
                    xf += v * (r2["Team"] == loser)
                    xa += v * (r2["Team"] == taker)
        row = dict(game=game_id(ev_path, release), team=f"{release}:{loser}", period=p,
                   x_rel=x_rel, event=r["Event"], shots_for=f, shots_against=a)
        if xg is not None:
            row.update(xg_for=xf, xg_against=xa)
        rows.append(row)
    return rows


def main(raw_dir, out_csv, lo=10.0, window=20, xg_csv=None):
    xg = load_xg(xg_csv) if xg_csv else None
    rows = []
    for path in sorted(glob.glob(os.path.join(raw_dir, "*Events*.csv"))):
        release = os.path.basename(path)[:4]
        for broad in (False, True):
            rows += [dict(r, definition="broad" if broad else "narrow")
                     for r in failed_carry_proxy(path, release, lo, window, broad, xg)]
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    cols = ["shots_for", "shots_against"] + (["xg_for", "xg_against"] if xg is not None else [])
    print(out.groupby("definition")[cols].agg(["size", "mean"]))
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(float(a) for a in sys.argv[3:4]), *(int(a) for a in sys.argv[4:5]),
         *sys.argv[5:6])

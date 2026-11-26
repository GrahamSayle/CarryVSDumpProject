# Carry or dump? Replicating the zone-entry shot gap and testing how much survives adjustment

STA2112H (Statistical Learning I), University of Toronto, Fall 2026 · Group 10
Graham Sayle and Alec Neme

## Question

Tulsky et al. (2013) found that carried zone entries produce more than twice the shots of dump-ins. Yet heavy-forechecking, dump-and-chase teams have won the last three Stanley Cups. Is the shot gap caused by the choice to carry, or by the situations in which teams get to carry?

**Terms.** A *zone entry* is the moment the attacking team gets the puck over the opponent's blue line. A *carry* skates the puck in with control. A *dump* shoots it into the corner and chases it.

## Approach

1. **Replicate** the raw carry-vs-dump gap in shot attempts on public Big Data Cup data.
2. **Adjust for game situation.** Compare carries and dumps made in the same situation: manpower, score, period and time, home/away, lane, time since faceoff, shift length, team.
   - Logistic model of P(carry | situation), cross-fitted by game. Entries outside 0.05–0.95 are trimmed.
   - Regression of shots on carry, situation and their interactions, giving the average effect (ATE).
   - 95% intervals from 1,000 resamples of whole games, refitting every step.
   - Checks: leave-one-game-out refits, and a placebo (shots in the 20 s *before* the entry).
3. **Next:** a sensitivity analysis for unmeasured confounding (robustness value, Cinelli & Hazlett 2020) and double ML (AIPW, boosted trees) as a robustness check. Optional: xG outcome, shots against, and tracking covariates at the entry frame.

## Midway results (Nov 20, 2026)

| Estimate | Carried − dumped, shot attempts per entry (next 20 s) | 95% game-bootstrap CI |
|---|---|---|
| Raw gap, all 1,748 entries | 0.36 | [0.31, 0.41] |
| Raw gap, trimmed (1,724) | 0.37 | [0.31, 0.42] |
| Adjusted for game situation (ATE) | 0.37 | [0.31, 0.43] |
| Placebo: shots in 20 s before entry | −0.00 | [−0.09, 0.09] |

- Raw means are 0.61 vs 0.25 shots per entry (2.4×). Carries lead in all 13 games.
- The 24 trimmed entries are all overtime carries, so the adjusted estimate excludes overtime.
- Game situation barely predicts who carries (AUC 0.60). Adjustment therefore had little to work with, and this is **not yet** evidence that the gap is causal. The main open issue is unmeasured confounding (how the defence is set, carrier skill) and selection (failed carries are never logged as entries).

## Data

| Source | Games | Used for |
|---|---|---|
| Big Data Cup 2026 (Stathletes) | 10 | Events, shifts, tracking |
| Big Data Cup 2025 (Stathletes) | 3 | Events, shifts, tracking |

- Teams are anonymised. Event coordinates run x ∈ [−100, 100], y ∈ [−42.5, 42.5] (feet).

## How to run

Python 3.10+ with numpy, pandas, scikit-learn and matplotlib (`pip install -r requirements.txt`). From the repo root:

```
python code/tests.py                                                # 10 tests, ~10 s
python code/build_entries.py data/raw data/processed/entries.csv    # 1,748 entries, a few seconds
python code/analyze.py data/processed/entries.csv output 1000       # midway results + figures, ~1 min
```

| File | What it does |
|---|---|
| `code/build_entries.py` | Main pipeline, step 1: raw events + shifts -> one row per carried or dumped entry |
| `code/analyze.py` | Main pipeline, step 2: raw gap, cross-fitted propensity and trimming, interacted regression (ATE), game bootstrap, leave-one-game-out, balance, placebo, figures |
| `code/wrangle.py` | Event-level cleaner (coordinates rotated to the attacking direction) for later tracking / xG / zone-exit work, and an independent second entry builder |
| `code/analyze_xg.py` | Same raw gap, interacted regression, double ML, placebos and sensitivity analysis with expected goals (xG) in the 20 s after the entry as the outcome, plus the xG version of the failed-carry break-even and decision-level gap; needs `entries.csv` and `failed_carries.csv` built with `xg_bdc_predictions.csv` (see `code/run_all.sh`) |
| `code/tests.py` | Known-answer tests on small hand-built inputs, plus a check that the two entry builders agree on every real entry |

`analyze.py` writes `output/results_table.csv`, `summary.json`, `per_game.csv`, `leave_one_game_out.csv`, `balance.csv`, `fig_overlap.png` and `fig_estimates.png`. A fresh run reproduces the midway table above exactly.

**Sensitivity placebo.** `y_pre_own` counts only the entering team's shots in the 20 s before the entry. Its adjusted estimate is −0.03 [−0.07, 0.01] (500 game resamples, reported in `summary.json`). So carries do not follow more shooting by the team that carries; if anything, they follow slightly less.

## Wrangling decisions

- **Outcome (`y`).** The entering team's shot attempts in the 20 s after the entry. That means `Shot` rows (on net, missed or blocked) plus `Goal` rows, which are a separate event in these files. The count stops at the next zone entry by either team, the next faceoff, or the end of the period, so a shot is never credited to two entries. Without the stop rule the means would be 0.72 vs 0.33.
- **Placebo (`y_pre`).** Shot attempts by either team in the 20 s before the entry, same period, no stop rule. This is the version used in the midway slides.
- **Event order.** About 4,900 events share a clock second, so both builders walk the raw file in row order rather than sorting by clock.
- **Teams.** The Cup anonymises teams per competition, so teams are prefixed with the season (`2025:Team D` and `2026:Team D` are different teams).
- **Covariates** (measured at or before the entry): seconds elapsed in the period, skater difference and an even-strength flag, score difference (clipped at ±2), |y| at the entry (distance from the middle of the ice), seconds since the last faceoff, mean seconds into the current shift for the entering team's skaters on the ice (median plus a missing flag when no shift matches), home, period (OT separate), and team.
- **Two independent builders agree.** `build_entries.py` (Graham) and `wrangle.py` (Alec) were written separately and give the same carry label, outcome, own-team placebo, time since faceoff and |y| for all 1,748 entries (`tests.py`).
- **Known data quirks:**
  - In periods 1–3 the raw x, y are absolute rink coordinates and teams switch ends each period. In overtime the raw file already records each event in the acting team's own attacking direction. This matters only for covariates that use signed x or y; `wrangle.py` handles it.
  - 16 entries have no player logged.
  - 5 dumped entries in the 2026 Team G @ Team L game are logged near centre ice (|x| < 20) rather than at the blue line. Check this before using entry location beyond |y|.
- **Balance note.** After inverse-probability weighting, the home/away imbalance gets larger (SMD −0.06 → 0.15; `output/balance.csv`). The interacted regression adjusts for home directly, but this is worth a line in the report.

## References

- Tulsky, E., Detweiler, G., Spencer, R., & Sznajder, C. (2013). *Using zone entry data to separate offensive, neutral, and defensive zone performance.* MIT Sloan Sports Analytics Conference.
- Cinelli, C., & Hazlett, C. (2020). Making sense of sensitivity: Extending omitted variable bias. *Journal of the Royal Statistical Society: Series B*, 82(1), 39–67.
- Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Stathletes Big Data Cup: https://github.com/bigdatacup

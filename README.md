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

## References

- Tulsky, E., Detweiler, G., Spencer, R., & Sznajder, C. (2013). *Using zone entry data to separate offensive, neutral, and defensive zone performance.* MIT Sloan Sports Analytics Conference.
- Cinelli, C., & Hazlett, C. (2020). Making sense of sensitivity: Extending omitted variable bias. *Journal of the Royal Statistical Society: Series B*, 82(1), 39–67.
- Chernozhukov, V., et al. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.
- Stathletes Big Data Cup: https://github.com/bigdatacup

# Expected goals: MoneyPuck (NHL) to Big Data Cup feature mapping

Goal: train an xG model on MoneyPuck NHL shots and apply it to Big Data Cup (BDC) shots.
Code: `code/xg_features.py` builds the same columns for both sources. Tests: `code/tests.py`.
Data source to credit: **MoneyPuck.com** (free for non-commercial use with attribution).

## 1. Coordinates

Both sources already use the same rink in feet, so only one rotation is needed.

| | MoneyPuck | Big Data Cup 2025/26 |
|---|---|---|
| Origin | centre ice (0, 0) | centre ice (0, 0) |
| Range | x in [-100, 100], y in [-42.5, 42.5] | same |
| Goal lines | x = ±89 | ±89 (NHL rink: faceoff dots found at (±69, ±22), (±20, ±22), (0, 0)) |
| Raw frame | absolute rink coordinates | absolute in periods 1–3; **already in the acting team's own attacking frame in overtime** |
| Shooter frame | `xCordAdjusted`, `yCordAdjusted` | `x_att`, `y_att` from `wrangle.py` |

**Conversion.** Rotate 180° (flip x and y together) so the shooting team attacks +x and the net is at (89, 0). After rotation, +y is the shooter's left in both sources. The BDC README draws x to the right and y up, with the camera on the bottom side, so a 180° rotation keeps left and right. This was checked on 2,999 MoneyPuck 2024–25 shots: every flipped shot has `yCordAdjusted = -yCord`.

**Distance and angle** use MoneyPuck's exact formulas (reproduced to machine precision, including 44 shots from behind the net):

$$d = \sqrt{(89 - x)^2 + y^2}, \qquad \theta = \operatorname{sign}(y)\,\arctan\!\frac{|y|}{|89 - x|} \ \text{(degrees; 90 when } x = 89\text{)}.$$

MoneyPuck also gives arena-adjusted coordinates (Schuckers–Curro correction for NHL scorer bias). There is no BDC equivalent, so train on the **unadjusted** columns. BDC locations are the puck's **release point**, tracked from video by Stathletes.

## 2. Covariates

✓ same meaning in both · ≈ available but defined slightly differently · ✗ missing on one side

| Feature (our name) | MoneyPuck column(s) | BDC derivation | Status |
|---|---|---|---|
| `distance`, `angle`, `abs_angle`, `x_adj`, `y_adj` | `xCordAdjusted`, `yCordAdjusted` | `x_att`, `y_att` | ✓ |
| `goal` (target) | `goal` | `Goal` event | ✓ |
| `unblocked` | always 1 (blocked shots are **not** in MoneyPuck) | `Shot` with Detail 2 = On Net / Missed, or `Goal` | ✓ train and apply on unblocked shots only |
| `shot_group` | `shotType` | Detail 1 | ≈ see §3 |
| `skaters_for`, `skaters_against` | `home/awaySkatersOnIce` (goalie excluded, 6 = goalie pulled) | `Home/Away_Team_Skaters` (same coding, 3–6) | ✓ |
| `empty_net_against` | defined as defending skaters = 6 | same rule | ✓ (MoneyPuck's own `shotOnEmptyNet` disagrees with the skater counts on 45 of 2,999 shots, so it is not used) |
| `score_diff` (shooter's view) | `homeTeamGoals`, `awayTeamGoals` (before the shot) | goals columns (the score on a `Goal` row is before that goal) | ✓ |
| `period`, `t_game`, `is_home` | `period`, `time`, `isHomeTeam` | period, clock, home team | ✓ |
| `time_since_faceoff` | `timeSinceFaceoff` | last `Faceoff Win` in the period | ✓ |
| `shooter_toi` | `shooterTimeOnIce` | shooter's current shift (Shifts file) | ✓ (12 BDC shots missing) |
| `shoot_toi_mean/min/max`, `def_toi_mean/min/max` | `shootingTeam…TimeOnIce`, `defendingTeam…TimeOnIce` | skaters on the ice from Shifts (goalies excluded) | ✓ |
| `toi_min_diff` | = `timeDifferenceSinceChange` | shooting min − defending min | ✓ |
| `rebound` | `shotRebound` = previous event is SHOT or MISS by the same team within 3 s | same rule on the **common event stream** (§4) | ≈ |
| `rebound_angle_change` | \|θ − `lastEventShotAngle`\| for rebounds | same | ✓ (the file's `shotAnglePlusRebound` is also non-zero for 554 non-rebounds, so it is recomputed) |
| forwards / defence on ice, TOI by position | `…OfForwards`, `…OfDefencemen`, `shootingTeamForwardsOnIce` … | no positions in BDC | ✗ |
| shooter handedness, `offWing`, shooter position | `shooterLeftRight`, `offWing`, `playerPositionThatDidEvent` | not provided | ✗ |
| shooter / goalie identity | `shooterPlayerId`, `goalieIdForShot` | anonymised | ✗ |
| penalty time left | `home/awayPenalty1TimeLeft`, `…Length` | only `Penalty Taken` events, no length | ✗ (skater counts carry most of it) |
| last event type, time and distance since last event, speed, `shotRush` | `lastEventCategory`, `timeSinceLastEvent`, `distanceFromLastEvent`, `speedFromLastEvent`, `shotRush` | the two sources log different events (§4) | ✗ do not use |
| traffic, one-timer | not in MoneyPuck | Detail 3, Detail 4 (2025 files t/f, 2026 files 1/0) | BDC only |
| playoff game | `isPlayoffGame` | not provided | ✗ (train on regular season) |

## 3. Shot types

| Group | MoneyPuck | BDC | Share of unblocked shots, NHL (2,999) / BDC (1,145) |
|---|---|---|---|
| `wrist_snap` | WRIST, SNAP, BACK, blank | Wristshot, Snapshot, Fan, Poke | 78% / 82% |
| `slap` | SLAP | Slapshot | 12.5% / 10.7% |
| `tip` | TIP, DEFL | Deflection, Bat | 8.5% / 7.4% |
| `wrap` | WRAP | Wrap Around | 0.8% / 0.3% |

Wrist and snap shots are pooled on purpose. NHL scorers log 2.4 wrist shots per snap shot, while Stathletes logs 1.3, so the line between them is drawn differently. BDC also has no backhand label.

## 4. "Previous event" features

The NHL logs hits, giveaways and stoppages but no passes. Stathletes logs passes, puck recoveries, zone entries and dump-ins but no hits or giveaways. In MoneyPuck, 27% of shots follow a hit or giveaway, an event type that does not exist in BDC. Anything built on "the previous event" therefore means something different in each source.

- **Kept:** `rebound`, computed on the common stream (shots, goals, faceoffs, takeaways, penalties). Shot attempts are logged in both sources, so this is close to identical.
- **Dropped:** last-event type, time, distance and speed. `shotRush` is set on only 6 of 2,999 MoneyPuck shots, so it is unusable anyway.

## 5. Domain shift to check before trusting the model on BDC

| | MoneyPuck 2024–25 (2,999-shot sample) | BDC (1,145 unblocked shots) |
|---|---|---|
| Goal rate, unblocked shots | 7.6% | 6.1% (70 goals) |
| Distance quartiles (ft) | 17 / 32 / 48 | 17 / 33 / 51 |
| Median goal distance (ft) | 20.6 | 11.5 |
| Rebounds (share of unblocked shots) | 7.2% | 12.2% |

The BDC league is not named in the README. Overall shot locations look alike, but BDC goals come from much closer. Planned check: compare the model's predicted goals with the 70 actual goals (calibration-in-the-large, plus a binned calibration plot with game-bootstrap intervals). If the model is off, refit only the intercept on BDC shots, cross-fitted by game.

## 6. Getting the MoneyPuck data

The shots files are at <https://moneypuck.com/data.htm> (`shots_2024.zip` etc., about 20 MB per season). Download them in a browser into `data/raw/moneypuck/`, which is git-ignored. Then:

```
python code/xg_features.py moneypuck data/raw/moneypuck/shots_2024.zip out/xg_mp_2024.csv
python code/xg_features.py bdc data/raw out/xg_bdc_shots.csv
```

The MoneyPuck output keeps MoneyPuck's own `xGoal` as `mp_xGoal`, a ready-made benchmark for our model.

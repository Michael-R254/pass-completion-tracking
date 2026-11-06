# Pass completion probability from tracking data

**Question:** at the moment a pass is played, how likely is it to reach a teammate, given where all 22 players are and how they're moving?

**Answer, in one line:** an XGBoost model using start-frame tracking features cuts held-out log loss by 11-14% compared with a pass-length baseline on every one of three unseen matches (AUC 0.73-0.76 vs 0.55-0.57), and it's well calibrated. The biggest single improvement came from *not* guessing who the pass was for: the model scores the pass toward every teammate, weighted by how likely each one is the intended receiver.

Every number below comes from a file in `results/`, written by `scripts/run_pipeline.py`.

## Data
[Metrica Sports sample data](https://github.com/metrica-sports/sample-data): 3 anonymised matches, 25 Hz tracking of all players and the ball, plus synchronised event data. Thanks to Metrica Sports for making it public. Games 1-2 are CSV; game 3 is EPTS FIFA tracking with JSON events. Tracking is loaded with kloppy and events with pandas or kloppy, then all three go into one schema (`src/passmodel/load.py`).

Coordinates are converted to metres (105 × 68, origin at the centre spot) and rotated so the passing team always attacks left to right in both halves (`src/passmodel/clean.py`).

Data issues found and handled (evidence in `results/` and the tests):
- The CSV events (games 1-2) use the opposite y axis to the tracking. After flipping, the event start sits a median 0.35 m from the tracked ball (it was 35 m before).
- **Game 3, second half: Player6 and Player13 (same team) have each other's IDs in the events vs the tracking**, in all 144 of their second-half events and none in the first half. The tracking IDs are swapped to match the events. We can't tell which source was wrong, so player-level attribution for these two players in that half is uncertain.
- 14 of 3,645 attempted passes (0.4%) were dropped: 1 with no tracked ball at the start frame, and 13 where the named passer was more than 5 m from the ball (`results/step1_pass_drops.csv`).

## Label: what counts as a pass (decisions D1, D2)
Metrica has no single "pass outcome" column, so the label is a rule (`clean.label_passes`, counts in `results/d1_label_counts.csv`):
- **Completed** = a `PASS` event.
- **Failed** = `BALL LOST`, plus `BALL OUT` when a player in possession put it out (but not right after a shot, clearance, tackle or another ball loss, so one loose ball isn't counted twice).
- **Excluded** from both, applied to every event type alike: theft, forced, clearance, offside, end of half, woodwork, hand ball, referee hit.

That gives 2,878 completed, 767 failed and 181 excluded events (79.0% completion). The model uses **open play only**. The 289 set-piece passes (7.9%) are kept and flagged (`results/d2_set_piece_counts.csv`).

| Match | Passes | Completion | Open-play passes | Open-play completion |
|---|---|---|---|---|
| 1 | 1,049 | 75.4% | 952 | 75.0% |
| 2 | 1,176 | 81.7% | 1,083 | 81.8% |
| 3 | 1,406 | 79.6% | 1,309 | 79.2% |
| All | 3,631 | 79.1% | 3,344 | 78.9% |

## The two leakage traps, and how they're avoided
1. **The receiver leaks the label.** Completed passes have a receiver; failed ones don't, and their recorded end point is where the ball was cut out. So no feature uses the actual receiver or the end location. Instead, an **intended target** is chosen the same way for both outcomes (D3). A logistic regression scores every teammate from start-frame information (distance, alignment with the passer's run, forward distance, how open they are, their run), and the top-scoring teammate is the target. The target model is fitted **inside each fold and never on the match it scores** (cross-fitting). Its top choice is the actual receiver on 54-56% of completed passes in every match (`results/d3_target_agreement.csv`). Because that top choice is wrong almost half the time, the headline model doesn't rely on it: it uses all ten teammates, weighted by this model's probabilities (see [One guessed target, or all of them?](#one-guessed-target-or-all-of-them-d5)).
2. **Time leaks too.** Every feature uses only the pass start frame. Velocities are backward differences over 0.2 s, capped at 12 m/s. A test checks that no feature changes when future frames are altered.

Features (`src/passmodel/features.py`): pass geometry, location, pressure on the passer, space around the target and how fast the nearest defender is closing, opponents in the passing lane, player speeds, and cross/header flags. All are computed with NumPy broadcasting over passes × players, with no loops.

## Split
**Leave-one-match-out**: train on 2 matches, test on the third, 3 folds. Passes within a match are correlated, so a random split would leak. Hyperparameters are fixed defaults chosen before seeing any result (D4: depth 3, learning rate 0.05, 300 trees). As a sensitivity check, an inner CV grid is run on the single-target model using only the training matches.

## Results (held-out match per fold)
`results/metrics_per_fold.csv`. Lower log loss / Brier is better; higher AUC is better.

| Model | Log loss (1 / 2 / 3, mean) | ROC-AUC (1 / 2 / 3, mean) | Brier (1 / 2 / 3, mean) |
|---|---|---|---|
| Constant (training base rate) | 0.571 / 0.480 / 0.511, **0.521** | 0.50 | 0.190 / 0.151 / 0.165, **0.169** |
| Logistic, pass length only | 0.566 / 0.479 / 0.509, **0.518** | 0.572 / 0.553 / 0.554, **0.560** | 0.189 / 0.150 / 0.164, **0.168** |
| XGBoost, single intended target | 0.520 / 0.419 / 0.454, **0.464** | 0.727 / 0.748 / 0.741, **0.739** | 0.168 / 0.133 / 0.146, **0.149** |
| XGBoost, single target, inner-CV tuned (sensitivity only) | 0.511 / 0.422 / 0.446, **0.459** | 0.727 / 0.744 / 0.756, **0.742** | 0.168 / 0.132 / 0.144, **0.148** |
| **XGBoost, all targets weighted (headline)** | 0.504 / 0.412 / 0.450, **0.455** | 0.730 / 0.762 / 0.746, **0.746** | 0.164 / 0.130 / 0.145, **0.147** |

- **The headline model beats the baseline on every match.** Log loss vs the logistic baseline, with 95% paired bootstrap CIs (`results/step5_bootstrap_ci.csv`): −0.063 [−0.088, −0.039], −0.067 [−0.088, −0.047], −0.059 [−0.081, −0.042]. These CIs resample passes within a match, so they show sampling noise, not match-to-match variation. The per-fold spread shows the latter.
- **Pass length alone is nearly useless**: it barely beats the constant model (the CI includes zero on every match).
- **Tuning doesn't matter much.** On the single-target model, inner CV chose depth 2, learning rate 0.03, 200 trees in every fold (`results/d4_inner_cv.csv`), and is within 0.01 log loss of the defaults on each match. All models use the pre-registered defaults.

### One guessed target, or all of them? (D5)
The first version of the model (the single-target row above) bet on one intended target per pass: the teammate the target model rated most likely. Its biggest mistakes were mostly that guess being wrong. Of its 10 riskiest completed passes, the guessed target was the real receiver on only 2. It also gave its 10 safest failures an average of 98%.

So the headline model stops betting. Every pass is scored toward each of the 10 teammates, and the results are averaged using the target model's probabilities:

> P(complete) = Σ over teammates j of P(j is the intended target) × P(complete | pass aimed at j)

Training works the same way: every (pass, teammate) pair is a training row, weighted by P(j is the target), so each pass still counts once in total (`features.build_target_mixture_features`, `models.run_lomo_target_mixture`). The target probabilities come from the same cross-fitted target model, so this adds no leakage. The method and the rule for keeping it (beat the single-target model on held-out log loss) were fixed before it was run, and only this one variant was tried.

- **It's better on all three matches**: log loss vs the single-target model −0.016 [−0.030, −0.003], −0.007 [−0.017, 0.003] and −0.004 [−0.014, 0.006] (`results/target_mixture_ci.csv`). The direction is consistent, but only match 1's CI excludes zero, so the gain is modest.
- **It fixes the failure mode it was aimed at** (`results/target_mixture_on_surprises.csv`): on the single-target model's 10 safest failures, the average prediction drops from 98.2% to 93.7%. On its 10 riskiest completed passes it rises from 15.8% to 20.7%.
- It costs about 11 s per pipeline run, because there are 10× as many feature rows (`results/benchmarks.csv`).

Everything below (calibration, SHAP, ablation, the pass plots and tables) is for this headline model.

### Calibration
![Calibration](results/figures/calibration.png)

Pooled held-out passes in 10 equal-count bins (`results/step5_calibration.csv`). **When the model says about 73%, 69% of passes come off.** The top end is close to exact: 94% predicted → 93% observed, and 97% → 98%. The bottom bin is slightly underconfident (43% predicted → 47% observed).

### What drives it (SHAP)
![SHAP](results/figures/shap_summary.png)

Mean |SHAP| on held-out passes (`results/step5_shap_importance.csv`). Each pass is explained as aimed at its most likely target, because a weighted sum over 10 teammates has no single feature row to attach SHAP values to. The top three, in football terms:
1. **Where the pass starts** (`start_x`): passes from deep in your own half are the riskiest; passes from midfield are the safest.
2. **Pressure on the passer** (`passer_nearest_opp`): with an opponent close by, passes go astray more often.
3. **Space around the target** (`target_nearest_opp`): the more room the receiver has, the more likely the pass arrives.

### Does each feature group earn its place? (ablation)
The headline model refitted without each group. Δ = increase in held-out log loss, so positive means the group helps (`results/step5_ablation.csv`, CIs in `results/step5_ablation_ci.csv`).

| Without | Δ match 1 | Δ match 2 | Δ match 3 | Mean | Helps on |
|---|---|---|---|---|---|
| location | 0.028 | 0.014 | 0.005 | 0.016 | 3/3 |
| target (space, closing speed, confidence) | 0.006 | 0.005 | 0.009 | 0.007 | 3/3 |
| event flags | 0.011 | 0.008 | −0.001 | 0.006 | 2/3 |
| geometry | 0.007 | 0.003 | 0.001 | 0.004 | 3/3 |
| speed | 0.004 | 0.005 | 0.003 | 0.004 | 3/3 |
| pressure | −0.002 | 0.008 | −0.001 | 0.002 | 1/3 |
| lane | 0.002 | −0.000 | −0.002 | 0.000 | 1/3 |

Location and the target features earn their place on every match. **Pressure-count and lane features don't**: each helps on only one match, and most of their CIs include zero. With three matches this ranking is fragile and shouldn't be over-read. The headline model keeps every group, because dropping groups based on these test-match scores would be tuning on the test set. (`passer_nearest_opp`, the second-ranked SHAP feature, is in the pressure group. Removing the group costs little because location and the other features largely cover it.)

## A pass, shown as football
![Two passes](results/figures/pass_stories.png)

Blue = passing team (attacking left to right), red = defenders, arrows = where each player will be in 1 s, thick ring and solid line = the most likely target, dashed arrow = where the ball actually went (from events; context only, never a model input).
- **Left:** the riskiest completed pass in the opponents' half where the most likely target was the real receiver. The model gave it **16%** and it came off: several defenders are close to the passer and one is on the line of the pass.
- **Right:** the failure the model was most sure about (**97%**). The most likely target is a short, free pass back and wide, but the ball went forward instead. Spreading the bet across all teammates lowers the model's confidence on passes like this, but it can't fully fix a target model that rates the wrong option highest.

### 10 riskiest completed passes (`results/step6_riskiest_completed.csv`)
| Match | ~Min | Passer | Most likely target | Actual receiver | P(complete) |
|---|---|---|---|---|---|
| 3 | 95 | Player17 | Player11 | Player10 | 10% |
| 3 | 88 | Player14 | Player5 | Player6 | 11% |
| 3 | 87 | Player32 | Player34 | Player31 | 11% |
| 2 | 74 | Player18 | Player26 | Player24 | 12% |
| 3 | 61 | Player5 | Player1 | Player1 | 16% |
| 3 | 59 | Player24 | Player28 | Player26 | 17% |
| 1 | 36 | Player7 | Player9 | Player8 | 19% |
| 3 | 53 | Player21 | Player26 | Player24 | 25% |
| 3 | 8 | Player28 | Player19 | Player23 | 25% |
| 3 | 33 | Player18 | Player22 | Player26 | 25% |

Minutes are minutes of play from kick-off (frame / 1,500), not the match clock. **The most likely target was the real receiver on only 1 of these 10**, against 54-56% overall. Even averaged over all teammates, the model's biggest surprises are passes that went somewhere the target model rated unlikely. The 10 safest failures (`results/step6_safest_failed.csv`, predicted 94.8-97.4%) all have nobody in the lane to the most likely target.

### Completion above expected, with a strong caveat
`results/step6_completion_above_expected.csv` gives each player's completion rate minus the model's expectation (players with at least 40 passes in a match). The mean per match is −3.6 pts (match 1), +4.8 (match 2) and +0.1 (match 3). That's a **match effect**, not player skill: match 2's base rate is higher and the model never saw it while training. With 40-90 passes per player, only 5 of 44 player-matches are more than 2 standard errors from zero. Not a player ranking.

## Stretch: does pitch control help? No, it isn't kept
Pitch control asks, for a point on the pitch, "if the ball were played there now, which team would win it?" Each player needs a reaction time (0.7 s at their current velocity) and then runs at up to 5 m/s. Once the ball arrives (at 15 m/s), each player who has got there wins it at a fixed rate, out of whatever chance is still unclaimed. This is the Spearman (2018) model with the parameters from [Laurie Shaw's Friends of Tracking tutorial](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking), credited and reimplemented vectorised over all passes (`features.pitch_control`). The feature is the passing team's control at the most likely target's position at the pass frame. It was tested on the single-target model, before the target mixture existed.

| Model | Log loss (1 / 2 / 3, mean) | ROC-AUC mean |
|---|---|---|
| Logistic, pass length only | 0.566 / 0.479 / 0.509, **0.518** | 0.560 |
| Logistic, pitch control only | 0.562 / 0.480 / 0.495, **0.512** | 0.625 |
| XGBoost, single target | 0.520 / 0.419 / 0.454, **0.464** | 0.739 |
| XGBoost, single target + pitch control | 0.516 / 0.421 / 0.460, **0.466** | 0.734 |

(`results/metrics_per_fold.csv`.) **On its own, pitch control is a better single feature than pass length** (AUC 0.62 vs 0.56). **On top of the existing features it adds nothing**: the change in log loss vs the single-target model is −0.004 [−0.011, 0.004], +0.003 [−0.002, 0.008] and +0.006 [−0.000, 0.013] on matches 1-3. It helps on one match and hurts on two, and every CI includes zero (`results/stretch_pitch_control_ci.csv`). The likely reason is that it is mostly a smarter version of "how much space does the target have", which the model already has (`target_nearest_opp`, rank correlation 0.90 with pitch control). Following the plan's rule, it was **not** kept. It roughly doubles the feature-build time, which is still under 2 s (`results/benchmarks.csv`).

## Scaling
Tracking is the biggest data in football: here, **9,727,275 position rows** for 3 matches (about 3.2M per match). The pipeline is built as if it had to run on a full season:
- tracking is stored as long-format Parquet, one file per match, processed one match at a time, so memory doesn't grow with the number of matches;
- velocities are computed for every frame at once with grouped NumPy differences;
- features read **only the pass start frames** from Parquet: 167,393 rows, 1.7% of the table;
- every stage logs wall time and peak memory to `results/benchmarks.csv`.

Measured on a Windows laptop during a full `--rebuild` run (6 Nov 2026), then projected to 380 matches (`results/step7_scaling.csv`; projection = measured × 380 / 3):

| Stage | Rows out (3 matches) | Wall time (3 matches) | Peak traced memory | Projected season |
|---|---|---|---|---|
| Load raw tracking → long format (kloppy) | 9,727,275 | 374.3 s | 2,315 MB | 790 min |
| Clean (ID fixes, metres, flip, velocities) | 9,727,275 | 1.0 s | 364 MB | 2.1 min |
| Write Parquet | | 2.9 s | 148 MB | 6.1 min |
| Read tracking at pass frames only | 167,393 | 0.2 s | 0.6 MB | 0.5 min |
| Build pass table | 3,645 | 0.4 s | 4.2 MB | 0.9 min |
| Build features incl. pitch control (all folds) | 10,893 | 1.6 s | 21 MB | 3.4 min |
| Build target-mixture features (all folds × 10 teammates) | 108,930 | 10.9 s | 54 MB | 23.0 min |
| **All stages** | | **391.3 s** | **2,315 MB** | **826 min** |

- **A full season is an estimated ~14 hours on one laptop core, and 96% of that is parsing the raw provider files** (790 of 826 min). Everything after the data is in Parquet, from cleaning to features, projects to about 36 minutes per season, most of it the 10×-larger target-mixture table. Matches are independent, so loading parallelises across cores or machines, and in a club it happens once per match as data arrives.
- **Memory stays at the single-match peak.** Peak traced memory is 2.3 GB, and the operating system's process peak is 7.0 GB, reached while kloppy parses game 3's EPTS file. The traced figure comes from `tracemalloc`, which can't see memory allocated inside Arrow's C++ pool, so Parquet stages are lower bounds. The process figure covers everything.
- **Storage:** about 57 MB of Parquet per match (`results/tracking_files.csv`), so about 21.6 GB per season.
- This is an estimate: it assumes season matches look like these three.

## Limitations
- **Three matches is a tiny sample.** Three test folds of 952-1,309 passes. The per-fold spread is a better guide to uncertainty than any single number, and results could change with more matches.
- **The target model is still the weak link.** Weighting all teammates removes the need to bet on one receiver, but the weights are only as good as the target model, whose top choice matches the real receiver on about 55% of completed passes. Better target probabilities, for example from passer body orientation or other start-frame cues, are the most promising next step.
- **Anonymised teams and players**, and player IDs may not refer to the same person across matches. Player-level results are not meaningful here.
- **Event-tracking sync error:** event frames are hand-tagged, so the start frame can be a few frames off. 13 passes were dropped where the passer was more than 5 m from the ball.
- **Match context is missing:** score, game state and team strength aren't features, which shows up as match-level miscalibration (match 2 completes more than predicted).
- **Pitch control is a simplified Spearman model:** no offside check, the goalkeeper is guessed as the defender nearest their own goal, and the parameters are Shaw's defaults, not fitted to this data.

## How to reproduce
```powershell
uv sync
uv run python scripts/download_data.py
uv run python scripts/run_pipeline.py --rebuild
uv run pytest
```
Tested from a fresh clone on 6 Nov 2026 (new environment, fresh download, `--rebuild`): every results table reproduced byte for byte. That test predates the target-mixture model; the full pipeline has been re-run since.

`--rebuild` rebuilds the tracking Parquet and rewrites `results/benchmarks.csv`. Without it, existing Parquet files are reused and timings go to `results/logs/`. Notebooks `notebooks/00`-`08` show each step's results.

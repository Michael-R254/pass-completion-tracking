# Pass completion probability from tracking data — weekend plan

**Question:** at the moment a pass is played, how likely is it to reach a teammate, given where all 22 players are and how they're moving?

**Why it matters for this application:** the Man City 1st Team Data Scientist role asks for experience with *spatio-temporal data* and for explaining technical findings to *non-technical audiences*. This project addresses both. Applications close **16 Nov 2026**, so the target is a finished repo by Sunday 8 Nov, leaving a week for the cover letter.

---

## The analysis: where this project can go wrong

These are the four places where a quick version of this project would give misleading results. Getting them right is what makes the project worth talking about at interview.

1. **The label isn't one column.** In Metrica events, `PASS` rows are completed passes. Failed passes appear under a different type (`BALL LOST`, with subtypes such as interceptions). We have to audit the event types and subtypes, then write down a rule for "attempted pass" before modelling.
2. **The receiver leaks the label.** For completed passes we know who received the ball. For failed passes we don't, and the recorded end location is *where it was cut out*, not where it was aimed. Features built from "the receiver" or "the end point" would therefore partly encode the outcome. Fix: define an **intended target** using one rule applied identically to both outcomes (for example, the teammate best aligned with the pass direction, weighted by distance). Check how often the rule picks the actual receiver on completed passes, and report that number.
3. **Time leaks too.** Every feature must be computable at the pass frame. Speeds use *backward-looking* differences only (frames before the pass), never centred windows that peek ahead.
4. **Three matches is a tiny sample.** Split by match. Run **leave-one-match-out** (3 folds), which includes the "train on 2, test on 1" split, and report each fold, not just the average. Tune hyperparameters only *inside* the training matches. Compare everything to a constant base-rate model, because roughly 80% of passes complete and a model has to beat that.

---

## Checklists

### 0. Setup (Fri evening or Sat first thing, ~1 hr)
- [x] Create folder structure, `pyproject.toml`, `.gitignore`, README stub and module stubs (done 5 Nov)
- [x] `git init`, first commit, create the private GitHub repo and push (done 5 Nov)
- [x] Install uv if needed, then `uv sync` (dependencies are already listed in `pyproject.toml`); check `uv run python -c "import kloppy, mplsoccer, xgboost"` works
- [x] `profiling.py`: decorator that logs wall time, peak memory and rows in/out per stage to `results/benchmarks.csv`
- [x] `scripts/download_data.py` fetches Metrica sample data (data/ is git-ignored, never committed)
- [x] Confirm kloppy loads tracking for games 1, 2 and 3; confirm how events load (pandas for game 1/2 CSV, kloppy for game 3 JSON)

### 1. Data audit and unified dataset (Sat morning, ~3 hrs)
- [x] Load events for all 3 games into **one schema**: match_id, period, start_frame, end_frame, team, from_player, to_player, type, subtype, start_xy, end_xy
- [x] Load tracking for all 3 games into one **long-format** schema: match_id, frame, period, player_id, team, x, y (ball as its own object)
- [x] Write tracking to Parquet **partitioned by match**, processing one match at a time so memory stays flat; record row counts and file sizes
- [x] Compute velocities for **every frame** in one vectorised pass per match (grouped backward differences), stored alongside positions
- [x] Convert to metres (105 × 68 or the pitch size given in the metadata) and **flip coordinates so each team always attacks the same direction** in both halves
- [x] Tabulate every event type and subtype; agree the **attempted pass** definition with the user (decision D1)
- [x] Decide on set pieces: exclude, or keep with a flag (decision D2)
- [x] Drop passes whose start frame has missing ball or player positions; log how many are dropped and why
- [x] Sanity plot: 3 random passes on a pitch, to check frames and coordinates line up
- [x] Output: `data/processed/passes.parquet` with one row per attempted pass and a `completed` label
- [x] Record counts per match and completion rate per match in the README notes

### 2. Feature engineering (Sat afternoon, ~3–4 hrs)
- [x] Implement intended-target rule (decision D3); report agreement with the actual receiver on completed passes
- [x] Pass geometry: distance, angle, forward/backward flag, progress toward goal
- [x] Location: start x, y; target x, y; zone (thirds / half-spaces)
- [x] Velocities (computed in step 1): backward finite difference over ~5 frames (0.2 s), light smoothing; cap physically implausible speeds (>12 m/s)
- [x] Build features by reading **only pass start frames** from Parquet, then computing passer/target/defender relationships with NumPy broadcasting (passes × players arrays), with no loops
- [x] Passer pressure: distance to nearest opponent, number of opponents within 3 m / 5 m
- [x] Target openness: target's distance to nearest defender; defenders closing on the target (relative velocity)
- [x] Lane congestion: number of opponents within X m of the passer→target line segment (only between passer and target)
- [x] Speeds: passer speed, target speed, nearest defender speed
- [x] Unit tests for geometry helpers (distance to segment, angle, flip direction)
- [x] Leakage check: every feature uses only frames ≤ start_frame
- [x] Output: `data/processed/features.parquet`

### 3. Baseline (Sun morning, ~30 min)
- [x] Constant model (training base rate): log loss
- [x] Logistic regression on pass distance only, leave-one-match-out

### 4. Model (Sun morning, ~2 hrs)
- [x] XGBoost on full feature set, leave-one-match-out
- [x] Hyperparameters: small grid tuned with inner CV grouped by match **within training matches only**, or fixed sensible defaults (decision D4)
- [x] Fixed random seeds; save fold predictions to `results/`

### 5. Honest evaluation (Sun midday, ~2 hrs)
- [x] Table: log loss, ROC-AUC, Brier score per fold and mean, for constant / logistic / XGBoost
- [x] Bootstrap confidence intervals on the test fold metrics (resample passes)
- [x] Calibration plot (reliability curve, ~10 quantile bins, with bin counts shown)
- [x] SHAP summary plot + one-sentence football reading of the top 3 features
- [x] Ablation: does each feature group (geometry / pressure / lane / speed) earn its place?

### 6. Show it as football (Sun afternoon, ~2 hrs)
- [x] mplsoccer plot of one real pass: all players, velocity arrows, pass line, intended target highlighted, predicted probability in the title
- [x] Table of the 10 riskiest completed passes (lowest predicted probability that came off) and the 10 "safest" failures
- [x] Optional: per-player "completion above expected" with an explicit caveat about sample size

### 7. README and polish (Sun evening, ~1.5 hrs)
- [x] Sections: question, data and source acknowledgement, label definition, split, results table, pitch plot, calibration plot, SHAP, **scaling**, limitations, how to reproduce
- [x] Scaling section: table from `benchmarks.csv` (rows, time, peak memory per stage) and a projected cost for a 380-match season, stated as an estimate
- [x] Limitations: 3 matches, anonymised teams, intended-target is an approximation, event and tracking sync error, no pitch control (unless stretch is done)
- [x] `make all` or a single `run_pipeline.py` reproduces everything from a clean clone
- [x] Fresh-clone test: delete the env, reinstall, run end to end

### Stretch: pitch control (only if steps 0–7 are done)
- [x] Implement Spearman-style pitch control following Laurie Shaw's Friends of Tracking tutorial (credit it)
- [x] Feature: attacking team's control at the intended target location at the pass frame
- [x] Re-run leave-one-match-out; keep it **only if** it improves held-out log loss beyond fold-to-fold noise. Report the result either way.

---

## Interview talking points this produces
- "I split by match because passes within a match are correlated; a random split would leak."
- "Failed passes don't have a receiver, so I defined an intended target the same way for both outcomes. Otherwise the receiver feature leaks the label."
- "A '70%' from my model completes about X% of the time" (from the calibration plot).
- "Pitch control didn't beat the simpler features on held-out matches (better on 1 of 3, every CI includes zero), so I dropped it."
- "The pipeline processes ~10M tracking rows in X minutes at Y GB peak memory, one match at a time, so a full season is about Z hours on a laptop" (fill in from `benchmarks.csv`).

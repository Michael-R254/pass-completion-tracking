# CLAUDE.md — Pass completion probability from tracking data

## What this project is
A weekend portfolio project for an application to Manchester City's **1st Team Data Scientist** role (closes 16 Nov 2026). The goal is a pass completion probability model built from Metrica Sports' public tracking and event data, evaluated honestly and presented as football.

The full step-by-step plan and checklists are in `PLAN.md`. **Tick items off in `PLAN.md` as they are completed.**

## Who you're working with
- The user is a data scientist strong in feature engineering, XGBoost and held-out evaluation, but new to **spatio-temporal / tracking data**. That's the gap this project fills.
- **Claude writes the code; the user reviews and runs it on their own machine.** The user doesn't write code during the weekend, so the user must still be able to **explain every line at interview**:
  - Prefer clear, readable code over clever code. Docstrings on every public function say what it does and *why*.
  - After finishing each checklist section, give a short walkthrough in chat: what each new function does, the key design choice, and one question an interviewer might ask about it, with the answer.
  - When introducing a tracking-specific idea (coordinate flipping, velocity estimation, distance to a line segment, pitch control), explain it briefly the first time.
- **Running locally:** the user runs commands in their own terminal. Give exact commands to run (`uv run ...`, `pytest`), say what output to expect, and ask the user to paste back errors or numbers rather than assuming success.
- **The user's CV strength to showcase: working with large datasets.** Tracking data is the largest data in football (25 Hz × 23 objects ≈ 3M+ position rows per match). Build the pipeline as if it had to run on a full season, not just 3 games:
  - Store tracking in **long format Parquet, partitioned by match**. Never load all matches into memory when one at a time will do.
  - **Vectorise.** Compute velocities for every frame at once with grouped NumPy/pandas operations, and compute per-pass features with array broadcasting across all 22 players. No Python loops over players or frames.
  - Read only the frames needed (pass start frames plus a short backward window) rather than the full table when building features.
  - **Measure, don't claim.** Log wall time and peak memory for each pipeline stage to `results/benchmarks.csv`, and project the cost to a 380-match season in the README.
  - DuckDB is allowed as an optional dependency for querying Parquet, if it makes a step simpler. Ask first.

## How to work with me
- Work in small steps: one checklist section at a time. At the end of each section, summarise what was done and show the key numbers or a plot before moving on.
- **Stop and ask me** before making any of these decisions (log the answer in the Decisions log below):
  - **D1** — what counts as an *attempted pass* and what counts as *failed* (after showing me the event type/subtype table)
  - **D2** — whether set pieces (free kicks, corners, throw-ins, goal kicks) are excluded or flagged
  - **D3** — the rule for choosing the *intended target* of a pass
  - **D4** — hyperparameter tuning approach (fixed defaults vs inner grouped CV)
  - Any time a data problem would drop more than ~5% of passes
  - Adding a new dependency beyond those listed below
  - Anything that changes the train/test split
- If something in the data looks off (sync errors, frames out of range, players missing), show me evidence rather than silently fixing it.
- Don't make up results. Every number in the README must come from a script output in `results/`.

## Non-negotiable rules (leakage and honesty)
1. **Split by match.** Leave-one-match-out over the 3 games. Never a random split across passes.
2. **No looking at the test match** when choosing features, hyperparameters or thresholds. Tuning happens only inside the training matches.
3. **Features use only frames ≤ the pass start frame.** Velocities use backward differences. No centred windows that see the future.
4. **No outcome-dependent features.** Don't use the actual receiver (`To` player) or the actual end location of failed passes to build features. Use the intended-target rule (D3), applied identically to completed and failed passes.
5. **Always compare against baselines:** constant base rate, then logistic regression on distance. Report per-fold results, not only the mean.
6. **Keep complexity only if it earns its place** on held-out log loss. Report negative results too.

## Data facts to verify (don't assume)
- Source: github.com/metrica-sports/sample-data. Acknowledge Metrica in the README. Data is anonymised.
- Games 1 and 2: CSV tracking (home and away files) + CSV events. Game 3: EPTS FIFA tracking (XML metadata + txt) + JSON events.
- kloppy `metrica.load_open_data(match_id=...)` loads **tracking** for all three. Events for games 1/2 are likely easiest with pandas; game 3 events via `kloppy.metrica.load_event`. Confirm against the installed kloppy version before writing loaders.
- Tracking is at 25 Hz. Metrica coordinates are normalised (0–1); convert to metres and flip so each team attacks one consistent direction across both halves.
- Events carry start/end frame numbers that link them to tracking frames. Spot-check the sync visually.
- Never commit raw data. `data/` is git-ignored; `scripts/download_data.py` fetches it.

## Repo structure
```
pass completion probability from tracking data/   (repo root)
├── CLAUDE.md
├── PLAN.md
├── README.md
├── .gitignore              # data/ ignored
├── pyproject.toml          # deps + src/passmodel as an installable package
├── scripts/
│   ├── download_data.py
│   └── run_pipeline.py     # reproduces everything end to end
├── src/passmodel/
│   ├── load.py             # unified event + tracking loaders
│   ├── clean.py            # units, direction flip, pass labelling
│   ├── features.py         # geometry, pressure, lane, velocity, target rule
│   ├── models.py           # baselines + XGBoost, LOMO CV
│   ├── evaluate.py         # metrics, bootstrap CIs, calibration, SHAP
│   ├── viz.py              # mplsoccer plots
│   ├── __init__.py
│   └── profiling.py        # timing + peak-memory decorator → results/benchmarks.csv
├── notebooks/              # exploration only; not the source of truth
├── tests/                  # pytest for geometry helpers and leakage checks
├── data/                   # git-ignored (raw/, processed/)
└── results/                # metrics CSVs, figures/ (committed)
```
The folders and empty module stubs already exist. Fill in the stubs; don't reorganise the layout without asking.

## Conventions
- **The user is on Windows** and runs commands in PowerShell. Give PowerShell-compatible commands (no `export`, `&&` chains are fine in PowerShell 7 but avoid them in 5.1; one command per line is safest).
- Python 3.11+, managed with **uv**. `pyproject.toml` already exists: use `uv sync` to install, `uv add <pkg>` for new dependencies (ask first), `uv run ...` to run. If uv isn't installed, give the install command first. Dependencies: kloppy, mplsoccer, pandas, numpy, scikit-learn, xgboost, shap, matplotlib, pyarrow, pytest.
- Paths go through `pathlib` relative to the repo root, so it runs on macOS, Linux or Windows.
- Logic lives in `src/`; notebooks only call it. Fixed random seeds everywhere.
- **One notebook per PLAN step** (`notebooks/00_setup_check.ipynb`, `01_...`, ...) that imports from `passmodel` and shows that step's results (counts, tables, plots) for the user to inspect in Jupyter. Create it as part of finishing each step.
- Coordinates in metres, origin at the pitch centre or bottom-left (pick one and document it in `clean.py`).
- Plots: consistent colours (attacking team vs defending team), titles that a coach would understand without explanation.
- Small, frequent git commits with clear messages.

## Definition of done
- `uv run python scripts/run_pipeline.py` reproduces all results from a fresh clone.
- README has: question, data, label definition, split, results table (constant / logistic / XGBoost, per fold), calibration plot, SHAP plot, one annotated pitch plot, riskiest completed passes, a **"Scaling" section** (rows processed, time and memory per stage, projected season cost), limitations (3 matches is a tiny sample, said plainly).
- All `PLAN.md` boxes for steps 0–7 ticked.

## Decisions log
<!-- Claude: record each decision here with date, the options shown, and the user's choice. -->
| ID | Decision | Choice | Date |
|----|----------|--------|------|
| D1 | Attempted / failed pass definition | **Completed** = PASS. **Failed** = BALL LOST, plus BALL OUT (a player in possession puts the ball out) if it has a From player and the previous event isn't a SHOT, CLEARANCE, tackle/THEFT or BALL LOST (the last stops one loose ball counting twice). **Excluded** from all three types alike: subtypes with THEFT, FORCED, CLEARANCE, OFFSIDE, END HALF, WOODWORK, HAND BALL, REFEREE HIT. Flags `is_cross`, `is_header`, `is_set_piece`. Code: `clean.label_passes`; counts: `results/d1_label_counts.csv` (2,878 completed / 767 failed / 181 excluded, 79.0%) | 2026-11-05 |
| D2 | Set pieces | _pending_ | |
| D3 | Intended target rule | _pending_ | |
| D4 | Hyperparameter tuning | _pending_ | |

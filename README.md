# Pass completion probability from tracking data

> **Work in progress.** This README is filled in at step 7 of `PLAN.md`, using only numbers produced by scripts in `results/`.

**Question:** at the moment a pass is played, how likely is it to reach a teammate, given where all 22 players are and how they're moving?

## Data
Metrica Sports sample data (3 anonymised matches, 25 Hz tracking + synchronised events): https://github.com/metrica-sports/sample-data. Thanks to Metrica Sports for making it public.

## How to reproduce
```bash
uv sync
uv run python scripts/download_data.py
uv run python scripts/run_pipeline.py
```

## Sections to come
- Label definition
- Train/test split (leave-one-match-out)
- Results: constant vs logistic vs XGBoost
- Calibration
- Feature importance (SHAP)
- A pass, shown as football
- Scaling
- Limitations

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

## Notes so far (step 1; numbers from `results/`)
Passes after labelling (D1), set-piece flag (D2) and data-quality drops (`results/step1_passes_per_match.csv`):

| Match | Passes | Completion | Open-play passes | Open-play completion |
|---|---|---|---|---|
| 1 | 1,049 | 75.4% | 952 | 75.0% |
| 2 | 1,176 | 81.7% | 1,083 | 81.8% |
| 3 | 1,406 | 79.6% | 1,309 | 79.2% |
| All | 3,631 | 79.1% | 3,344 | 78.9% |

14 of 3,645 attempted passes (0.4%) were dropped: 1 with no tracked ball at the start frame, 13 where the named passer was more than 5 m from the ball (`results/step1_pass_drops.csv`). 18 FORCED events (4 / 7 / 7 per match) were excluded by the label rule (`results/d1_label_counts.csv`).

Data issues found and handled:
- The CSV events (matches 1-2) use the opposite y axis to the tracking; flipped so the event start sits a median 0.35 m from the tracked ball (was 35 m).
- **Match 3, second half: Player6 and Player13 (same team) have each other's IDs in the events vs the tracking**, in all 144 of their second-half events and none in the first half. The tracking IDs are swapped to match the events. We can't tell which source was wrong, so player-level attribution for these two players in that half is uncertain. Team-level features are unaffected.

## Sections to come
- Label definition
- Train/test split (leave-one-match-out)
- Results: constant vs logistic vs XGBoost
- Calibration
- Feature importance (SHAP)
- A pass, shown as football
- Scaling
- Limitations

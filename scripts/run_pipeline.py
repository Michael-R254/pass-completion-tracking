"""Run every stage end to end and reproduce all results (PLAN.md step 7).

Built up step by step. Currently (PLAN step 1):
  1. tracking: one match at a time -> long format, ID fixes, metres, direction flip, velocities -> Parquet per match
  2. events -> D1/D2 pass labels -> results/d1_label_counts.csv, d2_set_piece_counts.csv
  3. passes: read tracking at pass start frames only -> quality checks -> drop -> data/processed/passes.parquet
     -> results/step1_pass_drops.csv, step1_passes_per_match.csv, figures/sanity_3_passes.png
  4. features (PLAN step 2): per leave-one-match-out fold, intended target (D3) + features
     -> data/processed/features.parquet, results/d3_target_agreement.csv
  5. models (PLAN steps 3-4): leave-one-match-out, open play
     -> results/predictions.csv (one row per model x test pass), results/metrics_per_fold.csv,
        results/d4_inner_cv.csv (sensitivity check: inner-CV grid scores and the setting chosen per fold)

Every run logs to results/logs/pipeline_<time>.log (git-ignored). On a crash the full
traceback and the process's peak memory go into that log before the error is re-raised.

Run:  uv run python scripts/run_pipeline.py            (skips matches whose tracking Parquet exists)
      uv run python scripts/run_pipeline.py --rebuild  (rebuilds tracking; needed for fresh benchmarks)
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files; no window needed
import pandas as pd

from passmodel.clean import (build_passes, clean_tracking, drop_bad_passes, home_attack_sign, label_counts,
                             label_passes, pass_summary, set_piece_counts, to_metres)
from passmodel.evaluate import fold_metrics
from passmodel.features import build_features, read_pass_frames, target_agreement
from passmodel.load import MATCH_IDS, load_events, load_tracking_long, read_frames, tracking_path, write_tracking
from passmodel.models import MODELS, inner_cv_sensitivity, run_lomo
from passmodel import profiling
from passmodel.profiling import BENCHMARK_CSV, process_peak_mb
from passmodel.viz import plot_sanity_passes

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PASSES_PATH = ROOT / "data" / "processed" / "passes.parquet"
FEATURES_PATH = ROOT / "data" / "processed" / "features.parquet"
log = logging.getLogger("pipeline")


def setup_logging() -> Path:
    """Log to the console and to a timestamped file in results/logs/."""
    path = RESULTS / "logs" / f"pipeline_{datetime.now():%Y%m%d-%H%M%S}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(path), logging.StreamHandler()])
    return path


def home_sign_path(match_id: int) -> Path:
    return tracking_path(match_id).parent / "home_attack_sign.csv"


def build_tracking(rebuild: bool) -> None:
    """Tracking Parquet per match, one match in memory at a time; logs sizes to results/tracking_files.csv.

    The per-period rotation (home_attack_sign) is saved next to each match's Parquet, because
    the passes table must apply the same rotation to the event coordinates.
    """
    files = []
    for m in MATCH_IDS:
        path = tracking_path(m)
        if path.exists() and not rebuild:
            log.info(f"tracking match {m}: exists, skipping ({path.stat().st_size / 1e6:.0f} MB)")
            continue
        raw = load_tracking_long(match_id=m)
        metres = raw.assign(x=to_metres(raw["x"], raw["y"])[0])  # sign is decided from unflipped metres
        sign = home_attack_sign(metres)
        tracking = clean_tracking(raw, match_id=m)
        del raw, metres
        write_tracking(tracking, match_id=m)
        sign.to_csv(home_sign_path(m))
        files.append({"match_id": m, "rows": len(tracking), "parquet_mb": round(path.stat().st_size / 1e6, 1),
                      "players_tracked": tracking.loc[tracking["team"] != "ball", "player_id"].nunique(),
                      "speed_capped_rows": tracking.attrs["n_speed_capped"]})
        log.info(f"tracking match {m}: {files[-1]}  process peak {process_peak_mb():.0f} MB")
        del tracking
    if files:
        pd.DataFrame(files).to_csv(RESULTS / "tracking_files.csv", index=False)


def build_pass_table(labelled: pd.DataFrame) -> pd.DataFrame:
    """passes.parquet: one row per attempted pass, reading only the tracking frames where passes start."""
    parts = []
    for m in MATCH_IDS:
        lm = labelled[labelled["match_id"] == m]
        sign = pd.read_csv(home_sign_path(m), index_col="period")["home_attack_sign"]
        starts = lm.loc[lm["pass_bucket"].isin(["completed", "failed"]), "start_frame"]
        frames = read_frames(match_id=m, frames=starts)
        parts.append(build_passes(lm, match_id=m, home_sign=sign, frames=frames))
    passes, drops = drop_bad_passes(pd.concat(parts, ignore_index=True))
    passes.to_parquet(PASSES_PATH, index=False)
    for name, table in [("step1_pass_drops", drops), ("step1_passes_per_match", pass_summary(passes))]:
        table.to_csv(RESULTS / f"{name}.csv")
        log.info(f"\n{name}\n{table.to_string()}")
    return passes


def main() -> None:
    rebuild = "--rebuild" in sys.argv
    if rebuild:  # a full rebuild starts a fresh benchmarks file, so every stage is timed in one run
        BENCHMARK_CSV.unlink(missing_ok=True)
    else:  # partial runs skip the tracking stages, so their timings go to the logs, not the committed file
        profiling.BENCHMARK_CSV = RESULTS / "logs" / "benchmarks_partial_run.csv"
    build_tracking(rebuild)

    events = pd.concat([load_events(m) for m in MATCH_IDS], ignore_index=True)
    labelled = label_passes(events)
    for name, table in [("d1_label_counts", label_counts(labelled)), ("d2_set_piece_counts", set_piece_counts(labelled))]:
        table.to_csv(RESULTS / f"{name}.csv")
        log.info(f"\n{name}\n{table.to_string()}")

    passes = build_pass_table(labelled)
    plot_sanity_passes(passes, read_frames, n=3, seed=0, path=RESULTS / "figures" / "sanity_3_passes.png")
    features = build_features(passes, read_pass_frames(passes))
    features.to_parquet(FEATURES_PATH, index=False)
    agreement = target_agreement(features)
    agreement.to_csv(RESULTS / "d3_target_agreement.csv")
    log.info(f"features: {len(features):,} rows (passes x folds)")
    log.info(f"\nd3_target_agreement\n{agreement.to_string()}")

    inner_scores, tuned_predictions = inner_cv_sensitivity(features)
    inner_scores.to_csv(RESULTS / "d4_inner_cv.csv", index=False)
    chosen = inner_scores[inner_scores["chosen"]].to_string(index=False)
    log.info(f"\nd4 inner CV, chosen setting per fold\n{chosen}")
    predictions = pd.concat([run_lomo(features, name, factory, cols) for name, (factory, cols) in MODELS.items()]
                            + [tuned_predictions], ignore_index=True)
    predictions.to_csv(RESULTS / "predictions.csv", index=False)
    metrics = fold_metrics(predictions)
    metrics.to_csv(RESULTS / "metrics_per_fold.csv", index=False)
    log.info(f"\nmetrics_per_fold\n{metrics.to_string(index=False)}")
    log.info(f"done. process peak memory {process_peak_mb():.0f} MB")


if __name__ == "__main__":
    log_path = setup_logging()
    try:
        main()
    except BaseException:
        log.exception(f"pipeline crashed. process peak memory {process_peak_mb():.0f} MB. log: {log_path}")
        raise

"""Metrics, bootstrap CIs, calibration and SHAP (PLAN.md step 5).

Log loss is the headline metric: it rewards honest probabilities, which is what a coach
reading "70% chance" needs. ROC-AUC (ranking) and Brier score are reported alongside.
"""

import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def fold_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Log loss, ROC-AUC and Brier score per model and fold, plus the mean over folds.

    Reported per fold, not only the mean (CLAUDE.md rule 5): with 3 matches the spread between
    folds is as informative as the average.
    """
    rows = []
    for (model, fold), g in predictions.groupby(["model", "fold"], sort=False):
        rows.append({"model": model, "fold": str(fold), "n": len(g),
                     "log_loss": log_loss(g["completed"], g["p_complete"], labels=[False, True]),
                     "roc_auc": roc_auc_score(g["completed"], g["p_complete"]),
                     "brier": brier_score_loss(g["completed"], g["p_complete"])})
    m = pd.DataFrame(rows)
    mean = m.groupby("model", sort=False)[["n", "log_loss", "roc_auc", "brier"]].mean().assign(fold="mean").reset_index()
    return pd.concat([m, mean], ignore_index=True).sort_values(["model", "fold"], kind="stable").round(4)

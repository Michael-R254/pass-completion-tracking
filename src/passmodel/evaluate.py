"""Metrics, bootstrap CIs, calibration and SHAP (PLAN.md step 5).

Log loss is the headline metric: it rewards honest probabilities, which is what a coach
reading "70% chance" needs. ROC-AUC (ranking) and Brier score are reported alongside.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from xgboost import DMatrix

from passmodel.features import FEATURES
from passmodel.load import MATCH_IDS
from passmodel.models import fold_split, xgb_model

SEED = 0
EPS = np.finfo(float).eps  # same clipping as sklearn's log_loss, so log(0) never happens


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


# ---------------------------------------------------------------- bootstrap

def batch_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """Log loss, ROC-AUC and Brier for many resamples at once. y, p: (B, n), one resample per row.

    AUC uses the Mann-Whitney identity: AUC = P(a random completed pass scores higher than a random
    failed one) = (sum of completed passes' ranks - n1(n1+1)/2) / (n1 * n0). Ties get average ranks,
    so this equals sklearn's roc_auc_score (checked in tests/test_evaluate.py), but runs on all
    resamples in one NumPy call instead of 1,000 sklearn calls.
    """
    pc = np.clip(p, EPS, 1 - EPS)
    n1 = y.sum(axis=1)
    n0 = y.shape[1] - n1
    ranks = rankdata(p, axis=1)
    return {"log_loss": -(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean(axis=1),
            "roc_auc": ((ranks * y).sum(axis=1) - n1 * (n1 + 1) / 2) / (n1 * n0),
            "brier": ((p - y) ** 2).mean(axis=1)}


def bootstrap_ci(predictions: pd.DataFrame, baseline: str, n_boot: int = 1000, seed: int = SEED) -> pd.DataFrame:
    """95% bootstrap CIs per model and test fold, resampling that fold's test passes with replacement.

    Paired: every model is scored on the SAME resampled passes, so `delta_log_loss` (model minus
    `baseline`, negative = model better) has a much tighter CI than comparing two separate CIs.
    Caveat: resampling passes treats them as independent; passes in one match are not, so these
    CIs show sampling noise within a match, not match-to-match variation (the per-fold spread shows that).
    Returns rows: model, fold, metric, estimate (on all passes), ci_low, ci_high.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for fold, g in predictions.groupby("fold"):
        wide = g.pivot(index="event_id", columns="model", values="p_complete")
        y = g.groupby("event_id")["completed"].first().loc[wide.index].to_numpy(float)
        idx = rng.integers(0, len(y), size=(n_boot, len(y)))
        full = {m: batch_metrics(y[None, :], wide[m].to_numpy()[None, :]) for m in wide.columns}
        boot = {m: batch_metrics(y[idx], wide[m].to_numpy()[idx]) for m in wide.columns}
        for m in wide.columns:
            full[m]["delta_log_loss"] = full[m]["log_loss"] - full[baseline]["log_loss"]
            boot[m]["delta_log_loss"] = boot[m]["log_loss"] - boot[baseline]["log_loss"]
            for metric, values in boot[m].items():
                lo, hi = np.percentile(values, [2.5, 97.5])
                rows.append({"model": m, "fold": fold, "metric": metric, "estimate": full[m][metric][0],
                             "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows).round(4)


# ---------------------------------------------------------------- calibration

def calibration_table(predictions: pd.DataFrame, model: str, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table: held-out passes of all folds in `n_bins` equal-count bins of predicted probability.

    Quantile bins (not fixed 0.1-wide ones) because most predictions sit above 0.7; fixed bins would
    leave the low end with a handful of passes. Each row: mean predicted vs observed completion rate, n.
    """
    p = predictions[predictions["model"] == model]
    bins = pd.qcut(p["p_complete"], n_bins, duplicates="drop")
    return (p.groupby(bins, observed=True)
             .agg(mean_predicted=("p_complete", "mean"), observed_rate=("completed", "mean"), n=("completed", "size"))
             .reset_index(drop=True).round(4))


# ---------------------------------------------------------------- SHAP

def shap_values(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SHAP values of the headline XGBoost (D4 defaults) on each held-out match.

    Refits the same seeded model as models.run_lomo per fold, then asks XGBoost for exact TreeSHAP
    contributions (pred_contribs=True; the same algorithm as shap.TreeExplainer, without its
    version-compatibility issues). Units: log-odds of completion; per pass, the contributions plus
    the bias sum to the model's raw prediction.
    Returns (shap values, feature values), both one row per test pass with columns FEATURES.
    """
    values, X_all = [], []
    for fold in MATCH_IDS:
        train, test = fold_split(features, fold)
        model = xgb_model().fit(train[FEATURES].astype(float), train["completed"])
        X = test[FEATURES].astype(float)
        contribs = model.get_booster().predict(DMatrix(X), pred_contribs=True)[:, :-1]  # last column = bias
        values.append(pd.DataFrame(contribs, columns=FEATURES, index=X.index))
        X_all.append(X)
    return pd.concat(values), pd.concat(X_all)


def shap_importance(values: pd.DataFrame) -> pd.DataFrame:
    """Mean |SHAP| per feature over all held-out passes: how much each feature moves predictions on average."""
    return values.abs().mean().sort_values(ascending=False).rename("mean_abs_shap").round(4).to_frame()


def ablation_table(predictions: pd.DataFrame, full_model: str) -> pd.DataFrame:
    """Held-out log loss per fold for each 'without <group>' model, minus the full model's.

    Positive delta = the model got worse without the group, i.e. the group earns its place.
    """
    ll = fold_metrics(predictions).pivot(index="model", columns="fold", values="log_loss")
    delta = (ll - ll.loc[full_model]).drop(index=full_model)
    return delta.add_prefix("delta_log_loss_fold_").assign(
        folds_worse_without=lambda d: (d.filter(like="fold_").drop(columns="delta_log_loss_fold_mean") > 0).sum(axis=1)
    ).sort_values("delta_log_loss_fold_mean", ascending=False).round(4)


# ---------------------------------------------------------------- step 6: passes as football

FRAMES_PER_MINUTE = 25 * 60


def held_out_passes(predictions: pd.DataFrame, features: pd.DataFrame, model: str) -> pd.DataFrame:
    """One row per open-play pass with its held-out prediction and the features of the fold that predicted it.

    features has one row per (fold, pass); we keep fold == match_id, i.e. the intended target and features
    as the model saw them when that match was the test match.
    """
    test_rows = features[features["fold"] == features["match_id"]].drop(columns="completed")
    p = predictions[predictions["model"] == model][["match_id", "event_id", "completed", "p_complete"]]
    out = p.merge(test_rows, on=["match_id", "event_id"], validate="one_to_one")
    # Frames run continuously from kick-off at 25 Hz, so this is minutes of play, not the match clock.
    return out.assign(approx_minute=out["start_frame"] / FRAMES_PER_MINUTE)


SURPRISE_COLUMNS = ["match_id", "period", "approx_minute", "passer", "target_player", "receiver", "p_complete",
                    "pass_length", "target_nearest_opp", "passer_nearest_opp", "lane_opps", "start_x", "event_id"]


def surprising_passes(held_out: pd.DataFrame, n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(riskiest completed passes, safest failed passes): where the outcome most surprised the model.

    Riskiest completed = lowest predicted chance among passes that came off; safest failures = highest
    predicted chance among passes that were lost. These are the clips an analyst would pull first.
    """
    risky = held_out[held_out["completed"]].nsmallest(n, "p_complete")[SURPRISE_COLUMNS]
    safe = held_out[~held_out["completed"]].nlargest(n, "p_complete")[SURPRISE_COLUMNS]
    return risky.round(3).reset_index(drop=True), safe.round(3).reset_index(drop=True)


def completion_above_expected(held_out: pd.DataFrame, min_passes: int = 40) -> pd.DataFrame:
    """Per player and match: completion rate minus the model's expected rate, for players with >= min_passes.

    se = binomial standard error of the difference if the model's probabilities were exactly right,
    sqrt(sum p(1-p)) / n. A player is only notable if |above_expected| is well beyond ~2 se.
    Per match, because Metrica's anonymised player IDs are not guaranteed to mean the same person across games.
    """
    g = held_out.assign(var=held_out["p_complete"] * (1 - held_out["p_complete"])).groupby(["match_id", "passer"])
    t = g.agg(passes=("completed", "size"), completion=("completed", "mean"), expected=("p_complete", "mean"),
              var=("var", "sum"))
    t = t[t["passes"] >= min_passes]
    t = t.assign(above_expected=t["completion"] - t["expected"], se=np.sqrt(t["var"]) / t["passes"]).drop(columns="var")
    return t.sort_values("above_expected", ascending=False).round(3)

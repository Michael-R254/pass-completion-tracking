"""Baselines and XGBoost with leave-one-match-out CV (PLAN.md steps 3-4).

Split: 3 folds, each holds out one whole match (CLAUDE.md rule 1). Passes within a match are
correlated (same teams, same game state), so a random pass-level split would let the model
see the test match during training and overstate how well it generalises to a new match.
The main model uses open-play passes only (D2).
"""

from itertools import product

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

from passmodel.features import FEATURE_GROUPS, FEATURES, PITCH_CONTROL
from passmodel.load import MATCH_IDS

SEED = 0


def fold_split(features: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) for one leave-one-match-out fold, open play only.

    features has one row per (fold, pass); we take this fold's rows, so the intended target of every
    pass was learned without its own match (D3). Train = the other matches, test = match `fold`.
    """
    f = features[(features["fold"] == fold) & ~features["is_set_piece"]]
    return f[f["match_id"] != fold], f[f["match_id"] == fold]


def constant_model():
    """Predicts the training base rate for every pass: the floor any model has to beat."""
    return DummyClassifier(strategy="prior")


def logistic_model():
    """Plain logistic regression; used on pass length only as the first real baseline."""
    return LogisticRegression(random_state=SEED)


# D4: fixed, conservative settings chosen BEFORE seeing any XGBoost result (no tuning).
# Shallow trees + slow learning rate + row/column subsampling, because each fold trains on only ~2,300 passes.
XGB_DEFAULTS = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    min_child_weight=5)
# D4 sensitivity check only: the small grid searched by inner leave-one-match-out CV.
XGB_GRID = {"max_depth": [2, 3, 4], "learning_rate": [0.03, 0.1], "n_estimators": [200, 500]}


def xgb_model(**params):
    """XGBoost classifier with the D4 defaults, overridable for the sensitivity check. Fixed seed."""
    return XGBClassifier(**{**XGB_DEFAULTS, **params}, random_state=SEED, eval_metric="logloss", n_jobs=4)


MODELS = {
    # name: (factory, feature columns)
    "constant": (constant_model, ["pass_length"]),   # the column is ignored by DummyClassifier
    "logistic (pass length)": (logistic_model, ["pass_length"]),
    # The original model: commits to the single most likely intended target. Now the comparison row; the
    # headline is the target mixture (run_lomo_target_mixture), which beat it on all 3 held-out matches.
    "xgboost (single target)": (xgb_model, FEATURES),
    # Stretch (PLAN): pitch control, tested on the single-target model; kept only if it beats it (it didn't).
    "logistic (pitch control)": (logistic_model, [PITCH_CONTROL]),
    "xgboost (single target + pitch control)": (xgb_model, FEATURES + [PITCH_CONTROL]),
}


def run_lomo(features: pd.DataFrame, name: str, factory, columns: list[str]) -> pd.DataFrame:
    """Fit on the training matches and predict the held-out match, for every fold.

    Returns one row per test pass: model, fold, match_id, event_id, completed, p_complete.
    """
    out = []
    for fold in MATCH_IDS:
        train, test = fold_split(features, fold)
        model = factory().fit(train[columns].astype(float), train["completed"])
        out.append(pd.DataFrame({
            "model": name, "fold": fold, "match_id": test["match_id"].to_numpy(),
            "event_id": test["event_id"].to_numpy(), "completed": test["completed"].to_numpy(),
            "p_complete": model.predict_proba(test[columns].astype(float))[:, 1],
        }))
    return pd.concat(out, ignore_index=True)


def fit_target_mixture(mixture: pd.DataFrame, fold: int, columns: list[str] = FEATURES):
    """Fit one fold's target-mixture XGBoost (D4 defaults); returns (model, that fold's test rows).

    mixture: build_target_mixture_features output, one row per (fold, pass, teammate) with weight q.
    Train: every (pass, teammate) row of the training matches, sample weight q (each pass totals weight 1),
    so the model learns from all plausible targets instead of betting on one. Open play only.
    """
    f = mixture[(mixture["fold"] == fold) & ~mixture["is_set_piece"]]
    train, test = f[f["match_id"] != fold], f[f["match_id"] == fold]
    model = xgb_model().fit(train[columns].astype(float), train["completed"], sample_weight=train["weight"])
    return model, test


def run_lomo_target_mixture(mixture: pd.DataFrame, name: str = "xgboost (target mixture)",
                            columns: list[str] = FEATURES) -> pd.DataFrame:
    """LOMO headline model: P(complete) = sum_j q_j * P(complete | aimed at teammate j), per test pass.

    q_j = P(teammate j is the intended target) from the cross-fitted D3 target model. Same output schema as run_lomo.
    """
    out = []
    for fold in MATCH_IDS:
        model, test = fit_target_mixture(mixture, fold, columns)
        test = test.assign(wp=test["weight"] * model.predict_proba(test[columns].astype(float))[:, 1])
        g = test.groupby(["match_id", "event_id"], sort=False).agg(
            completed=("completed", "first"), wp=("wp", "sum"), w=("weight", "sum")).reset_index()
        out.append(pd.DataFrame({"model": name, "fold": fold, "match_id": g["match_id"], "event_id": g["event_id"],
                                 "completed": g["completed"], "p_complete": g["wp"] / g["w"]}))
    return pd.concat(out, ignore_index=True)


def ablation(mixture: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-group-out ablation of the headline target-mixture model, refitted without each feature group.

    Answers "does each group earn its place?" (CLAUDE.md rule 6) on held-out log loss.
    Reported only: the headline model keeps all features, because choosing features from these
    test-fold scores would be tuning on the test matches (rule 2).
    """
    return pd.concat([run_lomo_target_mixture(mixture, f"without {group}", [f for f in FEATURES if f not in cols])
                      for group, cols in FEATURE_GROUPS.items()], ignore_index=True)


def inner_cv_sensitivity(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """D4 sensitivity check: what would inner CV have picked, and how would it have scored?

    For each outer fold, using ONLY its two training matches: train on one, validate on the other
    (both ways), for every XGB_GRID setting; pick the lowest mean inner log loss; refit on both
    training matches; predict the outer test match once.
    Caveat (documented in CLAUDE.md D4): in the inner split, the training match's intended targets
    were learned from the validation match's receivers (D3 cross-fitting), so inner scores are
    slightly optimistic. No completion labels or outer-test data are involved.
    Returns (one row per outer fold and grid setting with its inner log loss, test predictions).
    """
    grid = [dict(zip(XGB_GRID, values)) for values in product(*XGB_GRID.values())]
    scores, preds = [], []
    for fold in MATCH_IDS:
        train, test = fold_split(features, fold)
        X = train[FEATURES].astype(float)
        inner = []
        for params in grid:
            losses = []
            for val_match in train["match_id"].unique():
                fit, val = train["match_id"] != val_match, train["match_id"] == val_match
                m = xgb_model(**params).fit(X[fit], train.loc[fit, "completed"])
                losses.append(log_loss(train.loc[val, "completed"], m.predict_proba(X[val])[:, 1], labels=[False, True]))
            inner.append({"fold": fold, **params, "inner_log_loss": sum(losses) / len(losses)})
        inner = pd.DataFrame(inner)
        inner["chosen"] = inner["inner_log_loss"] == inner["inner_log_loss"].min()
        best = inner.loc[inner["chosen"], list(XGB_GRID)].iloc[0].to_dict()
        best = {k: (int(v) if k != "learning_rate" else float(v)) for k, v in best.items()}
        model = xgb_model(**best).fit(X, train["completed"])
        preds.append(pd.DataFrame({
            "model": "xgboost (inner-CV tuned, sensitivity)", "fold": fold, "match_id": test["match_id"].to_numpy(),
            "event_id": test["event_id"].to_numpy(), "completed": test["completed"].to_numpy(),
            "p_complete": model.predict_proba(test[FEATURES].astype(float))[:, 1]}))
        scores.append(inner)
    return pd.concat(scores, ignore_index=True).round(4), pd.concat(preds, ignore_index=True)

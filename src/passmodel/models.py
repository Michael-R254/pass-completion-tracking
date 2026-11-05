"""Baselines and XGBoost with leave-one-match-out CV (PLAN.md steps 3-4).

Split: 3 folds, each holds out one whole match (CLAUDE.md rule 1). Passes within a match are
correlated (same teams, same game state), so a random pass-level split would let the model
see the test match during training and overstate how well it generalises to a new match.
The main model uses open-play passes only (D2).
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

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


MODELS = {
    # name: (factory, feature columns)
    "constant": (constant_model, ["pass_length"]),   # the column is ignored by DummyClassifier
    "logistic (pass length)": (logistic_model, ["pass_length"]),
}


def run_lomo(features: pd.DataFrame, name: str, factory, columns: list[str]) -> pd.DataFrame:
    """Fit on the training matches and predict the held-out match, for every fold.

    Returns one row per test pass: model, fold, match_id, event_id, completed, p_complete.
    """
    out = []
    for fold in MATCH_IDS:
        train, test = fold_split(features, fold)
        model = factory().fit(train[columns], train["completed"])
        out.append(pd.DataFrame({
            "model": name, "fold": fold, "match_id": test["match_id"].to_numpy(),
            "event_id": test["event_id"].to_numpy(), "completed": test["completed"].to_numpy(),
            "p_complete": model.predict_proba(test[columns])[:, 1],
        }))
    return pd.concat(out, ignore_index=True)

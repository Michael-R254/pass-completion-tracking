"""The vectorised bootstrap metrics agree with sklearn's."""

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from passmodel.evaluate import batch_metrics


def test_batch_metrics_match_sklearn():
    rng = np.random.default_rng(0)
    y = (rng.random((3, 200)) < 0.8).astype(float)
    p = np.round(rng.uniform(0.05, 0.95, (3, 200)), 1)  # rounded so there are ties, which AUC must handle
    m = batch_metrics(y, p)
    for b in range(3):
        assert np.isclose(m["log_loss"][b], log_loss(y[b], p[b]))
        assert np.isclose(m["roc_auc"][b], roc_auc_score(y[b], p[b]))
        assert np.isclose(m["brier"][b], brier_score_loss(y[b], p[b]))

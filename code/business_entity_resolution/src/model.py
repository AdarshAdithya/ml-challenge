"""LightGBM matchers with grouped out-of-fold training and isotonic calibration.

Folds group by Source 1 entity so no entity leaks between train and validation.
Out-of-fold (OOF) predictions feed stage 2, calibration, and decoder tuning, so
every number we report is measured on data the model did not train on.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

PARAMS = dict(
    objective="binary", learning_rate=0.08, num_leaves=63, min_child_samples=30,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbose=-1, seed=42,
)


def oof_predict(X, y, groups, n_folds=5, rounds=500, params=None):
    params = params or PARAMS
    oof = np.zeros(len(y))
    best_iters = []
    gkf = GroupKFold(n_splits=n_folds)
    for tr, va in gkf.split(X, y, groups):
        dtr = lgb.Dataset(X.iloc[tr], y[tr])
        dva = lgb.Dataset(X.iloc[va], y[va])
        m = lgb.train(params, dtr, rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        best_iters.append(m.best_iteration or rounds)
    return oof, int(np.mean(best_iters))


def fit_full(X, y, rounds, params=None):
    return lgb.train(params or PARAMS, lgb.Dataset(X, y), max(rounds, 50))


def fit_calibrator(p, y):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y)
    return iso


def feature_importance(model, top=15):
    imp = model.feature_importance("gain")
    order = np.argsort(-imp)[:top]
    names = model.feature_name()
    return [(names[i], float(imp[i])) for i in order]

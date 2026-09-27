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
    oof = np.zeros(len(y), dtype=np.float32)
    best_iters = []
    gkf = GroupKFold(n_splits=n_folds)
    feature_names = list(X.columns) if hasattr(X, "columns") else None
    X_arr = np.ascontiguousarray(X.to_numpy(dtype=np.float32) if hasattr(X, "to_numpy") else X, dtype=np.float32)
    y_arr = np.ascontiguousarray(y, dtype=np.float32)

    for tr, va in gkf.split(X_arr, y_arr, groups):
        dtr = lgb.Dataset(X_arr[tr], label=y_arr[tr], feature_name=feature_names, free_raw_data=False)
        dva = lgb.Dataset(X_arr[va], label=y_arr[va], reference=dtr, feature_name=feature_names, free_raw_data=False)
        m = lgb.train(params, dtr, rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        oof[va] = m.predict(X_arr[va], num_iteration=m.best_iteration)
        best_iters.append(m.best_iteration or rounds)
    return oof, int(np.mean(best_iters))


def fit_full(X, y, rounds, params=None):
    feature_names = list(X.columns) if hasattr(X, "columns") else None
    X_arr = np.ascontiguousarray(X.to_numpy(dtype=np.float32) if hasattr(X, "to_numpy") else X, dtype=np.float32)
    y_arr = np.ascontiguousarray(y, dtype=np.float32)
    ds = lgb.Dataset(X_arr, label=y_arr, feature_name=feature_names, free_raw_data=False)
    return lgb.train(params or PARAMS, ds, max(rounds, 50))


def fit_calibrator(p, y):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y)
    return iso


def feature_importance(model, top=15):
    imp = model.feature_importance("gain")
    order = np.argsort(-imp)[:top]
    names = model.feature_name()
    return [(names[i], float(imp[i])) for i in order]

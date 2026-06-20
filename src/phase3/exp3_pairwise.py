"""
Phase 3 - Experiment 3: pairwise crop->crop transfer grid.

Train on ONE crop, test on ANOTHER, for all ordered pairs (12 off-diagonal).
The diagonal (train==test) is the within-crop 5-fold CV score for reference.

This maps the STRUCTURE of the generalization failure that Exp 2 showed in
aggregate: which specific sensor/crop pairs transfer least badly? Does similar
DM range or similar camera help? Output is a 4x4 matrix of pearson_r (does the
prediction TRACK the target) plus R2, saved as CSVs and ready for a heatmap.

Feature space per pair = bands shared by BOTH crops (so apple pairs use 141,
leek+mushroom use 204, etc.). Diagonal uses that crop's own common-with-all bands
for comparability with Exp 1.

Reuses the Phase-1 pipeline and the data module.

Usage:
    python exp3_pairwise.py --csv data/SpectroFood_dataset.csv \
        --out results/phase3/exp3_pairwise.csv
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error

from data import load_csv, common_bands, get_Xy, CROPS
from exp2_leave_one_out import make_models


def rpd(y, p):
    rmse = np.sqrt(mean_squared_error(y, p))
    return np.std(y) / rmse if rmse > 0 else np.nan


def best_pair_score(df, train_crop, test_crop, model_names):
    """Train on train_crop, test on test_crop, on their shared bands.
    Return best (by pearson_r) across models: (r, R2, model)."""
    bands = common_bands(df, [train_crop, test_crop])
    Xtr, ytr, _ = get_Xy(df, train_crop, bands=bands)
    Xte, yte, _ = get_Xy(df, test_crop, bands=bands)
    best = (-2.0, np.nan, None)
    models = make_models()
    for name in model_names:
        m = models[name]
        m.fit(Xtr, ytr)
        pred = np.asarray(m.predict(Xte)).ravel()
        r = pearsonr(pred, yte)[0] if np.std(pred) > 1e-9 else 0.0
        if r > best[0]:
            best = (r, r2_score(yte, pred), name)
    return best


def diagonal_score(df, crop, model_names):
    """Within-crop 5-fold CV (pearson_r, R2) on all-four common bands."""
    bands = common_bands(df, CROPS)
    X, y, _ = get_Xy(df, crop, bands=bands)
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    best = (-2.0, np.nan, None)
    models = make_models()
    for name in model_names:
        pred = cross_val_predict(models[name], X, y, cv=cv, n_jobs=-1).ravel()
        r = pearsonr(pred, y)[0] if np.std(pred) > 1e-9 else 0.0
        if r > best[0]:
            best = (r, r2_score(y, pred), name)
    return best


def run(csv_path, out_path):
    df = load_csv(csv_path)
    model_names = list(make_models().keys())

    r_mat = pd.DataFrame(index=CROPS, columns=CROPS, dtype=float)   # rows=train, cols=test
    r2_mat = pd.DataFrame(index=CROPS, columns=CROPS, dtype=float)
    long_rows = []

    for tr in CROPS:
        for te in CROPS:
            if tr == te:
                r, r2, mdl = diagonal_score(df, tr, model_names)
            else:
                r, r2, mdl = best_pair_score(df, tr, te, model_names)
            r_mat.loc[tr, te] = r
            r2_mat.loc[tr, te] = r2
            long_rows.append({'train': tr, 'test': te, 'pearson_r': r,
                              'R2': r2, 'best_model': mdl})

    res = pd.DataFrame(long_rows)
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    res.to_csv(out_path, index=False)
    r_mat.to_csv(out_path.replace('.csv', '_pearson_matrix.csv'))
    r2_mat.to_csv(out_path.replace('.csv', '_r2_matrix.csv'))

    print('=== Pairwise transfer: pearson_r (rows=train, cols=test) ===')
    print('  diagonal = within-crop CV; off-diagonal = cross-crop transfer\n')
    print(r_mat.astype(float).round(2).to_string())
    print('\nReading guide:')
    print('  diagonal high, off-diagonal low  -> transfer fails (expected).')
    print('  any off-diagonal cell notably higher -> those two crops/cameras share structure.')
    print(f'\nsaved {out_path} (+ _pearson_matrix.csv, _r2_matrix.csv)')
    return r_mat, r2_mat


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='exp3_pairwise.csv')
    args = ap.parse_args()
    run(args.csv, args.out)

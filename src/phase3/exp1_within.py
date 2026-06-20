"""
Phase 3 - Experiment 1: within-crop ceiling.

Normal 5-fold CV INSIDE each crop (train and test are the same crop+camera).
This is the best achievable per crop and the reference the few-shot curves climb
toward. Computed on the SAME all-four common 141-band space as the transfer
experiments, so the numbers are directly comparable (this differs slightly from
the Phase-1 benchmark, which used each crop's full native bands).

Usage:
    python exp1_within.py --csv data/SpectroFood_dataset.csv \
        --out results/phase3/exp1_within_crop.csv
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from data import load_csv, common_bands, get_Xy, CROPS
from exp2_leave_one_out import make_models


def rpd(y, p):
    rmse = np.sqrt(mean_squared_error(y, p))
    return np.std(y) / rmse if rmse > 0 else np.nan


def run(csv_path, out_path, folds=5):
    df = load_csv(csv_path)
    bands = common_bands(df, CROPS)            # 141 shared bands, same as transfer exps
    rows = []

    for crop in CROPS:
        X, y, _ = get_Xy(df, crop, bands=bands)
        cv = KFold(n_splits=folds, shuffle=True, random_state=0)
        for name, model in make_models().items():
            pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1).ravel()
            rows.append({
                'crop': crop, 'model': name, 'bands': len(bands), 'n': len(y),
                'R2': r2_score(y, pred), 'MAE': mean_absolute_error(y, pred),
                'RMSE': np.sqrt(mean_squared_error(y, pred)), 'RPD': rpd(y, pred),
            })

    res = pd.DataFrame(rows)
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    res.to_csv(out_path, index=False)

    print(f'=== Within-crop ceiling (best model per crop, {folds}-fold CV, 141 bands) ===')
    ceiling = {}
    for crop in CROPS:
        sub = res[res['crop'] == crop]
        b = sub.loc[sub['R2'].idxmax()]
        ceiling[crop] = b['R2']
        print(f'  {crop:10s} | {b["model"]:7s} R2={b["R2"]:.3f}  RPD={b["RPD"]:.2f}')
    print(f'\nsaved {out_path}')
    # also save a compact ceiling lookup for the notebook
    pd.Series(ceiling).to_csv(out_path.replace('.csv', '_ceiling.csv'), header=['R2'])
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='exp1_within_crop.csv')
    ap.add_argument('--folds', type=int, default=5)
    args = ap.parse_args()
    run(args.csv, args.out, args.folds)

"""
Phase 3 - statistical uncertainty utilities.

Three ways to put error bars on the metrics, matched to each experiment:

1) bootstrap_ci(y, pred)        -> for single held-out sets (leave-one-out, transfer):
   resample test samples with replacement, recompute the metric, take percentiles.
   Gives R2 = point [lo, hi] without needing repeated training.

2) cv_mean_std(...)             -> for cross-validated experiments (Phase 1, exp1):
   compute the metric PER FOLD, report mean +/- std across folds.

3) seed_mean_std(...)           -> already used by exp_fewshot.py (multiple seeds).

Bootstrap is the general fallback because it needs only (y_true, y_pred) and makes
no assumption about how the predictions were produced.

Usage (attach CIs to an existing leave-one-out result):
    python uncertainty.py --csv data/SpectroFood_dataset.csv \
        --out results/phase3/exp2_with_ci.csv
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error

from data import load_csv, train_test_bands, common_bands, get_Xy, CROPS
from exp2_leave_one_out import make_models


def rpd(y, p):
    rmse = np.sqrt(mean_squared_error(y, p))
    return np.std(y) / rmse if rmse > 0 else np.nan


def bootstrap_ci(y, pred, metric='r2', n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for a metric over (y, pred). Returns (point, lo, hi)."""
    y = np.asarray(y); pred = np.asarray(pred)
    rng = np.random.default_rng(seed)
    n = len(y)
    def m(yy, pp):
        if metric == 'r2': return r2_score(yy, pp)
        if metric == 'rmse': return np.sqrt(mean_squared_error(yy, pp))
        if metric == 'rpd': return rpd(yy, pp)
    point = m(y, pred)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        # guard: bootstrap sample with zero target variance breaks R2
        if np.std(y[idx]) < 1e-12:
            stats[b] = np.nan
        else:
            stats[b] = m(y[idx], pred[idx])
    lo, hi = np.nanpercentile(stats, [100*alpha/2, 100*(1-alpha/2)])
    return point, lo, hi


def cv_mean_std(model, X, y, folds=5, seed=0, metric='r2'):
    """Per-fold metric -> (mean, std). Reports variability across CV folds."""
    cv = KFold(n_splits=folds, shuffle=True, random_state=seed)
    vals = []
    for tr, te in cv.split(X):
        m = model
        m.fit(X[tr], y[tr])
        p = np.asarray(m.predict(X[te])).ravel()
        if metric == 'r2':
            vals.append(r2_score(y[te], p))
        elif metric == 'rpd':
            vals.append(rpd(y[te], p))
    return float(np.mean(vals)), float(np.std(vals))


def leave_one_out_with_ci(csv_path, out_path, n_boot=2000):
    """Re-run leave-one-out, but attach bootstrap CIs to each crop's best model."""
    df = load_csv(csv_path)
    rows = []
    for test_crop in CROPS:
        train_crops = [c for c in CROPS if c != test_crop]
        bands, Xtr, ytr, Xte, yte = train_test_bands(df, train_crops, test_crop)
        # pick best model by pearson (same as exp2), then bootstrap it
        from scipy.stats import pearsonr
        best = None
        for name, model in make_models().items():
            model.fit(Xtr, ytr)
            pred = np.asarray(model.predict(Xte)).ravel()
            r = pearsonr(pred, yte)[0] if np.std(pred) > 1e-9 else 0.0
            if best is None or r > best[1]:
                best = (name, r, pred)
        name, r, pred = best
        r2_pt, r2_lo, r2_hi = bootstrap_ci(yte, pred, 'r2', n_boot)
        rpd_pt, rpd_lo, rpd_hi = bootstrap_ci(yte, pred, 'rpd', n_boot)
        rows.append({
            'held_out': test_crop, 'best_model': name, 'pearson_r': r,
            'R2': r2_pt, 'R2_lo': r2_lo, 'R2_hi': r2_hi,
            'RPD': rpd_pt, 'RPD_lo': rpd_lo, 'RPD_hi': rpd_hi,
        })
    res = pd.DataFrame(rows)
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    res.to_csv(out_path, index=False)
    print('=== Leave-one-crop-out with 95% bootstrap CIs ===')
    for _, r in res.iterrows():
        print(f'  {r.held_out:9s} {r.best_model:7s} '
              f'R2={r.R2:+.2f} [{r.R2_lo:+.2f}, {r.R2_hi:+.2f}]  '
              f'r={r.pearson_r:+.2f}')
    print(f'\nsaved {out_path}')
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='exp2_with_ci.csv')
    ap.add_argument('--n_boot', type=int, default=2000)
    args = ap.parse_args()
    leave_one_out_with_ci(args.csv, args.out, args.n_boot)

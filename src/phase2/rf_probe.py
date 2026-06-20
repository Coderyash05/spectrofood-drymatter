"""
Phase 2 - DIAGNOSTIC: Random Forest on patch center spectra, cube-grouped.

Purpose: decide whether the CNN's R2~=0 means
  (a) signal exists but CNN training is failing      -> RF R2 clearly > 0
  (b) no signal at patch level / label problem        -> RF R2 ~= 0

Mirrors train.py exactly EXCEPT the model: same patch dataset, same center-pixel
spectrum, same GroupKFold by cube, same patch->cube averaging, cube-level metrics.

It runs THREE preprocessing variants so we also learn whether the 2nd-derivative
(great on Phase-1 mean spectra) is hurting noisy single-pixel spectra:
    raw    : center spectrum as-is
    snv    : SNV only
    snv_d2 : SNV + 2nd derivative  (what the CNN used)

Usage:
    python rf_probe.py --data results/phase2/apple_patches.npz --folds 5
"""
import argparse
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from preprocessing import snv, savgol_deriv


def rpd(y, p):
    rmse = np.sqrt(mean_squared_error(y, p))
    return np.std(y) / rmse if rmse > 0 else np.nan


def prep(spec, mode):
    if mode == 'raw':
        return spec
    if mode == 'snv':
        return snv(spec, axis=-1)
    if mode == 'snv_d2':
        return savgol_deriv(snv(spec, axis=-1), window=11, poly=2, deriv=2, axis=-1)
    raise ValueError(mode)


def run(data_path, folds):
    d = np.load(data_path, allow_pickle=True)
    X, y, groups = d['X'], d['y'], d['groups']
    P = X.shape[1]; c = P // 2
    center = X[:, c, c, :].astype(np.float32)          # (N, B) center-pixel spectrum
    print(f'patches={X.shape}  center-spectra={center.shape}  cubes={len(set(groups))}')

    for mode in ['raw', 'snv', 'snv_d2']:
        feats = prep(center, mode)
        gkf = GroupKFold(n_splits=folds)
        cube_true, cube_pred = {}, {}
        for tr, te in gkf.split(feats, y, groups=groups):
            rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0)
            rf.fit(feats[tr], y[tr])
            pred = rf.predict(feats[te])
            g_te = groups[te]
            for cid in np.unique(g_te):
                m = g_te == cid
                cube_pred[cid] = float(pred[m].mean())
                cube_true[cid] = float(y[te][m][0])
        ids = sorted(cube_true, key=lambda s: int(''.join(ch for ch in s if ch.isdigit())))
        yt = np.array([cube_true[i] for i in ids])
        yp = np.array([cube_pred[i] for i in ids])
        print(f'  [{mode:7s}] cube-level  R2={r2_score(yt, yp):+.3f}  '
              f'MAE={mean_absolute_error(yt, yp):.4f}  '
              f'RMSE={np.sqrt(mean_squared_error(yt, yp)):.4f}  RPD={rpd(yt, yp):.2f}')

    print('\nInterpretation:')
    print('  any RF R2 clearly > 0  -> signal EXISTS at patch level; CNN training is the problem')
    print('  all RF R2 ~= 0         -> no usable patch-level signal / check label assignment')
    print('  raw or snv >> snv_d2   -> 2nd derivative is destroying single-pixel signal')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--folds', type=int, default=5)
    args = ap.parse_args()
    run(args.data, args.folds)

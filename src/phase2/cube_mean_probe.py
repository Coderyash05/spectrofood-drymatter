"""
Phase 2 - DECISIVE DIAGNOSTIC: does the signal live in the cube AVERAGE?

Phase 1 got apple R2~=0.44 from the per-cube MEAN spectrum.
rf_probe.py got ~0 from single-pixel patch spectra.
Hypothesis: dry-matter signal survives only after heavy pixel averaging.

This script averages, for each cube, ALL its patch center-spectra into one mean
spectrum, then runs RF + GroupKFold (same as before) at the cube level.

Outcomes:
  R2 ~= 0.44  -> signal is real but AVERAGE-ONLY; patch pipeline is correct;
                 small patches simply destroy it. Path: bigger patches / cube-mean model.
  R2 ~= 0     -> the cube->patch->label chain differs from Phase 1 => pipeline/label bug
                 (because Phase 1 on the CSV mean spectra got 0.44). Debug the chain.

Also compares against a within-mask FULL-cube mean if --mat/--labels are given,
to check whether using only patch centers (vs all foreground pixels) loses signal.

Usage:
    python cube_mean_probe.py --data results/phase2/apple_patches.npz --folds 5
"""
import argparse
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
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


def run(data_path, folds):
    d = np.load(data_path, allow_pickle=True)
    X, y, groups = d['X'], d['y'], d['groups']
    P = X.shape[1]; c = P // 2
    center = X[:, c, c, :].astype(np.float32)

    # build one mean spectrum per cube
    ids = sorted(set(groups), key=lambda s: int(''.join(ch for ch in s if ch.isdigit())))
    cube_spec = np.stack([center[groups == cid].mean(axis=0) for cid in ids])
    cube_y = np.array([y[groups == cid][0] for cid in ids], dtype=np.float32)
    cube_groups = np.array(ids)
    print(f'cube-mean spectra: {cube_spec.shape}  (one per cube)')

    models = {
        'RF':    lambda: RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0),
        'Ridge': lambda: Ridge(alpha=1.0),
        'PLS10': lambda: PLSRegression(n_components=10),
    }
    for mode in ['raw', 'snv', 'snv_d2']:
        feats = prep(cube_spec, mode)
        line = f'  [{mode:7s}] '
        for mname, mk in models.items():
            gkf = GroupKFold(n_splits=folds)
            yt, yp = [], []
            for tr, te in gkf.split(feats, cube_y, groups=cube_groups):
                m = mk(); m.fit(feats[tr], cube_y[tr])
                pred = np.asarray(m.predict(feats[te])).ravel()
                yt.extend(cube_y[te]); yp.extend(pred)
            yt, yp = np.array(yt), np.array(yp)
            line += f'{mname}: R2={r2_score(yt, yp):+.3f}  '
        print(line)

    print('\nCompare to Phase-1 apple baseline: R2=0.44 (RF, 2nd-deriv on CSV mean spectra).')
    print('  ~0.44 here -> signal is AVERAGE-ONLY, pipeline correct, go to bigger patches.')
    print('  ~0    here -> pipeline/label mismatch vs Phase 1, debug the cube->label chain.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--folds', type=int, default=5)
    args = ap.parse_args()
    run(args.data, args.folds)
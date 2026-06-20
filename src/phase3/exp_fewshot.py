"""
Phase 3 - Few-shot transfer: how much target-crop data rescues generalization?

Experiment 2 showed naive transfer fails: train on 3 crops, test on the 4th -> R2<0.
Here we ask the useful follow-up: if we add a FEW samples of the held-out crop to
the training set, how quickly does prediction recover?

For each held-out crop and each k in [0, 5, 10, 20, 40, 80]:
  - train on (other 3 crops)  +  k randomly chosen samples of the held-out crop
  - test on the REMAINING held-out-crop samples (never the k used for training)
  - repeat with several random seeds and average (k samples is noisy)

k=0 reproduces Experiment 2 (pure cross-crop). Rising R2 with k shows how little
target data is needed. A sharp jump at small k = cheap to adapt; a flat line = the
crop is fundamentally hard to transfer to.

Reuses the Phase-1 pipeline (SNV + SG 2nd-deriv + scale) and the data module.

Usage:
    python exp_fewshot.py --csv data/SpectroFood_dataset.csv \
        --out results/phase3/exp_fewshot.csv --seeds 5
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error

from data import load_csv, common_bands, get_Xy, CROPS
from exp2_leave_one_out import make_models   # same pipelines/models


def rpd(y, p):
    rmse = np.sqrt(mean_squared_error(y, p))
    return np.std(y) / rmse if rmse > 0 else np.nan


def run(csv_path, out_path, k_list, seeds, model_name):
    df = load_csv(csv_path)
    rows = []

    for test_crop in CROPS:
        train_crops = [c for c in CROPS if c != test_crop]
        bands = common_bands(df, CROPS)            # all-four shared bands (141)

        Xtr_base, ytr_base, _ = get_Xy(df, train_crops, bands=bands)
        Xte_all, yte_all, _ = get_Xy(df, test_crop, bands=bands)
        n_target = len(yte_all)

        for k in k_list:
            if k >= n_target:
                continue
            r2s, rpds = [], []
            for s in range(seeds):
                rng = np.random.default_rng(1000 * s + k)
                if k == 0:
                    add_idx = np.array([], dtype=int)
                    test_idx = np.arange(n_target)
                else:
                    add_idx = rng.choice(n_target, size=k, replace=False)
                    test_idx = np.setdiff1d(np.arange(n_target), add_idx)

                Xtr = np.vstack([Xtr_base, Xte_all[add_idx]]) if k else Xtr_base
                ytr = np.concatenate([ytr_base, yte_all[add_idx]]) if k else ytr_base
                Xte, yte = Xte_all[test_idx], yte_all[test_idx]

                model = make_models()[model_name]
                model.fit(Xtr, ytr)
                pred = np.asarray(model.predict(Xte)).ravel()
                r2s.append(r2_score(yte, pred))
                rpds.append(rpd(yte, pred))

            rows.append({
                'held_out': test_crop, 'k_target_samples': k,
                'R2_mean': np.mean(r2s), 'R2_std': np.std(r2s),
                'RPD_mean': np.mean(rpds), 'seeds': seeds, 'model': model_name,
            })

    res = pd.DataFrame(rows)
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    res.to_csv(out_path, index=False)

    print(f'=== Few-shot transfer ({model_name}, {seeds} seeds avg) ===')
    print(f'{"crop":10s}' + ''.join(f'k={k:<7d}' for k in k_list))
    for tc in CROPS:
        sub = res[res['held_out'] == tc].set_index('k_target_samples')
        cells = ''.join(f'{sub.loc[k, "R2_mean"]:+.2f}    ' if k in sub.index else '  -     '
                        for k in k_list)
        print(f'{tc:10s}{cells}')
    print(f'\nsaved {out_path}')
    print('\nReading guide:')
    print('  k=0 column = Experiment 2 (naive transfer, expect R2<0).')
    print('  R2 climbing toward the Phase-1 ceiling as k grows = target data rescues it.')
    print('  fast rise at small k = cheap to adapt; flat = fundamentally hard crop.')
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='exp_fewshot.csv')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--model', default='RF', help='PLS/Ridge/RF/XGBoost')
    ap.add_argument('--ks', default='0,5,10,20,40,80')
    args = ap.parse_args()
    k_list = [int(x) for x in args.ks.split(',')]
    run(args.csv, args.out, k_list, args.seeds, args.model)

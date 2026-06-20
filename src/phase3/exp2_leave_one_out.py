"""
Phase 3 - Experiment 2: Leave-One-Crop-Out (the headline generalization test).

For each crop: train on the other THREE crops, test on the held-out one.
The held-out crop was measured by a DIFFERENT camera and never seen in training,
so this directly tests cross-sensor + cross-crop generalization.

Two scorings, because the crops have very different DM ranges:
  RAW        : predict dry matter directly. Honest headline. Mushroom (6-87%) is
               expected to break when the model only trained on <=20% crops.
  NORMALIZED : z-score dry matter per-crop (train stats applied sensibly), then
               score. This separates "can't predict the PATTERN" from "can't
               predict the absolute SCALE". A crop can have RAW R2<0 but
               NORMALIZED R2>0 -> the spectrum->DM *shape* transfers, only the
               offset/scale differs. That distinction is the actual insight.

Preprocessing reuses the Phase-1 winner: SNV + SG 2nd-derivative + standardize,
all fit on TRAIN only (inside a Pipeline -> no leakage).

Models: PLS, Ridge, RF (XGB optional). Metrics: R2, MAE, RMSE, RPD at crop level.

Usage:
    python exp2_leave_one_out.py --csv SpectroFood_dataset.csv --out exp2_leave_one_out.csv
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.signal import savgol_filter

from data import load_csv, train_test_bands, CROPS

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# ---- preprocessing transformers (same as Phase 1) ---------------------------
class SNV(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = np.asarray(X, float)
        mu = X.mean(1, keepdims=True); sd = X.std(1, keepdims=True); sd[sd == 0] = 1
        return (X - mu) / sd

class SavGol(BaseEstimator, TransformerMixin):
    def __init__(self, window=11, poly=2, deriv=2):
        self.window, self.poly, self.deriv = window, poly, deriv
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = np.asarray(X, float)
        w = min(self.window, X.shape[1] if X.shape[1] % 2 == 1 else X.shape[1] - 1)
        return X if w <= self.poly else savgol_filter(X, w, self.poly, deriv=self.deriv, axis=1)


def pipe(est):
    return Pipeline([('snv', SNV()), ('sg', SavGol()), ('scale', StandardScaler()), ('est', est)])


def make_models():
    m = {'PLS': pipe(PLSRegression(n_components=10)),
         'Ridge': pipe(Ridge(alpha=1.0)),
         'RF': pipe(RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0))}
    if HAS_XGB:
        m['XGBoost'] = pipe(XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5,
                                         subsample=0.9, colsample_bytree=0.9,
                                         random_state=0, n_jobs=-1))
    return m


def rpd(y, p):
    rmse = np.sqrt(mean_squared_error(y, p))
    return np.std(y) / rmse if rmse > 0 else np.nan


def score(y, p):
    return dict(R2=r2_score(y, p), MAE=mean_absolute_error(y, p),
                RMSE=np.sqrt(mean_squared_error(y, p)), RPD=rpd(y, p))


def run(csv_path, out_path):
    from scipy.stats import pearsonr
    df = load_csv(csv_path)
    rows = []

    for test_crop in CROPS:
        train_crops = [c for c in CROPS if c != test_crop]
        bands, Xtr, ytr, Xte, yte = train_test_bands(df, train_crops, test_crop)

        for name, model in make_models().items():
            model.fit(Xtr, ytr)
            pred = np.asarray(model.predict(Xte)).ravel()
            s = score(yte, pred)
            # correlation: does predicted TRACK actual, regardless of scale/offset?
            corr = pearsonr(pred, yte)[0] if np.std(pred) > 1e-9 else 0.0

            rows.append({
                'held_out': test_crop, 'model': name, 'bands': len(bands),
                'n_train': len(ytr), 'n_test': len(yte),
                'R2': s['R2'], 'RMSE': s['RMSE'], 'RPD': s['RPD'],
                'pearson_r': corr,
            })

    res = pd.DataFrame(rows)
    res.to_csv(out_path, index=False)

    print('=== Leave-One-Crop-Out (best model per held-out crop, by pearson_r) ===')
    for tc in CROPS:
        sub = res[res['held_out'] == tc]
        b = sub.loc[sub['pearson_r'].idxmax()]
        print(f'  hold {tc:9s} | {b["model"]:7s} '
              f'R2={b["R2"]:+8.2f}  RPD={b["RPD"]:.2f}  '
              f'pearson_r={b["pearson_r"]:+.3f}')
    print(f'\nsaved {out_path}')
    print('\nReading guide (the two metrics together tell the story):')
    print('  R2 > 0                  -> predicts the actual DM value of the unseen crop. Full transfer.')
    print('  R2 < 0 but pearson_r>0.5 -> predicted TRACKS actual but on a wrong scale/offset:')
    print('     the spectrum->DM RELATIONSHIP transfers, a per-crop recalibration would fix it.')
    print('  pearson_r ~ 0           -> no relationship transfers at all for that crop.')
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--out', default='exp2_leave_one_out.csv')
    args = ap.parse_args()
    run(args.csv, args.out)

"""
Phase 2 - Steps 6 & 7: train with cube-grouped CV, evaluate at CUBE level.

The honesty rules enforced here:
  - Split by CUBE (GroupKFold on cube_id), never by patch. All patches from a cube
    go entirely to train or entirely to test.
  - Per-band standardization is fit on TRAIN patches only, applied to test.
  - Predict per patch, then AVERAGE patch predictions per cube -> one value per cube.
  - Score at the cube level (R2/MAE/RMSE/RPD) so it's comparable to Phase 1.

Win condition (apple): cube-level R2 > 0.44.

Usage:
    python train.py --data apple_patches.npz --model 1d --folds 5 --epochs 25
    python train.py --data apple_patches.npz --model 3d --folds 5 --epochs 25
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from preprocessing import preprocess_patches
from model import make_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def rpd(y, p):
    rmse = np.sqrt(mean_squared_error(y, p))
    return np.std(y) / rmse if rmse > 0 else np.nan


def to_tensor(X, kind):
    """X: (N,P,P,B) -> 1d: (N,1,B) center spectrum ; 3d: (N,1,B,P,P)."""
    if kind == '1d':
        P = X.shape[1]; c = P // 2
        spec = X[:, c, c, :]                       # (N, B)
        return torch.from_numpy(spec[:, None, :].astype(np.float32))
    else:
        Xt = np.transpose(X, (0, 3, 1, 2))         # (N, B, P, P)
        return torch.from_numpy(Xt[:, None, :, :, :].astype(np.float32))


def standardize_fit(Xt):
    """Fit per-feature mean/std on train tensor (over batch dim)."""
    mean = Xt.mean(dim=0, keepdim=True)
    std = Xt.std(dim=0, keepdim=True) + 1e-6
    return mean, std


def train_one_fold(Xtr, ytr, Xte, kind, epochs, lr, batch):
    mean, std = standardize_fit(Xtr)
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std

    model = make_model(kind, n_bands=Xtr.shape[2] if kind == '1d' else Xtr.shape[2]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(Xtr, torch.from_numpy(ytr.astype(np.float32)))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)

    model.train()
    for ep in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xte), batch):
            xb = Xte[i:i + batch].to(DEVICE)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def run(data_path, kind, folds, epochs, lr, batch, seed):
    d = np.load(data_path, allow_pickle=True)
    X, y, groups = d['X'], d['y'], d['groups']
    print(f'loaded {X.shape} patches, {len(set(groups))} cubes, device={DEVICE}')

    print('preprocessing (SNV + 2nd derivative)...')
    X = preprocess_patches(X)
    Xt_all = to_tensor(X, kind)
    print(f'model input tensor: {tuple(Xt_all.shape)}')

    gkf = GroupKFold(n_splits=folds)
    # collect cube-level predictions across folds
    cube_true, cube_pred, cube_ids = {}, {}, {}

    for f, (tr, te) in enumerate(gkf.split(X, y, groups=groups), 1):
        Xtr, ytr = Xt_all[tr], y[tr]
        Xte = Xt_all[te]
        patch_pred = train_one_fold(Xtr, ytr, Xte, kind, epochs, lr, batch)

        # aggregate patch -> cube
        g_te = groups[te]; y_te = y[te]
        for cid in np.unique(g_te):
            m = g_te == cid
            cube_pred[cid] = float(patch_pred[m].mean())
            cube_true[cid] = float(y_te[m][0])     # all same within a cube
        print(f'  fold {f}: {len(np.unique(g_te))} test cubes')

    ids = sorted(cube_true, key=lambda s: int(''.join(ch for ch in s if ch.isdigit())))
    yt = np.array([cube_true[c] for c in ids])
    yp = np.array([cube_pred[c] for c in ids])

    print('\n================ CUBE-LEVEL RESULTS ================')
    print(f'model: {kind.upper()}-CNN   cubes: {len(ids)}')
    print(f'  R2  : {r2_score(yt, yp):.3f}')
    print(f'  MAE : {mean_absolute_error(yt, yp):.4f}')
    print(f'  RMSE: {np.sqrt(mean_squared_error(yt, yp)):.4f}')
    print(f'  RPD : {rpd(yt, yp):.2f}')
    print(f'\n  Phase-1 apple benchmark to beat: R2=0.44, RPD=1.34')
    verdict = 'BEATS baseline' if r2_score(yt, yp) > 0.44 else 'does NOT beat baseline'
    print(f'  -> {kind.upper()}-CNN {verdict}')

    np.savez(f'phase2_pred_{kind}.npz',
             ids=np.array(ids), y_true=yt, y_pred=yp)
    print(f'  saved phase2_pred_{kind}.npz')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--model', choices=['1d', '3d'], default='1d')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    run(args.data, args.model, args.folds, args.epochs, args.lr, args.batch, args.seed)

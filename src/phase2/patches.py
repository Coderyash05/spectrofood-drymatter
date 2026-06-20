"""
Phase 2 - Step 3: extract spatial-spectral patches from masked cubes.

Each cube (H, W, 141) becomes many small patches (P, P, 141). Every patch:
  - inherits the cube's dry-matter label
  - carries its cube_id as a GROUP tag  <-- critical for honest train/test splits

Why patches: 240 cubes is far too few to train a 3D-CNN. Foreground is ~85-89%
of each ~57x62 cube => thousands of valid patch centers per cube. With a stride
we subsample to keep the set balanced and manageable.

A patch is kept only if its center pixel is foreground AND at least `min_fg_frac`
of the patch window is foreground (so we don't train on half-background patches).

Output: a compressed .npz per run containing
    X         float32 (N, P, P, 141)
    y         float32 (N,)              dry matter
    groups    object  (N,)              cube_id per patch  -> use with GroupKFold

Usage:
    python patches.py --mat F:\\project\\data\\cubes\\Apple.mat \
        --labels apple_labels.csv --prefix A \
        --patch 7 --stride 4 --max-per-cube 200 --out apple_patches.npz
"""
import argparse
import numpy as np
import pandas as pd
from scipy.io import loadmat

from cube_loader import load_cube, foreground_mask


def extract_from_cube(cube, mask, patch=7, stride=4, min_fg_frac=0.6,
                      max_per_cube=None, rng=None):
    """Yield (patch_array, ) for valid centers. Returns list of (P,P,B) arrays."""
    H, W, B = cube.shape
    half = patch // 2
    coords = []
    for i in range(half, H - half, stride):
        for j in range(half, W - half, stride):
            if not mask[i, j]:
                continue
            win = mask[i - half:i + half + 1, j - half:j + half + 1]
            if win.mean() >= min_fg_frac:
                coords.append((i, j))

    if max_per_cube and len(coords) > max_per_cube:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(len(coords), size=max_per_cube, replace=False)
        coords = [coords[k] for k in idx]

    patches = np.empty((len(coords), patch, patch, B), dtype=np.float32)
    for k, (i, j) in enumerate(coords):
        patches[k] = cube[i - half:i + half + 1, j - half:j + half + 1, :]
    return patches


def build_patch_dataset(mat_path, labels_csv, prefix, patch=7, stride=4,
                        min_fg_frac=0.6, max_per_cube=200, out_path='patches.npz',
                        seed=0):
    labels = pd.read_csv(labels_csv).set_index('cube_id')['dry_matter'].to_dict()
    mat = loadmat(mat_path)
    rng = np.random.default_rng(seed)

    cube_ids = [k for k in labels if k in mat]   # only cubes we have AND have labels
    cube_ids.sort(key=lambda s: int(s[len(prefix):]))

    X_list, y_list, g_list = [], [], []
    for cid in cube_ids:
        cube = load_cube(mat, cid)
        mask = foreground_mask(cube)
        P = extract_from_cube(cube, mask, patch, stride, min_fg_frac, max_per_cube, rng)
        if len(P) == 0:
            print(f'  {cid}: 0 patches (skipped)')
            continue
        X_list.append(P)
        y_list.append(np.full(len(P), labels[cid], dtype=np.float32))
        g_list.append(np.array([cid] * len(P)))
        print(f'  {cid}: {len(P)} patches')

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    groups = np.concatenate(g_list, axis=0)

    np.savez_compressed(out_path, X=X, y=y, groups=groups)
    print(f'\nwrote {out_path}')
    print(f'  X={X.shape}  y={y.shape}  groups={groups.shape}')
    print(f'  cubes used: {len(cube_ids)}  patches/cube avg: {len(y)/len(cube_ids):.0f}')
    print(f'  total patches: {len(y)}')
    return out_path


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mat', required=True)
    ap.add_argument('--labels', required=True, help='cube_id,dry_matter csv from labels.py')
    ap.add_argument('--prefix', default='A')
    ap.add_argument('--patch', type=int, default=7)
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--min-fg-frac', type=float, default=0.6)
    ap.add_argument('--max-per-cube', type=int, default=200)
    ap.add_argument('--out', default='apple_patches.npz')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    build_patch_dataset(args.mat, args.labels, args.prefix, args.patch, args.stride,
                        args.min_fg_frac, args.max_per_cube, args.out, args.seed)

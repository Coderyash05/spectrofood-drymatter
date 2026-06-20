"""
Phase 2 - Step 2: load a single hyperspectral cube and mask out background.

A cube is (H, W, Bands) uint16. Most of the frame is dark background with no
sample. We build a foreground mask from per-pixel mean reflectance and keep only
sample pixels. Loading is lazy (one cube at a time) so this scales to the large
leek/broccoli files later without blowing memory.

Apple specifics (verified from Apple.mat):
  - dtype uint16, values ~0..9300
  - 141 bands, wavelengths 430-990 nm (visible + very-near-IR only)

Usage (self-test on one cube):
    python cube_loader.py --mat path/to/Apple.mat --cube A1
"""
import argparse
import numpy as np
from scipy.io import loadmat


def load_cube(mat_or_path, cube_id):
    """Load one cube as float32 (H, W, Bands). Accepts a loaded mat dict or a path."""
    mat = loadmat(mat_or_path) if isinstance(mat_or_path, str) else mat_or_path
    if cube_id not in mat:
        raise KeyError(f'{cube_id} not in .mat file')
    return np.asarray(mat[cube_id], dtype=np.float32)


def foreground_mask(cube, method='otsu', min_frac=0.02):
    """
    Return a boolean (H, W) mask of sample pixels.
    Strategy: collapse spectral axis to a brightness image (mean over bands),
    then threshold. 'otsu' picks the threshold automatically; 'mean' uses the
    image mean. Background in these cubes is near-zero, so this separates cleanly.
    """
    brightness = cube.mean(axis=2)  # (H, W)
    b = brightness
    if method == 'otsu':
        # lightweight Otsu without skimage
        hist, edges = np.histogram(b.ravel(), bins=256)
        centers = (edges[:-1] + edges[1:]) / 2
        w = hist.astype(float)
        total = w.sum()
        sumv = (w * centers).sum()
        sumB = 0.0
        wB = 0.0
        best_t, best_var = centers[0], -1.0
        for i in range(256):
            wB += w[i]
            if wB == 0:
                continue
            wF = total - wB
            if wF == 0:
                break
            sumB += w[i] * centers[i]
            mB = sumB / wB
            mF = (sumv - sumB) / wF
            var_between = wB * wF * (mB - mF) ** 2
            if var_between > best_var:
                best_var, best_t = var_between, centers[i]
        thr = best_t
    else:
        thr = b.mean()

    mask = brightness > thr
    # guard: if mask is degenerate, fall back to mean threshold
    if mask.mean() < min_frac:
        mask = brightness > brightness.mean()
    return mask


def masked_pixels(cube, mask):
    """Return (n_pixels, bands) array of foreground spectra."""
    return cube[mask]  # boolean index over (H, W) -> (n, bands)


def cube_summary(mat_path, cube_id):
    mat = loadmat(mat_path)
    cube = load_cube(mat, cube_id)
    mask = foreground_mask(cube)
    fg = masked_pixels(cube, mask)
    print(f'{cube_id}: shape={cube.shape} dtype_in=uint16->float32')
    print(f'  brightness range: {cube.mean(2).min():.1f} - {cube.mean(2).max():.1f}')
    print(f'  foreground pixels: {mask.sum()} / {mask.size} ({100*mask.mean():.1f}%)')
    print(f'  foreground spectra array: {fg.shape}')
    print(f'  mean spectrum (first 5 bands): {fg.mean(0)[:5].round(1)}')
    return cube, mask


def save_mask_preview(mat_path, cube_id, out_png=None):
    """Save a side-by-side PNG: brightness image | mask | masked sample. For eyeballing."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    mat = loadmat(mat_path)
    cube = load_cube(mat, cube_id)
    brightness = cube.mean(axis=2)
    mask = foreground_mask(cube)
    out_png = out_png or f'{cube_id}_mask.png'

    fig, ax = plt.subplots(1, 3, figsize=(11, 4))
    ax[0].imshow(brightness, cmap='gray'); ax[0].set_title(f'{cube_id} brightness')
    ax[1].imshow(mask, cmap='gray'); ax[1].set_title(f'mask ({100*mask.mean():.0f}% fg)')
    overlay = brightness.copy()
    overlay[~mask] = 0
    ax[2].imshow(overlay, cmap='gray'); ax[2].set_title('foreground only')
    for a in ax: a.axis('off')
    plt.tight_layout(); plt.savefig(out_png, dpi=90); plt.close()
    print(f'saved {out_png}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mat', required=True)
    ap.add_argument('--cube', default='A1')
    ap.add_argument('--preview', action='store_true', help='save a mask preview PNG')
    args = ap.parse_args()
    cube_summary(args.mat, args.cube)
    if args.preview:
        save_mask_preview(args.mat, args.cube)
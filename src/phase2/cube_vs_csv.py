"""
Phase 2 - DEBUG: is the cube spectrum the same data as the CSV spectrum?

cube_mean_probe got ~0 (not 0.44), so the patch pipeline does NOT reproduce the
Phase-1 CSV baseline. Prime suspect: the raw uint16 cube counts are NOT the same
as the calibrated reflectance in the CSV (different wavelength range too: cube
430-990 nm / 141 bands  vs  CSV 397-1716 nm / 421 cols, apple filling 142).

This script overlays, for a few cubes, the cube mean spectrum vs the CSV row,
and reports correlation on the overlapping wavelength range. It needs the .mat,
the CSV, and matplotlib.

Reads cube wavelengths from the .mat 'Wavelengths' variable and CSV wavelengths
from the CSV header (numeric column names).

Usage:
    python cube_vs_csv.py --mat F:\\project\\data\\cubes\\Apple.mat \
        --csv F:\\project\\data\\SpectroFood_dataset.csv --prefix A --cubes A1 A50 A240
"""
import argparse
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import pearsonr

from cube_loader import load_cube, foreground_mask


def csv_apple_spectra(csv_path, prefix):
    df = pd.read_csv(csv_path)
    df.columns.values[0] = 'label'
    df['label'] = df['label'].astype(str).str.strip()
    df = df[df['label'].str.match(rf'^{prefix}\d+$')].copy()
    wl_cols = [c for c in df.columns if c not in ('label', 'DRY MATTER')]
    # numeric wavelength values from column names
    wls = []
    keep = []
    for c in wl_cols:
        try:
            wls.append(float(c)); keep.append(c)
        except ValueError:
            pass
    spec = df[keep].apply(pd.to_numeric, errors='coerce')
    return df['label'].values, np.array(wls), spec.values


def run(mat_path, csv_path, prefix, cube_ids):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    mat = loadmat(mat_path)
    cube_wl = np.asarray(mat['Wavelengths']).ravel().astype(float)
    labels, csv_wl, csv_spec = csv_apple_spectra(csv_path, prefix)
    lab2row = {l.strip(): i for i, l in enumerate(labels)}

    print(f'cube bands: {len(cube_wl)}  ({cube_wl.min():.0f}-{cube_wl.max():.0f} nm)')
    print(f'csv  bands: {len(csv_wl)}  ({csv_wl.min():.0f}-{csv_wl.max():.0f} nm)')

    # overlap range
    lo, hi = max(cube_wl.min(), csv_wl.min()), min(cube_wl.max(), csv_wl.max())
    print(f'overlap: {lo:.0f}-{hi:.0f} nm\n')

    fig, axes = plt.subplots(len(cube_ids), 2, figsize=(12, 3.2 * len(cube_ids)))
    if len(cube_ids) == 1:
        axes = axes[None, :]

    for r, cid in enumerate(cube_ids):
        cube = load_cube(mat, cid)
        mask = foreground_mask(cube)
        cube_mean = cube[mask].mean(axis=0)            # (141,) raw counts

        row = lab2row.get(cid)
        csv_row = csv_spec[row]

        # interpolate both onto common overlap grid for correlation
        grid = np.linspace(lo, hi, 100)
        c_i = np.interp(grid, cube_wl, cube_mean)
        # csv has NaNs outside apple's native range -> mask them
        ok = ~np.isnan(csv_row)
        s_i = np.interp(grid, csv_wl[ok], csv_row[ok])
        # correlation of shape (z-scored)
        cz = (c_i - c_i.mean()) / (c_i.std() + 1e-9)
        sz = (s_i - s_i.mean()) / (s_i.std() + 1e-9)
        r_pear = pearsonr(cz, sz)[0]

        axes[r, 0].plot(cube_wl, cube_mean, color='tab:red')
        axes[r, 0].set_title(f'{cid} CUBE mean (raw counts)')
        axes[r, 0].set_xlabel('nm')
        axes[r, 1].plot(csv_wl[ok], csv_row[ok], color='tab:blue')
        axes[r, 1].set_title(f'{cid} CSV row  | shape corr={r_pear:+.2f}')
        axes[r, 1].set_xlabel('nm')
        print(f'{cid}: shape correlation on overlap = {r_pear:+.2f}')

    plt.tight_layout(); plt.savefig('cube_vs_csv.png', dpi=90); plt.close()
    print('\nsaved cube_vs_csv.png')
    print('\nInterpretation:')
    print('  |corr| high (>0.8) -> same underlying spectrum; mismatch is scaling -> calibrate cubes')
    print('  |corr| low         -> different data entirely; check band order / reflectance vs raw counts')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mat', required=True)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--prefix', default='A')
    ap.add_argument('--cubes', nargs='+', default=['A1', 'A50', 'A240'])
    args = ap.parse_args()
    run(args.mat, args.csv, args.prefix, args.cubes)
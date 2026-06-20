"""
Phase 2 - Step 1: build the cube -> dry-matter label table.

Join key is the CUBE VARIABLE NAME (e.g. 'A1'), NOT the .mat 'Labels' array.
The 'Labels' variable in Apple.mat only stores the category ('A' x240) and is
useless for matching individual samples. scipy loads .mat vars into a dict whose
order is arbitrary, so we map by parsing the variable name, never by position.

Usage:
    python labels.py --mat path/to/Apple.mat --csv path/to/SpectroFood_dataset.csv \
                     --prefix A --out results/phase2/apple_labels.csv
"""
import argparse
import re
import numpy as np
import pandas as pd
from scipy.io import loadmat


def cube_names_from_mat(mat_path, prefix):
    """Return sorted list of cube variable names like ['A1','A2',...] from a .mat file."""
    mat = loadmat(mat_path)
    pat = re.compile(rf'^{prefix}(\d+)$')
    names = [k for k in mat.keys() if pat.match(k)]
    # sort by the integer index, not lexicographically (so A2 < A10)
    names.sort(key=lambda k: int(pat.match(k).group(1)))
    return names, mat


def dry_matter_from_csv(csv_path, prefix):
    """Return DataFrame [cube_id, dry_matter] for one category from the spectral CSV."""
    df = pd.read_csv(csv_path)
    df.columns.values[0] = 'label'
    df['label'] = df['label'].astype(str).str.strip()
    df = df[df['label'].str.match(rf'^{prefix}\d+$')].copy()
    df['dry_matter'] = pd.to_numeric(df['DRY MATTER'], errors='coerce')
    df = df.dropna(subset=['dry_matter'])
    return df[['label', 'dry_matter']].rename(columns={'label': 'cube_id'})


def build_label_table(mat_path, csv_path, prefix, out_path):
    cube_names, _ = cube_names_from_mat(mat_path, prefix)
    dm = dry_matter_from_csv(csv_path, prefix)

    # join on name
    table = pd.DataFrame({'cube_id': cube_names}).merge(dm, on='cube_id', how='left')

    missing = table[table['dry_matter'].isna()]['cube_id'].tolist()
    if missing:
        print(f'WARNING: {len(missing)} cubes have no dry-matter match: {missing[:10]}...')
    else:
        print(f'OK: all {len(table)} cubes matched a dry-matter value.')

    table = table.dropna(subset=['dry_matter']).reset_index(drop=True)
    table.to_csv(out_path, index=False)
    print(f'wrote {out_path}  ({len(table)} rows)')
    print(f'DM range: {table.dry_matter.min():.4f} - {table.dry_matter.max():.4f} '
          f'(mean {table.dry_matter.mean():.4f})')
    return table


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mat', required=True, help='path to category .mat file')
    ap.add_argument('--csv', required=True, help='path to SpectroFood_dataset.csv')
    ap.add_argument('--prefix', default='A', help='category prefix: A/B/L/M')
    ap.add_argument('--out', default='apple_labels.csv')
    args = ap.parse_args()
    build_label_table(args.mat, args.csv, args.prefix, args.out)

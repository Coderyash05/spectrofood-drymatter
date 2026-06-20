"""
Phase 3 - data module: cross-sensor generalization splits.

The CSV stacks four crops, each measured by a DIFFERENT camera, onto one shared
398-1717 nm grid. Each crop only fills its own native bands; the rest are NaN.
To train ONE model across crops we must restrict to bands they all share.

Verified common-band counts (no-NaN intersection):
    all four          -> 141   (apple is the limiter: 430-775 nm)
    leek + mushroom   -> 204   (richest pair, no apple)
    broccoli + leek   -> 150
DM ranges (%): apple 13.5-17.4, broccoli 11.9-20.2, leek 8.1-19.1, mushroom 5.6-86.9
(mushroom is the range outlier -> the leave-one-out "break" case.)

Core API:
    load_csv(path)                       -> cleaned DataFrame (label, crop, dry_matter, +bands)
    common_bands(df, crops)              -> list of wavelength-column names shared by all `crops`
    get_Xy(df, crops, bands=None)        -> (X, y, crop_labels) on the common bands
"""
import numpy as np
import pandas as pd

PREFIX_MAP = {'A': 'apple', 'B': 'broccoli', 'L': 'leek', 'M': 'mushroom'}
CROPS = ['apple', 'broccoli', 'leek', 'mushroom']


def load_csv(path):
    """Clean the SpectroFood CSV: drop header rows, parse crop, numeric DM + bands."""
    df = pd.read_csv(path)
    df.columns.values[0] = 'label'
    df['label'] = df['label'].astype(str).str.strip()
    df = df[df['label'].str.match(r'^[ABLM]\d+$')].copy()
    df['crop'] = df['label'].str.extract(r'^([ABLM])')[0].map(PREFIX_MAP)
    df['dry_matter'] = pd.to_numeric(df['DRY MATTER'], errors='coerce')
    df = df.dropna(subset=['dry_matter']).reset_index(drop=True)

    band_cols = [c for c in df.columns
                 if c not in ('label', 'DRY MATTER', 'crop', 'dry_matter')]
    df[band_cols] = df[band_cols].apply(pd.to_numeric, errors='coerce')
    # attach the band-column list as an attribute for convenience
    df.attrs['band_cols'] = band_cols
    return df


def _band_cols(df):
    return df.attrs.get('band_cols',
                        [c for c in df.columns
                         if c not in ('label', 'DRY MATTER', 'crop', 'dry_matter')])


def common_bands(df, crops):
    """Wavelength columns with NO NaN for every crop in `crops` (their shared range)."""
    band_cols = np.array(_band_cols(df))
    present = None
    for c in crops:
        sub = df.loc[df['crop'] == c, band_cols]
        ok = set(band_cols[(sub.isna().mean() == 0).values])
        present = ok if present is None else (present & ok)
    # keep original order, sorted by wavelength
    return sorted(present, key=lambda s: float(s))


def get_Xy(df, crops, bands=None):
    """
    Return (X, y, crop_labels) for the given crops on their common bands.
    `crops` may be a single crop name or a list. If `bands` is given, use those
    (so train and test can share an identical feature space); else compute common.
    """
    if isinstance(crops, str):
        crops = [crops]
    if bands is None:
        bands = common_bands(df, crops)
    sub = df[df['crop'].isin(crops)]
    X = sub[bands].values.astype(np.float32)
    y = sub['dry_matter'].values.astype(np.float32)
    crop_labels = sub['crop'].values
    return X, y, crop_labels


def train_test_bands(df, train_crops, test_crop):
    """
    For leave-one-out: feature space must be bands shared by BOTH the training
    crops AND the test crop (so the model's inputs exist for the test crop).
    Returns (bands, Xtr, ytr, Xte, yte).
    """
    crops_all = list(train_crops) + [test_crop]
    bands = common_bands(df, crops_all)
    Xtr, ytr, _ = get_Xy(df, list(train_crops), bands=bands)
    Xte, yte, _ = get_Xy(df, test_crop, bands=bands)
    return bands, Xtr, ytr, Xte, yte


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'SpectroFood_dataset.csv'
    df = load_csv(path)
    print('rows:', len(df), '| crops:', df['crop'].value_counts().to_dict())
    print('all-four common bands:', len(common_bands(df, CROPS)))
    for tc in CROPS:
        tr = [c for c in CROPS if c != tc]
        bands, Xtr, ytr, Xte, yte = train_test_bands(df, tr, tc)
        print(f'hold out {tc:9s}: bands={len(bands):3d}  '
              f'train={Xtr.shape}  test={Xte.shape}  '
              f'test DM {yte.min()*100:.0f}-{yte.max()*100:.0f}%')
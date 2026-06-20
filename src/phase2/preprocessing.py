"""
Phase 2 - Step 4: spectral preprocessing for patches.

Reuses the Phase-1 winning transform for apple: SNV + Savitzky-Golay 2nd derivative,
applied along the 141-band spectral axis of every pixel in every patch.

These two transforms are STATELESS / per-spectrum (no parameters learned from the
training set), so applying them to all patches up front does NOT leak test info.
The only fitted step (per-band standardization) is fit on TRAIN patches inside each
CV fold - see train.py. Keep it that way.

Shapes: patches arrive as (N, P, P, B). We transform along the last axis.
"""
import numpy as np
from scipy.signal import savgol_filter


def snv(x, axis=-1, eps=1e-8):
    """Standard Normal Variate along `axis`. Per-spectrum mean-center + scale."""
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    return (x - mu) / (sd + eps)


def savgol_deriv(x, window=11, poly=2, deriv=2, axis=-1):
    """Savitzky-Golay derivative along spectral axis."""
    B = x.shape[axis]
    w = window if window % 2 == 1 else window - 1
    w = min(w, B if B % 2 == 1 else B - 1)
    if w <= poly:
        return x
    return savgol_filter(x, window_length=w, polyorder=poly, deriv=deriv, axis=axis)


def preprocess_patches(X, window=11, poly=2, deriv=2):
    """
    X: (N, P, P, B) float32. Apply SNV then SG-2nd-derivative along band axis.
    Returns float32, same shape.
    """
    X = np.asarray(X, dtype=np.float32)
    X = snv(X, axis=-1)
    X = savgol_deriv(X, window=window, poly=poly, deriv=deriv, axis=-1)
    return X.astype(np.float32)


if __name__ == '__main__':
    # smoke test
    rng = np.random.default_rng(0)
    X = rng.random((8, 7, 7, 141)).astype(np.float32) * 5000
    Y = preprocess_patches(X)
    print('in', X.shape, '-> out', Y.shape, Y.dtype)
    print('per-spectrum mean after SNV+deriv ~0:', round(float(Y.mean()), 4))

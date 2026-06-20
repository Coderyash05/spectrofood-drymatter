"""
Phase 2 - Step 5: models (PyTorch).

Two models, used in order:

1) SpectralCNN1D - sanity baseline. Operates on the CENTER-PIXEL spectrum of each
   patch only (ignores spatial info). If this does NOT beat the Phase-1 baseline
   (apple R2=0.44), then spatial features are unlikely to help either.

2) SpectralSpatialCNN3D - the real model. Shallow Conv3D over the full
   (1, B, P, P) patch volume. Kept shallow on purpose: 24k correlated patches
   overfit a deep net quickly.

Input convention:
  patches are (N, P, P, B) in numpy. In the Dataset we move to torch as:
    1D model: center spectrum -> (N, 1, B)
    3D model: (N, 1, B, P, P)   [channels=1, depth=bands, H=P, W=P]
"""
import torch
import torch.nn as nn


class SpectralCNN1D(nn.Module):
    """1D-CNN over the spectral axis of the center pixel. Sanity baseline."""
    def __init__(self, n_bands=141):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32, 32), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, x):           # x: (N, 1, B)
        return self.head(self.net(x)).squeeze(-1)


class SpectralSpatialCNN3D(nn.Module):
    """Shallow 3D-CNN over (1, B, P, P) patch volumes."""
    def __init__(self, n_bands=141, patch=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3), padding=(3, 1, 1)),
            nn.BatchNorm3d(8), nn.ReLU(),
            nn.MaxPool3d((2, 1, 1)),                 # pool spectral only, keep small spatial dims
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            nn.BatchNorm3d(16), nn.ReLU(),
            nn.MaxPool3d((2, 1, 1)),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(16, 32), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, x):           # x: (N, 1, B, P, P)
        return self.head(self.features(x)).squeeze(-1)


def make_model(kind, n_bands=141, patch=7):
    if kind == '1d':
        return SpectralCNN1D(n_bands)
    if kind == '3d':
        return SpectralSpatialCNN3D(n_bands, patch)
    raise ValueError(kind)


if __name__ == '__main__':
    # shape smoke test
    b = 4
    m1 = make_model('1d'); x1 = torch.randn(b, 1, 141)
    print('1D out:', m1(x1).shape)
    m3 = make_model('3d'); x3 = torch.randn(b, 1, 141, 7, 7)
    print('3D out:', m3(x3).shape)
    print('1D params:', sum(p.numel() for p in m1.parameters()))
    print('3D params:', sum(p.numel() for p in m3.parameters()))

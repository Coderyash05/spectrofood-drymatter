# SpectroFood — Dry Matter Estimation & Cross-Sensor Generalization

Predicting **dry matter content** in four food crops (apple, broccoli, leek, mushroom) from
hyperspectral data, and testing whether a single model **generalizes across different cameras**.

> **One-line finding:** a dry matter model trained on three crops cannot predict a fourth unseen crop
> (zero-shot transfer fails), but adding as few as **5–20 samples** of the target crop restores useful
> prediction for three of four crops — apple being the exception due to its intrinsically weak signal.

Dataset: [SpectroFood](https://doi.org/10.1016/j.dib.2024.110040) (Malounas et al., *Data in Brief*, 2024).

---

## Headline results

| Phase | Question | Result |
|-------|----------|--------|
| **1. Spectral baselines** | Can we predict dry matter per crop? | Yes for leek (R²=0.91) & mushroom (0.87); weakly for broccoli (0.58) & apple (0.44). |
| **2. Cube CNN** | Do spatial features in the raw image cubes help? | No — diagnosed that the cubes carry no dry-matter signal beyond the mean spectrum. A 3D-CNN was deliberately ruled out, with evidence. |
| **3. Cross-sensor transfer** | Does a model generalize to an unseen crop+camera? | Zero-shot fails (R²<0); few-shot (5–20 samples) rescues 3 of 4 crops; apple resists. |

Key figures (in `results/phase3/plots/`):
- `fewshot_recovery.png` — the headline: transfer failing at k=0, recovering with target data.
- `ceiling_vs_naive.png` — within-crop ceiling vs naive transfer.
- `pairwise_heatmap.png` — which crop/camera pairs share structure.

---

## Project structure

```
data/                         raw inputs (CSV + .mat cubes — see "Data" below)
src/
  phase1/phase1.ipynb         per-crop spectral regression baselines
  phase2/                     cube pipeline + the diagnostic chain that ruled out the CNN
  phase3/                     cross-sensor experiments + summary notebook
results/                      generated metrics (CSV) and plots (PNG)
PROJECT_REPORT.md             full write-up, start to end
```

---

## Data

The dataset is not committed (the cubes are large). Download from Zenodo and place as:

```
data/
  SpectroFood_dataset.csv         # extracted mean spectra + dry matter (Zenodo 10.5281/zenodo.8362947)
  cubes/
    Apple.mat                     # Zenodo 10.5281/zenodo.10301753 (+ broccoli/leek/mushroom records)
    Broccoli_15.mat
    Leek_1.mat
    Mushroom_1.mat
```

Phase 1 and Phase 3 need only `SpectroFood_dataset.csv`. The cubes are for Phase 2.

---

## Setup

```bash
pip install -r requirements.txt
```

`torch` is needed only for Phase 2. Phase 1 and Phase 3 run on numpy/pandas/scikit-learn/scipy/matplotlib.

---

## How to run

Run everything from the project root.

**Phase 1 — baselines:** open `src/phase1/phase1.ipynb` and run all cells.

**Phase 3 — cross-sensor study:**
```bash
python src/phase3/exp1_within.py        --csv data/SpectroFood_dataset.csv --out results/phase3/exp1_within_crop.csv
python src/phase3/exp2_leave_one_out.py --csv data/SpectroFood_dataset.csv --out results/phase3/exp2_leave_one_out.csv
python src/phase3/exp_fewshot.py        --csv data/SpectroFood_dataset.csv --out results/phase3/exp_fewshot.csv --seeds 5 --model RF
python src/phase3/exp3_pairwise.py      --csv data/SpectroFood_dataset.csv --out results/phase3/exp3_pairwise.csv
python src/phase3/uncertainty.py        --csv data/SpectroFood_dataset.csv --out results/phase3/exp2_with_ci.csv
```
Then open `src/phase3/phase3.ipynb` to assemble the figures.

**Phase 2 — cube CNN (optional, reproduces the negative result):**
```bash
python src/phase2/labels.py      --mat data/cubes/Apple.mat --csv data/SpectroFood_dataset.csv --prefix A --out results/phase2/apple_labels.csv
python src/phase2/patches.py     --mat data/cubes/Apple.mat --labels results/phase2/apple_labels.csv --prefix A --max-per-cube 100 --out results/phase2/apple_patches.npz
python src/phase2/train.py       --data results/phase2/apple_patches.npz --model 3d --folds 5 --epochs 25
# diagnostics: rf_probe.py, cube_mean_probe.py, cube_vs_csv.py
```

---

## Methodology notes

- **No data leakage.** Cross-validation is group-aware: by crop in Phase 3, by cube in Phase 2 (all patches from one cube stay on one side of the split). Preprocessing (SNV, Savitzky–Golay) is fit on training folds only.
- **Uncertainty.** `uncertainty.py` reports bootstrap 95% CIs for single held-out sets and per-fold std for cross-validated scores (e.g. mushroom zero-shot R² = −0.14 [−0.23, −0.07]).
- **Honest negatives.** Phase 2's "the CNN doesn't help" is a substantiated finding reached through a documented diagnostic chain, not an abandoned attempt.

See `PROJECT_REPORT.md` for the full narrative, all results, limitations, and future work.
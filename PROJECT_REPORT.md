# SpectroFood: Dry Matter Estimation and Cross-Sensor Generalization from Hyperspectral Data

## Abstract

This project investigates dry matter content estimation in four food crops — apple, broccoli, leek, and mushroom — from hyperspectral measurements in the SpectroFood dataset. The work proceeded in three phases. Phase 1 established per-crop spectral regression baselines from the dataset's extracted mean spectra. Phase 2 investigated whether a 3D convolutional neural network on the raw hyperspectral image cubes could improve on those baselines, and through a structured diagnostic process determined that the cube data does not carry recoverable per-pixel dry matter signal — a documented negative result. Phase 3 reframed the work around the dataset's intended purpose: cross-sensor generalization. It showed that a single model trained on three crops cannot predict a fourth unseen crop measured by a different camera (zero-shot transfer fails), but that adding as few as 5–20 target-crop samples restores useful prediction for three of the four crops. Apple, which has the weakest spectral signal throughout, is the consistent exception.

---

## 1. Dataset

The SpectroFood dataset (Malounas et al., *Data in Brief*, 2024) contains hyperspectral measurements of four crops, each captured by a different camera:

| Crop | Camera | Samples | Native bands | Spectral range |
|------|--------|---------|--------------|----------------|
| Apple | Cubert ULTRIS S20 | 240 | 141 | 430–990 nm |
| Broccoli | Imec Snapscan | 250 | 150 | 470–900 nm |
| Leek | Specim FX10/FX17 | 288 | 421 | 398–1717 nm |
| Mushroom | SpecimIQ | 250 | 204 | 400–998 nm |

The dataset is distributed in two forms:

- **`SpectroFood_dataset.csv`** — 1028 samples of *extracted mean reflectance spectra* resampled onto a shared 398–1717 nm grid (421 columns), each with a dry matter content value. Because each crop only covers its own instrument's range, every crop fills only its native bands; the remaining columns are NaN.
- **`.mat` hyperspectral cubes** — the per-sample 3D image cubes `(height, width, bands)` from which the CSV spectra were derived. The cubes are already radiometrically corrected to reflectance by the camera software.

Dry matter ranges differ sharply between crops: apple 13.5–17.4%, broccoli 11.9–20.2%, leek 8.1–19.1%, and mushroom 5.6–86.9%. Mushroom's wide range makes it the outlier in all cross-crop work.

---

## 2. Phase 1 — Per-crop spectral regression

### Objective
Establish baseline dry matter prediction accuracy per crop from the extracted mean spectra.

### Method
The CSV required cleaning: stray header rows mixed into the data were dropped, the text-encoded dry matter column was coerced to numeric, and the per-crop band structure was handled by modeling each crop only on its own non-NaN bands. Preprocessing followed standard near-infrared practice — Standard Normal Variate (SNV) scatter correction and a Savitzky–Golay derivative — fitted inside cross-validation folds to avoid leakage. Five models were compared with 5-fold cross-validation: PLS, Ridge, Random Forest, XGBoost, and an MLP. Metrics were R², MAE, RMSE, and RPD (ratio of standard deviation to RMSE).

### Results (best model per crop, native bands)

| Crop | Best model | R² | RPD |
|------|-----------|-----|-----|
| Leek | Ridge | 0.91 | 3.25 |
| Mushroom | Ridge | 0.87 | 2.80 |
| Broccoli | SNV + 2nd-deriv + RF | 0.58 | 1.54 |
| Apple | 2nd-deriv + RF | 0.44 | 1.34 |

### Findings
Leek and mushroom were predicted well; broccoli and apple weakly. A preprocessing grid (scatter × derivative) confirmed that the 2nd derivative helped the weak crops marginally but could not overcome a data limitation: apple's near-identical spectra, narrow dry matter range, and visible-only spectral coverage (no informative near-infrared water bands past ~1000 nm) limit how much signal is present. A pooled cross-crop model scored higher (R²≈0.74) but this was identified as inflated — it partly learns *which crop* a sample is, since the crops' dry matter ranges barely overlap, so per-crop results are the valid benchmark. The MLP diverged on the small per-crop sets and was excluded from the final table.

---

## 3. Phase 2 — Hyperspectral cube CNN investigation

### Objective
Test whether spatial-spectral features from the raw image cubes, modeled with a 3D-CNN, could improve on the mean-spectrum baselines — particularly for the weak crops where averaging may discard spatial information.

### Method
The work began with apple, the only crop with a complete cube download (240/240). The pipeline built:
- a **label table** joining each cube (`A1`…`A240`) to its dry matter value by name (the `.mat` `Labels` variable stored only the crop letter and was unusable for matching);
- a **cube loader** with Otsu foreground masking to separate sample pixels from background (verified visually at ~85–89% foreground, cleanly tracing the fruit);
- a **patch extractor** producing 24,000 spatial patches (7×7×141) with each cube's dry matter as the label and cube ID as a group tag;
- **1D-CNN and 3D-CNN models** (PyTorch) trained with GroupKFold so all patches from a cube stayed on one side of the train/test split.

### Results and diagnostic chain
Both CNNs returned R² ≈ 0. Rather than tuning blindly, a sequence of diagnostics was run:

1. **Random Forest on patch center spectra** (raw / SNV / SNV+derivative) — all ≈ 0. This ruled out CNN training failure: a model that almost always finds signal also found none, so the signal was absent at the pixel level, not merely unlearned.
2. **Random Forest on cube-mean spectra** — also ≈ 0, *not* the ~0.44 of Phase 1. This showed the patch pipeline did not reproduce the Phase 1 baseline, indicating a mismatch between the cube data and the CSV.
3. **Direct cube-vs-CSV comparison** — the cube mean spectrum and the CSV row shared the same broad shape but only correlated ~0.5, because the comparison range was wrong: the CSV's apple data stops at ~775 nm while the cube extends to 990 nm.
4. **Reading the dataset paper** — resolved the question. The cubes are *already* calibrated reflectance, and the CSV is the *processed output* of the cubes: the authors removed background, dead pixels, and spikes, then computed each sample's mean spectrum.

### Findings
The CNN approach could not work for this task. The dry matter signal in apple is weak even in the clean mean spectrum (R²=0.44), and a crude cube-derived mean — polluted by dead pixels, spikes, and rim pixels — destroys it. More fundamentally, predicting dry matter from cubes is an attempt to re-derive the clean spectra the authors had already extracted, with no spatial signal to exploit. A 3D-CNN was therefore deliberately *not* pursued further. This is a substantiated negative result: spatial-spectral modeling adds nothing for dry matter on this data, and the reason is understood.

---

## 4. Phase 3 — Cross-sensor generalization

### Motivation
Phase 2 established that the hyperspectral cubes carry no dry matter signal beyond what the extracted mean spectrum already provides — spatial modeling offered no improvement because the signal is global, not local. This redirects attention from *how the signal is encoded within one measurement* to *how the signal transfers across measurements*. The SpectroFood dataset's defining feature is that each crop was captured by a different camera, and its stated purpose is to study whether models generalize across sensors. Having confirmed that the per-sample spectrum is the sole carrier of dry matter information, the natural and higher-value question becomes whether a model built on that spectrum generalizes to crops and cameras it has never seen. Phase 3 addresses exactly this.

### Objective
Does a dry matter model generalize to a crop and camera it has never seen?

### Method
All cross-crop work used the 141 bands common to all four crops (apple's narrow range is the limiter). Three experiments were run with the Phase 1 model pipeline:

1. **Within-crop ceiling** — 5-fold CV inside each crop on the shared 141 bands, giving the best achievable per crop and the reference each transfer result is compared against.
2. **Leave-one-crop-out** — train on three crops, predict the held-out fourth. This is the core generalization test. Both R² (does it predict the actual value) and Pearson correlation (does the prediction track the target regardless of scale) were reported, because the crops' dry matter ranges differ.
3. **Few-shot transfer** — add *k* samples of the held-out crop to training (k = 0, 5, 10, 20, 40, 80), averaged over multiple random seeds, and measure how prediction recovers.

A pairwise transfer grid (train on one crop, test on another, for all pairs) mapped the structure of the failure.

### Results

**Within-crop ceiling (141-band space):** apple 0.34, broccoli 0.59, leek 0.61, mushroom 0.80. (Lower than Phase 1's native-band numbers because the shared space drops leek's rich >990 nm bands — the cost of a common feature space.)

**Leave-one-crop-out (zero-shot):** every held-out crop returned negative R² and weak correlation (r ≈ 0.18–0.34). A model trained on three crops cannot predict the unseen fourth.

**Few-shot transfer (Random Forest, R² on remaining held-out samples):**

| Held-out crop | k=0 | k=5 | k=20 | k=80 |
|---------------|-----|-----|------|------|
| Mushroom | −0.23 | +0.10 | +0.31 | +0.59 |
| Broccoli | −10.2 | −0.26 | +0.20 | +0.43 |
| Leek | −38.9 | −0.02 | +0.18 | +0.32 |
| Apple | −866 | −238 | −24 | −15 |

**Pairwise transfer (Pearson r, train→test):** the failure is not uniform. The diagonal (within-crop) is strong (0.59–0.90). Apple is a surprisingly effective *donor* — apple→broccoli 0.44, apple→mushroom 0.50 — likely because its broad smooth visible-range spectrum is a generic feature set. Leek is a spectral island: nearly all transfers to and from it are near zero or negative. Transfer is also directional (apple→mushroom 0.50 but mushroom→apple 0.18).

### Findings
Zero-shot cross-sensor generalization fails completely. Few-shot adaptation rescues broccoli, leek, and mushroom with only 5–20 target samples, climbing toward their within-crop ceilings. Apple never recovers, even at k=80 — consistent with its status as the weakest-signal crop in every phase. The pairwise grid shows generalization has structure tied to spectral-range and camera overlap rather than failing uniformly.

---

## 5. Overall conclusions

1. **Dry matter is predictable from mean spectra where the spectral signal is strong** (leek, mushroom) and weakly where it is not (apple, broccoli). Signal strength tracks spectral coverage and dry matter range.
2. **The hyperspectral cubes add no value for dry matter prediction.** They are the processed source of the CSV spectra; per-pixel modeling cannot recover signal that only survives whole-sample averaging. A 3D-CNN was correctly ruled out rather than forced.
3. **Cross-sensor generalization fails zero-shot but is largely fixable with minimal target data** — except where the underlying signal is intrinsically weak (apple).

A single sentence captures the project: *a dry matter model trained on three crops cannot predict a fourth unseen crop, but adding as few as 5–20 samples of the target crop restores useful prediction for three of four crops, apple being the exception due to its intrinsically weak spectral signal.*

The recurring through-line — apple is the hardest crop in every phase, and generalization difficulty tracks spectral signal strength — ties the three phases into one coherent study.

---

## 6. Limitations and future work

**Limitations.** Absolute accuracy is modest (best cross-sensor recovery R² ≈ 0.5–0.6). Only apple had a complete cube download; broccoli, leek, and mushroom cubes were partial. The common-band constraint (141 bands) discards leek's informative near-infrared range in all cross-crop experiments.

**Future work.**
- Report all headline metrics with uncertainty intervals: per-fold standard deviation for cross-validated experiments (Phase 1, within-crop ceiling), multi-seed standard deviation for few-shot transfer (already computed), and percentile bootstrap confidence intervals for single held-out sets (leave-one-crop-out). A `uncertainty.py` utility implements all three; e.g. mushroom zero-shot transfer is R² = −0.14 [−0.23, −0.07] (95% bootstrap CI), confirming the failure is consistent rather than noise.
- Apply established domain-adaptation methods (e.g. CORAL, MMD-based feature alignment) to the zero-shot setting rather than relying on few-shot samples.
- Investigate per-crop spectral standardization to remove camera-specific offsets before transfer.
- Extend the few-shot study to active sampling — which target samples are most informative to label.

---

## 7. Reproducibility

```
data/                         raw inputs (CSV + .mat cubes)
src/phase1/phase1.ipynb       per-crop baselines
src/phase2/                   cube pipeline + diagnostics
src/phase3/                   cross-sensor experiments + notebook
results/                      generated metrics and plots
```

Phase 1 and Phase 3 run end to end from the CSV with scikit-learn, scipy, and matplotlib; Phase 2's CNN requires PyTorch. All experiments use fixed random seeds and group-aware cross-validation (by crop in Phase 3, by cube in Phase 2) so that reported scores do not leak across the relevant grouping.

---

*Dataset: Malounas, I. et al. (2024). SpectroFood dataset: A comprehensive fruit and vegetable hyperspectral meta-dataset for dry matter estimation. Data in Brief 52, 110040.*

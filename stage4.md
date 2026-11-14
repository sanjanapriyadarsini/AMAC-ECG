# AMAC-ECG Stage 4 — Final Evaluation + Decision Layer

## What this package contains
- `artifacts/ds1.npz` and `artifacts/ds2.npz`: the user's processed MIT-BIH datasets.
- `src/folds.json`: frozen patient-grouped DS1 folds.
- `artifacts/preds/`: 50 DS1 out-of-fold prediction files (10 configurations × 5 folds).
- `stage4.py`: Stage 4 final-evaluation and decision-layer runner.
- This README.

The datasets and OOF prediction files are carried forward from the supplied project assets. No synthetic dataset or synthetic prediction file is added.

## Locked experiment
Five base model/imbalance configurations, each with RR features enabled and disabled:
1. Logistic Regression — unweighted
2. Logistic Regression — class weight
3. XGBoost — class weight
4. CNN — class weight
5. CNN — focal loss

This gives 10 configurations × 5 random seeds = 50 DS2 forward-pass evaluation runs.

All five seeds use the same fixed patient-held-out DS1 fold (fold 0) for validation; the seed changes independently of fold assignment. LR/XGBoost fit on the same DS1 training partition; CNN uses the same held-out fold for early stopping.

## Stage 3 → Stage 4 calibration bridge
- For each configuration, Stage-3-compatible leave-one-fold-out temperature scaling is reproduced on DS1 OOF predictions: each held-out fold receives a temperature fitted on the other four folds.
- The mean of the five fold-specific temperatures is transferred to the seed-0 DS2 probabilities only.
- Calibration comparisons are seed-0-specific. Seeds 1–4 are not claimed as calibrated and calibration robustness across seeds is not claimed.
- For CNN + Focal Loss + RR, the mean fold temperature should reproduce the saved Stage 3 temperature (~0.961941), subject to numerical precision.

## Class-wise thresholds and rejection
- One-vs-rest F1-optimal thresholds for N, S, V and F are selected exclusively from DS1 OOF probabilities.
- Thresholds are frozen before DS2 metrics are calculated.
- On DS2, the model's five-class argmax is retained when the predicted class is N/S/V/F and its class-specific probability meets that class's frozen threshold.
- Predictions of Q, or N/S/V/F predictions below their class threshold, are assigned to `REJECT`.
- DS2 coverage, rejection rate, selective accuracy, selective risk, selective Macro-F1 and per-class results are reported.
- DS2 coverage-risk sweeps are descriptive only; no threshold or operating point is selected from DS2.
- Class-wise precision-recall plots and threshold tables are generated from DS1 OOF only.

## Bootstrap and metrics
- 2,000 patient-cluster bootstrap resamples are used for 95% confidence intervals on raw DS2 Macro-F1.
- Headline classification metrics are N/S/V/F; Q remains a model output and is excluded from headline metrics, consistent with the project configuration.
- Accuracy, balanced accuracy, Macro-F1, weighted-F1, per-class precision/recall/F1, class-wise average precision, log loss, ECE and multiclass Brier score are recorded.

## Outputs
Created under `artifacts/stage4/`:
- `stage4_results.json`
- `stage4_metrics_by_seed.csv`
- `classwise_thresholds_DS1_OOF.csv`
- `selective_metrics_by_seed.csv`
- `STAGE4_REPORT.md`
- `RUN_RECEIPT.json`
- `figures/confusion_matrices/`: per-seed raw matrices, summed counts across the five repeated DS2 runs, mean row-normalized matrices, and per-seed matrices with a REJECT column.
- `figures/precision_recall/`: DS1 OOF class-wise precision-recall plots with selected thresholds.
- `figures/rejection/`: per-seed DS2 descriptive coverage-risk plots with the frozen DS1 threshold operating point.
- `figures/reliability/`: seed-0 calibration reliability diagrams.
- Comparison graphs for Macro-F1, per-class F1, calibration ECE, coverage and selective risk.

Confusion matrices and graphs are separate PNG files; they are not embedded in the Markdown report.

## Run instructions
Use the companion Colab notebook `AMAC_ECG_Stage4_Final_Colab.ipynb`. Run cells sequentially; do not use Run all. Stop if preflight fails. Section 6 is the first cell that loads DS2 and starts model fitting/evaluation.

## Dataset-use note
DS2 is not used for model fitting, model selection, calibration fitting, or threshold selection in this run. The repository's supplied documentation records earlier DS2 access, so this run does not claim that DS2 is historically untouched or pristine.

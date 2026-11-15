# AMAC-ECG Stage 4 — Final Evaluation Report

## Run identity and protocol
- Run completed: 2026-11-15T10:09:57Z (UTC).
- This archive contains the completed 15 November 2026 Stage 4 run: 10 configurations × 5 seeds = 50 DS2 evaluation runs.
- All five seeds use the same fixed patient-held-out DS1 fold (fold 0) for validation. Seeds vary model initialisation; they do not represent different fold assignments.
- LR and XGBoost fit on the DS1 training partition; CNN uses the DS1 held-out fold for early stopping.
- DS1 OOF predictions provide the Stage-3-compatible leave-one-fold-out temperature procedure and N/S/V/F class-wise thresholds. The mean fold temperature is applied to seed-0 DS2 probabilities only.
- DS2 was not used for fitting, model selection, calibration fitting or threshold tuning in this run; it was opened for final evaluation. Repository documentation records earlier DS2 access, so DS2 is not described as historically untouched or pristine.
- Q remains a model output. Headline metrics are calculated over N/S/V/F; there is zero Q ground-truth support in the filtered DS2 evaluation subset, so Q recall/F1 should not be interpreted.

## Dataset identity
- DS1: 53,111 beats, 22 patients; SHA-256 `74bfdda3fe8d2d73b13b00596271eb15418ee1bd13046e0aa7e42186eba54644`.
- DS2: 47,534 beats, 21 patients; SHA-256 `5e4269ad60fc37d6b9fefe9fc2c6d038ada7d989548ecb902053ab70c45b2fea`.

## Five-seed summary (mean ± sample SD, ddof=1)
| Configuration | Macro-F1 | Balanced Accuracy | Accuracy |
|---|---:|---:|---:|
| LR + Unweighted + RR | 0.6040 ± 0.0000 | 0.5954 ± 0.0000 | 0.9320 ± 0.0000 |
| LR + Unweighted + Waveform Only | 0.3504 ± 0.0000 | 0.3766 ± 0.0000 | 0.8380 ± 0.0000 |
| LR + Class Weight + RR | 0.5318 ± 0.0000 | 0.7543 ± 0.0000 | 0.7447 ± 0.0000 |
| LR + Class Weight + Waveform Only | 0.2873 ± 0.0000 | 0.4405 ± 0.0000 | 0.5250 ± 0.0000 |
| XGBoost + Class Weight + RR | 0.6084 ± 0.0038 | 0.6369 ± 0.0075 | 0.9336 ± 0.0014 |
| XGBoost + Class Weight + Waveform Only | 0.4475 ± 0.0037 | 0.4826 ± 0.0012 | 0.8835 ± 0.0024 |
| CNN + Class Weight + RR | 0.6228 ± 0.0254 | 0.8161 ± 0.0155 | 0.8617 ± 0.0420 |
| CNN + Class Weight + Waveform Only | 0.4696 ± 0.0482 | 0.6336 ± 0.0538 | 0.7460 ± 0.0734 |
| CNN + Focal Loss + RR | 0.6113 ± 0.0152 | 0.8410 ± 0.0266 | 0.8213 ± 0.0363 |
| CNN + Focal Loss + Waveform Only | 0.4774 ± 0.0238 | 0.6220 ± 0.0334 | 0.7758 ± 0.0588 |

## Calibration, thresholds and rejection
- Calibration is seed-0-specific. Seeds 1–4 are not reported as calibrated; no multi-seed calibration robustness claim is made.
- N/S/V/F one-vs-rest F1-optimal thresholds are selected from DS1 OOF predictions and frozen before DS2 selective/rejection evaluation.
- DS2 coverage-risk sweeps are descriptive only and are not used to select thresholds or operating points.

## Uncertainty and statistical reporting
- Macro-F1 SD is the sample standard deviation across the five model initialisations on the same fixed patient-held-out DS1 fold (fold 0). It does not measure variation across fold assignments and is not a patient-level confidence interval.
- Patient-cluster bootstrap uses 2,000 resamples per run; percentile 95% intervals are stored with each run in `stage4_results.json`.
- Population SD was not used in the summary table in this package. The table values were independently recalculated from the 50-row per-seed CSV using sample SD (`ddof=1`).

## Output inventory
- `stage4_results.json` and `stage4_results_partial.json`: saved run results.
- `stage4_metrics_by_seed.csv`: 50 per-seed rows.
- `classwise_thresholds_DS1_OOF.csv` and `selective_metrics_by_seed.csv`: frozen thresholds and selective metrics.
- `RUN_RECEIPT.json`: run receipt with dataset hashes, configuration list and completion count.
- `probabilities/`: raw DS2 probability outputs and DS1 OOF cross-calibration files saved by the run.
- `figures/`: 194 PNG figures, including confusion matrices, precision-recall plots, rejection/coverage-risk curves, reliability diagrams and comparison plots.

## Reproducibility checks performed for this bundle
- Confirmed 10 configurations with 5 stored runs each (50 total) and 50 rows in the per-seed CSV.
- Recomputed Accuracy, Balanced Accuracy and N/S/V/F Macro-F1 from all 50 stored raw DS2 confusion matrices. Maximum absolute difference from the stored JSON metrics was 0 for all three measures.
- No data, predictions, metrics or figures were simulated or generated for this bundle; result files and figures are carried directly from the supplied completed Stage 4 results archive. The summary table is recalculated from the archived per-seed metrics.

## Important interpretation
- The five runs per configuration are repeated model initialisations evaluated on the same fixed patient-held-out split. Do not describe them as five different validation folds.
- The updated 15 November run has different CNN outputs from the previously archived run. Do not merge the two runs; use this run consistently for Stage 4 reporting and retain the earlier archive separately for audit history.

## Environment recorded in results
- Python: 3.13.16; NumPy: 2.4.6.

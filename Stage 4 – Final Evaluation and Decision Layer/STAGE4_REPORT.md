# AMAC-ECG Stage 4 — Final Evaluation Report

## Protocol
- Five random seeds were evaluated for each of ten configurations (five model/imbalance settings × RR enabled/disabled).
- All five random seeds use the same fixed patient-held-out DS1 fold (fold 0) for validation across configurations; the remaining DS1 patients are used for fitting. Random seeds vary independently of fold assignment.
- LR and XGBoost use the DS1 training partition for fitting; CNN uses the DS1 validation partition for early stopping.
- Stage-3-compatible leave-one-fold-out temperature scaling is run on DS1 OOF predictions for each configuration; the mean of its five fold-specific temperatures is applied only to seed-0 DS2 probabilities. Calibration is reported only for seed 0, not averaged across seeds.
- Class Q remains a model output but is excluded from headline metrics; headline classification metrics use N, S, V and F.
- DS2 is not used for fitting, model selection, calibration fitting or threshold tuning in this run.
- Repository documentation notes prior Stage 4 access to DS2; therefore this report does not describe DS2 as historically untouched or pristine.

## Five-seed Macro-F1 summary

| Configuration | Macro-F1 mean | Macro-F1 SD (sample, n=5) |
|---|---:|---:|
| LR + Unweighted + RR | 0.6040 | 0.0000 |
| LR + Unweighted + Waveform Only | 0.3504 | 0.0000 |
| LR + Class Weight + RR | 0.5318 | 0.0000 |
| LR + Class Weight + Waveform Only | 0.2873 | 0.0000 |
| XGBoost + Class Weight + RR | 0.6084 | 0.0038 |
| XGBoost + Class Weight + Waveform Only | 0.4475 | 0.0037 |
| CNN + Class Weight + RR | 0.6227 | 0.0344 |
| CNN + Class Weight + Waveform Only | 0.4684 | 0.0515 |
| CNN + Focal Loss + RR | 0.6102 | 0.0153 |
| CNN + Focal Loss + Waveform Only | 0.4395 | 0.0633 |

## Calibration scope
Calibration analysis is seed-0-specific. Temperature is fitted from the corresponding five-fold DS1 OOF predictions. Raw and calibrated seed-0 DS2 metrics are both saved; seeds 1–4 remain uncalibrated in this analysis.

## Outputs
- `stage4_results.json`: complete numerical results, confusion matrices and per-seed metadata.
- `stage4_metrics_by_seed.csv`: one row per configuration and seed.
- `figures/confusion_matrices/`: per-seed raw confusion matrices, summed five-run counts, mean normalised matrices, and per-seed frozen-threshold matrices with a REJECT column.
- `figures/macro_f1_five_seed_comparison.png`: Macro-F1 mean ± SD.
- `figures/per_class_f1_five_seed_comparison.png`: per-class F1 mean ± SD.
- `figures/seed0_calibration_ece_comparison.png`: seed-0 calibration impact.
- `figures/reliability/`: seed-0 reliability diagrams.
- `classwise_thresholds_DS1_OOF.csv`: frozen N/S/V/F thresholds selected on DS1 OOF only.
- `selective_metrics_by_seed.csv`: DS2 coverage, selective risk and selective Macro-F1.
- `figures/precision_recall/`: DS1 OOF per-class precision-recall plots with selected thresholds.
- `figures/rejection/`: per-seed DS2 descriptive coverage-risk curves with the frozen DS1 operating point.
- `figures/coverage_selective_risk_comparison.png`: five-seed coverage and selective-risk comparison.
- `figures/confusion_matrices/*threshold_rejection*`: DS2 confusion matrices with a REJECT column.

## Interpretation cautions
- The five-seed SD is the sample standard deviation (ddof=1) across five model initialisations on the same fixed patient-held-out DS1 split (fold 0); it does not represent variation across validation-fold assignments and is not a patient-level confidence interval.
- Patient-cluster bootstrap 95% CIs are stored per seed in the JSON and CSV.
- Calibration conclusions are limited to seed 0 and must not be described as multi-seed calibration robustness.
- The RR ablation compares each model/imbalance configuration with RR features enabled versus disabled.
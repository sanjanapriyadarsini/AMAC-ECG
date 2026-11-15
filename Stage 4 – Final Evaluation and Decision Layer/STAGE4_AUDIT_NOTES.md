# AMAC-ECG Stage 4 — Updated Run Audit Notes

Audit basis: supplied completed results archive; run receipt timestamp `2026-11-15T10:09:57Z`. These notes refer specifically to the 15 November 2026 output set.

## Verified run inventory
- 10 configurations × 5 seeds = 50 completed DS2 evaluation runs.
- `stage4_metrics_by_seed.csv` contains 50 rows.
- The result JSON contains 10 configurations with 5 runs each.
- Dataset beat/patient counts and SHA-256 hashes are recorded in the result JSON and run receipt.
- Archive contains 70 NPZ files (probability and DS1 OOF cross-calibration files) and 194 PNG figures.

## Dataset and protocol disclosure
- DS1: 53,111 beats from 22 patients; SHA-256 `74bfdda3fe8d2d73b13b00596271eb15418ee1bd13046e0aa7e42186eba54644`.
- DS2: 47,534 beats from 21 patients; SHA-256 `5e4269ad60fc37d6b9fefe9fc2c6d038ada7d989548ecb902053ab70c45b2fea`.
- All seeds use the same fixed patient-held-out DS1 fold (fold 0). The seed variation is model initialisation variation, not fold-assignment variation.
- DS2 was not used for fitting, model selection, calibration fitting or threshold tuning in this run. Historical DS2 access is disclosed in repository documentation; do not describe DS2 as historically untouched/pristine.
- Calibration is seed-0-specific; thresholds are derived from DS1 OOF and frozen before DS2 evaluation.

## Independent numerical consistency check
- Recomputed Accuracy, Balanced Accuracy and N/S/V/F Macro-F1 from each of the 50 saved raw DS2 confusion matrices.
- Maximum absolute difference against the stored JSON values: 0 for Accuracy, 0 for Balanced Accuracy and 0 for Macro-F1.
- Q remains a model output but is excluded from headline N/S/V/F metrics; the filtered DS2 evaluation subset has no Q ground-truth support.

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

## SD and confidence-interval wording
- The table reports sample SD (`ddof=1`) across five initialisations on the same fixed DS1 fold 0. It is not variation across validation-fold assignments and is not a patient-level confidence interval.
- Patient-cluster bootstrap percentile 95% intervals are separately stored per run using 2,000 resamples.

## Run-to-run change note
- The 15 November run is a new completed execution. CNN result files differ from the earlier 14 November archive; this is not merely a documentation/path fix. Preserve the earlier archive separately and do not combine metrics across the two runs.

## Scope limits
- No clinical expert alignment audit is claimed.
- No claim of clinical validation is made.
- No synthetic ECG records, synthetic patients, synthetic model predictions, or fabricated measurements were added.

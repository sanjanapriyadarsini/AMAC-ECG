# AMAC-ECG Stage 4 — Consolidated Audit and Paper Handover

Audit basis: supplied Stage 4 Results ZIP, Stage 4 Complete Package ZIP, corrected dataset ZIP, and full project ZIP. No retraining was performed.

## Verified protocol facts
- DS1: 53,111 beats from 22 patients.
- DS2: 47,534 beats from 21 patients.
- 10 configurations × 5 seeds = 50 DS2 evaluation runs.
- The same fixed patient-held-out DS1 fold (fold 0) is used across configurations; random seeds do not change fold assignment.
- DS1 OOF predictions are stored for all 10 configurations × 5 folds = 50 files.
- DS1 OOF files were checked against frozen fold assignments, validation indices, labels, patient IDs and dataset hash.
- DS1 and DS2 NPZ files are byte-identical across the corrected-data archive, Stage 4 package and full project archive.
- DS2 was not used for model fitting, model selection, calibration fitting or threshold tuning in this Stage 4 run. Historical DS2 access is disclosed in project records; do not call DS2 historically untouched/pristine.

## Numerical checks
- All 50 stored confusion matrices were independently used to recompute per-class precision, recall, F1, Macro-F1 and Balanced Accuracy.
- Maximum absolute difference between recomputed and stored Macro-F1: 0.
- Maximum absolute difference between recomputed and stored Balanced Accuracy: 0.
- Per-class precision, recall and F1 also match the confusion matrices.
- Balanced Accuracy is the mean recall over N/S/V/F.
- Q has zero DS2 ground-truth support after the evaluation-label filter. Q remains a model output; predictions of Q on N/S/V/F examples are incorrect predictions and reduce the corresponding metrics. Do not report Q recall/F1 as meaningful.
- The sklearn warning about predicted labels not present in y_true is expected when Q is predicted although there are no Q ground-truth samples in the evaluated subset. The code retains all five output columns in the confusion matrix and computes N/S/V/F headline metrics with Q predictions treated as wrong labels.
- Patient bootstrap implementation resamples unique patient IDs with replacement, concatenates all beats belonging to each sampled patient, and computes Macro-F1 on N/S/V/F. It uses 2,000 bootstrap replicates and stores percentile 95% intervals for each run. The same fixed bootstrap RNG seed is reset for each run, giving paired patient-resampling draws across configurations/seeds.
- Calibration uses Stage-3-compatible leave-one-fold-out temperatures from DS1 OOF predictions. The mean of the five fold-specific temperatures is applied only to seed-0 DS2 probabilities; calibrated metrics are seed-0-specific, not a five-seed calibration result.
- N/S/V/F class-wise thresholds are selected from DS1 OOF predictions and frozen before DS2 selective/rejection evaluation.

## SD correction for paper consistency
The original report and original plot used population SD (ddof=0). The corrected paper table and new Figure 1 use sample SD (ddof=1) across the five initialisations. This is a reporting-statistic correction only; no model was retrained and no predictions changed. LR configurations are deterministic in this run, so their five values are identical and SD remains 0.0000.

## Stage 4 headline results (mean ± sample SD)
| Configuration | Macro-F1 | Balanced Accuracy | Accuracy |
|---|---:|---:|---:|
| LR + Unweighted + RR | 0.604 ± 0.000 | 0.595 ± 0.000 | 0.932 ± 0.000 |
| LR + Unweighted + Waveform Only | 0.350 ± 0.000 | 0.377 ± 0.000 | 0.838 ± 0.000 |
| LR + Class Weight + RR | 0.532 ± 0.000 | 0.754 ± 0.000 | 0.745 ± 0.000 |
| LR + Class Weight + Waveform Only | 0.287 ± 0.000 | 0.440 ± 0.000 | 0.525 ± 0.000 |
| XGBoost + Class Weight + RR | 0.608 ± 0.004 | 0.637 ± 0.008 | 0.934 ± 0.001 |
| XGBoost + Class Weight + Waveform Only | 0.448 ± 0.004 | 0.483 ± 0.001 | 0.883 ± 0.002 |
| CNN + Class Weight + RR | 0.623 ± 0.034 | 0.812 ± 0.024 | 0.848 ± 0.061 |
| CNN + Class Weight + Waveform Only | 0.468 ± 0.052 | 0.629 ± 0.029 | 0.747 ± 0.107 |
| CNN + Focal Loss + RR | 0.610 ± 0.015 | 0.840 ± 0.028 | 0.818 ± 0.019 |
| CNN + Focal Loss + Waveform Only | 0.440 ± 0.063 | 0.569 ± 0.094 | 0.688 ± 0.138 |

## Selected model for the main confusion matrix
Highest mean Macro-F1 configuration: **CNN + Class Weight + RR**.
Its DS1 OOF average precision values are:
- N: 0.9938
- S: 0.2256
- V: 0.8459
- F: 0.0402

## Recommended figures for a 6-page IEEE paper
1. `paper_figures/Figure1_MacroF1_10config_mean_sampleSD.png` — main comparison across all 10 configurations.
2. `paper_figures/Figure2_DS1_OOF_PR_CNN_ClassWeight_RR.png` — DS1 OOF class-wise PR curves for the selected best-Macro-F1 configuration.
3. `paper_figures/Figure3_DS2_Confusion_CNN_ClassWeight_RR.png` — five-seed mean-normalised confusion matrix for the selected configuration.
4. `paper_figures/Figure4_DS2_Coverage_SelectiveRisk.png` — decision-layer comparison; include only if space permits.
5. `AMAC_ECG_Pipeline_Updated_DS2_21.png` — updated pipeline diagram; include if it fits the methodology space, otherwise move to supplement.

For a strict 6-page paper, target 3 main figures (Macro-F1 comparison, PR curves, confusion matrix). Use coverage-risk and the pipeline diagram only if layout permits. Do not insert all 194 generated figures. Keep all original figures in the results archive as supplementary/audit material.

## Remaining human/editorial decisions
- Writing team must confirm which 3–4 figures fit the actual IEEE layout.
- Confirm paper terminology and citation format.
- Flowchart DS2 count corrected to 21 patients.
- Retain historical DS2-access disclosure.
- Do not claim clinical expert validation.

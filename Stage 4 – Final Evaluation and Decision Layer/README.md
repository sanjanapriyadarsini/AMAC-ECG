# AMAC-ECG Stage 4 — Final Evaluation + Decision Layer

This folder contains the completed Stage 4 run dated 15 November 2026 and its saved outputs.

## Run contents
- 10 configurations × 5 seeds = 50 completed DS2 evaluation runs.
- `stage4_results.json` and `stage4_results_partial.json` contain the archived numerical run outputs.
- `stage4_metrics_by_seed.csv` contains 50 per-seed result rows.
- `classwise_thresholds_DS1_OOF.csv` and `selective_metrics_by_seed.csv` contain the saved DS1-OOF thresholds and DS2 selective metrics.
- `probabilities/` contains saved raw DS2 probability and DS1 OOF cross-calibration NPZ outputs.
- `figures/` contains the 194 PNG figures from this run.
- `RUN_RECEIPT.json`, `STAGE4_REPORT.md`, and `STAGE4_AUDIT_NOTES.md` document the run and audit checks.

## Protocol notes
- The five seeds use the same fixed patient-held-out DS1 fold 0. Seed variation is model-initialisation variation, not fold-assignment variation.
- Calibration is seed-0-specific. The N/S/V/F thresholds are selected from DS1 OOF and frozen before DS2 evaluation.
- DS2 was not used for fitting, model selection, calibration fitting or threshold tuning in this run. Prior DS2 access is disclosed in repository documentation; this run does not call DS2 historically untouched or pristine.
- Q remains a model output but is excluded from headline N/S/V/F metrics.

## Data integrity
The result files and PNGs in this directory are copied from the supplied completed 15-November Stage 4 results archive. No synthetic datasets, patient records, predictions or fabricated measurements have been added. The summary report recalculates aggregate mean and sample SD from the archived 50-row metrics CSV.

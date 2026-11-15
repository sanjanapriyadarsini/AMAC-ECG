# AMAC-ECG — Stage 5: Explainability and Model Analysis

This folder contains the Stage 5 SHAP workflow documentation and result artifacts.

## Recorded run
- Selected model: CNN + Focal Loss + RR, seed 0.
- Model fitting and early stopping: DS1 only, using frozen patient-held-out fold 0.
- Explanations: KernelSHAP on a bounded DS2 sample (16 examples; four each for N/S/V/F).
- Background rows: 12; KernelSHAP nsamples: 100.
- Regional summaries: P (samples 0–53), QRS (54–107), T (108–215); RR features reported separately.
- Faithfulness: top 20% SHAP-ranked waveform masking compared with equal-size random masking.
- Efficiency: 146,189 trainable parameters; checkpoint size 602,550 bytes; CUDA batch-256 mean latency 2.6944 ms.

## Result files currently present in the repository
- `artifacts/stage5/STAGE5_REPORT.md`
- `artifacts/stage5/shap_classwise_feature_importance.csv`
- `artifacts/stage5/shap_feature_importance.csv`
- `artifacts/stage5/shap_region_importance.csv`
- `artifacts/stage5/shap_faithfulness_top20_vs_random.csv`
- `artifacts/stage5/stage5_efficiency.json`
- `artifacts/stage5/stage5_explainability.json`
- `PAPER_FIGURE_CAPTIONS.md`

The executed notebook supplied for this run is named `ECG_Stage5_SHAP(1).ipynb`. It contains 31 cells, 15 executed code cells, saved outputs, and no recorded execution-error outputs. It is not yet committed in this repository folder. The clean notebook and helper module should be added only after their exact source files are verified.

## Figure files
The six original PNGs are included in the results ZIP and the separately extracted paper-figures ZIP, but are not yet committed as standalone files in this repository:
- `shap_top20_features.png`
- `shap_region_importance.png`
- `shap_summary_class_N.png`
- `shap_summary_class_S.png`
- `shap_summary_class_V.png`
- `shap_summary_class_F.png`

The saved figures are 1620×1260 and 1620×900 at approximately 180 DPI for the global and regional plots, and 1212×1180 at approximately 160 DPI for each class plot. If the target publication requires 300 DPI, re-export from the notebook at the required dimensions; changing metadata alone does not add resolution.

## Research-integrity notes
- DS2 is used for post-fit explanations and perturbation analysis, not model fitting in this Stage 5 run.
- Project records disclose historical DS2 access. Do not describe DS2 as historically untouched.
- SHAP values describe model attribution, not causality or clinical validity.
- Faithfulness results are mixed and preliminary: mean top-20% masking probability drop 0.241150 versus random masking drop 0.235358; top-20% exceeded the random mean in 62.5% of examples. Zero masking can create out-of-distribution waveforms.
- No clinician-led clinical validation is claimed.
- Runtime latency is hardware- and batch-specific.

## Remaining repository tasks
1. Commit the executed notebook.
2. Commit the six PNG figures as standalone binary assets.
3. Verify the exact clean notebook and helper-module sources before committing them.
4. Check dataset redistribution and model-checkpoint publication permissions before publishing the checkpoint.
5. Perform final repository and paper-format checks.

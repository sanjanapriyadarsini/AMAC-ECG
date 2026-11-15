# AMAC-ECG — Stage 5: Explainability and Model Analysis

This folder contains the Stage 5 SHAP workflow, generated tabular results, model-efficiency measurements, and figure documentation.

## Scope
- Selected model: CNN + Focal Loss + RR, seed 0.
- Model fitting and early stopping: DS1 only, using frozen patient-held-out fold 0.
- Explanations: KernelSHAP on a bounded class-stratified DS2 sample (16 examples; four each for N/S/V/F).
- Regional summaries: P (samples 0–53), QRS (54–107), T (108–215); RR features reported separately.
- Faithfulness: top 20% SHAP-ranked waveform masking compared with equal-size random masking.
- Efficiency: parameter count, checkpoint size and CUDA inference latency at batch size 256.

## Repository contents
- `AMAC_ECG_Stage5_SHAP_FINAL_CLEAN.ipynb`: clean Colab workflow notebook.
- `stage5_utils.py`: helper functions used by the workflow.
- `STAGE5_REPORT.md`: recorded Stage 5 result summary.
- `artifacts/stage5/shap_classwise_feature_importance.csv`: class-wise feature attribution.
- `artifacts/stage5/shap_feature_importance.csv`: global feature importance.
- `artifacts/stage5/shap_region_importance.csv`: P/QRS/T and RR attribution.
- `artifacts/stage5/shap_faithfulness_top20_vs_random.csv`: per-example perturbation comparison.
- `artifacts/stage5/stage5_efficiency.json`: efficiency measurements.
- `artifacts/stage5/stage5_explainability.json`: execution settings, provenance and caveats.
- `artifacts/stage5/RESULTS_ARCHIVE_MANIFEST.md`: supplied ZIP inventory.
- `PAPER_FIGURE_CAPTIONS.md`: suggested captions and paper-use notes for the six generated PNG figures.

## Figure files
The six original PNG figures are present in the supplied Colab results ZIP and in the extracted paper-figures bundle shared separately:
1. `shap_top20_features.png`
2. `shap_region_importance.png`
3. `shap_summary_class_N.png`
4. `shap_summary_class_S.png`
5. `shap_summary_class_V.png`
6. `shap_summary_class_F.png`

They are not yet committed as standalone binary files in this repository. The extracted PNGs preserve the saved outputs; they have not been upscaled. If the target publication requires 300 DPI, re-export the plots from the notebook at the required size rather than merely changing the DPI metadata.

## Run protocol
1. Use the complete Stage 4 companion runtime package in Colab. It must contain processed `artifacts/ds1.npz`, `artifacts/ds2.npz`, frozen `src/folds.json`, DS1 OOF prediction NPZ files and project source.
2. Run notebook cells sequentially; do not run all cells before preflight passes.
3. Use a CUDA runtime where available.
4. Stop if dataset hashes, shapes, patient counts, fold identity, patient disjointness or OOF-file count do not match the notebook assertions.

## Research-integrity notes
- DS2 is not passed to model fitting in the recorded Stage 5 workflow. A bounded DS2 sample is used for post-fit explanations and perturbation analysis.
- Project records disclose historical DS2 access. Do not describe DS2 as historically untouched or pristine.
- SHAP values describe model attribution, not causality or clinical validity.
- Faithfulness results are mixed and preliminary; zero masking may create out-of-distribution waveforms.
- No clinician-led clinical-alignment audit or clinical validation is claimed.
- Runtime latency is hardware- and batch-specific.

## Remaining repository tasks
- Commit the six PNG figures as standalone binary assets.
- Add the executed-output notebook as a separate file, preserving the clean notebook.
- Check permission/redistribution terms before publishing the trained checkpoint.

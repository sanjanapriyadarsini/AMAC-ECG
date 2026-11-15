# AMAC-ECG — Stage 5: Explainability and Model Analysis

This folder contains the Stage 5 SHAP handover and the generated text-based results.

## Scope
- Selected model: CNN + Focal Loss + RR, seed 0.
- Fit and early stopping: DS1 only, using the existing patient-held-out fold 0.
- Explanations: KernelSHAP class-wise attributions on a bounded, class-stratified sample of DS2 (16 examples; four each for N/S/V/F).
- Regional summaries: P (samples 0–53), QRS (54–107), T (108–215), with four RR features reported separately.
- Perturbation comparison: zero-mask the top 20% SHAP-ranked waveform samples and compare with 20 equal-size random masks per example.
- Efficiency measurements: parameter count, checkpoint size and CUDA inference latency at batch size 256.

## Files
- `AMAC_ECG_Stage5_SHAP_FINAL_CLEAN.ipynb`: clean Colab workflow notebook.
- `stage5_utils.py`: helper functions used for SHAP output normalization, masking comparison and latency measurement.
- `STAGE5_REPORT.md`: Stage 5 result summary.
- `artifacts/stage5/shap_classwise_feature_importance.csv`: class-wise feature attributions.
- `artifacts/stage5/shap_feature_importance.csv`: global feature importance.
- `artifacts/stage5/shap_region_importance.csv`: P/QRS/T and RR attribution.
- `artifacts/stage5/shap_faithfulness_top20_vs_random.csv`: per-example perturbation comparison.
- `artifacts/stage5/stage5_efficiency.json`: model efficiency measurements.
- `artifacts/stage5/stage5_explainability.json`: execution provenance, settings, summary metrics and limitations.
- `artifacts/stage5/RESULTS_ARCHIVE_MANIFEST.md`: inventory of the supplied results ZIP and publication notes.

## Run protocol
1. Use the complete Stage 4 companion runtime package in Colab. It must contain processed `artifacts/ds1.npz`, `artifacts/ds2.npz`, the frozen `src/folds.json`, project source files and all 50 DS1 OOF prediction NPZ files.
2. Run notebook cells sequentially; do not use Run all before preflight passes.
3. Select a CUDA GPU runtime where available.
4. Stop if dataset hashes, array shapes, patient counts, fold identity, patient disjointness or OOF-file count do not match the notebook assertions.

## Research-integrity notes
- DS2 is not passed to model fitting in the recorded Stage 5 workflow. Its bounded class-stratified sample is used for post-fit explanations and perturbation analysis only.
- Project records disclose historical DS2 access. Do not describe DS2 as historically untouched or pristine.
- SHAP values describe model attribution, not causality or clinical validity.
- The faithfulness results are mixed and preliminary. Zero masking may create out-of-distribution waveforms.
- No clinician-led clinical-alignment audit or clinical validation is claimed.
- Runtime latency is hardware- and batch-specific.

## Remaining binary artifacts
The supplied results ZIP contains six PNG figures and a model checkpoint. The text/CSV/JSON outputs have been committed. The PNGs and checkpoint are not included in this repository update; confirm model-weight redistribution permissions before publishing the checkpoint.

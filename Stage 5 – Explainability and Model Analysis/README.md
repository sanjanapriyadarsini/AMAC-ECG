# AMAC-ECG — Stage 5: Explainability and Model Analysis

This folder contains the Stage 5 SHAP handover for the AMAC-ECG project.

## Scope
- Selected model: CNN + Focal Loss + RR, seed 0.
- Fit and early stopping: DS1 only, using the existing patient-held-out fold 0.
- Explanations: KernelSHAP class-wise attributions on a bounded, class-stratified sample of DS2 (16 examples; four each for N/S/V/F).
- Regional summaries: P (samples 0–53), QRS (54–107), T (108–215), with four RR features reported separately.
- Perturbation comparison: zero-mask the top 20% SHAP-ranked waveform samples and compare with 20 equal-size random masks per example.
- Efficiency measurements: parameter count, checkpoint size and CUDA inference latency at batch size 256.

## Files
- `AMAC_ECG_Stage5_SHAP_FINAL_CLEAN.ipynb`: clean Colab workflow notebook. Its handover note says the notebook passed static checks and syntax validation; it does not certify a complete Colab run of this exact clean notebook version.
- `stage5_utils.py`: helper functions used for SHAP output normalization, masking comparison and latency measurement.
- `STAGE5_REPORT.md`: Stage 5 result summary transcribed from the project's combined Stage 4–Stage 5 technical handover.

## Run protocol
1. Use the complete Stage 4 companion runtime package in Colab. It must contain processed `artifacts/ds1.npz`, `artifacts/ds2.npz`, the frozen `src/folds.json`, project source files and all 50 DS1 OOF prediction NPZ files.
2. Run notebook cells sequentially; do not use Run all before preflight passes.
3. Select a CUDA GPU runtime where available.
4. Stop if dataset hashes, array shapes, patient counts, fold identity, patient disjointness or OOF-file count do not match the notebook assertions.

## Research-integrity notes
- DS2 is not passed to model fitting. Its bounded class-stratified sample is used for post-fit explanations and perturbation analysis only.
- Project records disclose historical DS2 access. Do not describe DS2 as historically untouched or pristine.
- SHAP values describe model attribution, not causality or clinical validity.
- The faithfulness results are mixed and preliminary. Zero masking may create out-of-distribution waveforms.
- No clinician-led clinical-alignment audit or clinical validation is claimed.
- Runtime latency is hardware- and batch-specific.

## Results package status
The Colab-generated `AMAC_ECG_Stage5_SHAP_Results.zip` is described in the combined handover, but the exact ZIP was not available among the files used to prepare this repository update. Therefore the results ZIP, checkpoint, and individual generated CSV/PNG result files are not included in this commit. The report preserves the metrics documented in the handover; it should not be mistaken for a substitute for the underlying raw results archive.

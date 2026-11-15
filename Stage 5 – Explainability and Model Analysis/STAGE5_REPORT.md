# Stage 5 — SHAP Explainability and Model Analysis

## Run scope and provenance
The project's combined Stage 4–Stage 5 handover records a Stage 5 run for the selected **CNN + Focal Loss + RR, seed 0** model. Model fitting and early stopping used DS1 only, with frozen patient-held-out fold 0. A bounded class-stratified DS2 sample was used after fitting for explanation and perturbation analysis only.

The recorded explanation sample contains **16 examples: four each for N, S, V and F**. The handover reports the following measured outputs. The notebook file stored beside this report is the clean/static-validated handover version; its own validation note does not certify an end-to-end run of that exact notebook version.

## Training and checkpoint
| Measure | Recorded value |
|---|---:|
| Best epoch | 21 |
| Early stopping | Epoch 33 |
| Best DS1 validation Macro-F1 | 0.4618268262 |
| Trainable parameters | 146,189 |
| Checkpoint | `cnn_focal_rr_seed0_stage5.pt` |
| Checkpoint size | 602,550 bytes |

## KernelSHAP feature attribution
KernelSHAP completed for all 16 selected examples. The combined handover notes linear-regression singularity and convergence warnings; these should be disclosed as numerical-stability cautions.

Top displayed global features by mean absolute SHAP:
| Feature | Mean absolute SHAP |
|---|---:|
| wave_076 | 0.024114 |
| pre_rr | 0.023585 |
| wave_043 | 0.016461 |
| wave_033 | 0.016252 |
| wave_208 | 0.014507 |

## Regional attribution
The project configuration's waveform regions are P: samples 0–53, QRS: 54–107 and T: 108–215. RR attribution is reported separately.

| Class | n | P | QRS | T | RR |
|---|---:|---:|---:|---:|---:|
| N | 4 | 35.00% | 27.37% | 31.77% | 5.86% |
| S | 4 | 17.63% | 31.06% | 49.18% | 2.13% |
| V | 4 | 25.45% | 21.13% | 44.65% | 8.77% |
| F | 4 | 20.77% | 27.70% | 50.32% | 1.20% |

These are attribution summaries over four examples per class, not clinical validation, causal evidence or population-level estimates.

## Preliminary perturbation / faithfulness comparison
For each example, the top 20% waveform positions by absolute SHAP value for the true class were zero-masked and compared with 20 random masks of equal size.

| Measure | Recorded result |
|---|---:|
| Examples | 16 |
| Mean true-class probability drop — top-20% SHAP mask | 0.241150 |
| Mean probability drop — random mask | 0.235358 |
| Top-20% drop exceeded random-mask mean | 0.625 (10/16) |

Per-example outcomes were mixed: top-SHAP masking sometimes increased true-class probability, and random masks sometimes produced larger drops. These results do not establish consistently strong faithfulness. Zero masking is a simple perturbation baseline and may create out-of-distribution waveforms.

## Model efficiency
Measured on the recorded Colab CUDA runtime at batch size 256.

| Measure | Recorded result |
|---|---:|
| Device | CUDA |
| Trainable parameters | 146,189 |
| Checkpoint size | 602,550 bytes |
| Mean batch inference latency | 2.694358 ms |
| Mean latency per beat | 0.010525 ms |
| Warm-up batches | 10 |
| Timed repeats | 50 |

Latency is specific to the runtime, hardware and batch setting; it is not a hardware-independent benchmark.

## Interpretation and limitations
Stage 5 adds model-specific, seed-0 explanations and efficiency analysis; it is not a second Stage 4 evaluation and does not replace or alter Stage 4 metrics. SHAP outputs describe how the selected model responds to inputs. They do not establish causal importance or clinical validity. Regional summaries use a small class-stratified sample. No clinician-led clinical-alignment audit or clinical validation is claimed.

## Data-use disclosure
Project records disclose historical DS2 access. This report does not describe DS2 as historically untouched or pristine. In the recorded Stage 5 procedure, DS2 was not used for model training; only the bounded post-fit explanation/perturbation sample was used.

## Underlying artifact availability
The combined handover records creation of `AMAC_ECG_Stage5_SHAP_Results.zip` and describes its contents (checkpoint, SHAP CSVs, regional attribution CSV, faithfulness CSV, efficiency/explainability JSON and PNG figures). The exact results ZIP was not available among the source files used for this repository update, so those generated raw result files are not included here.

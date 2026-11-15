# Stage 5 — SHAP Explainability

## Scope
- SHAP-based class-wise feature attribution.
- Selected model: CNN + Focal Loss + RR, seed 0.
- DS1-only fitting with frozen patient-held-out fold 0 for early stopping.
- Bounded class-stratified DS2 sample used for explanations only.

## Execution settings
- DS1 SHA256: `74bfdda3fe8d2d73b13b00596271eb15418ee1bd13046e0aa7e42186eba54644`
- DS2 SHA256: `5e4269ad60fc37d6b9fefe9fc2c6d038ada7d989548ecb902053ab70c45b2fea`
- SHAP background rows: 12
- Explained examples: 16
- KernelSHAP nsamples: 100

## Region and RR attribution
| class | n_samples | P | QRS | T | RR_total | P_fraction | QRS_fraction | T_fraction | RR_total_fraction |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| N | 4 | 0.153088 | 0.119741 | 0.138977 | 0.0256446 | 0.349956 | 0.273724 | 0.317698 | 0.0586227 |
| S | 4 | 0.134891 | 0.237738 | 0.37641 | 0.0162838 | 0.176253 | 0.310637 | 0.491832 | 0.021277 |
| V | 4 | 0.218991 | 0.181801 | 0.384163 | 0.0754157 | 0.254531 | 0.211306 | 0.446509 | 0.0876548 |
| F | 4 | 0.0764521 | 0.101973 | 0.18523 | 0.00443133 | 0.207702 | 0.277035 | 0.503225 | 0.0120388 |

## Faithfulness summary
- Examples: 16
- Mean top-20% masking probability drop: 0.241150
- Mean random masking probability drop: 0.235358
- Fraction where top-20% drop exceeds random mean: 0.625

## Model efficiency
- Device: cuda
- Trainable parameters: 146189
- Checkpoint bytes: 602550
- Mean batch-256 latency: 2.6944 ms
- Mean latency per beat: 0.010525 ms

## Limitations
Attributions describe model behaviour and do not establish causality or clinical validity. The perturbation baseline uses zero masking. DS2 historical-access disclosure is retained.

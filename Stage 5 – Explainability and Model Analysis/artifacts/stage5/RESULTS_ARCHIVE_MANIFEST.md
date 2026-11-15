# Stage 5 Results Archive — contents and publication notes

The supplied Colab ZIP was inspected. It contains 15 entries:

## Text outputs
- `artifacts/stage5/STAGE5_REPORT.md`
- `artifacts/stage5/stage5_efficiency.json`
- `artifacts/stage5/stage5_explainability.json`
- `artifacts/stage5/shap_region_importance.csv`
- `artifacts/stage5/shap_feature_importance.csv`
- `artifacts/stage5/shap_classwise_feature_importance.csv`
- `artifacts/stage5/shap_faithfulness_top20_vs_random.csv`
- `stage5_utils.py`

## Binary outputs in the supplied ZIP
- `artifacts/stage5/shap_top20_features.png`
- `artifacts/stage5/shap_summary_class_F.png`
- `artifacts/stage5/shap_summary_class_N.png`
- `artifacts/stage5/shap_summary_class_S.png`
- `artifacts/stage5/shap_summary_class_V.png`
- `artifacts/stage5/shap_region_importance.png`
- `artifacts/stage5/models/cnn_focal_rr_seed0_stage5.pt`

The ZIP also contains the exact helper module `stage5_utils.py`, which matches the helper already committed in the Stage 5 folder.

## Integrity and privacy notes
- The archive contains no raw dataset files.
- It includes a trained model checkpoint. Confirm that the model weights are permitted for public redistribution under the dataset/project terms before publishing them.
- The source archive was approximately 1,020,175 bytes.
- The text result files are committed alongside this manifest. The ZIP itself and its PNG/checkpoint binary members are not included in this text-only repository update.

# Stage 3 --- Decision Layer

## 1. Objective

Stage 3 evaluates the decision layer applied to the selected Stage 2
model configuration. It does not retrain the CNN. Instead, it uses the
saved out-of-fold predictions to study probability calibration and
confidence-based selective prediction.

The configuration evaluated in the recorded run was:

-   **Model:** CNN
-   **Imbalance strategy:** Focal Loss
-   **RR features:** Enabled
-   **Seed:** 0
-   **Evaluation classes:** N, S, V and F

Class Q was excluded in accordance with the Stage 2 evaluation protocol.

## 2. Connection to Stage 2

Stage 2 trained and compared models using five-fold patient-grouped
cross-validation on DS1. The selected CNN + Focal Loss + RR
configuration produced out-of-fold predictions.

Stage 3 consumes those saved predictions and applies post-hoc decision
analysis:

1.  Temperature scaling for probability calibration.
2.  Expected Calibration Error (ECE) comparison.
3.  Reliability-diagram analysis.
4.  Confidence-based rejection analysis.
5.  Per-class F1 analysis along the rejection curve.
6.  Class S precision--recall analysis.

Because Stage 3 uses the same saved predictions, the zero-rejection
classification metrics should agree with the matching Stage 2
configuration, subject to rounding and the exact aggregation method.
Temperature scaling changes probability confidence but does not change
the argmax class labels in this run.

## 3. Calibration

Temperature scaling was fitted using the cross-fold procedure
implemented in `src/stage3_decision.py`.

Recorded temperature for seed 0:

-   **T = 0.961941**

Recorded ECE:

  Measure                           Value
  ---------------------------- ----------
  Before temperature scaling     0.081007
  After temperature scaling      0.074037
  Absolute reduction             0.006970

The ECE reduction indicates a modest improvement in the reported
calibration measure. The temperature is below 1, which sharpens the
predicted probability distribution.

The reliability diagram compares uncalibrated and temperature-scaled
confidence against observed accuracy.

**Metric scope:** The calibration procedure operates on the evaluated
classes N, S, V and F. Class S is additionally examined with a separate
precision--recall analysis because it is the study's focused
minority-class target. This does not mean calibration is performed only
for Class S.

## 4. Stage 2 baseline cross-check

The matching Stage 2 summary entry was located in:

`amac_ecg_cnn_outputs/artifacts/grid_summary.json`

Configuration: CNN / Focal Loss / RR enabled / seed 0.

  Metric         Stage 2 summary   Stage 3, 0% rejection
  ------------ ----------------- -----------------------
  Macro-F1              0.547050                  0.5471
  Class N F1              0.9183                  0.9183
  Class S F1              0.4379                  0.4379
  Class V F1              0.7500                  0.7501
  Class F F1              0.0891                  0.0891

The values agree to the displayed precision. Small differences can arise
from rounding and aggregation. This cross-check supports that Stage 3 is
analysing the intended Stage 2 prediction set rather than reporting a
newly trained model.

The Stage 2 summary records a macro-F1 standard deviation of
approximately 0.130224 across its folds. The Stage 3 run shown here
reports only seed 0; it must not be described as a five-seed Stage 3
experiment.

## 5. Confidence-based rejection

The decision layer rejects predictions with confidence below a threshold
and evaluates the remaining accepted beats.

Selected points from the recorded rejection curve:

    Rejection rate   Macro-F1   Class S F1
  ---------------- ---------- ------------
                0%     0.5471       0.4379
               10%     0.5547       0.4777
               20%     0.5745       0.5502
               30%     0.5915       0.5720
               40%     0.6252       0.5744
               45%     0.6319       0.5846
               50%     0.6356       0.5662

The area-under-accuracy-rejection-curve (AU-ARC) summary over coverage
0.50--1.00 was:

-   **Macro-F1 AU-ARC:** 0.5864
-   **Class S F1 AU-ARC:** 0.5341

These values describe selective performance over a range of coverage
levels. They do not mean that the original classifier's overall accuracy
or F1 increased for every beat. Higher rejection means fewer beats
receive an automatic prediction.

## 6. Per-class behaviour

The per-class F1 analysis includes all four evaluated classes: N, S, V
and F.

At 0% rejection, the pooled per-class F1 values were:

  Class         F1
  ------- --------
  N         0.9183
  S         0.4379
  V         0.7501
  F         0.0891

As rejection increases, N and V F1 generally rise in the recorded curve.
Class S improves over much of the curve, while Class F declines.
Therefore, rejection does not benefit all classes equally.

The script warns that beyond approximately 48% rejection, some evaluated
classes have fewer than 20 accepted beats. Points in that region are
statistically thin and should not be presented as robust operating
points.

## 7. Class S precision--recall analysis

The recorded Class S average precision was:

-   **Average Precision (AP): 0.3935**
-   **Class S prevalence:** approximately 0.0188

The precision--recall curve evaluates the ranking/detection trade-off
for Class S. Average precision is not classification accuracy and should
not be described as such.

Class S receives this additional analysis because it is the designated
minority-class focus of the study. Classes N, V and F remain included in
macro-F1 and per-class F1 reporting; they are not omitted from the
overall evaluation.

## 8. Generated outputs

The Stage 3 script writes the following files to:

`artifacts/stage3/`

-   `stage3_results.json` --- numerical results and configuration.
-   `reliability.png` --- reliability diagram before and after
    temperature scaling.
-   `arc.png` --- Macro-F1 and Class S F1 versus rejection rate.
-   `pr_class_s.png` --- Class S precision--recall curve.
-   `per_class_f1.png` --- per-class F1 versus rejection rate for N, S,
    V and F.

The plots and JSON should be retained together so the reported values
can be traced to the run.

## 9. Limitations

1.  The recorded Stage 3 run contains **one seed (seed 0)**. Five folds
    do not mean five independent random seeds.
2.  The rejection curve trades coverage for performance on accepted
    beats. Results at high rejection rates have fewer accepted examples.
3.  The script reports DS2 was not opened during this Stage 3 run. Stage
    4 must still audit the historical use of DS2 before describing it as
    an untouched final test set.
4.  Calibration and rejection analysis use saved Stage 2 out-of-fold
    predictions; Stage 3 does not retrain the CNN.
5.  The observed ECE reduction is modest and should not be overstated.
6.  The Stage 2/Stage 3 baseline comparison is a consistency check, not
    evidence that Stage 3 improved the underlying classifier.

## 10. Bridge to Stage 4

-   **Stage 2 --- Model development:** trains and compares models and
    produces out-of-fold predictions on DS1.
-   **Stage 3 --- Decision layer:** calibrates the selected model's
    probabilities and analyses confidence-based rejection using those
    predictions.
-   **Stage 4 --- Final evaluation:** evaluates the locked model and
    decision configuration on DS2, only after auditing whether DS2 has
    previously influenced model development or selection.

Before Stage 4, document the DS2 usage audit, freeze the model and
decision settings, and then follow the pre-defined evaluation protocol.
Do not tune the model or decision threshold on DS2 and subsequently
describe the same DS2 results as an untouched final test.

------------------------------------------------------------------------

**Run status:** Stage 3 executed successfully for CNN / Focal Loss / RR
enabled / seed 0. The JSON, reliability, ARC, Class S PR and per-class
F1 outputs were generated. Further protocol audit and final
documentation review remain necessary before Stage 3 is frozen.

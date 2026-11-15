"""Utility functions used by the AMAC-ECG Stage 5 SHAP notebook.

This module contains deterministic shape normalization and perturbation/
benchmark helpers. It does not load datasets, train models, or create data.
"""
from __future__ import annotations

import numpy as np


def normalize_shap_values(values, n_samples: int, n_features: int,
                          n_classes: int = 5) -> np.ndarray:
    """Return multi-output SHAP values in (samples, features, classes) order."""
    # shap.KernelExplainer normally returns a list (one array per output)
    # on older SHAP releases and a 3-D ndarray on newer releases.
    if hasattr(values, "values"):  # shap.Explanation
        values = values.values

    if isinstance(values, list):
        arrays = [np.asarray(v) for v in values]
        if len(arrays) != n_classes:
            raise ValueError(
                f"Expected {n_classes} output arrays, received {len(arrays)}"
            )
        if any(a.shape != (n_samples, n_features) for a in arrays):
            raise ValueError(
                "List SHAP output must contain arrays shaped "
                f"{(n_samples, n_features)}; got {[a.shape for a in arrays]}"
            )
        arr = np.stack(arrays, axis=-1)
    else:
        arr = np.asarray(values)
        candidates = []
        if arr.shape == (n_samples, n_features, n_classes):
            candidates.append(arr)
        if arr.shape == (n_classes, n_samples, n_features):
            candidates.append(np.moveaxis(arr, 0, -1))
        if arr.shape == (n_samples, n_classes, n_features):
            candidates.append(np.moveaxis(arr, 1, -1))
        if len(candidates) != 1:
            raise ValueError(
                "Ambiguous or unsupported SHAP output shape "
                f"{arr.shape}; expected samples={n_samples}, "
                f"features={n_features}, classes={n_classes}"
            )
        arr = candidates[0]

    if arr.shape != (n_samples, n_features, n_classes):
        raise ValueError(
            f"Normalized SHAP shape is {arr.shape}, expected "
            f"{(n_samples, n_features, n_classes)}"
        )
    if not np.isfinite(arr).all():
        raise ValueError("SHAP output contains NaN or infinite values")
    return arr.astype(np.float64, copy=False)


def masking_faithfulness(predict_fn, waveform: np.ndarray, rr: np.ndarray,
                         waveform_shap: np.ndarray, *, mask_fraction: float = 0.20,
                         random_repeats: int = 20,
                         rng: np.random.Generator) -> dict:
    """Compare true-class probability drop for SHAP-ranked vs random zero masks."""
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    rr = np.asarray(rr, dtype=np.float32).reshape(-1)
    attribution = np.asarray(waveform_shap, dtype=np.float64).reshape(-1)

    if waveform.shape != attribution.shape:
        raise ValueError("Waveform and waveform SHAP vectors must have equal length")
    if not 0 < mask_fraction < 1:
        raise ValueError("mask_fraction must be between 0 and 1")
    if random_repeats < 2:
        raise ValueError("Use at least two random masks to estimate SD")

    n = len(waveform)
    k = max(1, int(np.ceil(mask_fraction * n)))
    base = np.concatenate([waveform, rr])[None, :]
    base_prob = np.asarray(predict_fn(base))[0]
    if base_prob.ndim != 1:
        raise ValueError("Prediction function must return (rows, classes)")
    # Caller sets class_index after receiving this result; all class scores
    # are returned to keep the helper generic.
    ranked = np.argsort(np.abs(attribution))[-k:]
    x_top = waveform.copy()
    x_top[ranked] = 0.0
    top_prob = np.asarray(predict_fn(np.concatenate([x_top, rr])[None, :]))[0]

    random_drops = []
    for _ in range(random_repeats):
        idx = rng.choice(n, size=k, replace=False)
        x_random = waveform.copy()
        x_random[idx] = 0.0
        p_random = np.asarray(
            predict_fn(np.concatenate([x_random, rr])[None, :])
        )[0]
        random_drops.append(base_prob - p_random)

    return {
        "base_probabilities": base_prob,
        "top_mask_probabilities": top_prob,
        "random_probability_drops": np.asarray(random_drops),
        "masked_positions": int(k),
        "mask_fraction": float(mask_fraction),
    }


def benchmark_inference(model, x_tensor, rr_tensor, *, device: str,
                        warmup: int = 10, repeats: int = 50) -> float:
    """Return mean forward-pass milliseconds per batch; excludes warmup."""
    import time
    import torch

    if len(x_tensor) != len(rr_tensor) or len(x_tensor) == 0:
        raise ValueError("Benchmark tensors must have equal, non-zero batch size")
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(x_tensor, rr_tensor)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(x_tensor, rr_tensor)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / repeats
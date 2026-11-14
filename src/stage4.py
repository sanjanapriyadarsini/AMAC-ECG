"""
AMAC-ECG Stage 4 — locked final evaluation.
Run from the AMAC-ECG repository root: python stage4.py

Five seeds, five model/imbalance configurations, with and without RR features.
DS1 OOF predictions are used for seed-0-only temperature calibration.
DS2 is used only for final forward-pass evaluation in this run.
"""
import os, sys, json, time, hashlib, platform, csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import config as C
from models import build_features, make_lr, make_xgb
from imbalance import sample_weights
from run_grid import load_folds, file_sha256

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix,
    average_precision_score, accuracy_score, balanced_accuracy_score,
    log_loss, precision_recall_curve
)
from sklearn.preprocessing import label_binarize

ART = ROOT / "artifacts"
OUT = ART / "stage4"
FIG = OUT / "figures"
CM_DIR = FIG / "confusion_matrices"
REL_DIR = FIG / "reliability"
PROB_DIR = OUT / "probabilities"
PR_DIR = FIG / "precision_recall"
REJECT_DIR = FIG / "rejection"
for p in (OUT, FIG, CM_DIR, REL_DIR, PROB_DIR, PR_DIR, REJECT_DIR):
    p.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
EVAL = np.array(C.EVAL_LABELS, dtype=int)
ALL_LABELS = np.arange(len(C.CLASS_NAMES))
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20261114

BASE_CONFIGS = [
    ("lr", "none", "LR + Unweighted"),
    ("lr", "class_weight", "LR + Class Weight"),
    ("xgb", "class_weight", "XGBoost + Class Weight"),
    ("cnn", "class_weight", "CNN + Class Weight"),
    ("cnn", "focal", "CNN + Focal Loss"),
]
CONFIGS = [
    (model, strategy, use_rr, f"{label} + {'RR' if use_rr else 'Waveform Only'}")
    for model, strategy, label in BASE_CONFIGS
    for use_rr in (True, False)
]

def load_oof(model, strategy, use_rr):
    tag = "rr" if use_rr else "norr"
    probs, labels, fold_ids = [], [], []
    ds1_hash = file_sha256(ART / "ds1.npz")
    for fold in range(5):
        path = ART / "preds" / f"{model}_{strategy}_{tag}_f{fold}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing DS1 OOF prediction: {path}")
        d = np.load(path, allow_pickle=False)
        if "dataset_hash" not in d.files:
            raise ValueError(f"{path} has no dataset hash.")
        if str(d["dataset_hash"].item()) != ds1_hash:
            raise ValueError(f"DS1 hash mismatch in {path}.")
        probs.append(d["proba"].astype(np.float64))
        labels.append(d["y_true"].astype(int))
        fold_ids.append(np.full(len(d["y_true"]), fold, dtype=int))
    return np.vstack(probs), np.concatenate(labels), np.concatenate(fold_ids)

def fit_temperature(proba, y):
    keep = np.isin(y, EVAL)
    z = np.log(np.clip(proba[keep].astype(np.float64), 1e-12, 1.0))
    yy = y[keep]
    def objective(temp):
        q = z / temp
        q -= logsumexp(q, axis=1, keepdims=True)
        return float(-q[np.arange(len(yy)), yy].mean())
    result = minimize_scalar(objective, bounds=(0.05, 10.0), method="bounded")
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError("Temperature optimisation failed.")
    return float(result.x)

def apply_temperature(proba, temp):
    z = np.log(np.clip(proba.astype(np.float64), 1e-12, 1.0)) / temp
    z -= logsumexp(z, axis=1, keepdims=True)
    return np.exp(z)

def ece_score(proba, y, bins=15):
    keep = np.isin(y, EVAL)
    p, yy = proba[keep], y[keep]
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == yy)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)

def metric_pack(proba, y):
    keep = np.isin(y, EVAL)
    p = np.clip(proba[keep].astype(np.float64), 1e-12, 1.0)
    p /= p.sum(axis=1, keepdims=True)
    yy = y[keep]
    pred = p.argmax(axis=1)
    cm = confusion_matrix(yy, pred, labels=ALL_LABELS)
    per_class = {}
    for i, cls in enumerate(C.CLASS_NAMES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        per_class[cls] = {
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
            "support": int(cm[i, :].sum()),
        }
    y_bin = label_binarize(yy, classes=ALL_LABELS)
    ap = {}
    for i, cls in enumerate(C.CLASS_NAMES):
        if i in EVAL and y_bin[:, i].sum() > 0:
            ap[cls] = float(average_precision_score(y_bin[:, i], p[:, i]))
    return {
        "n_evaluated": int(len(yy)),
        "accuracy": float(accuracy_score(yy, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(yy, pred)),
        "macro_f1_N_S_V_F": float(f1_score(
            yy, pred, labels=EVAL, average="macro", zero_division=0)),
        "weighted_f1_N_S_V_F": float(f1_score(
            yy, pred, labels=EVAL, average="weighted", zero_division=0)),
        "log_loss_N_S_V_F": float(log_loss(yy, p, labels=ALL_LABELS)),
        "brier_multiclass": float(np.mean(np.sum(
            (p - np.eye(len(ALL_LABELS))[yy]) ** 2, axis=1))),
        "ece_15_bins": ece_score(p, yy),
        "per_class": per_class,
        "average_precision": ap,
        "confusion_matrix_labels": C.CLASS_NAMES,
        "confusion_matrix": cm.tolist(),
    }

def patient_bootstrap(proba, y, pid, n_boot=BOOTSTRAP_N):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    patients = np.unique(pid)
    by_patient = {patient: np.flatnonzero(pid == patient) for patient in patients}
    values = []
    pred = proba.argmax(axis=1)
    for _ in range(n_boot):
        chosen = rng.choice(patients, size=len(patients), replace=True)
        idx = np.concatenate([by_patient[x] for x in chosen])
        keep = np.isin(y[idx], EVAL)
        if keep.sum() == 0:
            continue
        values.append(f1_score(
            y[idx][keep], pred[idx][keep], labels=EVAL,
            average="macro", zero_division=0))
    if not values:
        return {"n_bootstrap": 0, "ci95": [None, None], "bootstrap_mean": None}
    return {
        "n_bootstrap": int(len(values)),
        "ci95": [float(np.percentile(values, 2.5)),
                 float(np.percentile(values, 97.5))],
        "bootstrap_mean": float(np.mean(values)),
    }

def fit_predict_one(model_kind, strategy, use_rr, seed, X, rr, y, X2, rr2, folds):
    # The same patient-held-out DS1 fold is used for all model families per seed.
    tr, va = folds[0]  # fixed patient-held-out fold; seed varies independently
    Xtr, rrtr, ytr = X[tr], rr[tr], y[tr]
    Xva, rrva, yva = X[va], rr[va], y[va]
    if model_kind == "cnn":
        from train_cnn import train_cnn
        _, meta = train_cnn(
            Xtr, rrtr, ytr, Xva, rrva, yva,
            use_rr=use_rr, strategy=strategy,
            epochs=60, batch=128, lr=1e-3,
            patience=12, min_epochs=20, seed=seed,
            verbose=False, predict_extra=(X2, rr2), return_model=False
        )
        return meta["proba_extra"], {
            "validation_fold": 0,
            "best_epoch": int(meta["best_epoch"]),
            "best_val_macro_f1": float(meta["best_val_macro_f1"]),
            "training_beats": int(len(tr)),
        }
    Ftr = build_features(Xtr, rrtr, use_rr, C.SEGMENTS, model_kind)
    Fte = build_features(X2, rr2, use_rr, C.SEGMENTS, model_kind)
    sw = sample_weights(ytr) if strategy == "class_weight" else None
    if model_kind == "lr":
        estimator = make_lr(strategy, seed)
        fit_kwargs = {} if sw is None else {"logisticregression__sample_weight": sw}
        estimator.fit(Ftr, ytr, **fit_kwargs)
    elif model_kind == "xgb":
        estimator = make_xgb(seed)
        estimator.fit(Ftr, ytr, sample_weight=sw, verbose=False)
    else:
        raise ValueError(model_kind)
    return estimator.predict_proba(Fte), {
        "validation_fold": 0,
        "training_beats": int(len(tr)),
    }

def save_cm_png(cm, labels, path, title, normalize=False):
    mat = np.asarray(cm, dtype=float)
    if normalize:
        den = mat.sum(axis=1, keepdims=True)
        mat = np.divide(mat, den, out=np.zeros_like(mat), where=den != 0)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(mat, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels,
           xlabel="Predicted class", ylabel="True class", title=title)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    fmt = ".2f" if normalize else "d"
    thresh = mat.max() / 2 if mat.size else 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, (format(int(round(mat[i, j])), "d") if not normalize else format(mat[i, j], ".2f")), ha="center", va="center",
                    color="white" if mat[i, j] > thresh else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

def save_reliability_plot(raw, calibrated, y, path, title):
    keep = np.isin(y, EVAL)
    series = []
    for p in (raw, calibrated):
        pp = p[keep]
        conf = pp.max(axis=1)
        corr = (pp.argmax(axis=1) == y[keep]).astype(float)
        edges = np.linspace(0, 1, 16)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.any():
                xs.append(float(conf[m].mean()))
                ys.append(float(corr[m].mean()))
        series.append((xs, ys))
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    ax.plot(series[0][0], series[0][1], "o-", label="Before scaling")
    ax.plot(series[1][0], series[1][1], "s-", label="After scaling")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted confidence",
           ylabel="Empirical accuracy", title=title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)



def crossfit_temperature_oof(proba, y, fold):
    """Stage-3-compatible leave-one-fold-out temperature calibration."""
    z = np.log(np.clip(proba.astype(np.float64), 1e-12, 1.0))
    out = np.empty_like(proba, dtype=np.float64)
    temperatures = {}
    for f in range(5):
        held = fold == f
        fit = ~held
        fit_keep = np.isin(y[fit], EVAL)
        zfit = z[fit][fit_keep]
        yfit = y[fit][fit_keep]
        def objective(temp):
            q = zfit / temp
            q -= logsumexp(q, axis=1, keepdims=True)
            return float(-q[np.arange(len(yfit)), yfit].mean())
        opt = minimize_scalar(objective, bounds=(0.05, 10.0), method="bounded")
        if not opt.success:
            raise RuntimeError(f"Cross-fold temperature fit failed for fold {f}")
        temperatures[str(f)] = float(opt.x)
        zh = z[held] / opt.x
        zh -= logsumexp(zh, axis=1, keepdims=True)
        out[held] = np.exp(zh)
    return out, temperatures


def classwise_f1_thresholds(proba, y):
    """One-vs-rest F1-optimal class thresholds, fitted only on DS1 OOF."""
    thresholds, curves = {}, {}
    for cls_idx, cls_name in enumerate(C.CLASS_NAMES):
        if cls_idx not in EVAL:
            continue
        binary = (y == cls_idx).astype(int)
        precision, recall, thr = precision_recall_curve(binary, proba[:, cls_idx])
        if len(thr) == 0:
            threshold = 1.0
            best_f1 = 0.0
        else:
            p, r = precision[:-1], recall[:-1]
            f1 = np.divide(2*p*r, p+r, out=np.zeros_like(p), where=(p+r)>0)
            best = int(np.nanargmax(f1))
            threshold, best_f1 = float(thr[best]), float(f1[best])
        thresholds[cls_name] = threshold
        curves[cls_name] = {
            "precision": precision.astype(float).tolist(),
            "recall": recall.astype(float).tolist(),
            "thresholds": thr.astype(float).tolist(),
            "selected_threshold": threshold,
            "best_oof_f1": best_f1,
            "support_positive": int(binary.sum()),
            "support_negative": int((binary == 0).sum()),
        }
    return thresholds, curves


def apply_classwise_rejection(proba, thresholds):
    """Argmax prediction is accepted only if its class-specific DS1 threshold is met.
    Q predictions and below-threshold predictions are abstentions.
    """
    pred = np.argmax(proba, axis=1)
    decided = np.full(len(pred), -1, dtype=int)  # -1 = REJECT
    for cls_idx, cls_name in enumerate(C.CLASS_NAMES):
        if cls_idx not in EVAL:
            continue
        m = pred == cls_idx
        accepted = m & (proba[:, cls_idx] >= thresholds[cls_name])
        decided[accepted] = cls_idx
    return pred, decided


def selective_metrics(proba, y, thresholds):
    keep = np.isin(y, EVAL)
    yy = y[keep]
    pp = proba[keep]
    raw_pred, decision = apply_classwise_rejection(pp, thresholds)
    accepted = decision >= 0
    cm = confusion_matrix(yy, decision, labels=[0, 1, 2, 3, -1])
    per_class = {}
    for idx, cls in enumerate(C.CLASS_NAMES):
        if idx not in EVAL:
            continue
        tp = int(cm[idx, idx])
        fp = int(cm[:, idx].sum() - tp)
        fn = int(cm[idx, :].sum() - tp)
        per_class[cls] = {
            "precision": float(tp/(tp+fp)) if tp+fp else 0.0,
            "recall": float(tp/(tp+fn)) if tp+fn else 0.0,
            "f1": float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.0,
            "support": int(cm[idx, :].sum()),
            "accepted_true": int((decision[yy == idx] == idx).sum()),
            "rejected_true": int((decision[yy == idx] == -1).sum()),
        }
    if accepted.any():
        accepted_y, accepted_pred = yy[accepted], decision[accepted]
        selective_acc = float(accuracy_score(accepted_y, accepted_pred))
        selective_macro = float(f1_score(
            accepted_y, accepted_pred, labels=EVAL, average="macro", zero_division=0))
    else:
        selective_acc, selective_macro = None, None
    coverage = float(accepted.mean()) if len(accepted) else 0.0
    return {
        "n_eval": int(len(yy)),
        "accepted": int(accepted.sum()),
        "rejected": int((~accepted).sum()),
        "coverage": coverage,
        "rejection_rate": float(1-coverage),
        "selective_accuracy": selective_acc,
        "selective_risk": float(1-selective_acc) if selective_acc is not None else None,
        "selective_macro_f1_N_S_V_F": selective_macro,
        "per_class": per_class,
        "confusion_matrix_labels": ["N", "S", "V", "F", "REJECT"],
        "confusion_matrix": cm.tolist(),
        "q_argmax_count_among_eval": int((raw_pred == 4).sum()),
    }


def save_pr_threshold_plot(curves, path, title):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, cls in zip(axes.flat, ["N", "S", "V", "F"]):
        c = curves[cls]
        ax.plot(c["recall"], c["precision"], lw=1.8)
        ax.axvline(
            next((c["recall"][i] for i, t in enumerate(c["thresholds"])
                  if abs(t-c["selected_threshold"]) < 1e-12), 0.0),
            linestyle="--", alpha=0.7, label=f"F1-opt threshold={c['selected_threshold']:.3f}"
        )
        ax.set(xlim=(0,1), ylim=(0,1), xlabel="Recall", ylabel="Precision", title=f"Class {cls}")
        ax.grid(alpha=0.2); ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_rejection_curve(proba, y, thresholds, path, title):
    """Descriptive DS2 coverage-risk curve; no DS2 operating point is selected."""
    keep = np.isin(y, EVAL)
    yy, pp = y[keep], proba[keep]
    pred = pp.argmax(axis=1)
    conf = pp.max(axis=1)
    order = np.argsort(-conf, kind="stable")
    fractions = np.linspace(0, 0.5, 51)
    coverage, risk, macro = [], [], []
    for reject_frac in fractions:
        n_keep = max(1, int(round(len(order)*(1-reject_frac))))
        idx = order[:n_keep]
        accepted_y, accepted_pred = yy[idx], pred[idx]
        coverage.append(n_keep/len(yy))
        risk.append(float(np.mean(accepted_y != accepted_pred)))
        macro.append(float(f1_score(
            accepted_y, accepted_pred, labels=EVAL, average="macro", zero_division=0)))
    # Frozen DS1 class-wise threshold operating point
    _, decided = apply_classwise_rejection(pp, thresholds)
    accepted = decided >= 0
    frozen_cov = float(accepted.mean())
    frozen_risk = float(np.mean(decided[accepted] != yy[accepted])) if accepted.any() else None
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(coverage, risk, label="DS2 descriptive confidence sweep")
    if frozen_risk is not None:
        ax.scatter([frozen_cov], [frozen_risk], marker="*", s=150,
                   label="Frozen DS1 class-wise thresholds")
    ax.set(xlabel="Coverage", ylabel="Selective risk (1 - accuracy)",
           title=title, xlim=(0.5,1.01), ylim=(0,1))
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)
    return {
        "ds2_descriptive_sweep": {
            "coverage": coverage, "selective_risk": risk, "macro_f1": macro,
            "note": "Descriptive curve only; no threshold selected using DS2."
        },
        "frozen_ds1_threshold_operating_point": {
            "coverage": frozen_cov, "selective_risk": frozen_risk
        }
    }


def run_decision_layer(result):
    """Fit thresholds on DS1 OOF only; apply frozen thresholds to DS2 predictions."""
    thresholds_by_cfg = {}
    threshold_rows = []
    selective_rows = []
    for cfg in result["configurations"]:
        tag = "rr" if cfg["use_rr"] else "norr"
        oof_p, oof_y, oof_fold = load_oof(cfg["model"], cfg["strategy"], cfg["use_rr"])
        thresholds, curves = classwise_f1_thresholds(oof_p, oof_y)
        thresholds_by_cfg[cfg["slug"]] = thresholds
        cfg["decision_layer"] = {
            "threshold_source": "DS1 OOF only",
            "threshold_objective": "one-vs-rest F1 maximisation per class",
            "thresholds": thresholds,
            "classwise_oof_pr": {
                cls: {k: v for k, v in curves[cls].items()
                      if k not in ("precision", "recall", "thresholds")}
                for cls in curves
            },
        }
        save_pr_threshold_plot(
            curves, PR_DIR / f"{cfg['slug']}_ds1_oof_classwise_pr.png",
            f"{cfg['name']} — DS1 OOF class-wise precision-recall"
        )
        # DS1 OOF confidence-reject sweep, Stage-3-style diagnostic.
        ds1_conf = oof_p.max(axis=1)
        ds1_order = np.argsort(-ds1_conf, kind="stable")
        ds1_cov, ds1_risk = [], []
        for frac in np.linspace(0, 0.5, 51):
            n_keep = max(1, int(round(len(ds1_order)*(1-frac))))
            idx = ds1_order[:n_keep]
            ds1_cov.append(n_keep/len(oof_y))
            ds1_risk.append(float(np.mean(oof_y[idx] != oof_p[idx].argmax(axis=1))))
        cfg["decision_layer"]["ds1_oof_confidence_reject_curve"] = {
            "coverage": ds1_cov, "selective_risk": ds1_risk,
            "note": "DS1 OOF diagnostic; class-wise thresholds are separately selected per class."
        }
        for cls, threshold in thresholds.items():
            threshold_rows.append({
                "configuration": cfg["name"], "class": cls,
                "threshold": threshold,
                "ds1_oof_best_f1": curves[cls]["best_oof_f1"],
                "positive_support": curves[cls]["support_positive"],
                "negative_support": curves[cls]["support_negative"],
            })
        for run in cfg["runs"]:
            seed = run["seed"]
            pfile = PROB_DIR / f"{cfg['slug']}_seed{seed}_raw.npz"
            if not pfile.exists():
                raise FileNotFoundError(f"Missing saved DS2 probabilities: {pfile}")
            dprob = np.load(pfile, allow_pickle=False)
            p2, y2 = dprob["proba"], dprob["y"]
            sel = selective_metrics(p2, y2, thresholds)
            run["classwise_threshold_rejection_DS2"] = sel
            selective_rows.append({
                "configuration": cfg["name"], "seed": seed,
                "coverage": sel["coverage"], "rejection_rate": sel["rejection_rate"],
                "selective_accuracy": sel["selective_accuracy"],
                "selective_risk": sel["selective_risk"],
                "selective_macro_f1_N_S_V_F": sel["selective_macro_f1_N_S_V_F"],
                "threshold_source": "DS1 OOF only",
            })
            curve_meta = save_rejection_curve(
                p2, y2, thresholds,
                REJECT_DIR / f"{cfg['slug']}_seed{seed}_ds2_coverage_risk.png",
                f"{cfg['name']} — seed {seed} DS2 coverage-risk (descriptive)"
            )
            run["rejection_curve_DS2"] = curve_meta
            # Save a dedicated 5x5 with REJECT column for every seed.
            cm = np.asarray(sel["confusion_matrix"], dtype=int)
            save_cm_png(
                cm, ["N", "S", "V", "F", "REJECT"],
                CM_DIR / f"{cfg['slug']}_seed{seed}_threshold_rejection_counts.png",
                f"{cfg['name']} — seed {seed} DS2 frozen-threshold decisions",
                normalize=False
            )
    with open(OUT / "classwise_thresholds_DS1_OOF.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(threshold_rows[0].keys()))
        w.writeheader(); w.writerows(threshold_rows)
    with open(OUT / "selective_metrics_by_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(selective_rows[0].keys()))
        w.writeheader(); w.writerows(selective_rows)
    return result



def write_outputs(result):
    rows = []
    for cfg in result["configurations"]:
        for run in cfg["runs"]:
            raw = run["raw_DS2"]
            cal = run["calibrated_DS2"]
            row = {
                "configuration": cfg["name"], "model": cfg["model"],
                "strategy": cfg["strategy"], "use_rr": cfg["use_rr"],
                "seed": run["seed"],
                "accuracy": raw["accuracy"],
                "balanced_accuracy": raw["balanced_accuracy"],
                "macro_f1_N_S_V_F": raw["macro_f1_N_S_V_F"],
                "weighted_f1_N_S_V_F": raw["weighted_f1_N_S_V_F"],
                "ece_raw": raw["ece_15_bins"],
                "brier_raw": raw["brier_multiclass"],
                "temperature_seed0_DS1_OOF": run["temperature_DS1_OOF"],
                "ece_calibrated_seed0": cal["ece_15_bins"] if cal else None,
                "brier_calibrated_seed0": cal["brier_multiclass"] if cal else None,
                "bootstrap_ci95_low": run["patient_bootstrap_95CI_macro_f1_raw"]["ci95"][0],
                "bootstrap_ci95_high": run["patient_bootstrap_95CI_macro_f1_raw"]["ci95"][1],
                "coverage_after_DS1_threshold": run.get("classwise_threshold_rejection_DS2", {}).get("coverage"),
                "rejection_rate_after_DS1_threshold": run.get("classwise_threshold_rejection_DS2", {}).get("rejection_rate"),
                "selective_macro_f1_after_DS1_threshold": run.get("classwise_threshold_rejection_DS2", {}).get("selective_macro_f1_N_S_V_F"),
                "selective_risk_after_DS1_threshold": run.get("classwise_threshold_rejection_DS2", {}).get("selective_risk"),
            }
            for cls in C.CLASS_NAMES:
                row[f"f1_{cls}"] = raw["per_class"][cls]["f1"]
                row[f"precision_{cls}"] = raw["per_class"][cls]["precision"]
                row[f"recall_{cls}"] = raw["per_class"][cls]["recall"]
            rows.append(row)
    csv_path = OUT / "stage4_metrics_by_seed.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Mean and standard deviation comparison graph.
    names = [c["name"] for c in result["configurations"]]
    means = [c["summary"]["macro_f1_mean"] for c in result["configurations"]]
    stds = [c["summary"]["macro_f1_std"] for c in result["configurations"]]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, capsize=4)
    ax.set_xticks(x, names, rotation=35, ha="right")
    ax.set_ylabel("Macro-F1 (N/S/V/F)")
    ax.set_title("Stage 4: DS2 Macro-F1 across five seeds (mean ± SD)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "macro_f1_five_seed_comparison.png", dpi=180)
    plt.close(fig)

    # Per-class F1 means and SD across seeds, by configuration.
    fig, ax = plt.subplots(figsize=(13, 6))
    width = 0.16
    x = np.arange(len(names))
    for j, cls in enumerate(C.EVAL_CLASSES):
        vals = []
        errs = []
        for cfg in result["configurations"]:
            v = [r["raw_DS2"]["per_class"][cls]["f1"] for r in cfg["runs"]]
            vals.append(float(np.mean(v)))
            errs.append(float(np.std(v)))
        ax.bar(x + (j - 1.5) * width, vals, width, yerr=errs, capsize=2, label=cls)
    ax.set_xticks(x, names, rotation=35, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1 across five seeds (mean ± SD)")
    ax.legend(title="AAMI class")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "per_class_f1_five_seed_comparison.png", dpi=180)
    plt.close(fig)

    # Seed-0 calibration impact comparison.
    labels, before, after = [], [], []
    for cfg in result["configurations"]:
        seed0 = next(r for r in cfg["runs"] if r["seed"] == 0)
        if seed0["calibrated_DS2"] is not None:
            labels.append(cfg["name"])
            before.append(seed0["raw_DS2"]["ece_15_bins"])
            after.append(seed0["calibrated_DS2"]["ece_15_bins"])
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(labels)); w = 0.36
    ax.bar(x - w/2, before, w, label="Before")
    ax.bar(x + w/2, after, w, label="After")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel("ECE (15 bins)")
    ax.set_title("Seed-0 calibration impact (temperature fitted on DS1 OOF only)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "seed0_calibration_ece_comparison.png", dpi=180)
    plt.close(fig)

    # Coverage and selective-risk comparison from DS1-frozen class thresholds.
    cov_means, cov_stds, risk_means, risk_stds = [], [], [], []
    for cfg in result["configurations"]:
        cov = [r["classwise_threshold_rejection_DS2"]["coverage"] for r in cfg["runs"]]
        risk = [r["classwise_threshold_rejection_DS2"]["selective_risk"]
                for r in cfg["runs"] if r["classwise_threshold_rejection_DS2"]["selective_risk"] is not None]
        cov_means.append(float(np.mean(cov))); cov_stds.append(float(np.std(cov)))
        risk_means.append(float(np.mean(risk)) if risk else 0.0)
        risk_stds.append(float(np.std(risk)) if risk else 0.0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(np.arange(len(names)), cov_means, yerr=cov_stds, capsize=3)
    axes[0].set_title("Coverage after DS1-frozen class thresholds")
    axes[0].set_ylabel("Coverage")
    axes[1].bar(np.arange(len(names)), risk_means, yerr=risk_stds, capsize=3)
    axes[1].set_title("Selective risk on accepted DS2 beats")
    axes[1].set_ylabel("Selective risk (1 - accuracy)")
    for ax in axes:
        ax.set_xticks(np.arange(len(names)), names, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "coverage_selective_risk_comparison.png", dpi=180)
    plt.close(fig)

    # Mean row-normalised confusion matrix across seeds; individual raw matrices saved separately.
    for cfg in result["configurations"]:
        cms = [np.asarray(r["raw_DS2"]["confusion_matrix"], dtype=float) for r in cfg["runs"]]
        normed = []
        for cm in cms:
            den = cm.sum(axis=1, keepdims=True)
            normed.append(np.divide(cm, den, out=np.zeros_like(cm), where=den != 0))
        mean_norm = np.mean(normed, axis=0)
        slug = cfg["slug"]
        save_cm_png(mean_norm, C.CLASS_NAMES,
                    CM_DIR / f"{slug}_mean_normalized_across_seeds.png",
                    f"{cfg['name']} — mean row-normalised CM (5 seeds)", normalize=True)
        summed = np.sum(cms, axis=0).astype(int)
        save_cm_png(summed, C.CLASS_NAMES,
                    CM_DIR / f"{slug}_summed_counts_across_5_seed_runs.png",
                    f"{cfg['name']} — summed counts across 5 runs (repeated DS2)", normalize=False)

    # Seed-0 reliability plots for all configs.
    for cfg in result["configurations"]:
        seed0 = next(r for r in cfg["runs"] if r["seed"] == 0)
        if seed0["calibrated_DS2"] is None:
            continue
        # Store only probabilities for seed 0 in an internal compressed archive.
        pfile = OUT / f"{cfg['slug']}_seed0_probabilities.npz"
        d = np.load(pfile, allow_pickle=False)
        save_reliability_plot(
            d["raw"], d["calibrated"], d["y"],
            REL_DIR / f"{cfg['slug']}_seed0_reliability.png",
            f"{cfg['name']} — seed 0 reliability"
        )

    # Report, separate from figures.
    lines = [
        "# AMAC-ECG Stage 4 — Final Evaluation Report", "",
        "## Protocol",
        "- Five random seeds were evaluated for each of ten configurations (five model/imbalance settings × RR enabled/disabled).",
        "- All five random seeds use the same fixed patient-held-out DS1 fold (fold 0) for validation across configurations; the remaining DS1 patients are used for fitting. Random seeds vary independently of fold assignment.",
        "- LR and XGBoost use the DS1 training partition for fitting; CNN uses the DS1 validation partition for early stopping.",
        "- Stage-3-compatible leave-one-fold-out temperature scaling is run on DS1 OOF predictions for each configuration; the mean of its five fold-specific temperatures is applied only to seed-0 DS2 probabilities. Calibration is reported only for seed 0, not averaged across seeds.",
        "- Class Q remains a model output but is excluded from headline metrics; headline classification metrics use N, S, V and F.",
        "- DS2 is not used for fitting, model selection, calibration fitting or threshold tuning in this run.",
        "- Repository documentation notes prior Stage 4 access to DS2; therefore this report does not describe DS2 as historically untouched or pristine.", "",
        "## Five-seed Macro-F1 summary", "",
        "| Configuration | Macro-F1 mean | Macro-F1 SD |", "|---|---:|---:|"
    ]
    for cfg in result["configurations"]:
        lines.append(f"| {cfg['name']} | {cfg['summary']['macro_f1_mean']:.4f} | {cfg['summary']['macro_f1_std']:.4f} |")
    lines += [
        "", "## Calibration scope",
        "Calibration analysis is seed-0-specific. Temperature is fitted from the corresponding five-fold DS1 OOF predictions. Raw and calibrated seed-0 DS2 metrics are both saved; seeds 1–4 remain uncalibrated in this analysis.",
        "", "## Outputs",
        "- `stage4_results.json`: complete numerical results, confusion matrices and per-seed metadata.",
        "- `stage4_metrics_by_seed.csv`: one row per configuration and seed.",
        "- `figures/confusion_matrices/`: per-seed raw confusion matrices, summed five-run counts, mean normalised matrices, and per-seed frozen-threshold matrices with a REJECT column.",
        "- `figures/macro_f1_five_seed_comparison.png`: Macro-F1 mean ± SD.",
        "- `figures/per_class_f1_five_seed_comparison.png`: per-class F1 mean ± SD.",
        "- `figures/seed0_calibration_ece_comparison.png`: seed-0 calibration impact.",
        "- `figures/reliability/`: seed-0 reliability diagrams.",
        "- `classwise_thresholds_DS1_OOF.csv`: frozen N/S/V/F thresholds selected on DS1 OOF only.",
        "- `selective_metrics_by_seed.csv`: DS2 coverage, selective risk and selective Macro-F1.",
        "- `figures/precision_recall/`: DS1 OOF per-class precision-recall plots with selected thresholds.",
        "- `figures/rejection/`: per-seed DS2 descriptive coverage-risk curves with the frozen DS1 operating point.",
        "- `figures/coverage_selective_risk_comparison.png`: five-seed coverage and selective-risk comparison.",
        "- `figures/confusion_matrices/*threshold_rejection*`: DS2 confusion matrices with a REJECT column.",
        "", "## Interpretation cautions",
        "- The five-seed SD describes variation across model initialisations and validation-fold assignments; it is not a patient-level confidence interval.",
        "- Patient-cluster bootstrap 95% CIs are stored per seed in the JSON and CSV.",
        "- Calibration conclusions are limited to seed 0 and must not be described as multi-seed calibration robustness.",
        "- The RR ablation compares each model/imbalance configuration with RR features enabled versus disabled.",
    ]
    (OUT / "STAGE4_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

def main():
    started = time.time()
    ds1_path, ds2_path = ART / "ds1.npz", ART / "ds2.npz"
    if not ds1_path.exists() or not ds2_path.exists():
        raise FileNotFoundError("Both artifacts/ds1.npz and artifacts/ds2.npz must exist before final evaluation.")
    ds1 = np.load(ds1_path, allow_pickle=False)
    X, rr, y, pid = ds1["X"], ds1["rr"], ds1["y"], ds1["pid"]
    ds1_hash = file_sha256(ds1_path)
    folds = load_folds(y, pid, ds1_hash)
    for i, (tr, va) in enumerate(folds):
        assert set(pid[tr]).isdisjoint(set(pid[va])), f"Patient overlap in DS1 fold {i}"
    # Only now, in the final evaluation cell, open DS2.
    ds2 = np.load(ds2_path, allow_pickle=False)
    X2, rr2, y2, pid2 = ds2["X"], ds2["rr"], ds2["y"], ds2["pid"]
    assert X2.shape == (47534, 216) and rr2.shape == (47534, 4)
    assert set(np.unique(pid)).isdisjoint(set(np.unique(pid2))), "DS1/DS2 patient overlap detected."
    assert set(np.unique(y2)).issubset(set(range(len(C.CLASS_NAMES))))
    ds2_hash = file_sha256(ds2_path)

    result = {
        "stage": "4",
        "protocol": "DS1 model fitting and seed-0 OOF calibration; DS2 final forward-pass evaluation",
        "ds2_usage_in_this_run": "not used for fitting, model selection, calibration fitting, or threshold tuning; opened only in final evaluation",
        "historical_note": "Repository DATA_README documents prior Stage 4 access; this run does not claim DS2 is historically untouched.",
        "dataset": {
            "ds1_beats": int(len(y)), "ds1_patients": int(len(np.unique(pid))),
            "ds1_sha256": ds1_hash, "ds2_beats": int(len(y2)),
            "ds2_patients": int(len(np.unique(pid2))), "ds2_sha256": ds2_hash
        },
        "seeds": SEEDS, "configurations": []
    }

    for model, strategy, use_rr, name in CONFIGS:
        print(f"\n=== {name} ===")
        oof_p, oof_y, oof_fold = load_oof(model, strategy, use_rr)
        oof_calibrated, fold_temperatures = crossfit_temperature_oof(oof_p, oof_y, oof_fold)
        temperature = float(np.mean(list(fold_temperatures.values())))
        np.savez_compressed(
            PROB_DIR / f"{model}_{strategy}_{'rr' if use_rr else 'norr'}_ds1_oof_crosscal.npz",
            proba=oof_calibrated.astype(np.float32),
            y=oof_y.astype(np.int16), fold=oof_fold.astype(np.int8)
        )
        print(f"Seed-0 DS1 OOF temperature: {temperature:.5f}")
        cfg = {
            "model": model, "strategy": strategy, "use_rr": use_rr,
            "name": name,
            "slug": f"{model}_{strategy}_{'rr' if use_rr else 'norr'}",
            "seed0_temperature_DS1_OOF": temperature,
            "stage3_crossfit_fold_temperatures_DS1_OOF": fold_temperatures,
            "runs": []
        }
        for seed in SEEDS:
            print(f"  seed {seed}: fitting on DS1 and evaluating on DS2...")
            p2, fit_meta = fit_predict_one(
                model, strategy, use_rr, seed, X, rr, y, X2, rr2, folds
            )
            raw = metric_pack(p2, y2)
            np.savez_compressed(
                PROB_DIR / f"{cfg['slug']}_seed{seed}_raw.npz",
                proba=p2.astype(np.float32), y=y2.astype(np.int16)
            )
            calibrated = None
            if seed == 0:
                p2_cal = apply_temperature(p2, temperature)
                calibrated = metric_pack(p2_cal, y2)
                np.savez_compressed(
                    OUT / f"{cfg['slug']}_seed0_probabilities.npz",
                    raw=p2.astype(np.float32),
                    calibrated=p2_cal.astype(np.float32),
                    y=y2.astype(np.int16)
                )
                save_reliability_plot(
                    p2, p2_cal, y2,
                    REL_DIR / f"{cfg['slug']}_seed0_reliability.png",
                    f"{name} — seed 0 reliability"
                )
            cm = np.asarray(raw["confusion_matrix"], dtype=int)
            save_cm_png(
                cm, C.CLASS_NAMES,
                CM_DIR / f"{cfg['slug']}_seed{seed}_counts.png",
                f"{name} — seed {seed} confusion matrix (counts)",
                normalize=False
            )
            ci = patient_bootstrap(p2, y2, pid2)
            cfg["runs"].append({
                "seed": seed,
                "fit_metadata": fit_meta,
                "temperature_DS1_OOF": temperature if seed == 0 else None,
                "raw_DS2": raw,
                "calibrated_DS2": calibrated,
                "patient_bootstrap_95CI_macro_f1_raw": ci
            })
            print(
                f"    Macro-F1={raw['macro_f1_N_S_V_F']:.4f}; "
                f"Accuracy={raw['accuracy']:.4f}; ECE={raw['ece_15_bins']:.4f}"
            )
        values = [r["raw_DS2"]["macro_f1_N_S_V_F"] for r in cfg["runs"]]
        cfg["summary"] = {
            "macro_f1_mean": float(np.mean(values)),
            "macro_f1_std": float(np.std(values, ddof=0)),
        }
        result["configurations"].append(cfg)
        with open(OUT / "stage4_results_partial.json", "w") as f:
            json.dump(result, f, indent=2)

    result["elapsed_seconds"] = float(time.time() - started)
    result["environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    # DS1-only threshold fitting and frozen DS2 rejection evaluation.
    result = run_decision_layer(result)
    with open(OUT / "stage4_results.json", "w") as f:
        json.dump(result, f, indent=2)
    receipt = {
        "status": "completed",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": result["elapsed_seconds"],
        "ds1_sha256": ds1_hash, "ds2_sha256": ds2_hash,
        "seeds": SEEDS,
        "configurations": [c["name"] for c in CONFIGS],
        "runs_completed": sum(len(c["runs"]) for c in result["configurations"]),
        "ds2_usage_in_this_run": result["ds2_usage_in_this_run"],
        "historical_note": result["historical_note"],
    }
    with open(OUT / "RUN_RECEIPT.json", "w") as f:
        json.dump(receipt, f, indent=2)
    write_outputs(result)
    print(f"\nSTAGE 4 COMPLETE — {receipt['runs_completed']} runs.")
    print(f"Results: {OUT}")
    print(f"Figures: {FIG}")

if __name__ == "__main__":
    main()

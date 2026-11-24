"""
JEV QA Statistical Evaluators: Calibration (ECE / Brier), Confusion Matrix,
Feature/Prediction Drift (PSI / JS Divergence), and P95/P99 Latency SLA Verification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np


def compute_brier_score(y_true_indices: Sequence[int], probs_matrix: Sequence[Sequence[float]]) -> float:
    """Multi-class Brier score: mean squared error between predicted prob vector and one-hot truth."""
    y_idx = np.asarray(y_true_indices, dtype=int)
    p = np.asarray(probs_matrix, dtype=np.float64)
    if p.ndim != 2 or len(y_idx) != p.shape[0]:
        raise ValueError("Shape mismatch between y_true_indices and probs_matrix")
    one_hot = np.zeros_like(p)
    one_hot[np.arange(len(y_idx)), y_idx] = 1.0
    return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))


def compute_ece(confidences: Sequence[float], is_correct: Sequence[bool], n_bins: int = 10) -> Dict[str, Any]:
    """Expected Calibration Error (ECE) over equal-width probability bins in [0, 1]."""
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(is_correct, dtype=np.float64)
    if len(conf) == 0 or len(conf) != len(corr):
        raise ValueError("Invalid confidence/correctness arrays for ECE")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins_summary: List[Dict[str, Any]] = []
    n = len(conf)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        count = int(np.sum(mask))
        if count == 0:
            continue
        avg_conf = float(np.mean(conf[mask]))
        avg_acc = float(np.mean(corr[mask]))
        ece += (count / n) * abs(avg_acc - avg_conf)
        bins_summary.append(
            {
                "bin_range": [round(float(lo), 2), round(float(hi), 2)],
                "count": count,
                "avg_confidence": round(avg_conf, 4),
                "empirical_accuracy": round(avg_acc, 4),
            }
        )

    return {
        "ece": round(float(ece), 6),
        "accuracy": round(float(np.mean(corr)), 6),
        "mean_confidence": round(float(np.mean(conf)), 6),
        "bins": bins_summary,
    }


def compute_classification_report(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> Dict[str, Any]:
    """Confusion matrix + per-class and macro Precision/Recall/F1."""
    label_list = list(labels)
    idx_map = {lbl: i for i, lbl in enumerate(label_list)}
    k = len(label_list)
    cm = np.zeros((k, k), dtype=int)

    for yt, yp in zip(y_true, y_pred):
        cm[idx_map[yt], idx_map[yp]] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    f1s: List[float] = []
    for i, lbl in enumerate(label_list):
        tp = int(cm[i, i])
        fp = int(np.sum(cm[:, i]) - tp)
        fn = int(np.sum(cm[i, :]) - tp)
        support = int(np.sum(cm[i, :]))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        if support > 0:
            f1s.append(f1)
        per_class[lbl] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }

    total = int(np.sum(cm))
    acc = float(np.trace(cm) / total) if total > 0 else 0.0
    return {
        "labels": label_list,
        "confusion_matrix": cm.tolist(),
        "accuracy": round(acc, 4),
        "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 4),
        "per_class": per_class,
    }


def compute_psi(baseline_values: Sequence[float], current_values: Sequence[float], n_bins: int = 10) -> Dict[str, Any]:
    """Population Stability Index (PSI) to detect confidence or numeric distribution drift."""
    base = np.asarray(baseline_values, dtype=np.float64)
    curr = np.asarray(current_values, dtype=np.float64)
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(base, quantiles)
    # Handle degenerate constant baseline distributions cleanly
    edges[0] = min(float(np.min(base)), float(np.min(curr))) - 1e-9
    edges[-1] = max(float(np.max(base)), float(np.max(curr))) + 1e-9
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(float(edges[0]), float(edges[-1]) + 1e-6, 4)

    base_counts, _ = np.histogram(base, bins=edges)
    curr_counts, _ = np.histogram(curr, bins=edges)
    eps = 1e-6
    base_pct = np.clip(base_counts / max(len(base), 1), eps, None)
    curr_pct = np.clip(curr_counts / max(len(curr), 1), eps, None)
    base_pct /= np.sum(base_pct)
    curr_pct /= np.sum(curr_pct)

    psi = float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))
    if psi < 0.10:
        severity = "STABLE"
    elif psi < 0.25:
        severity = "MODERATE_SHIFT"
    else:
        severity = "CRITICAL_DRIFT"
    return {"psi": round(psi, 6), "severity": severity}


def compute_js_divergence(p_dist: Dict[str, float], q_dist: Dict[str, float]) -> float:
    """Base-2 Jensen-Shannon divergence in [0, 1] for categorical class-mix drift."""
    keys = sorted(set(p_dist.keys()) | set(q_dist.keys()))
    p = np.array([max(p_dist.get(k, 0.0), 0.0) for k in keys], dtype=np.float64) + 1e-12
    q = np.array([max(q_dist.get(k, 0.0), 0.0) for k in keys], dtype=np.float64) + 1e-12
    p /= np.sum(p)
    q /= np.sum(q)
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log2(p / m)))
    kl_qm = float(np.sum(q * np.log2(q / m)))
    return round(0.5 * (kl_pm + kl_qm), 6)


def verify_latency_sla(
    latencies_ms: Sequence[float],
    p95_limit_ms: float = 500.0,
    p99_limit_ms: float = 800.0,
) -> Dict[str, Any]:
    """Computes P50/P95/P99 latency quantiles and verifies System-1 SLA bounds."""
    arr = np.asarray(latencies_ms, dtype=np.float64)
    p50 = float(np.percentile(arr, 50))
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    max_v = float(np.max(arr))
    return {
        "count": int(len(arr)),
        "p50_ms": round(p50, 4),
        "p95_ms": round(p95, 4),
        "p99_ms": round(p99, 4),
        "max_ms": round(max_v, 4),
        "p95_limit_ms": p95_limit_ms,
        "p99_limit_ms": p99_limit_ms,
        "sla_passed": bool(p95 <= p95_limit_ms and p99 <= p99_limit_ms),
    }

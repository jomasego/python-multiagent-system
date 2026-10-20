from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        if obj.size > 2000:
            return {"shape": list(obj.shape), "dtype": str(obj.dtype)}
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        total += float(np.sum(mask)) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return total / len(probs)


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def risk_coverage(m: np.ndarray, correct: np.ndarray) -> Dict[str, object]:
    order = np.argsort(-m)
    c = correct[order].astype(np.float64)
    cum = np.cumsum(c)
    k = np.arange(1, len(c) + 1)
    risk = 1.0 - cum / k
    coverage = k / len(c)
    grid = np.linspace(0.0, 1.0, 101)
    idx = np.searchsorted(coverage, grid, side="left").clip(0, len(coverage) - 1)
    risk_grid = risk[idx]
    return {
        "coverage": coverage.tolist() if len(coverage) <= 2000 else None,
        "risk": risk.tolist() if len(risk) <= 2000 else None,
        "aurc": float(np.mean(risk_grid)),
        "risk_at_coverage": {str(round(float(g), 2)): float(r) for g, r in zip(grid, risk_grid)},
    }


def per_class_roc(probs: np.ndarray, labels: np.ndarray, class_names: Optional[Sequence[str]] = None) -> List[Dict[str, object]]:
    from sklearn.metrics import roc_auc_score, roc_curve

    out = []
    c = probs.shape[1]
    for k in range(c):
        y = (labels == k).astype(int)
        if y.min() == y.max():
            continue
        score = probs[:, k]
        auc = float(roc_auc_score(y, score))
        fpr, tpr, thr = roc_curve(y, score, drop_intermediate=False)
        i05 = int(np.argmin(np.abs(thr - 0.5)))
        prevalence = float(y.mean())
        precision = tpr * prevalence / np.maximum(tpr * prevalence + fpr * (1 - prevalence), 1e-12)
        f1 = 2 * precision * tpr / np.maximum(precision + tpr, 1e-12)
        ibest = int(np.argmax(f1))
        name = class_names[k] if class_names is not None and k < len(class_names) else str(k)
        out.append(
            {
                "class": name,
                "auc": auc,
                "threshold_at_0.5": float(thr[i05]),
                "tpr_at_0.5": float(tpr[i05]),
                "fpr_at_0.5": float(fpr[i05]),
                "f1_at_0.5": float(f1[i05]),
                "best_f1_threshold": float(thr[ibest]),
                "best_f1": float(f1[ibest]),
                "tpr_at_best_f1": float(tpr[ibest]),
                "fpr_at_best_f1": float(fpr[ibest]),
                "prevalence": float(y.mean()),
            }
        )
    return out


def summarize_probs(
    probs: np.ndarray,
    name: str,
    links: Dict[str, str],
    labels: Optional[np.ndarray] = None,
    logits: Optional[np.ndarray] = None,
    class_names: Optional[Sequence[str]] = None,
    temperatures: Sequence[float] = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
) -> Dict[str, object]:
    probs = np.asarray(probs, dtype=np.float64)
    probs = probs / probs.sum(axis=1, keepdims=True)
    n, c = probs.shape
    arg = probs.argmax(axis=1)
    m = probs.max(axis=1)
    delta = float(np.mean(m < 0.5))
    coverage = float(np.mean(m >= 0.5))
    sorted_p = np.sort(probs, axis=1)
    tie_rate = float(np.mean(sorted_p[:, -1] - sorted_p[:, -2] < 1e-12))
    selected = probs >= 0.5
    k = selected.sum(axis=1)
    ambiguous = float(np.mean(k >= 2))
    wrong_selected = None
    if labels is not None:
        idx = np.flatnonzero(k == 1)
        if len(idx):
            wrong_selected = int(np.sum(selected[idx].argmax(axis=1) != arg[idx]))
    q = probs / (2.0 * m[:, None])
    q_top = q[np.arange(n), arg]
    loser = q.copy()
    loser[np.arange(n), arg] = -1.0
    adaptive_ok = float(np.mean(((q >= 0.5).sum(axis=1) == 1) & ((q >= 0.5).argmax(axis=1) == arg)))
    out: Dict[str, object] = {
        "name": name,
        "links": links,
        "n": int(n),
        "C": int(c),
        "class_names": list(class_names) if class_names is not None else [str(i) for i in range(c)],
        "argmax_entropy": float(-np.mean(np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1))),
        "mean_max_prob": float(m.mean()),
        "median_max_prob": float(np.median(m)),
        "delta_0.5": delta,
        "coverage_0.5": coverage,
        "tie_rate": tie_rate,
        "ambiguous_rate": ambiguous,
        "wrong_when_selected": wrong_selected,
        "adaptive_winner_gap": float(np.max(np.abs(q_top - 0.5))),
        "adaptive_loser_max": float(np.max(loser)),
        "adaptive_decision_match": adaptive_ok,
        "exact_adaptive": bool(adaptive_ok >= 1.0 - 1e-12),
    }
    order = np.argsort(-m)
    out["top_confidence"] = {
        "p50": float(m[order[int(0.50 * (n - 1))]]),
        "p90": float(m[order[int(0.10 * (n - 1))]]),
        "p99": float(m[order[int(0.01 * (n - 1))]]),
        "max": float(m.max()),
        "min": float(m.min()),
    }
    if labels is not None:
        labels = np.asarray(labels, dtype=int)
        correct = (arg == labels).astype(np.float64)
        covered = m >= 0.5
        out.update(
            {
                "accuracy_argmax": float(correct.mean()),
                "accuracy_0.5_abstain_as_error": float(np.mean(covered & (arg == labels))),
                "selective_accuracy": float(correct[covered].mean()) if np.any(covered) else None,
                "coverage": coverage,
                "ece_15": ece(probs, labels),
                "brier": brier(probs, labels),
                "n_correct": int(correct.sum()),
                "n_wrong": int(n - correct.sum()),
            }
        )
        out["risk_coverage"] = risk_coverage(m, correct)
        out["per_class_roc"] = per_class_roc(probs, labels, class_names)
        freq = np.bincount(labels, minlength=c).astype(np.float64)
        freq = freq / freq.sum()
        w = freq / freq.mean()
        arg_w = (probs * w).argmax(axis=1)
        out["prior_reweighting"] = {
            "weights": w.tolist(),
            "flip_rate_vs_argmax": float(np.mean(arg_w != arg)),
            "accuracy": float(np.mean(arg_w == labels)),
            "accuracy_unweighted": float(correct.mean()),
        }
        out["per_class_counts"] = np.bincount(labels, minlength=c).tolist()
    if logits is not None:
        logits = np.asarray(logits, dtype=np.float64)
        rows = []
        base_arg = logits.argmax(axis=1)
        for t in temperatures:
            z = logits / float(t)
            z = z - z.max(axis=1, keepdims=True)
            e = np.exp(z)
            pt = e / e.sum(axis=1, keepdims=True)
            mt = pt.max(axis=1)
            row = {
                "temperature": float(t),
                "delta_0.5": float(np.mean(mt < 0.5)),
                "coverage_0.5": float(np.mean(mt >= 0.5)),
                "mean_max_prob": float(mt.mean()),
                "argmax_agreement": float(np.mean(pt.argmax(axis=1) == base_arg)),
                "entropy": float(-np.mean(np.sum(pt * np.log(np.maximum(pt, 1e-12)), axis=1))),
            }
            if labels is not None:
                row["accuracy_argmax"] = float(np.mean(pt.argmax(axis=1) == labels))
                row["ece_15"] = ece(pt, labels)
                row["accuracy_0.5_abstain_as_error"] = float(np.mean((mt >= 0.5) & (pt.argmax(axis=1) == labels)))
            rows.append(row)
        out["temperature_sweep"] = rows
    return to_jsonable(out)


def linear_conversion_fit(probs: np.ndarray, name: str = "") -> Dict[str, object]:
    import warnings

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    y = probs.argmax(axis=1)
    if len(np.unique(y)) < 2:
        return {"name": name, "error": "degenerate labels"}
    x_tr, x_te, y_tr, y_te = train_test_split(probs, y, test_size=0.3, random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=3000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(x_tr, y_tr)
    return to_jsonable(
        {
            "name": name,
            "train_accuracy": float(clf.score(x_tr, y_tr)),
            "test_accuracy": float(clf.score(x_te, y_te)),
            "chance": float(1.0 / probs.shape[1]),
        }
    )

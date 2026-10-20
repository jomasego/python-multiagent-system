from __future__ import annotations

import json
import os
import warnings
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine, make_classification
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .diagnostics import ece, to_jsonable


def _splits(x: np.ndarray, y: np.ndarray, seed: int) -> Tuple[np.ndarray, ...]:
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(x, y, test_size=0.5, stratify=y, random_state=seed)
    x_va, x_te, y_va, y_te = train_test_split(x_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)
    return x_tr, x_va, x_te, y_tr, y_va, y_te


def _datasets(seed: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for name, loader in (("breast_cancer", load_breast_cancer), ("wine", load_wine), ("iris", load_iris), ("digits", load_digits)):
        x, y = loader(return_X_y=True)
        out.append({"name": name, "x": np.asarray(x, dtype=np.float64), "y": np.asarray(y)})
    for classes, samples, features in ((5, 3000, 20), (20, 4000, 30)):
        x, y = make_classification(
            n_samples=samples,
            n_features=features,
            n_informative=max(4, features // 2),
            n_redundant=2,
            n_classes=classes,
            n_clusters_per_class=1,
            class_sep=1.0,
            flip_y=0.03,
            random_state=seed,
        )
        out.append({"name": f"synthetic_{classes}", "x": np.asarray(x, dtype=np.float64), "y": np.asarray(y)})
    return out


def _classifiers(seed: int) -> List[Tuple[str, Any]]:
    return [
        ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=seed))),
        ("random_forest", RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)),
        ("hist_gb", HistGradientBoostingClassifier(random_state=seed)),
        ("gaussian_nb", GaussianNB()),
        ("knn", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance"))),
    ]


def _best_f1(y: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    order = np.argsort(-scores, kind="mergesort")
    ys = y[order].astype(np.float64)
    tp = np.cumsum(ys)
    fp = np.cumsum(1.0 - ys)
    fn = float(ys.sum()) - tp
    f1 = 2.0 * tp / np.maximum(2.0 * tp + fp + fn, 1e-12)
    last = np.r_[np.where(np.diff(scores[order]) != 0.0)[0], len(scores) - 1]
    f1_at = f1[last]
    tau_at = scores[order][last]
    best = np.max(f1_at)
    close = np.isclose(f1_at, best, rtol=1e-12, atol=1e-12)
    candidates = tau_at[close]
    tau = float(candidates[np.argmin(np.abs(candidates - 0.5))])
    return {"tau": tau, "f1": float(best), "f1_at_0.5": float(f1_score(y, (scores >= 0.5).astype(int), zero_division=0))}


def _convert(scores: np.ndarray, tau: float) -> np.ndarray:
    tau = float(np.clip(tau, 1e-9, 1.0 - 1e-9))
    return 0.5 + (scores - tau) / (2.0 * (1.0 - tau))


def run_tabular(out_dir: str, seed: int = 0) -> Dict[str, Any]:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    rows: List[Dict[str, Any]] = []
    for data in _datasets(seed):
        y = data["y"]
        x_tr, x_va, x_te, y_tr, y_va, y_te = _splits(data["x"], y, seed)
        for name, model in _classifiers(seed):
            model.fit(x_tr, y_tr)
            proba = np.asarray(model.predict_proba(np.vstack([x_va, x_te])), dtype=np.float64)
            c = len(np.unique(y))
            p_va = proba[: len(x_va)]
            p_te = proba[len(x_va) :]
            if c == 2:
                detail = _binary_from_proba(y_va, y_te, p_va, p_te, seed)
                kind = "binary"
            else:
                detail = _multiclass_from_proba(y_va, y_te, np.vstack([p_va, p_te]), len(x_va), seed)
                kind = "multiclass"
            row = {
                "dataset": data["name"],
                "classifier": name,
                "kind": kind,
                "C": int(c),
                "n_train": int(len(y_tr)),
                "n_test": int(len(y_te)),
                **detail,
            }
            rows.append(row)
            if kind == "binary":
                print(
                    f"[tabular] {row['dataset']}/{name} | C=2 n={row['n_test']} | F1@0.5={row['f1_at_0.5']:.4f} "
                    f"| F1@tau={row['f1_at_tau']:.4f} | tau={row['tau']:.4f} | exact={row['conversion_exact']} "
                    f"| delta={100*row['delta_0.5']:.2f}%",
                    flush=True,
                )
            else:
                print(
                    f"[tabular] {row['dataset']}/{name} | C={row['C']} n={row['n_test']} | macroF1@0.5={row['macro_f1_at_0.5']:.4f} "
                    f"| macroF1@tau={row['macro_f1_at_tau']:.4f} | argmaxF1={row['argmax_f1']:.4f} "
                    f"| argmaxConvF1={row['argmax_converted_f1']:.4f} | agree={100*row['argmax_agreement']:.1f}% "
                    f"| tauSpread={row['tau_spread']:.3f} | exact={row['conversion_exact']} | delta={100*row['delta_0.5']:.2f}%",
                    flush=True,
                )
    result = {"experiments": rows}
    with open(os.path.join(out_dir, "tabular_results.json"), "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(result), fh, indent=2, sort_keys=True)
    _write_markdown(rows, os.path.join(out_dir, "tabular_summary.md"))
    _plots(rows, out_dir)
    return result


def _binary_from_proba(y_va: np.ndarray, y_te: np.ndarray, p_va: np.ndarray, p_te: np.ndarray, seed: int) -> Dict[str, Any]:
    del seed
    best = _best_f1(y_va, p_va[:, 1])
    tau = float(np.clip(best["tau"], 1e-9, 1.0 - 1e-9))
    pred_tau = (p_te[:, 1] >= tau).astype(int)
    pred_conv = (_convert(p_te[:, 1], tau) >= 0.5).astype(int)
    pred_half = (p_te[:, 1] >= 0.5).astype(int)
    return {
        "tau": tau,
        "f1_at_0.5": float(f1_score(y_te, pred_half, zero_division=0)),
        "f1_val_at_0.5": float(f1_score(y_va, (p_va[:, 1] >= 0.5).astype(int), zero_division=0)),
        "f1_at_tau_val": float(best["f1"]),
        "f1_at_tau": float(f1_score(y_te, pred_tau, zero_division=0)),
        "f1_at_converted_0.5": float(f1_score(y_te, pred_conv, zero_division=0)),
        "conversion_exact": bool(np.array_equal(pred_tau, pred_conv)),
        "accuracy_at_0.5": float(np.mean(pred_half == y_te)),
        "accuracy_at_tau": float(np.mean(pred_tau == y_te)),
        "delta_0.5": float(np.mean(np.max(p_te, axis=1) < 0.5)),
        "ece": ece(p_te, y_te),
        "support_positive": int(y_te.sum()),
    }


def _multiclass_from_proba(y_va: np.ndarray, y_te: np.ndarray, proba: np.ndarray, n_va: int, seed: int) -> Dict[str, Any]:
    del seed
    classes = np.unique(y_va)
    rows: List[Dict[str, Any]] = []
    pred_half = np.zeros((len(y_te), len(classes)), dtype=bool)
    pred_conv = np.zeros_like(pred_half)
    for j, cls in enumerate(classes):
        yv = (y_va == cls).astype(int)
        yt = (y_te == cls).astype(int)
        best = _best_f1(yv, proba[:n_va, j])
        tau = float(np.clip(best["tau"], 1e-9, 1.0 - 1e-9))
        pred_half[:, j] = proba[n_va:, j] >= 0.5
        pred_conv[:, j] = _convert(proba[n_va:, j], tau) >= 0.5
        rows.append(
            {
                "class": str(cls),
                "tau": tau,
                "prevalence_test": float(yt.mean()),
                "f1_at_0.5": float(f1_score(yt, pred_half[:, j].astype(int), zero_division=0)),
                "f1_val_at_0.5": float(f1_score(yv, (proba[:n_va, j] >= 0.5).astype(int), zero_division=0)),
                "f1_at_tau_val": float(best["f1"]),
                "f1_at_tau": float(f1_score(yt, (proba[n_va:, j] >= tau).astype(int), zero_division=0)),
                "f1_at_converted_0.5": float(f1_score(yt, pred_conv[:, j].astype(int), zero_division=0)),
                "exact": bool(np.array_equal((proba[n_va:, j] >= tau), pred_conv[:, j])),
            }
        )
    pred_argmax = classes[np.argmax(proba[n_va:], axis=1)]
    converted = np.column_stack([_convert(proba[n_va:, j], rows[j]["tau"]) for j in range(len(classes))])
    pred_conv_argmax = classes[np.argmax(converted, axis=1)]
    return {
        "tau_min": float(min(r["tau"] for r in rows)),
        "tau_max": float(max(r["tau"] for r in rows)),
        "tau_spread": float(max(r["tau"] for r in rows) - min(r["tau"] for r in rows)),
        "n_tau_not_0.5": int(sum(abs(r["tau"] - 0.5) > 1e-9 for r in rows)),
        "macro_f1_at_0.5": float(np.mean([r["f1_at_0.5"] for r in rows])),
        "macro_f1_val_at_0.5": float(np.mean([r["f1_val_at_0.5"] for r in rows])),
        "macro_f1_at_tau_val": float(np.mean([r["f1_at_tau_val"] for r in rows])),
        "macro_f1_at_tau": float(np.mean([r["f1_at_tau"] for r in rows])),
        "macro_f1_at_converted_0.5": float(np.mean([r["f1_at_converted_0.5"] for r in rows])),
        "conversion_exact": bool(all(r["exact"] for r in rows)),
        "argmax_f1": float(f1_score(y_te, pred_argmax, average="macro", zero_division=0)),
        "argmax_converted_f1": float(f1_score(y_te, pred_conv_argmax, average="macro", zero_division=0)),
        "argmax_agreement": float(np.mean(pred_argmax == pred_conv_argmax)),
        "delta_0.5": float(np.mean(np.max(proba[n_va:], axis=1) < 0.5)),
        "ece": ece(proba[n_va:], y_te),
        "class_rows": rows,
    }


def _write_markdown(rows: Sequence[Dict[str, Any]], path: str) -> None:
    lines = ["# Tabular max-F1 conversion", "", "| dataset | classifier | C | n | tau | F1val@0.5 | F1val@tau | F1test@0.5 | F1test@tau | F1test@conv0.5 | exact | delta |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for r in rows:
        if r["kind"] == "binary":
            fval05, fval, f05, ftau = r["f1_val_at_0.5"], r["f1_at_tau_val"], r["f1_at_0.5"], r["f1_at_tau"]
            tau = f"{r['tau']:.4f}"
        else:
            fval05, fval, f05, ftau = r["macro_f1_val_at_0.5"], r["macro_f1_at_tau_val"], r["macro_f1_at_0.5"], r["macro_f1_at_tau"]
            tau = f"{r['tau_min']:.4f}-{r['tau_max']:.4f}"
        lines.append(
            f"| {r['dataset']} | {r['classifier']} | {r['C']} | {r['n_test']} | {tau} | {fval05:.4f} | {fval:.4f} | {f05:.4f} | {ftau:.4f} | "
            f"{r.get('f1_at_converted_0.5', r.get('macro_f1_at_converted_0.5')):.4f} | {r['conversion_exact']} | {100*r['delta_0.5']:.2f}% |"
        )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def _plots(rows: Sequence[Dict[str, Any]], out_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [f"{r['dataset']}\n{r['classifier']}" for r in rows]
    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(rows)), 4.2), dpi=160)
    for i, r in enumerate(rows):
        taus = [x["tau"] for x in r["class_rows"]] if r["kind"] == "multiclass" else [r["tau"]]
        ax.scatter([i] * len(taus), taus, s=12, color="#2b6cb0", alpha=0.75)
    ax.axhline(0.5, color="#c53030", lw=1.2, ls="--", label="0.5")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("F1-optimal threshold")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_tabular_tau.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(rows)), 4.2), dpi=160)
    width = 0.38
    xs = np.arange(len(rows))
    f05 = [r["f1_at_0.5"] if r["kind"] == "binary" else r["macro_f1_at_0.5"] for r in rows]
    ftau = [r["f1_at_tau"] if r["kind"] == "binary" else r["macro_f1_at_tau"] for r in rows]
    ax.bar(xs - width / 2, f05, width, label="0.5", color="#a0aec0")
    ax.bar(xs + width / 2, ftau, width, label="max-F1 tau", color="#2b6cb0")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("F1 (binary positive or macro one-vs-rest)")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_tabular_f1.png"))
    plt.close(fig)

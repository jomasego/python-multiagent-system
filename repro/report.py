from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from .diagnostics import to_jsonable


def _save(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    import matplotlib.pyplot as plt

    plt.close(fig)


def plot_model_delta(summaries: Dict[str, Dict[str, Any]], out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = []
    deltas = []
    cov = []
    for key, s in summaries.items():
        names.append(f"{s['name']}\n(C={s['C']}, n={s['n']})")
        deltas.append(100.0 * float(s["delta_0.5"]))
        cov.append(100.0 * float(s["coverage_0.5"]))
    order = np.argsort(deltas)
    fig, ax = plt.subplots(figsize=(8.4, 0.7 * len(names) + 1.6))
    y = np.arange(len(names))
    ax.barh(y, [deltas[i] for i in order], color="#c44e52")
    for yi, i in zip(y, order):
        ax.text(deltas[i] + 1.0, yi, f"{deltas[i]:.1f}%", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=8)
    ax.set_xlim(0, 105)
    ax.set_xlabel("P(max_c p_c < 0.5)  =  abstention rate of the 0.5 rule (%)")
    ax.set_title("Where the 0.5 rule stops being argmax")
    _save(fig, out_path)


def plot_temperature(summaries: Dict[str, Dict[str, Any]], out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    for key, s in summaries.items():
        rows = s.get("temperature_sweep")
        if not rows:
            continue
        t = [r["temperature"] for r in rows]
        axes[0].plot(t, [100 * r["delta_0.5"] for r in rows], marker="o", ms=3, label=s["name"])
        if "accuracy_argmax" in rows[0]:
            axes[1].plot(t, [100 * r["accuracy_argmax"] for r in rows], marker="o", ms=3, label=s["name"])
    axes[0].set_xscale("log")
    axes[0].set_xlabel("softmax temperature T")
    axes[0].set_ylabel("P(max p < 0.5) (%)")
    axes[0].set_title("the 0.5 rule is calibration-dependent")
    axes[0].axhline(50, color="gray", lw=0.8, ls=":")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("softmax temperature T")
    axes[1].set_ylabel("argmax accuracy (%)")
    axes[1].set_title("argmax is invariant to T > 0")
    axes[1].legend(frameon=False, fontsize=7)
    _save(fig, out_path)


def plot_roc(model: Dict[str, Any], out_path: str, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    probs = model["arrays"]["probs"]
    labels = model["arrays"]["labels"]
    if labels is None or len(labels) == 0:
        return
    c = probs.shape[1]
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    roc_rows = {row["class"]: row for row in model["summary"].get("per_class_roc", [])}
    shown = 0
    for k in range(min(c, 6)):
        y = (labels == k).astype(int)
        if y.min() == y.max():
            continue
        fpr, tpr, thr = roc_curve(y, probs[:, k], drop_intermediate=False)
        i05 = int(np.argmin(np.abs(thr - 0.5)))
        cname = model["summary"]["class_names"][k]
        auc = roc_rows.get(cname, {}).get("auc", float("nan"))
        ax.plot(fpr, tpr, lw=1.4, label=f"{cname} (AUC={auc:.3f})")
        ax.scatter([fpr[i05]], [tpr[i05]], s=28, zorder=5, edgecolor="black", facecolor="white")
        shown += 1
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(title or "one-vs-rest ROC; circles = the 0.5 operating point")
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    _save(fig, out_path)


def plot_risk_coverage(summaries: Dict[str, Dict[str, Any]], out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for key, s in summaries.items():
        arrays = s.get("arrays")
        if not arrays or "labels" not in arrays or arrays["labels"] is None or len(arrays["labels"]) == 0:
            continue
        probs = arrays["probs"]
        labels = arrays["labels"]
        m = probs.max(axis=1)
        correct = (probs.argmax(axis=1) == labels).astype(float)
        order = np.argsort(-m)
        cum = np.cumsum(correct[order])
        k = np.arange(1, len(correct) + 1)
        ax.plot(k / len(correct), 1 - cum / k, lw=1.3, label=f"{s['name']} (C={s['C']})")
        cov = float(s["coverage_0.5"])
        idx = max(0, int(cov * len(correct)) - 1)
        ax.scatter([cov], [1 - cum[idx] / (idx + 1)], s=22, zorder=5)
    ax.set_xlabel("coverage (fraction of instances kept)")
    ax.set_ylabel("risk (error on covered instances)")
    ax.set_title("Risk-coverage; dots mark the 0.5 operating point")
    ax.legend(frameon=False, fontsize=7)
    _save(fig, out_path)


def write_summary_md(results: Dict[str, Any], out_path: str) -> None:
    meta = results.get("meta", {})
    lines = ["# Threshold-conversion reproduction: raw summary", ""]
    lines.append(f"- generated: {meta.get('timestamp', '')}")
    lines.append(f"- upstream repo: {meta.get('upstream_repo', '')}")
    lines.append(f"- execution fork: {meta.get('fork_repo', '')}")
    lines.append(f"- branch: `{meta.get('branch', '')}`")
    lines.append(f"- commit: `{meta.get('commit', '')}`")
    lines.append(f"- device: {meta.get('device', '')}")
    lines.append("")
    lines.append("## Theory")
    lines.append("")
    theory = results.get("theory")
    if theory:
        lines.append(f"- whiteboard example: `{json.dumps(theory['whiteboard'], sort_keys=True)}`")
        lines.append(f"- binary conversion max decision mismatch: {max(r['decision_mismatch'] for r in theory['binary']['tau_rows'])}")
        lines.append(f"- binary conversion max AUC difference: {theory['binary']['max_auc_abs_diff']:.3g}")
        lines.append(f"- uniform-simplex closed form: `{json.dumps(theory['closed_form_uniform'], sort_keys=True)}`")
        lines.append("")
        lines.append("| C | delta_0.5 | coverage | ambiguous | adaptive decision match |")
        lines.append("|---:|---:|---:|---:|---:|")
        for c, row in theory["uniform_table"].items():
            lines.append(f"| {c} | {row['delta']:.4f} | {row['coverage']:.4f} | {row['ambiguous_rate']:.4f} | {row['adaptive_decision_match']:.4f} |")
        lines.append("")
        lines.append("Counterexamples to `argmax = max_c p_c >= 0.5`:")
        lines.append("")
        for ex in theory["counterexamples"]:
            lines.append(f"- p={ex['p']}, max={ex['max']:.2f}, argmax={ex['argmax']}, classes>=0.5: {ex['passes_0.5']}, adaptive q={[round(v, 3) for v in ex['adaptive_conversion']]}")
    if results.get("models"):
        lines.append("")
        lines.append("## Models")
        lines.append("")
        lines.append("| model | C | n | argmax acc | delta_0.5 | coverage | selective acc | ECE | exact adaptive |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for key, entry in results["models"].items():
            s = entry["summary"]
            if "delta_0.5" not in s:
                continue
            lines.append(
                "| {name} | {C} | {n} | {acc} | {d} | {cov} | {sel} | {ece} | {ad} |".format(
                    name=s["name"],
                    C=s["C"],
                    n=s["n"],
                    acc=f"{100*s['accuracy_argmax']:.2f}%" if "accuracy_argmax" in s else "n/a",
                    d=f"{100*s['delta_0.5']:.2f}%",
                    cov=f"{100*s['coverage_0.5']:.2f}%",
                    sel=f"{100*s['selective_accuracy']:.2f}%" if s.get("selective_accuracy") is not None else "n/a",
                    ece=f"{s['ece_15']:.4f}" if "ece_15" in s else "n/a",
                    ad=s["exact_adaptive"],
                )
            )
        lines.append("")
        lines.append("### Temperature sweep")
        lines.append("")
        lines.append("| model | T | delta_0.5 | argmax acc | ECE |")
        lines.append("|---|---:|---:|---:|---:|")
        for key, entry in results["models"].items():
            for row in entry["summary"].get("temperature_sweep", []):
                lines.append(
                    f"| {entry['summary']['name']} | {row['temperature']} | {100*row['delta_0.5']:.2f}% | "
                    f"{100*row['accuracy_argmax']:.2f}% | {row.get('ece_15', float('nan')):.4f} |"
                )
        lines.append("")
        lines.append("### Per-class ROC operating points (0.5 vs best F1)")
        lines.append("")
        lines.append("| model | class | AUC | TPR@0.5 | FPR@0.5 | best F1 thr | best F1 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for key, entry in results["models"].items():
            for row in entry["summary"].get("per_class_roc", []):
                lines.append(
                    f"| {entry['summary']['name']} | {row['class']} | {row['auc']:.3f} | {row['tpr_at_0.5']:.3f} | "
                    f"{row['fpr_at_0.5']:.3f} | {row['best_f1_threshold']:.3f} | {row['best_f1']:.3f} |"
                )
    if results.get("conversion_fit"):
        lines.append("")
        lines.append("## Fixed CxC linear map followed by argmax (not a 0.5 threshold)")
        lines.append("")
        lines.append("| model | train acc | test acc | chance |")
        lines.append("|---|---:|---:|---:|")
        for key, fit in results["conversion_fit"].items():
            lines.append(f"| {key} | {fit.get('train_accuracy')} | {fit.get('test_accuracy')} | {fit.get('chance')} |")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def write_outputs(results: Dict[str, Any], out_dir: str, include_arrays: bool = False) -> None:
    os.makedirs(out_dir, exist_ok=True)
    summaries = {}
    for key, entry in results.get("models", {}).items():
        if "delta_0.5" in entry["summary"]:
            summaries[key] = entry["summary"]
    if summaries:
        plot_model_delta(summaries, os.path.join(out_dir, "fig_model_delta.png"))
        plot_temperature(summaries, os.path.join(out_dir, "fig_temperature.png"))
        plot_risk_coverage(summaries, os.path.join(out_dir, "fig_risk_coverage.png"))
        roc_model = None
        preferred = ["sentiment3", "emotion6", "binary_sst2", "digits10"]
        ordered = [results["models"][k] for k in preferred if k in results["models"]]
        ordered += [v for k, v in results["models"].items() if k not in preferred]
        for entry in ordered:
            if entry["summary"].get("per_class_roc") and "accuracy_argmax" in entry["summary"]:
                roc_model = entry
                break
        if roc_model is not None:
            plot_roc(roc_model, os.path.join(out_dir, "fig_roc.png"), f"one-vs-rest ROC: {roc_model['summary']['name']}")
    clean = {
        "meta": results.get("meta", {}),
        "theory": results.get("theory"),
        "models": {k: {"summary": v["summary"]} for k, v in results.get("models", {}).items()},
        "conversion_fit": results.get("conversion_fit", {}),
    }
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(clean), fh, indent=2, sort_keys=True)
    write_summary_md(clean, os.path.join(out_dir, "summary.md"))

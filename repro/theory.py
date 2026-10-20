from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import betainc

from .diagnostics import to_jsonable


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def sample_simplex(rng: np.random.Generator, n: int, c: int, alpha: float = 1.0) -> np.ndarray:
    return rng.dirichlet(np.full(c, alpha, dtype=np.float64), size=n)


def uniform_delta(c: int) -> float:
    return 1.0 - c / (2.0 ** (c - 1))


def dirichlet_delta(c: int, alpha: float) -> float:
    return 1.0 - c + c * float(betainc(alpha, (c - 1) * alpha, 0.5))


def binary_conversion(c: int = 2) -> Dict[str, object]:
    p1 = np.linspace(1e-6, 1.0 - 1e-6, 20001)
    p2 = 1.0 - p1
    taus = [0.1, 0.25, 0.5, 0.75, 0.9]
    rows = []
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=p1.size)
    auc_raw = float(roc_auc_score(y, p1))
    auc_diffs = []
    for tau in taus:
        w = 1.0 / (2.0 * (1.0 - tau))
        b = (1.0 - 2.0 * tau) / (2.0 * (1.0 - tau))
        s = w * p1 + b
        decision = (s >= 0.5).astype(int)
        direct = (p1 >= tau).astype(int)
        auc_diffs.append(abs(auc_raw - float(roc_auc_score(y, s))))
        rows.append(
            {
                "tau": tau,
                "w1": w,
                "w2": 0.0,
                "b": b,
                "decision_mismatch": int(np.sum(decision != direct)),
                "max_abs_score_gap": float(np.max(np.abs(s - (p1 - tau) / (2.0 * (1.0 - tau)) - 0.5))),
                "score_at_tau": float(w * tau + b),
                "auc_abs_diff": auc_diffs[-1],
            }
        )
    return {
        "tau_rows": rows,
        "auc_raw": auc_raw,
        "max_auc_abs_diff": float(max(auc_diffs)),
        "p1": p1,
        "p2": p2,
    }


def simplex_truth_table(rng: np.random.Generator, c: int, n: int, alpha: float = 1.0) -> Dict[str, object]:
    p = sample_simplex(rng, n, c, alpha)
    m = p.max(axis=1)
    arg = p.argmax(axis=1)
    selected = p >= 0.5
    k = selected.sum(axis=1)
    delta = float(np.mean(m < 0.5))
    coverage = float(np.mean(m >= 0.5))
    ambiguous = float(np.mean(k >= 2))
    wrong_when_selected = 0
    if np.any(k == 1):
        idx = np.flatnonzero(k == 1)
        wrong_when_selected = int(np.sum(selected[idx].argmax(axis=1) != arg[idx]))
    q = p / (2.0 * m[:, None])
    q_top = q[np.arange(n), arg]
    winner_gap = float(np.max(np.abs(q_top - 0.5)))
    loser = q.copy()
    loser[np.arange(n), arg] = -1.0
    loser_max = float(np.max(loser))
    adaptive_ok = float(np.mean(((q >= 0.5).sum(axis=1) == 1) & ((q >= 0.5).argmax(axis=1) == arg)))
    return {
        "C": c,
        "alpha": alpha,
        "n": n,
        "delta": delta,
        "coverage": coverage,
        "ambiguous_rate": ambiguous,
        "wrong_when_selected": wrong_when_selected,
        "adaptive_winner_gap": winner_gap,
        "adaptive_loser_max": loser_max,
        "adaptive_decision_match": adaptive_ok,
        "m": m,
        "arg": arg,
        "p": p,
    }


def scan_fixed_elementwise(rng: np.random.Generator, c: int = 3, n: int = 20000) -> Dict[str, object]:
    p = sample_simplex(rng, n, c, 1.0)
    arg = p.argmax(axis=1)
    grid = np.linspace(0.2, 1.8, 33)
    best = {"agreement": -1.0}
    for a0 in grid:
        for a1 in grid:
            a = np.ones(c)
            a[0] = a0
            a[1] = a1
            q = p * a
            sel = q >= 0.5
            k = sel.sum(axis=1)
            pred = np.where(k == 1, sel.argmax(axis=1), np.where(k == 0, -1, q.argmax(axis=1)))
            agreement = float(np.mean(pred == arg))
            if agreement > best["agreement"]:
                best = {"agreement": agreement, "a": a.tolist(), "abstain_rate": float(np.mean(k == 0)), "wrong_selected": float(np.mean((k == 1) & (pred != arg)))}
    tau = 0.5 / np.asarray(best["a"])
    return {"best": best, "tau": tau.tolist(), "identity_delta": float(np.mean(p.max(axis=1) < 0.5))}


def whiteboard_example() -> Dict[str, object]:
    p = np.array([0.07, 0.93])
    w = np.array([0.6, 0.4])
    dot = float(w @ p)
    reweighted = w * p
    norm = reweighted / reweighted.sum()
    return {
        "p": p.tolist(),
        "w": w.tolist(),
        "dot": dot,
        "dot_threshold_pred": int(dot >= 0.5),
        "argmax": int(p.argmax()),
        "hadamard_argmax": int(reweighted.argmax()),
        "renormalized": norm.tolist(),
        "renormalized_argmax": int(norm.argmax()),
    }


def fixed_conversion_fit(rng: np.random.Generator, c: int, n: int = 30000) -> Dict[str, object]:
    import warnings

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    p = sample_simplex(rng, n, c, 1.0)
    y = p.argmax(axis=1)
    x_tr, x_te, y_tr, y_te = train_test_split(p, y, test_size=0.3, random_state=0, stratify=y)
    clf = LogisticRegression(max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(x_tr, y_tr)
    tr = float(clf.score(x_tr, y_tr))
    te = float(clf.score(x_te, y_te))
    return {"C": c, "train_accuracy": tr, "test_accuracy": te, "n": n}


def temperature_theory(rng: np.random.Generator, c: int = 6, n: int = 2000) -> Dict[str, object]:
    logits = rng.normal(size=(n, c)) * 2.0
    temps = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    base = softmax(logits).argmax(axis=1)
    rows = []
    for t in temps:
        p = softmax(logits / t)
        rows.append(
            {
                "temperature": t,
                "delta": float(np.mean(p.max(axis=1) < 0.5)),
                "mean_max": float(p.max(axis=1).mean()),
                "argmax_agreement": float(np.mean(p.argmax(axis=1) == base)),
            }
        )
    return {"C": c, "n": n, "rows": rows}


def plot_simplex_c3(p: np.ndarray, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = p[:, 1] + 0.5 * p[:, 2]
    y = (math.sqrt(3.0) / 2.0) * p[:, 2]
    m = p.max(axis=1)
    arg = p.argmax(axis=1)
    colors = np.array(["#1f77b4", "#d62728", "#2ca02c"])
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(x, y, c=colors[arg], s=3, alpha=0.55, linewidths=0)
    tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, math.sqrt(3.0) / 2.0], [0.0, 0.0]])
    ax.plot(tri[:, 0], tri[:, 1], color="black", lw=1.2)
    low = m < 0.5
    ax.scatter(x[low], y[low], facecolors="none", edgecolors="black", s=6, linewidths=0.3, alpha=0.35)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("C=3 simplex: argmax regions (colors)\nand the 0.5-threshold caps (outlined)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_delta_vs_c(out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cs = np.arange(2, 13)
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for alpha, color in [(0.3, "#9467bd"), (1.0, "#1f77b4"), (3.0, "#ff7f0e")]:
        ax.plot(cs, [dirichlet_delta(int(c), alpha) for c in cs], color=color, lw=1.8, label=f"Dirichlet alpha={alpha}")
        mc = [float(np.mean(sample_simplex(rng, 20000, int(c), alpha).max(axis=1) < 0.5)) for c in cs]
        ax.plot(cs, mc, "o", color=color, ms=4)
    ax.plot(cs, [uniform_delta(int(c)) for c in cs], "k--", lw=1.0, label="uniform (alpha=1) closed form")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("number of classes C")
    ax.set_ylabel("P(max_c p_c < 0.5)")
    ax.set_title("The 0.5 rule abstains on 1 - C/2^(C-1) of a uniform simplex")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def plot_binary_conversion(binary: Dict[str, object], out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p1 = np.asarray(binary["p1"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for row in binary["tau_rows"]:
        tau = float(row["tau"])
        s = float(row["w1"]) * p1 + float(row["b"])
        ax.plot(p1, s, lw=1.5, label=f"tau={tau}")
        ax.axvline(tau, color="gray", lw=0.5, ls=":")
    ax.axhline(0.5, color="black", lw=1.0)
    ax.set_xlabel("p(class 1)")
    ax.set_ylabel("converted score s_tau(p)")
    ax.set_title("Binary case: every threshold tau is the 0.5 level set of an affine conversion")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def run_theory(out_dir: str, seed: int = 0) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    binary = binary_conversion()
    tables = {}
    for c in [2, 3, 4, 5, 6, 10, 20]:
        tables[str(c)] = {k: v for k, v in simplex_truth_table(rng, c, 40000, 1.0).items() if k not in {"p", "m", "arg"}}
    dirichlet_rows = []
    for c in [2, 3, 4, 5, 6, 10]:
        row = {"C": c}
        for alpha in [0.3, 1.0, 3.0]:
            mc = float(np.mean(sample_simplex(rng, 40000, c, alpha).max(axis=1) < 0.5))
            row[f"mc_alpha_{alpha}"] = mc
            row[f"formula_alpha_{alpha}"] = dirichlet_delta(c, alpha)
        dirichlet_rows.append(row)
    fixed = scan_fixed_elementwise(rng, c=3, n=20000)
    fits = [fixed_conversion_fit(rng, c, n=20000) for c in [3, 4, 6, 10]]
    temp = temperature_theory(rng)
    import os

    os.makedirs(out_dir, exist_ok=True)
    p3 = sample_simplex(rng, 30000, 3, 1.0)
    plot_simplex_c3(p3, os.path.join(out_dir, "fig_simplex_c3.png"))
    plot_delta_vs_c(os.path.join(out_dir, "fig_delta_vs_c.png"))
    plot_binary_conversion(binary, os.path.join(out_dir, "fig_binary_conversion.png"))
    examples = []
    for vec in ([0.4, 0.35, 0.25], [0.34, 0.33, 0.33], [0.3, 0.3, 0.2, 0.1, 0.1], [0.25, 0.2, 0.2, 0.2, 0.15]):
        p = np.asarray(vec)
        examples.append(
            {
                "p": p.tolist(),
                "C": len(vec),
                "max": float(p.max()),
                "argmax": int(p.argmax()),
                "passes_0.5": int(np.sum(p >= 0.5)),
                "adaptive_conversion": (p / (2.0 * p.max())).tolist(),
            }
        )
    return to_jsonable(
        {
            "binary": binary,
            "uniform_table": tables,
            "dirichlet_formula": dirichlet_rows,
            "fixed_elementwise_search": fixed,
            "fixed_affine_fit": fits,
            "temperature": temp,
            "whiteboard": whiteboard_example(),
            "counterexamples": examples,
            "closed_form_uniform": {str(c): uniform_delta(c) for c in [2, 3, 4, 5, 6, 10, 20]},
        }
    )

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np

from .diagnostics import linear_conversion_fit, to_jsonable
from .models import run_digits, run_text, run_vision_track
from .report import write_outputs
from .theory import run_theory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _load_config() -> Dict[str, Any]:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _versions() -> Dict[str, str]:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch

        out["torch"] = torch.__version__
        out["mps"] = str(torch.backends.mps.is_available())
    except Exception:
        out["torch"] = "missing"
    try:
        import transformers

        out["transformers"] = transformers.__version__
    except Exception:
        out["transformers"] = "missing"
    try:
        import datasets

        out["datasets"] = datasets.__version__
    except Exception:
        out["datasets"] = "missing"
    try:
        import sklearn

        out["sklearn"] = sklearn.__version__
    except Exception:
        out["sklearn"] = "missing"
    return out


def _print_model(s: Dict[str, Any]) -> None:
    print(
        "[model] {name} | C={C} n={n} | argmax_acc={acc} | delta_0.5={d} | coverage={cov} | selective_acc={sel} | ECE={ece} | adaptive_exact={ae}".format(
            name=s["name"],
            C=s["C"],
            n=s["n"],
            acc=f"{100*s['accuracy_argmax']:.2f}%" if "accuracy_argmax" in s else "n/a",
            d=f"{100*s['delta_0.5']:.2f}%",
            cov=f"{100*s['coverage_0.5']:.2f}%",
            sel=f"{100*s['selective_accuracy']:.2f}%" if s.get("selective_accuracy") is not None else "n/a",
            ece=f"{s['ece_15']:.4f}" if "ece_15" in s else "n/a",
            ae=s["exact_adaptive"],
        ),
        flush=True,
    )


def main() -> int:
    os.chdir(ROOT)
    np.random.seed(0)
    cfg = _load_config()
    tracks = list(cfg.get("tracks", ["theory", "digits"]))
    seed = int(cfg.get("seed", 0))
    out_dir = os.path.join(ROOT, cfg.get("out_dir", "repro/out"))
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cwd": ROOT,
        "branch": _git("branch", "--show-current"),
        "commit": _git("rev-parse", "HEAD"),
        "upstream_repo": "https://github.com/isllabres/python-multiagent-system",
        "fork_repo": "https://github.com/jomasego/python-multiagent-system",
        "config": cfg,
        "versions": _versions(),
        "device": "auto (text=cpu, vision=mps if available, digits=cpu)",
        "job_url": os.environ.get("REPRO_JOB_URL", ""),
        "bucket_url": os.environ.get("REPRO_BUCKET_URL", ""),
    }
    print("=== ENV ===", flush=True)
    print(json.dumps(meta, indent=2, sort_keys=True), flush=True)
    results: Dict[str, Any] = {"meta": meta, "models": {}, "errors": []}
    if "theory" in tracks:
        print("=== TRACK: theory ===", flush=True)
        results["theory"] = run_theory(out_dir, seed)
        t = results["theory"]
        print(
            "[theory] uniform-simplex delta: "
            + ", ".join(f"C={c}:{100*row['delta']:.1f}%" for c, row in t["uniform_table"].items()),
            flush=True,
        )
        print(f"[theory] adaptive conversion exact: {all(row['adaptive_decision_match'] >= 1.0 - 1e-12 for row in t['uniform_table'].values())}", flush=True)
        print(f"[theory] whiteboard: {json.dumps(t['whiteboard'], sort_keys=True)}", flush=True)
    if "digits" in tracks:
        print("=== TRACK: digits ===", flush=True)
        try:
            summary, arrays = run_digits()
            results["models"]["digits10"] = {"summary": summary, "arrays": arrays}
            _print_model(summary)
        except Exception:
            traceback.print_exc()
            results["errors"].append("digits")
    text_cache: Dict[str, Any] = {}
    if "text" in tracks or "conversion_fit" in tracks:
        print("=== TRACK: text ===", flush=True)
        try:
            text_cache = run_text(cfg.get("text_specs"))
            for key, entry in text_cache.items():
                if "error" in entry["summary"]:
                    print(f"[text] {key} failed: {entry['summary']['error']}", flush=True)
                    results["errors"].append(f"text:{key}")
                    continue
                results["models"][key] = entry
                _print_model(entry["summary"])
        except Exception:
            traceback.print_exc()
            results["errors"].append("text")
    if "conversion_fit" in tracks:
        print("=== TRACK: conversion_fit ===", flush=True)
        results["conversion_fit"] = {}
        for key, entry in text_cache.items():
            probs = entry["arrays"]["probs"]
            fit = linear_conversion_fit(probs, key)
            results["conversion_fit"][key] = fit
            print(f"[fit] {key}: train={fit.get('train_accuracy')} test={fit.get('test_accuracy')}", flush=True)
    if "vision" in tracks:
        print("=== TRACK: vision ===", flush=True)
        try:
            vis = run_vision_track()
            for key, entry in vis.items():
                if "error" in entry["summary"]:
                    print(f"[vision] {key} failed: {entry['summary']['error']}", flush=True)
                    results["errors"].append(f"vision:{key}")
                    continue
                results["models"][key] = entry
                _print_model(entry["summary"])
        except Exception:
            traceback.print_exc()
            results["errors"].append("vision")
    write_outputs(results, out_dir)
    clean = {
        "meta": meta,
        "theory": results.get("theory"),
        "models": {k: {"summary": v["summary"]} for k, v in results.get("models", {}).items()},
        "conversion_fit": results.get("conversion_fit", {}),
        "errors": results["errors"],
    }
    with open(os.path.join(out_dir, "results_full.json"), "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(clean), fh, indent=2, sort_keys=True)
    print("=== SUMMARY ===", flush=True)
    for key, entry in results.get("models", {}).items():
        _print_model(entry["summary"])
    if results.get("conversion_fit"):
        print("=== FIXED-AFFINE FIT ===", flush=True)
        for key, fit in results["conversion_fit"].items():
            print(f"{key}: train={fit.get('train_accuracy')} test={fit.get('test_accuracy')} chance={fit.get('chance')}", flush=True)
    print("=== ARTIFACTS ===", flush=True)
    print(out_dir, flush=True)
    for name in sorted(os.listdir(out_dir)):
        print(f"  {name}", flush=True)
    print("=== LINKS ===", flush=True)
    print("upstream: https://github.com/isllabres/python-multiagent-system", flush=True)
    print("fork:     https://github.com/jomasego/python-multiagent-system", flush=True)
    print("models:   https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest", flush=True)
    print("          https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion", flush=True)
    print("          https://huggingface.co/google/vit-base-patch16-224", flush=True)
    print("          https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english", flush=True)
    print("datasets: https://huggingface.co/datasets/cardiffnlp/tweet_eval", flush=True)
    print("          https://huggingface.co/datasets/dair-ai/emotion", flush=True)
    print("          https://huggingface.co/datasets/uoft-cs/cifar10", flush=True)
    print("          https://huggingface.co/datasets/stanfordnlp/sst2", flush=True)
    print("=== DONE ===", flush=True)
    return 0 if results.get("theory") or results.get("models") else 1


if __name__ == "__main__":
    sys.exit(main())

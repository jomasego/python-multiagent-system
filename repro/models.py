from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .diagnostics import summarize_probs, to_jsonable

TEXT_SPECS = [
    {
        "key": "binary_sst2",
        "name": "binary control: DistilBERT SST-2",
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "dataset": "stanfordnlp/sst2",
        "dataset_fallback": "imdb",
        "config": None,
        "split": "validation",
        "fallback_split": "test",
        "n": 1000,
        "links": {
            "model": "https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english",
            "dataset": "https://huggingface.co/datasets/stanfordnlp/sst2",
            "dataset_fallback": "https://huggingface.co/datasets/stanfordnlp/imdb",
        },
        "label_order": ["negative", "positive"],
    },
    {
        "key": "sentiment3",
        "name": "3-class sentiment: Twitter-RoBERTa",
        "model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "dataset": "cardiffnlp/tweet_eval",
        "dataset_fallback": "tweet_eval",
        "config": "sentiment",
        "split": "test",
        "fallback_split": "test",
        "n": 2000,
        "links": {
            "model": "https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest",
            "dataset": "https://huggingface.co/datasets/cardiffnlp/tweet_eval",
        },
        "label_order": ["negative", "neutral", "positive"],
    },
    {
        "key": "emotion6",
        "name": "6-class emotion: DistilBERT emotion",
        "model": "bhadresh-savani/distilbert-base-uncased-emotion",
        "dataset": "dair-ai/emotion",
        "dataset_fallback": "emotion",
        "config": None,
        "split": "test",
        "fallback_split": "test",
        "n": 2000,
        "links": {
            "model": "https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion",
            "dataset": "https://huggingface.co/datasets/dair-ai/emotion",
        },
        "label_order": ["sadness", "joy", "love", "anger", "fear", "surprise"],
    },
]

VISION_SPEC = {
    "key": "vit1000",
    "name": "1000-class vision: ViT-B/16 (ImageNet head on CIFAR-10)",
    "model": "google/vit-base-patch16-224",
    "dataset": "uoft-cs/cifar10",
    "config": None,
    "split": "test",
    "n": 1000,
    "links": {
        "model": "https://huggingface.co/google/vit-base-patch16-224",
        "dataset": "https://huggingface.co/datasets/uoft-cs/cifar10",
    },
}


def _device() -> str:
    import torch

    if torch.backends.mps.is_available() and os.environ.get("REPRO_DEVICE", "auto") in {"auto", "mps"}:
        return "mps"
    return "cpu"


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def _load_dataset(spec: Dict[str, Any]) -> Tuple[Any, str, Optional[List[str]]]:
    from datasets import load_dataset

    errors = []
    candidates = [
        (spec["dataset"], spec.get("config"), spec.get("split", "test")),
        (spec.get("dataset_fallback", spec["dataset"]), spec.get("config"), spec.get("fallback_split", spec.get("split", "test"))),
    ]
    for ds_name, cfg, split in candidates:
        if ds_name is None:
            continue
        try:
            n = int(spec.get("n", 1000))
            split_arg = f"{split}[:{n}]"
            ds = load_dataset(ds_name, cfg, split=split_arg) if cfg else load_dataset(ds_name, split=split_arg)
            names = None
            if "label" in ds.features and hasattr(ds.features["label"], "names"):
                names = list(ds.features["label"].names)
            return ds, ds_name, names
        except Exception as exc:
            errors.append(f"{ds_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("could not load dataset; " + " | ".join(errors))


def _align_labels(ds: Any, id2label: Dict[int, str], manual_order: Sequence[str]) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    raw = np.asarray(ds["label"], dtype=int)
    names = None
    if "label" in ds.features and hasattr(ds.features["label"], "names"):
        names = list(ds.features["label"].names)
    model_names = [id2label[i] for i in sorted(id2label)]
    if names is not None:
        ds_by_norm = {_norm(n): i for i, n in enumerate(names)}
        ds_to_model = {}
        for mi, mn in id2label.items():
            if _norm(mn) in ds_by_norm:
                ds_to_model[ds_by_norm[_norm(mn)]] = int(mi)
        if len(ds_to_model) == len(model_names):
            aligned = np.array([ds_to_model[int(v)] for v in raw], dtype=int)
            return aligned, names
    if manual_order is not None and len(manual_order) == len(model_names):
        if names is not None:
            ds_by_norm = {_norm(n): i for i, n in enumerate(names)}
            order = [_norm(n) for n in manual_order]
            if all(n in ds_by_norm for n in order):
                ds_to_model = {ds_by_norm[order[mi]]: mi for mi in range(len(order))}
                aligned = np.array([ds_to_model[int(v)] for v in raw], dtype=int)
                return aligned, names
        else:
            return raw, [str(i) for i in range(len(model_names))]
    if names is not None and len(names) == len(model_names):
        return raw, names
    return None, names


def run_text_spec(spec: Dict[str, Any], batch_size: int = 32) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(0)
    np.random.seed(0)
    ds, ds_name, _ = _load_dataset(spec)
    text_col = None
    for candidate in ("text", "sentence", "premise", "document"):
        if candidate in ds.column_names:
            text_col = candidate
            break
    if text_col is None:
        for column in ds.column_names:
            if isinstance(ds[0][column], str):
                text_col = column
                break
    if text_col is None:
        raise RuntimeError(f"no text column in {ds.column_names}")
    texts = list(ds[text_col])
    tokenizer = AutoTokenizer.from_pretrained(spec["model"])
    model = AutoModelForSequenceClassification.from_pretrained(spec["model"])
    model.eval()
    device = "cpu"
    model.to(device)
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    display_order = spec.get("label_order", [])
    labels, ds_names = _align_labels(ds, id2label, display_order)
    logits_chunks = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc).logits.detach().cpu()
            logits_chunks.append(out)
    logits = torch.cat(logits_chunks, dim=0).numpy()
    z = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(z)
    probs = probs / probs.sum(axis=1, keepdims=True)
    class_names = [id2label[i] for i in sorted(id2label)]
    if display_order and all(str(n).startswith("LABEL_") for n in class_names) and len(display_order) == len(class_names):
        class_names = list(display_order)
    links = dict(spec["links"])
    if ds_name != spec["dataset"]:
        links["dataset_used"] = f"https://huggingface.co/datasets/{ds_name}"
    summary = summarize_probs(
        probs,
        name=spec["name"],
        links=links,
        labels=labels,
        logits=logits,
        class_names=class_names,
    )
    summary["dataset_used"] = ds_name
    summary["label_names_dataset"] = ds_names
    summary["label_names_model"] = class_names
    summary["label_alignment"] = None if labels is None else "aligned by name or fallback order"
    summary["device"] = device
    return to_jsonable(summary), {"probs": probs, "logits": logits, "labels": labels if labels is not None else np.array([], dtype=int)}


def run_vision(batch_size: int = 32) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    import torch
    from datasets import load_dataset
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    torch.manual_seed(0)
    np.random.seed(0)
    spec = VISION_SPEC
    ds = load_dataset(spec["dataset"], split=f"{spec['split']}[:{spec['n']}]")
    processor = AutoImageProcessor.from_pretrained(spec["model"])
    model = AutoModelForImageClassification.from_pretrained(spec["model"])
    model.eval()
    device = _device()
    model.to(device)
    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    images = [img.convert("RGB") for img in ds["img"]]
    logits_chunks = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            enc = processor(images=batch, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc).logits.detach().cpu()
            logits_chunks.append(out)
    if device == "mps":
        torch.mps.synchronize()
    logits = torch.cat(logits_chunks, dim=0).numpy()
    z = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(z)
    probs = probs / probs.sum(axis=1, keepdims=True)
    class_names = [id2label[i] for i in sorted(id2label)]
    summary = summarize_probs(
        probs,
        name=spec["name"],
        links=spec["links"],
        labels=None,
        logits=logits,
        class_names=class_names,
    )
    summary["dataset_used"] = spec["dataset"]
    summary["label_alignment"] = "none: the 1000-way ImageNet head is evaluated out-of-domain on CIFAR-10 images"
    summary["device"] = device
    return to_jsonable(summary), {"probs": probs, "logits": logits, "labels": np.array([], dtype=int)}


def run_digits() -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    from sklearn.datasets import load_digits
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    digits = load_digits()
    x = digits.data
    y = digits.target.astype(int)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.3, random_state=0, stratify=y)
    scaler = StandardScaler().fit(x_tr)
    clf = LogisticRegression(max_iter=5000, random_state=0)
    clf.fit(scaler.transform(x_tr), y_tr)
    probs = clf.predict_proba(scaler.transform(x_te))
    summary = summarize_probs(
        probs,
        name="10-class tabular: logistic regression on sklearn digits",
        links={
            "dataset": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
            "code": "https://github.com/jomasego/python-multiagent-system",
        },
        labels=y_te,
        logits=None,
        class_names=[str(i) for i in range(10)],
    )
    summary["dataset_used"] = "sklearn.datasets.load_digits"
    summary["device"] = "cpu"
    return to_jsonable(summary), {"probs": probs, "labels": y_te}


def run_text(specs: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    import traceback

    out = {}
    for spec in TEXT_SPECS:
        if specs is not None and spec["key"] not in specs:
            continue
        try:
            summary, arrays = run_text_spec(spec)
            out[spec["key"]] = {"summary": summary, "arrays": arrays}
        except Exception:
            traceback.print_exc()
            out[spec["key"]] = {
                "summary": {
                    "name": spec["name"],
                    "links": spec["links"],
                    "C": None,
                    "n": 0,
                    "error": traceback.format_exc().splitlines()[-1],
                },
                "arrays": {"probs": np.zeros((0, 2)), "labels": np.array([], dtype=int)},
            }
    return out


def run_vision_track() -> Dict[str, Any]:
    summary, arrays = run_vision()
    return {VISION_SPEC["key"]: {"summary": summary, "arrays": arrays}}

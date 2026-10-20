#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export REPRO_DEVICE="${REPRO_DEVICE:-auto}"
python3 - <<'PY' || {
import importlib.util as u, sys
required = ["numpy", "scipy", "sklearn", "matplotlib"]
sys.exit(1 if any(u.find_spec(m) is None for m in required) else 0)
PY
  python3 -m pip install --quiet --disable-pip-version-check --upgrade pip
  python3 -m pip install --quiet --disable-pip-version-check numpy scipy scikit-learn matplotlib
}
if ! python3 -c "import torch" >/dev/null 2>&1; then
  python3 -m pip install --quiet --disable-pip-version-check torch --index-url https://download.pytorch.org/whl/cpu
fi
if ! python3 -c "import transformers, datasets" >/dev/null 2>&1; then
  python3 -m pip install --quiet --disable-pip-version-check transformers datasets
fi
python3 repro/run.py

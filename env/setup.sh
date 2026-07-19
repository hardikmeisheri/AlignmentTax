#!/usr/bin/env bash
# Phase-0 environment setup. Idempotent: safe to re-run.
#
# Creates a single virtualenv. If you run this across several machines, put the
# repo on shared storage so they import the same .venv, or run this once on
# each machine.
#
# We deliberately install into a plain venv rather than building a container so
# there is one less moving part to debug. If you prefer a container, the same
# pip line goes into a Dockerfile FROM nvcr.io/nvidia/pytorch:25.xx-py3.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Default to `python` (often a fuller conda/toolchain interpreter that HAS the
# venv/ensurepip module, unlike a bare system `python3`). Override with
# ATAX_PYTHON=python3.10 etc. if you need a specific one.
PY="${ATAX_PYTHON:-python}"
VENV="${ATAX_VENV:-$REPO_ROOT/.venv}"
# Which dependency set to install. Default = the pinned, reproducible stack.
# Set ATAX_REQUIREMENTS=requirements-next.txt (together with
# ATAX_VENV=$PWD/.venv-next) to build the SEPARATE next-gen env
# (transformers 5.x / vLLM 0.23) used only by the NEXTGEN ladders -- this keeps
# the pinned .venv that reproduces the committed deck numbers untouched.
REQ="${ATAX_REQUIREMENTS:-requirements.txt}"

echo "[setup] repo   = $REPO_ROOT"
echo "[setup] python = $($PY --version 2>&1)"
echo "[setup] venv   = $VENV"
echo "[setup] reqs   = $REQ"

# ---------------------------------------------------------------------------
# Create the virtualenv, self-healing on machine images whose python can't
# bootstrap pip ("ensurepip is not available"). On those, `python -m venv`
# "succeeds" but leaves a BROKEN venv with no bin/activate. We detect that and,
# WITHOUT sudo: recreate the venv --without-pip, then bootstrap pip via ensurepip
# and, failing that, get-pip.py.
# ---------------------------------------------------------------------------
venv_ok() { [[ -f "$VENV/bin/activate" && -x "$VENV/bin/python" ]]; }

if venv_ok && "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "[setup] reusing existing working venv"
else
  # Remove any half-created venv from a previous failed attempt.
  [[ -e "$VENV" ]] && rm -rf "$VENV"

  echo "[setup] creating venv: $PY -m venv"
  if "$PY" -m venv "$VENV" 2>/tmp/atax_venv_err.log \
       && "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
    echo "[setup] venv created (pip bootstrapped normally)"
  else
    echo "[setup] standard venv has no pip (ensurepip missing); retrying --without-pip"
    sed 's/^/[setup]   /' /tmp/atax_venv_err.log 2>/dev/null || true
    rm -rf "$VENV"
    if ! "$PY" -m venv --without-pip "$VENV"; then
      echo "[setup] ERROR: '$PY -m venv' is unavailable on this machine." >&2
      echo "[setup]   Fix: set ATAX_PYTHON to a python that has venv (e.g. a conda" >&2
      echo "[setup]   interpreter), or 'sudo apt install python3.10-venv', then re-run." >&2
      exit 1
    fi
    if "$VENV/bin/python" -m ensurepip --upgrade >/dev/null 2>&1; then
      echo "[setup] pip bootstrapped via ensurepip"
    else
      echo "[setup] ensurepip unavailable; fetching get-pip.py"
      GETPIP="/tmp/atax_get-pip.py"
      if command -v curl >/dev/null 2>&1; then
        curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$GETPIP"
      elif command -v wget >/dev/null 2>&1; then
        wget -qO "$GETPIP" https://bootstrap.pypa.io/get-pip.py
      else
        echo "[setup] ERROR: need curl or wget to bootstrap pip" >&2
        exit 1
      fi
      "$VENV/bin/python" "$GETPIP"
    fi
    echo "[setup] venv created (pip bootstrapped manually)"
  fi
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

if ! python -m pip --version >/dev/null 2>&1; then
  echo "[setup] ERROR: pip still unavailable in the venv after bootstrap" >&2
  exit 1
fi

python -m pip install --upgrade pip wheel setuptools

# Speeds up large model downloads dramatically.
export HF_HUB_ENABLE_HF_TRANSFER=1

# --- CUDA note -------------------------------------------------------------
# torch/vLLM wheels must match your machine's CUDA. The pinned versions in
# requirements.txt target CUDA 12.4 wheels (the default index). If your machine
# uses a different CUDA, set ATAX_TORCH_INDEX, e.g.:
#   export ATAX_TORCH_INDEX="https://download.pytorch.org/whl/cu121"
# ---------------------------------------------------------------------------
if [[ -n "${ATAX_TORCH_INDEX:-}" ]]; then
  echo "[setup] installing torch from $ATAX_TORCH_INDEX"
  pip install "torch==2.6.0" --index-url "$ATAX_TORCH_INDEX"
fi

pip install -r "$REQ"

# Make the `atax` package importable without packaging ceremony.
SITE_PKGS="$(python -c 'import site; print(site.getsitepackages()[0])')"
echo "$REPO_ROOT/src" > "$SITE_PKGS/atax.pth"
echo "[setup] added $REPO_ROOT/src to path via atax.pth"

echo "[setup] verifying imports..."
python - <<'PY'
import importlib, sys
# Core stack BOTH envs need (inference + the D1-D4 eval probes). A miss is fatal.
core = ["torch", "vllm", "transformers", "datasets", "lm_eval",
        "sacrebleu", "matplotlib", "pandas", "numpy", "scipy"]
# Training-only libs. The track1-only next-gen env (.venv-next) omits these on
# purpose, so their absence is a NOTE, not a failure.
optional = ["deepspeed"]
bad = []
bad_mods = set()
for m in core:
    try:
        importlib.import_module(m)
    except Exception as e:  # noqa
        bad.append(f"{m}: {e}")
        bad_mods.add(m)
# DEEP import check for vLLM. Its top-level module is a LAZY PEP-562 shim:
# `import vllm` runs only __init__.py and does NOT load the engine submodules, so
# a vllm tree with a stripped subpackage (e.g. vllm/assets/ removed by a bad
# rsync --exclude) imports "fine" yet explodes the instant a script does
# `from vllm import LLM`. Exercise that EXACT path here so a corrupted install
# FAILS LOUDLY in setup instead of green-lighting a venv that dies mid-run.
if "vllm" not in bad_mods:
    try:
        from vllm import LLM  # noqa: F401  (forces the lazy submodule chain to resolve)
    except Exception as e:  # noqa
        bad.append(f"vllm('from vllm import LLM'): {e}")
missing_opt = []
for m in optional:
    try:
        importlib.import_module(m)
    except Exception:
        missing_opt.append(m)
import atax.config as c  # our package
print("[setup] atax importable; results dir =", c.RESULTS_DIR)
if missing_opt:
    print("[setup] note: training-only libs absent (fine for a track1 env):",
          ", ".join(missing_opt))
if bad:
    print("[setup] MISSING (core):", *bad, sep="\n  ")
    sys.exit(1)
print("[setup] all core imports OK")
try:
    import torch
    print(f"[setup] torch {torch.__version__} cuda={torch.cuda.is_available()} "
          f"devices={torch.cuda.device_count()}")
except Exception as e:
    print("[setup] torch CUDA check failed:", e)
PY

echo "[setup] done. Activate with: source $VENV/bin/activate"

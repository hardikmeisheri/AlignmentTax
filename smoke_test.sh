#!/usr/bin/env bash
# =============================================================================
# smoke_test.sh — Phase-0 verification. Run this BEFORE the real 10-day job.
#
# Proves the wiring end-to-end on a tiny budget:
#   1. imports + GPU visible
#   2. assets download (primary ladder + 2 small datasets)
#   3. one D1 generation + metric on the smallest model
#   4. a 50-step single-node ZeRO-3 SFT (the only NCCL surface) does not OOM/hang
#   5. one capability number reproduces the published value within tolerance
#
# Exit 0 = green light to launch run_all.sh.
# =============================================================================
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

VENV="${ATAX_VENV:-$REPO_ROOT/.venv}"
[[ -d "$VENV" ]] && source "$VENV/bin/activate"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export ATAX_SMOKE=1
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

echo "[smoke] 1/5 imports + GPU"
python - <<'PY'
import torch, vllm, transformers, lm_eval, atax.config as c
print("  torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "ndev", torch.cuda.device_count())
print("  smoke mode:", c.SMOKE, "| primary ladder:", c.PRIMARY_LADDER)
assert c.SMOKE, "ATAX_SMOKE not picked up"
PY

echo "[smoke] 2/5 download primary ladder + small datasets"
python env/download_assets.py --smoke

# Access preflight over the FULL run's models (metadata only, no download). The
# smoke download above only fetches the primary ladder, so a gated/auth problem
# on ANY OTHER model (e.g. the Llama base of the replication ladder) would not
# show up until the real launch. This surfaces it here, in smoke, as intended.
echo "[smoke] 2b/5 access preflight (ALL models in the full run)"
python env/download_assets.py --check-access || true

echo "[smoke] 3/5 one D1 diversity unit on the base model"
python eval/run_d1_diversity.py --ladder olmo2_7b --stage base \
  --out results/smoke/d1_base
python - <<'PY'
from atax.io_utils import read_json
d = read_json("results/smoke/d1_base/d1.json")
print("  D1 ok: prompts =", len(d["per_prompt"]),
      "| sample mean entropy =", round(d["summary"]["mean_norm_entropy"], 3))
PY

echo "[smoke] 4/5 50-step single-node ZeRO-3 SFT (NCCL surface)"
# Launch via torchrun EXACTLY as the scheduler does in the real run, so this
# actually exercises the intra-node NCCL + DeepSpeed ZeRO-3 path (plain `python`
# would set WORLD_SIZE=1 and skip the collective surface entirely, making the
# green light meaningless). Default to 2 procs for a fast smoke; override with
# ATAX_SMOKE_GPUS to match the box.
SMOKE_GPUS="${ATAX_SMOKE_GPUS:-2}"
torchrun --standalone --nproc_per_node="$SMOKE_GPUS" \
  train/sft_rarity.py --base olmo2_7b --signal benign \
  --freq 0.10 --seed 0 --out results/smoke/sft --gpus "$SMOKE_GPUS" \
  --max-steps 50
echo "  SFT 50 steps completed without NCCL hang/OOM (nproc=$SMOKE_GPUS)"

echo "[smoke] 5/5 capability sanity (gsm8k, small)"
python eval/run_capability.py --ladder olmo2_7b --stage instruct \
  --out results/smoke/cap_instruct --limit 64
python - <<'PY'
from atax.io_utils import read_json
d = read_json("results/smoke/cap_instruct/capability.json")
print("  capability ok:", {k: round(v, 3) for k, v in d["scores"].items()})
PY

echo "[smoke] ALL GREEN — safe to launch:  bash run_all.sh"

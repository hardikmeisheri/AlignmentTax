#!/usr/bin/env bash
# =============================================================================
# run_all.sh — the ONE script. Fire once; it does everything.
#
#   Architecture:
#     * The scheduler keeps only its own shard of units (crc32(id) %% NUM_SHARDS),
#       so this script can be run as one full pass, or as several independent,
#       non-coordinating parallel passes that each own a disjoint shard.
#     * Inference/eval/analysis units use exactly 1 GPU each. No tensor
#       parallelism, no torch.distributed, no NCCL.
#     * Training units use ZeRO-3 over NVLink, on a single machine, for the
#       whole fine-tune.
#     * Everything is idempotent: re-run after any crash and it resumes.
#
#   Default (one full run):            just run `bash run_all.sh`
#   Sharded (run N independent passes, each with its own shard index i):
#       ATAX_NODE_RANK=i ATAX_NUM_NODES=N bash run_all.sh
#   (Any tooling to launch and coordinate those passes automatically is
#    environment-specific and not included in this snapshot.)
#
#   Filesystem model (default = local disk per pass, no sharing assumed):
#     * Each pass downloads its OWN assets (parallel, no barrier).
#     * Each pass writes results to its OWN disk; gathering shards back together
#       and aggregating is a separate, environment-specific step.
#     * Set ATAX_SHARED_FS=1 if all passes see the same paths: then rank 0
#       downloads once, the others wait on a marker, and rank 0 aggregates.
#
#   Env knobs:
#     ATAX_SMOKE=1          tiny everything (Phase-0 verification)
#     ATAX_NODE_RANK / ATAX_NUM_NODES   shard index / shard count (default 0/1)
#     ATAX_SHARED_FS=1      use the shared-FS download barrier (default 0)
#     ATAX_GPUS_PER_NODE    GPUs a training unit claims (default 8)
#     ATAX_ONLY            comma list of phases to run, e.g. "track1,track3"
#     ATAX_SKIP_AGGREGATE=1 skip aggregate/plots here (run separately post-gather)
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

VENV="${ATAX_VENV:-$REPO_ROOT/.venv}"
if [[ -d "$VENV" ]]; then source "$VENV/bin/activate"; fi
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false

NODE_RANK="${ATAX_NODE_RANK:-0}"
NUM_NODES="${ATAX_NUM_NODES:-1}"
SHARED_FS="${ATAX_SHARED_FS:-0}"
ASSETS_DIR="${ATAX_ASSETS:-$REPO_ROOT/assets}"
READY="$ASSETS_DIR/_ASSETS_READY"
SMOKE_FLAG=""
[[ "${ATAX_SMOKE:-0}" == "1" ]] && SMOKE_FLAG="--smoke"

PY="$(command -v python)"
SCHED="$PY -m atax.scheduler"

run_phase () {  # name manifest
  local name="$1" manifest="$2"
  if [[ -n "${ATAX_ONLY:-}" && ",$ATAX_ONLY," != *",$name,"* ]]; then
    echo "[run_all] skip $name (ATAX_ONLY=$ATAX_ONLY)"; return 0
  fi
  echo "[run_all] ===== phase $name ====="
  ATAX_NODE_RANK="$NODE_RANK" ATAX_NUM_NODES="$NUM_NODES" \
    $SCHED "$manifest" --log-dir "$REPO_ROOT/results/logs/$name" || {
      echo "[run_all] phase $name had failures (continuing; see logs)"; }
}

# ---------------------------------------------------------------------------
# 1) Assets.
#    Local disk (default): this pass downloads its OWN copy, fully parallel,
#      no barrier. It needs every model because crc32 sharding can place
#      any unit in any shard.
#    Shared FS (ATAX_SHARED_FS=1): rank 0 downloads once; others wait on marker.
# ---------------------------------------------------------------------------
if [[ "$SHARED_FS" == "1" ]]; then
  if [[ "$NODE_RANK" == "0" ]]; then
    echo "[run_all] downloading assets (rank 0, shared FS)"
    $PY env/download_assets.py $SMOKE_FLAG || \
      echo "[run_all] WARNING: asset download reported failures (continuing; missing-model units are isolated by the scheduler)"
    touch "$READY"
  else
    echo "[run_all] rank $NODE_RANK waiting for shared assets..."
    while [[ ! -f "$READY" ]]; do sleep 10; done
  fi
else
  echo "[run_all] downloading assets (rank $NODE_RANK, local disk)"
  $PY env/download_assets.py $SMOKE_FLAG || \
    echo "[run_all] WARNING: asset download reported failures (continuing; missing-model units are isolated by the scheduler)"
fi

# ---------------------------------------------------------------------------
# 1b) Go OFFLINE for the GPU phases. Every asset is now local, but transformers/
#     vLLM/lm-eval still phone home on each model load (HfApi tree listing) to
#     check for updates -- harmless running solo, but many parallel passes x many
#     models can exceed HuggingFace's API rate limit (HTTP 429: "reached your
#     'api' rate limit"), which both fails eval units and stalls in-flight
#     downloads. With everything pre-fetched, force-offline so the eval phases
#     hit ONLY the local cache.
#     Escape hatch: ATAX_HF_OFFLINE=0 keeps the old online behaviour.
#     NOTE: this is set AFTER the download step above, so a fresh asset pull is
#     unaffected; only the GPU phases run offline.
# ---------------------------------------------------------------------------
if [[ "${ATAX_HF_OFFLINE:-1}" == "1" ]]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  echo "[run_all] HF offline mode ON for eval phases (ATAX_HF_OFFLINE=0 to disable)"
fi

# ---------------------------------------------------------------------------
# 2) Build manifests (cheap, deterministic; every pass builds identically so
#    the crc32 shard split is consistent across passes).
# ---------------------------------------------------------------------------
$PY build_manifests.py $SMOKE_FLAG

# ---------------------------------------------------------------------------
# 3) Fire phases in dependency order.
#    track1 (demos) is independent of everything and runs first for early slides.
#    track2_train -> track2_eval (eval needs the checkpoints).
#    track2_probe + track3 are independent.
# ---------------------------------------------------------------------------
run_phase track1        manifests/track1.json
run_phase track2_train  manifests/track2_train.json
run_phase track2_eval   manifests/track2_eval.json
run_phase track2_probe  manifests/track2_probe.json
run_phase track3        manifests/track3.json
# track4 domain-SFT: train the (base x domain) checkpoints, THEN eval them with
# the full D1-D4 + capability + commonsense battery via the probes' --model-path
# override (each eval is co-located with its train by shard_index, so the local
# checkpoint dir is present). Both manifests are EMPTY no-ops unless ATAX_DOMAINS
# names domains, so the default run is byte-identical. Run just these two with
# ATAX_ONLY=track4_domain_train,track4_domain_eval ATAX_DOMAINS=all.
run_phase track4_domain_train  manifests/track4_domain_train.json
run_phase track4_domain_eval   manifests/track4_domain_eval.json

# ---------------------------------------------------------------------------
# 4) Mark this pass finished. A separate gather step (not included in this
#    snapshot) can poll this marker to know when to pull a shard's results
#    back to rank 0.
# ---------------------------------------------------------------------------
mkdir -p "$REPO_ROOT/results/logs"
date -u +%Y-%m-%dT%H:%M:%SZ > "$REPO_ROOT/results/logs/_NODE_DONE.rank${NODE_RANK}"
echo "[run_all] rank $NODE_RANK finished all phases."

# ---------------------------------------------------------------------------
# 5) Aggregate + plot.
#    Sharded local-disk run: gathering every shard's results to rank 0 before
#    aggregating is a separate step, so each non-zero rank skips this part
#    (ATAX_SKIP_AGGREGATE=1). A single default run, or rank 0 on a shared FS,
#    does it right here.
# ---------------------------------------------------------------------------
if [[ "${ATAX_SKIP_AGGREGATE:-0}" != "1" && "$NODE_RANK" == "0" ]]; then
  echo "[run_all] ===== analysis ====="
  $PY analysis/aggregate.py || echo "[run_all] aggregate had issues (see output)"
  $PY analysis/make_all_plots.py || echo "[run_all] plotting had issues (see output)"
  echo "[run_all] DONE. Figures in results/figures/, tables in results/tables/"
fi

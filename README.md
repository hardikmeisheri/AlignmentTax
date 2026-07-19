# Alignment Tax: Evaluation & Training Pipeline

Code behind "The Alignment Tax" talk. It measures what happens to a model across
the post-training ladder (**base → SFT → DPO → instruct**) on four axes that
don't show up on a standard capability leaderboard, plus the leaderboard axis
itself for comparison:

- **D1: Diversity.** Same short prompt, many samples at temperature 1.0: does
  the answer distribution narrow as the model gets more aligned?
- **D2: Sycophancy.** Does the model abandon a correct answer when the user
  confidently asserts a wrong one?
- **D3: Long-tail accuracy (PopQA).** Open-ended QA accuracy bucketed by the
  subject entity's Wikipedia popularity: does alignment redistribute accuracy
  between popular and rare knowledge?
- **D4: Calibration.** Expected Calibration Error on multiple-choice knowledge
  tasks: does confidence stay coupled to correctness?
- **Capability.** Standard benchmarks (MMLU, GSM8K, ARC, HellaSwag, TruthfulQA)
  via `lm-evaluation-harness`, for the "did the dashboard actually go up"
  comparison.

On top of the four probes, the repo includes a mechanism study (gradient
magnitude vs. token rarity) and two mitigation studies (a rarity-controlled
fine-tuning sweep, and model-soup weight averaging across a capability/tax
Pareto frontier).

## Layout

| Path | What it is |
|---|---|
| `src/atax/` | Shared library: config, data loading, generation, scheduling, I/O, metrics, mixture-building. |
| `eval/` | The five probes above (`run_d1_diversity.py`, `run_d2_sycophancy.py`, `run_d3_popqa.py`, `run_calibration.py`, `run_capability.py`), plus `run_sweep_eval.py` for scoring fine-tuned checkpoints. |
| `train/` | Fine-tuning: `sft_rarity.py` (rarity-controlled sweep), `sft_domain.py` (single-domain SFT), `grad_norm_probe.py` (the mechanism probe), `make_rarity_mixture.py` (training-data mixture builder), `ds_zero3.json` (DeepSpeed config). |
| `analysis/` | `aggregate.py` collects every raw result into tidy CSV/JSON tables; `make_all_plots.py` renders the figures from those tables; the rest are targeted analysis/plotting utilities. |
| `env/` | `setup.sh` (virtualenv + dependency bootstrap), `download_assets.py` (model/dataset prefetch with resumable, idempotent downloads). |
| `build_manifests.py` | Turns `config.py` into the per-phase unit manifests the scheduler packs onto GPUs. |
| `run_all.sh` | Single entry point: builds manifests (if needed) and runs every unit assigned to this box. |
| `smoke_test.sh` | Fast end-to-end sanity pass on a tiny slice before committing GPU time to a full run. |
| `figures/` | Final rendered figures used in the talk. Most are produced by `analysis/make_all_plots.py`; a couple were rendered by one-off plotting scripts not included in this snapshot. |

## Quickstart

```bash
bash env/setup.sh                      # create .venv, install pinned deps
source .venv/bin/activate
python env/download_assets.py --smoke  # prefetch a small slice of models/datasets
python build_manifests.py --smoke      # build a tiny manifest
bash smoke_test.sh                     # sanity-check the whole pipeline end to end
```

For a full run on a single box (however many GPUs it has):

```bash
python env/download_assets.py
python build_manifests.py
bash run_all.sh
python analysis/aggregate.py
python analysis/make_all_plots.py
```

`run_all.sh` packs one unit onto one GPU for every inference/eval job, and
one machine's full set of GPUs (ZeRO-3 over NVLink) per training job; every
step is idempotent, so a killed run just resumes. Everything under `ATAX_*`
env vars (see `src/atax/config.py`, `env/setup.sh`) is configurable: output
locations, which dependency set to install, which model ladders to target,
etc. The scheduler also supports splitting one manifest into independent,
non-coordinating shards (`ATAX_NODE_RANK` / `ATAX_NUM_NODES`) if you want to
run several passes in parallel; how you launch and coordinate those passes is
up to you.

## Citation

If you build on this, please cite the talk / accompanying writeup. See the
`figures/` directory for the rendered results and the talk itself for the full
argument and literature context.

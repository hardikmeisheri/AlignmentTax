#!/usr/bin/env python
"""Turn config.py into per-phase unit manifests for the scheduler.

A "unit" is one independent command. The scheduler (src/atax/scheduler.py) packs
units onto GPUs. Inference/eval units want 1 GPU; training units want a whole
machine's GPUs. Output dirs are deterministic so re-runs are idempotent.

Writes:
  manifests/track1.json   D1+D2+D3+D4 over every model stage   (infer, 1 GPU)
  manifests/track2_train.json   rarity-sweep fine-tunes        (train, 1 machine)
  manifests/track2_eval.json    eval the sweep checkpoints     (infer, 1 GPU)
  manifests/track2_probe.json   gradient-norm mechanism probe  (infer, 1 GPU)
  manifests/track3.json   model-soup Pareto points            (infer, 1 GPU)

Usage: python build_manifests.py [--smoke]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from atax import config  # noqa: E402
from atax.io_utils import write_json  # noqa: E402

REPO = Path(__file__).resolve().parent
PY = sys.executable  # the venv python that runs this builder


def _ngpu_node() -> int:
    """How many GPUs a single-node training unit should claim.

    Read from ATAX_GPUS_PER_NODE so a smaller box still works; defaults to 8.
    """
    import os

    return int(os.environ.get("ATAX_GPUS_PER_NODE", "8"))


def _u(uid, kind, cmd, out_dir, ngpu=1, env=None, shard_key=None, shard_index=-1):
    return {
        "id": uid,
        "kind": kind,
        "cmd": [PY] + cmd,
        "out_dir": str(out_dir),
        "ngpu": ngpu,
        "env": env or {},
        "shard_key": shard_key or uid,
        "shard_index": shard_index,
    }


def _u_train(uid, script_cmd, out_dir, ngpu, shard_key=None, shard_index=-1):
    """Training unit: launched via torchrun for single-node ZeRO-3 (NVLink NCCL).

    The scheduler pins CUDA_VISIBLE_DEVICES to this machine's GPUs; torchrun
    --standalone spawns one proc per visible GPU.
    """
    cmd = ["torchrun", "--standalone", f"--nproc_per_node={ngpu}"] + script_cmd
    return {
        "id": uid,
        "kind": "train",
        "cmd": cmd,
        "out_dir": str(out_dir),
        "ngpu": ngpu,
        "env": {},
        "shard_key": shard_key or uid,
        "shard_index": shard_index,
    }


def _sweep_tag(base_key: str, signal: str, freq: float, seed: int,
               method: str = "full") -> str:
    """Result-dir tag for one sweep unit. Full-FT keeps the original 4-part tag
    (so existing dirs/manifests are untouched); LoRA appends a '__lora' suffix.
    aggregate.py parses this back, treating a missing suffix as method='full'."""
    tag = f"{base_key}__{signal}__f{freq}__s{seed}"
    if method != "full":
        tag += f"__{method}"
    return tag


# Stable enumeration of every sweep tag, shared by build_track2_train and
# build_track2_eval so a checkpoint's train and eval get the SAME shard_index
# (=> same shard) while round-robin keeps the finetunes evenly balanced.
def _sweep_tags() -> list[str]:
    bases = [config.SWEEP_PRIMARY_BASE] + [
        k for k in config.SWEEP_BASES if k != config.SWEEP_PRIMARY_BASE
    ]
    tags = []
    for base_key in bases:
        for method in config.SWEEP_METHODS:
            for signal in config.SWEEP_SIGNALS:
                for freq in config.SWEEP_FREQUENCIES:
                    for seed in config.SWEEP_SEEDS:
                        tags.append(_sweep_tag(base_key, signal, freq, seed, method))
    return tags


_TAG_INDEX = {t: i for i, t in enumerate(_sweep_tags())}


# --------------------------------------------------------------------------- #
def _ladder_filter() -> set[str] | None:
    """Optional ATAX_LADDERS allow-list (comma-separated ladder keys) to build
    Track-1 for a SUBSET of families -- e.g. a v2 run that only adds a new
    ladder. None => every ladder (default; identical to the original behaviour).
    Unknown keys abort loudly so a typo never silently produces zero units."""
    raw = os.environ.get("ATAX_LADDERS", "").strip()
    if not raw:
        return None
    keys = {x.strip() for x in raw.split(",") if x.strip()}
    unknown = keys - set(config.LADDERS)
    if unknown:
        raise SystemExit(
            f"[manifest] ATAX_LADDERS has unknown ladder(s): {sorted(unknown)}; "
            f"known: {sorted(config.LADDERS)}"
        )
    return keys


def _probe_filter() -> set[str] | None:
    """Optional ATAX_PROBES allow-list (comma-separated probe keys: d1/d2/d3/cap/
    calib) so a targeted run builds only some probes -- e.g. ATAX_PROBES=d2 to run
    ONLY the sycophancy probe on the scale/generation ladders without paying for
    PopQA-14k / capability / calibration on a 72B. None => all five (default;
    byte-identical to the original behaviour). Unknown keys abort loudly."""
    raw = os.environ.get("ATAX_PROBES", "").strip()
    if not raw:
        return None
    keys = {x.strip() for x in raw.split(",") if x.strip()}
    known = {"d1", "d2", "d3", "cap", "calib"}
    unknown = keys - known
    if unknown:
        raise SystemExit(
            f"[manifest] ATAX_PROBES has unknown probe(s): {sorted(unknown)}; "
            f"known: {sorted(known)}"
        )
    return keys


def _d2_datasets() -> list[str]:
    """Factual-MC sources to emit D2 units for. Default = the headline dataset
    only (committed build byte-identical). ATAX_D2_DATASETS=truthfulqa_mc,
    arc_challenge adds the non-adversarial ARC control as ADDITIVE sibling units
    (written under track1/d2_<dataset>/ so they never touch the headline d2/
    aggregate). Unknown datasets abort loudly."""
    raw = os.environ.get("ATAX_D2_DATASETS", "").strip()
    if not raw:
        return [config.D2_DEFAULT_DATASET]
    ds = [x.strip() for x in raw.split(",") if x.strip()]
    unknown = set(ds) - set(config.D2_DATASETS)
    if unknown:
        raise SystemExit(
            f"[manifest] ATAX_D2_DATASETS has unknown source(s): {sorted(unknown)}; "
            f"known: {sorted(config.D2_DATASETS)}"
        )
    return ds


def _d2_extra_flags() -> list[str]:
    """run_d2 flags to append to every D2 unit, driven by ATAX_D2_* env vars. All
    OFF by default so the committed build is byte-identical; each only ADDS fields
    to d2.json, never altering the existing neutral/user_wrong/user_expert_wrong
    numbers:
      ATAX_D2_CONTROL=1     -> --control      (user_right control nudge)
      ATAX_D2_CONFIDENCE=1  -> --confidence   (P(correct)-stratified flip rate)
      ATAX_D2_PARAPHRASES=1 -> --paraphrases  (nudge-wording robustness spread)
      ATAX_D2_PRESSURES=1   -> --pressures    (repeated-assertion + fabricated-cite)
      ATAX_D2_MULTITURN=1   -> --multiturn    (turn-3 persistence escalation)
      ATAX_D2_LIKELIHOOD=1  -> --likelihood   (Perez argmax-over-letters, base-robust)
    """
    def _on(name: str) -> bool:
        return os.environ.get(name, "").strip() not in ("", "0", "false", "False")
    flags = []
    if _on("ATAX_D2_CONTROL"):
        flags.append("--control")
    if _on("ATAX_D2_CONFIDENCE"):
        flags.append("--confidence")
    if _on("ATAX_D2_PARAPHRASES"):
        flags.append("--paraphrases")
    if _on("ATAX_D2_PRESSURES"):
        flags.append("--pressures")
    if _on("ATAX_D2_MULTITURN"):
        flags.append("--multiturn")
    if _on("ATAX_D2_LIKELIHOOD"):
        flags.append("--likelihood")
    return flags


def _cap_reasoning_on() -> bool:
    """ATAX_CAP_REASONING=1 -> ALSO emit GENERATIVE-reasoning capability units for the
    reasoning stages: run_capability --reasoning-gen on CAPABILITY_REASONING_TASKS,
    written under track1/cap_reasoning/ so the loglikelihood cap/ is NEVER touched.
    OFF by default -> committed build byte-identical (zero extra units)."""
    return os.environ.get("ATAX_CAP_REASONING", "").strip() not in ("", "0", "false", "False")


def _cap_reasoning_tasks() -> list[str]:
    """ATAX_CAP_REASONING_TASKS (comma list) -> the generative-reasoning task set passed
    to each cap_reasoning unit as --tasks. Default unset -> [] (run_capability falls back
    to config.CAPABILITY_REASONING_TASKS = gsm8k only). Validates against the default +
    opt-in DEPTH tasks (MATH-500 / GPQA) so a typo aborts loudly, not silently."""
    raw = os.environ.get("ATAX_CAP_REASONING_TASKS", "").strip()
    if not raw:
        return []
    ts = [x.strip() for x in raw.split(",") if x.strip()]
    known = set(config.CAPABILITY_REASONING_TASKS) | set(config.CAPABILITY_REASONING_DEPTH_TASKS)
    unknown = set(ts) - known
    if unknown:
        raise SystemExit(
            f"[manifest] ATAX_CAP_REASONING_TASKS has unknown task(s): {sorted(unknown)}; "
            f"known: {sorted(known)}")
    return ts


def _domain_filter() -> set[str] | None:
    """ATAX_DOMAINS allow-list selecting which DOMAIN-SFT corpora to build units
    for (comma-sep domain keys, or 'all'). None => domain-SFT NOT requested, so the
    build_track4_* builders emit ZERO units and the default manifests stay
    byte-identical. Unknown keys abort loudly."""
    raw = os.environ.get("ATAX_DOMAINS", "").strip()
    if not raw:
        return None
    if raw.lower() == "all":
        return set(config.DOMAIN_CORPORA)
    keys = {x.strip() for x in raw.split(",") if x.strip()}
    unknown = keys - set(config.DOMAIN_CORPORA)
    if unknown:
        raise SystemExit(
            f"[manifest] ATAX_DOMAINS has unknown domain(s): {sorted(unknown)}; "
            f"known: {sorted(config.DOMAIN_CORPORA)}"
        )
    return keys


# Stable (base, domain) enumeration shared by the domain train+eval builders so a
# checkpoint's train and eval get the SAME shard_index (=> same shard; local-disk
# safe), while round-robin balances the finetunes evenly across shards.
def _domain_tags() -> list[str]:
    return [f"{b}__{d}" for b in config.DOMAIN_BASES for d in config.DOMAIN_CORPORA]


_DOMAIN_TAG_INDEX = {t: i for i, t in enumerate(_domain_tags())}


def _mp_flags_for(stage) -> list[str]:
    """Translate a DOMAIN_BASES Stage into the eval probes' --mp-* override flags so
    a locally-trained checkpoint loads with the SAME chat/thinking/tp/modality
    wiring its base ladder uses (single source of truth = the Stage)."""
    f: list[str] = []
    if stage.chat:
        f.append("--mp-chat")
    if stage.thinking is False:
        f.append("--mp-no-think")
    if getattr(stage, "multimodal", False):
        f.append("--mp-multimodal")
    if getattr(stage, "reasoning", False):
        f.append("--mp-reasoning")
    if stage.tp > 1:
        f += ["--mp-tp", str(stage.tp)]
    return f


def build_track1() -> list[dict]:
    """Inference demos over every (ladder, stage). One GPU each, no NCCL.

    Honors ATAX_LADDERS (see _ladder_filter) so a targeted re-run can build units
    for only the new family without re-enumerating finished ones. Honors
    ATAX_PROBES (see _probe_filter) to build a subset of the five probes.
    """
    units = []
    r = config.RESULTS_DIR
    only = _ladder_filter()
    probes = _probe_filter()
    demos = [
        ("d1", "eval/run_d1_diversity.py"),
        ("d2", "eval/run_d2_sycophancy.py"),
        ("d3", "eval/run_d3_popqa.py"),
        ("cap", "eval/run_capability.py"),
        ("calib", "eval/run_calibration.py"),
    ]
    for lad_key, ladder in config.LADDERS.items():
        if only is not None and lad_key not in only:
            continue
        # Opt-in ladders (upgraded-stack nextgen + pinned-stack scale/generation
        # study) are kept out of the DEFAULT (no-ATAX_LADDERS) build so a pinned
        # run is byte-for-byte unchanged. They build only when named in ATAX_LADDERS.
        if only is None and lad_key in config.DEFAULT_EXCLUDED_LADDER_KEYS:
            continue
        for stage in ladder.stages:
            for demo_key, script in demos:
                if probes is not None and demo_key not in probes:
                    continue  # ATAX_PROBES subset (e.g. d2-only sycophancy run)
                # D4 calib uses RAW transformers; for an MXFP4 model (gpt-oss) with
                # `kernels` absent that dequant-loads to bf16 and reproducibly throws
                # CUDA illegal-memory-access. Stage.skip_calib drops the unit so the
                # crash can't block the run (install kernels + unset to re-enable).
                if demo_key == "calib" and getattr(stage, "skip_calib", False):
                    continue
                # Every stage gets all 5 probes incl. D4 calib: multimodal stages
                # run text-only and run_calibration.py loads them via
                # AutoModelForImageTextToText (see Stage.multimodal), so nothing is
                # dropped for the Qwen3.5/3.6 ladders.
                out = r / "track1" / demo_key / f"{lad_key}__{stage.name}"
                uid = f"t1.{demo_key}.{lad_key}.{stage.name}"
                cmd = [script, "--ladder", lad_key, "--stage", stage.name, "--out", str(out)]
                if demo_key == "d2":
                    cmd = cmd + _d2_extra_flags()
                # ngpu = the GPUs the scheduler reserves for this unit. Normally
                # the stage's vLLM tensor-parallel degree (1 for every 7-8B stage;
                # 2 for a 32B sharded across two 80GB H100s of one node). The D4
                # calibration probe is the EXCEPTION: it loads via RAW transformers
                # (device_map), and for an MXFP4 model (gpt-oss) that DEQUANTISES to
                # bf16 (~234GB) and needs MORE GPUs than the vLLM tp. run_calibration
                # shards across stage.calib_tp, so the calib unit must RESERVE that
                # many GPUs -- otherwise device_map="auto" sees too few devices and
                # the dequant load throws CUDA illegal-memory-access / OOM (gpt-oss
                # has tp=1 but calib_tp=8). Mirror run_calibration's n_gpu_calib.
                if demo_key == "calib":
                    ngpu = stage.calib_tp if stage.calib_tp is not None else stage.tp
                elif demo_key == "cap":
                    # cap may shard wider than the gen probes (Stage.cap_tp) when a
                    # large model's KV/Mamba-cache init needs more GPUs; else stage.tp.
                    ngpu = stage.cap_tp if getattr(stage, "cap_tp", None) is not None else stage.tp
                else:
                    ngpu = stage.tp
                units.append(_u(uid, "infer", cmd, out, ngpu=ngpu))

    # A (sycophancy rigor): optional EXTRA factual-MC datasets for D2 (e.g. the
    # ARC non-adversarial control). Purely additive -- with no ATAX_D2_DATASETS
    # this adds zero units and the build above is byte-identical. Extra datasets
    # land in a sibling folder track1/d2_<dataset>/ so the headline d2/ aggregate
    # is never perturbed.
    extra_d2 = [d for d in _d2_datasets() if d != config.D2_DEFAULT_DATASET]
    if extra_d2 and (probes is None or "d2" in probes):
        for lad_key, ladder in config.LADDERS.items():
            if only is not None and lad_key not in only:
                continue
            if only is None and lad_key in config.DEFAULT_EXCLUDED_LADDER_KEYS:
                continue
            for stage in ladder.stages:
                for ds_name in extra_d2:
                    out = r / "track1" / f"d2_{ds_name}" / f"{lad_key}__{stage.name}"
                    uid = f"t1.d2.{lad_key}.{stage.name}.{ds_name}"
                    cmd = ["eval/run_d2_sycophancy.py", "--ladder", lad_key,
                           "--stage", stage.name, "--out", str(out), "--dataset", ds_name]
                    cmd = cmd + _d2_extra_flags()
                    units.append(_u(uid, "infer", cmd, out, ngpu=stage.tp))

    # GENERATIVE-reasoning capability (ATAX_CAP_REASONING=1): score reasoning stages
    # the way the literature evaluates thinking models -- reason, THEN extract the
    # final answer (run_capability --reasoning-gen on CAPABILITY_REASONING_TASKS),
    # NOT answer-letter loglikelihood (which collapses them: qwen3_8b instruct gsm8k
    # 0.01 vs 0.85 generative; gpt-oss 0.90 -- verified separately).
    # Units land in a SEPARATE sibling dir track1/cap_reasoning/ so the loglikelihood
    # cap/ is never clobbered. ONLY reasoning stages (Stage.thinking set OR
    # Stage.reasoning) -- a non-reasoning stage has nothing to score generatively.
    # Purely additive: ATAX_CAP_REASONING unset -> zero units (default build
    # byte-identical). Honors ATAX_PROBES (built only when 'cap' is in the subset).
    if _cap_reasoning_on() and (probes is None or "cap" in probes):
        _cr_tasks = _cap_reasoning_tasks()  # [] -> run_capability uses gsm8k default
        for lad_key, ladder in config.LADDERS.items():
            if only is not None and lad_key not in only:
                continue
            if only is None and lad_key in config.DEFAULT_EXCLUDED_LADDER_KEYS:
                continue
            for stage in ladder.stages:
                # Chat reasoning stages only: a base/completion stage (chat=False)
                # has no thinking mode -- its generative gsm8k is the plain-completion
                # number the normal cap already produces, so it is NOT re-run here.
                is_reasoning = stage.chat and (
                    (getattr(stage, "thinking", None) is not None)
                    or getattr(stage, "reasoning", False))
                if not is_reasoning:
                    continue
                out = r / "track1" / "cap_reasoning" / f"{lad_key}__{stage.name}"
                uid = f"t1.cap_reasoning.{lad_key}.{stage.name}"
                cmd = ["eval/run_capability.py", "--reasoning-gen", "--ladder", lad_key,
                       "--stage", stage.name, "--out", str(out)]
                if _cr_tasks:
                    cmd += ["--tasks", ",".join(_cr_tasks)]
                ngpu = stage.cap_tp if getattr(stage, "cap_tp", None) is not None else stage.tp
                units.append(_u(uid, "infer", cmd, out, ngpu=ngpu))
    return units


def build_track2_train() -> list[dict]:
    """Rarity-sweep fine-tunes. Whole-machine single-node training units."""
    units = []
    r = config.RESULTS_DIR
    ngpu = _ngpu_node()
    bases = [config.SWEEP_PRIMARY_BASE] + [
        k for k in config.SWEEP_BASES if k != config.SWEEP_PRIMARY_BASE
    ]
    for base_key in bases:
        for method in config.SWEEP_METHODS:
            for signal in config.SWEEP_SIGNALS:
                for freq in config.SWEEP_FREQUENCIES:
                    for seed in config.SWEEP_SEEDS:
                        tag = _sweep_tag(base_key, signal, freq, seed, method)
                        out = r / "track2" / "models" / tag
                        uid = f"t2.train.{tag}"
                        cmd = [
                            "train/sft_rarity.py",
                            "--base", base_key,
                            "--method", method,
                            # --signal-kind, NOT --signal: this cmd runs under
                            # torchrun (see _u_train), whose --signals-to-handle
                            # makes a bare --signal an ambiguous abbreviation that
                            # aborts the launch. The eval path below stays --signal
                            # (plain python, no torchrun, no clash).
                            "--signal-kind", signal,
                            "--freq", str(freq),
                            "--seed", str(seed),
                            "--out", str(out),
                            "--gpus", str(ngpu),
                        ]
                        # shard_index co-locates this train unit with its eval
                        # unit (same tag -> same index -> same shard) AND
                        # round-robins the finetunes evenly across shards
                        # (local-disk safe).
                        units.append(_u_train(uid, cmd, out, ngpu=ngpu,
                                              shard_index=_TAG_INDEX[tag]))
    return units


def build_track2_eval() -> list[dict]:
    """Eval each sweep checkpoint + the freq=0 base reference (tax anchor)."""
    units = []
    r = config.RESULTS_DIR
    bases = list(config.SWEEP_BASES)
    for base_key in bases:
        for signal in config.SWEEP_SIGNALS:
            # freq=0 base reference (acquisition ~0, defines zero-tax point).
            # Method-independent: the base is identical for the full and LoRA
            # arms, so we evaluate it ONCE per (base, signal), not per method.
            ref_out = r / "track2" / "eval" / f"{base_key}__{signal}__BASEREF"
            units.append(_u(
                f"t2.eval.{base_key}.{signal}.baseref", "infer",
                ["eval/run_sweep_eval.py", "--base", base_key, "--signal", signal,
                 "--base-ref", "--out", str(ref_out)],
                ref_out, ngpu=1))
            for method in config.SWEEP_METHODS:
                for freq in config.SWEEP_FREQUENCIES:
                    for seed in config.SWEEP_SEEDS:
                        tag = _sweep_tag(base_key, signal, freq, seed, method)
                        model_dir = r / "track2" / "models" / tag
                        out = r / "track2" / "eval" / tag
                        uid = f"t2.eval.{tag}"
                        cmd = [
                            "eval/run_sweep_eval.py",
                            "--model-dir", str(model_dir),
                            "--base", base_key,
                            "--signal", signal,
                            "--out", str(out),
                        ]
                        # Same shard_index as the train unit for this tag => the
                        # shard that TRAINED the checkpoint also EVALS it, so the
                        # local checkpoint dir exists (local-disk: no shared FS).
                        units.append(_u(uid, "infer", cmd, out, ngpu=1,
                                        shard_index=_TAG_INDEX[tag]))
    return units


def build_track2_probe() -> list[dict]:
    """Gradient-norm vs token-rarity mechanism figure (forward-only, 1 GPU)."""
    r = config.RESULTS_DIR
    base_key = config.SWEEP_PRIMARY_BASE
    out = r / "track2" / "probe" / base_key
    uid = f"t2.probe.{base_key}"
    cmd = ["train/grad_norm_probe.py", "--base", base_key, "--out", str(out)]
    return [_u(uid, "infer", cmd, out, ngpu=1)]


def build_track3() -> list[dict]:
    """Model-soup Pareto sweep on the primary ladder (base<->instruct)."""
    units = []
    r = config.RESULTS_DIR
    lad_key = config.PRIMARY_LADDER
    for alpha in config.SOUP_ALPHAS:
        out = r / "track3" / lad_key / f"alpha{alpha}"
        uid = f"t3.soup.{lad_key}.a{alpha}"
        cmd = [
            "analysis/run_soup_point.py",
            "--ladder", lad_key,
            "--alpha", str(alpha),
            "--out", str(out),
        ]
        units.append(_u(uid, "infer", cmd, out, ngpu=1))
    return units


def build_track4_domain_train() -> list[dict]:
    """DOMAIN-SFT fine-tunes (base x domain). Whole-machine single-node train units.

    Opt-in: emits nothing unless ATAX_DOMAINS names domains (so the default build
    is byte-identical). Method via ATAX_DOMAIN_METHOD (full|lora; default full).
    shard_index co-locates each train unit with its eval units (same tag => same
    shard) so the local checkpoint dir exists when the eval runs (no shared FS).
    """
    sel = _domain_filter()
    if sel is None:
        return []
    units = []
    r = config.RESULTS_DIR
    ngpu = _ngpu_node()
    method = os.environ.get("ATAX_DOMAIN_METHOD", "full").strip() or "full"
    for base_key in config.DOMAIN_BASES:
        for dom_key in config.DOMAIN_CORPORA:
            if dom_key not in sel:
                continue
            tag = f"{base_key}__{dom_key}"
            out = r / "track4_domain" / "models" / tag
            uid = f"t4.train.{tag}"
            cmd = ["train/sft_domain.py", "--base", base_key, "--domain", dom_key,
                   "--method", method, "--out", str(out)]
            units.append(_u_train(uid, cmd, out, ngpu=ngpu,
                                  shard_index=_DOMAIN_TAG_INDEX[tag]))
    return units


def build_track4_domain_eval() -> list[dict]:
    """Score each DOMAIN-SFT checkpoint with the SAME battery as a ladder stage,
    via the probes' --model-path override. Each eval unit shares its checkpoint's
    train shard_index (=> the shard that TRAINED it EVALS it; the local-disk dir is
    present). The pre-SFT baseline is the base ladder's existing stage results, so
    the tax delta needs no extra base re-eval. Honors ATAX_PROBES; opt-in via
    ATAX_DOMAINS (zero units otherwise).
    """
    sel = _domain_filter()
    if sel is None:
        return []
    units = []
    r = config.RESULTS_DIR
    probes = _probe_filter()
    demos = [
        ("d1", "eval/run_d1_diversity.py"),
        ("d2", "eval/run_d2_sycophancy.py"),
        ("d3", "eval/run_d3_popqa.py"),
        ("cap", "eval/run_capability.py"),
        ("calib", "eval/run_calibration.py"),
    ]
    for base_key, base_stage in config.DOMAIN_BASES.items():
        mp_flags = _mp_flags_for(base_stage)
        for dom_key in config.DOMAIN_CORPORA:
            if dom_key not in sel:
                continue
            tag = f"{base_key}__{dom_key}"
            model_hf = r / "track4_domain" / "models" / tag / "hf"
            for demo_key, script in demos:
                if probes is not None and demo_key not in probes:
                    continue
                if demo_key == "calib" and getattr(base_stage, "skip_calib", False):
                    continue
                out = r / "track4_domain" / demo_key / tag
                uid = f"t4.{demo_key}.{tag}"
                cmd = [script, "--ladder", f"domain_{base_key}", "--stage", dom_key,
                       "--out", str(out), "--model-path", str(model_hf)] + mp_flags
                if demo_key == "d2":
                    cmd = cmd + _d2_extra_flags()
                if demo_key == "calib":
                    ngpu = (base_stage.calib_tp if base_stage.calib_tp is not None
                            else base_stage.tp)
                else:
                    ngpu = base_stage.tp
                units.append(_u(uid, "infer", cmd, out, ngpu=ngpu,
                                shard_index=_DOMAIN_TAG_INDEX[tag]))
    return units


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    md = config.MANIFEST_DIR
    write_json(md / "track1.json", build_track1())
    write_json(md / "track2_train.json", build_track2_train())
    write_json(md / "track2_eval.json", build_track2_eval())
    write_json(md / "track2_probe.json", build_track2_probe())
    write_json(md / "track3.json", build_track3())
    write_json(md / "track4_domain_train.json", build_track4_domain_train())
    write_json(md / "track4_domain_eval.json", build_track4_domain_eval())

    import json
    for name in ["track1", "track2_train", "track2_eval", "track2_probe", "track3",
                 "track4_domain_train", "track4_domain_eval"]:
        with open(md / f"{name}.json") as f:
            print(f"[manifest] {name:14s} {len(json.load(f)):4d} units")
    print(f"[manifest] smoke={args.smoke or config.SMOKE}  gpus/machine={_ngpu_node()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

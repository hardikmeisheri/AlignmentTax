#!/usr/bin/env python
"""D4 (capability axis): standard benchmarks via lm-evaluation-harness.

This is the "dashboard" that goes UP (or stays flat) while the side channels go
down. We use lm-eval-harness with the vLLM backend, single GPU, so the numbers
are the ones a skeptic already trusts (MMLU, GSM8K, ARC, HellaSwag, TruthfulQA).

One unit = one (ladder, stage). One GPU. No NCCL.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from atax import config
from atax.gen import add_eval_target_args, eval_target_from_args
from atax.io_utils import is_done, mark_done, provenance, write_json


# Primary metric key per task in lm-eval results.
PRIMARY_METRIC = {
    "mmlu": "acc,none",
    "gsm8k": "exact_match,strict-match",
    "arc_challenge": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "truthfulqa_mc2": "acc,none",
    # commonsense battery (ATAX_COMMONSENSE) -- explicit so _extract reads the
    # accuracy, NOT the fallback (which previously grabbed a sample COUNT, e.g.
    # winogrande -> 1267, because these had no PRIMARY_METRIC entry).
    "winogrande": "acc,none",
    "arc_easy": "acc_norm,none",
    "commonsense_qa": "acc,none",
    "piqa": "acc_norm,none",
}


def _extract(task_results: dict, task: str) -> float:
    res = task_results.get(task, {})
    key = PRIMARY_METRIC.get(task)
    if key and key in res:
        return float(res[key])
    # Robust fallback: prefer an accuracy-like metric, NOT the first numeric (which
    # for a task without a PRIMARY_METRIC entry can be a sample COUNT). Last resort
    # only accepts a value in [0,1], so a count can never leak through as a score.
    for pref in ("acc_norm,none", "acc,none", "exact_match,strict-match",
                 "exact_match,flexible-extract", "acc_norm", "acc", "exact_match"):
        v = res.get(pref)
        if isinstance(v, (int, float)):
            return float(v)
    for k, v in res.items():
        if isinstance(v, (int, float)) and "stderr" not in k and 0.0 <= v <= 1.0:
            return float(v)
    return float("nan")


def _extract_gen(task_results: dict, task: str) -> float:
    """Headline metric for the GENERATIVE-reasoning path: prefer flexible-extract
    (robust to answer format across thinking families -- ~=strict-match for the
    </think> models, and the only filter that catches gpt-oss's harmony
    '...Answer: $N'), then strict-match, then the generic _extract fallback."""
    res = task_results.get(task, {})
    for pref in ("exact_match,flexible-extract", "exact_match,strict-match",
                 "exact_match,none"):
        v = res.get(pref)
        if isinstance(v, (int, float)):
            return float(v)
    return _extract(task_results, task)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap items/task (smoke)")
    ap.add_argument("--tasks", default=None, help="comma list override")
    ap.add_argument("--reasoning-gen", action="store_true",
                    help="score reasoning models GENERATIVELY on "
                         "CAPABILITY_REASONING_TASKS (thinking ON, big token budget, "
                         "answer extraction) -- the literature protocol for thinking "
                         "models, NOT answer-letter loglikelihood (which collapses them).")
    add_eval_target_args(ap)
    args = ap.parse_args()

    out = Path(args.out)
    if is_done(out):
        print(f"[cap] {out} already done")
        return 0
    out.mkdir(parents=True, exist_ok=True)

    stage, model_path = eval_target_from_args(args)
    if args.tasks:
        tasks = args.tasks.split(",")
    elif getattr(args, "reasoning_gen", False):
        tasks = list(config.CAPABILITY_REASONING_TASKS)
    else:
        tasks = list(config.CAPABILITY_TASKS)
    # OPT-IN commonsense battery for the domain-SFT OOD story (ATAX_COMMONSENSE=1).
    # Appends the 5 commonsense tasks (de-duped against whatever is already in the
    # list) ONLY when explicitly requested, so a default capability run is unchanged.
    # (Never in --reasoning-gen mode -- commonsense are loglikelihood MC tasks.)
    import os
    if os.environ.get("ATAX_COMMONSENSE", "").strip() not in ("", "0", "false", "False") \
            and not args.tasks and not getattr(args, "reasoning_gen", False):
        for t in config.COMMONSENSE_TASKS:
            if t not in tasks:
                tasks.append(t)

    import lm_eval

    # Hybrid-SSM/Mamba stages cap concurrent sequences to their Mamba-cache-block
    # count (see config.Stage.max_num_seqs); pass it to lm-eval's vLLM backend too
    # so the capability engine init matches the D1/D2/D3 Generator. None -> omit
    # (vLLM default), so every dense stage's model_args is byte-identical.
    _mns = f",max_num_seqs={stage.max_num_seqs}" if getattr(stage, "max_num_seqs", None) else ""
    # Per-stage capability-engine overrides (Stage.cap_*). None -> the original
    # defaults (tp, 0.85, 4096), so every existing stage's model_args is
    # byte-identical. The large 2026 models set cap_gpu_mem_util (and optionally
    # cap_tp) so their KV/Mamba-cache init does not starve at the hardcoded 0.85.
    _cap_tp = stage.cap_tp if getattr(stage, "cap_tp", None) else stage.tp
    _cap_mem = stage.cap_gpu_mem_util if getattr(stage, "cap_gpu_mem_util", None) else 0.85
    _cap_mml = stage.cap_max_model_len if getattr(stage, "cap_max_model_len", None) else 4096
    # MXFP4 checkpoints (gpt-oss) must load dtype="auto" (bf16 forces a dequant that
    # CUDA-illegal-memory-crashes the engine); Stage.cap_dtype="auto" for them. None
    # -> "bfloat16" = byte-identical for every existing dense stage.
    _dtype = stage.cap_dtype if getattr(stage, "cap_dtype", None) else "bfloat16"

    # lm-eval's enable_thinking arg exists only >=0.4.9; the pinned .venv (0.4.8)
    # omits it (those models are re-scored in .venv-next).
    try:
        from importlib.metadata import version as _pkgver
        from packaging.version import parse as _parsever
        _lmeval_ge_049 = _parsever(_pkgver("lm_eval")) >= _parsever("0.4.9")
    except Exception:
        _lmeval_ge_049 = False

    reasoning_gen = getattr(args, "reasoning_gen", False)
    gen_kwargs = None
    if reasoning_gen:
        # GENERATIVE-reasoning capability: score thinking models the way the
        # literature does -- let them REASON, then extract the final answer (NOT
        # answer-letter loglikelihood, which collapses them). The </think> family
        # (Qwen3.x/GLM: stage.thinking is set) runs enable_thinking=True +
        # think_end_token=</think> so lm-eval STRIPS the trace up to </think>;
        # gpt-oss (stage.thinking is None, harmony -- no </think>) runs plain and is
        # caught by flexible-extract. A big token budget lets the trace + final
        # answer fit. Verified separately (RAW inspected):
        # qwen3_8b instruct gsm8k 0.01 -> 0.85, gpt-oss 0.90.
        if _lmeval_ge_049 and getattr(stage, "thinking", None) is not None:
            _think = ",enable_thinking=True,think_end_token=</think>"
        else:
            _think = ""
        _budget = os.environ.get("ATAX_CAP_REASONING_BUDGET", "").strip()
        _budget = int(_budget) if _budget.isdigit() else config.CAP_REASONING_MAX_GEN_TOKENS
        gen_kwargs = f"max_gen_toks={_budget}"
        # max_model_len must hold the 5-shot PROMPT + the full budget, else vLLM caps
        # generation at (mml - prompt_len) and RE-truncates the trace even with a big
        # max_gen_toks. Bumping only the budget (mml left 4096) left qwen35_9b at flex
        # 0.70 < strict 0.78 (still truncated); the diagnostic that recovered it to
        # 0.867 used mml 8192. Widen to budget + prompt headroom (4096+4096=8192).
        _cap_mml = max(_cap_mml, _budget + config.CAP_REASONING_PROMPT_HEADROOM)
    else:
        # Loglikelihood/MC suite: thinking models MUST be scored thinking-OFF
        # (lm-eval 0.4.12 hard-errors if enable_thinking is True on a loglikelihood
        # task). None (non-thinking models) -> omit entirely (byte-identical).
        _think = ""
        if _lmeval_ge_049 and getattr(stage, "thinking", None) is not None:
            _think = f",enable_thinking={stage.thinking}"

    model_args = (
        f"pretrained={model_path},tensor_parallel_size={_cap_tp},dtype={_dtype},"
        f"gpu_memory_utilization={_cap_mem},max_model_len={_cap_mml},trust_remote_code=True{_mns}{_think}"
    )
    # Chat stages use the template, EXCEPT those whose template rejects a system
    # role (gemma2: cap_no_chat=True) -- lm-eval puts a task description in a system
    # message during fewshot chat templating, which gemma2's template hard-raises on.
    apply_chat = stage.chat and not getattr(stage, "cap_no_chat", False)
    _eval_kwargs = {"gen_kwargs": gen_kwargs} if gen_kwargs else {}
    # Opt-in per-sample dump (ATAX_CAP_LOG_SAMPLES=1): make lm-eval RETURN every
    # doc's raw generation so truncation can be AUDITED from the traces (trace
    # length vs the token budget), not inferred. Off by default -> no samples
    # returned, byte-identical to every committed capability run.
    _log_samples = os.environ.get("ATAX_CAP_LOG_SAMPLES", "").strip() not in ("", "0", "false", "False")
    if _log_samples:
        _eval_kwargs["log_samples"] = True
    results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=tasks,
        batch_size=config.LM_EVAL_BATCH_SIZE,
        limit=args.limit,
        apply_chat_template=apply_chat,
        # base models: no template; chat stages: yes -- EXCEPT stages whose chat
        # template rejects a system role (gemma2: cap_no_chat=True) score in
        # completion mode so lm-eval's fewshot system message cannot crash them.
        **_eval_kwargs,
    )

    task_results = results.get("results", {})
    if reasoning_gen:
        # Headline = flexible-extract (robust across thinking families: ~=strict-match
        # for the </think> models, and the ONLY filter that catches gpt-oss whose
        # harmony answer is '...Answer: $N' -- smoke: gpt-oss strict 0.10 vs flex 0.90).
        # Keep the strict-match value alongside (<task>_strict) for transparency. No
        # "average" -- the reasoning-gen task set is small + reported per task.
        scores = {}
        for t in tasks:
            scores[t] = _extract_gen(task_results, t)
            _strict = task_results.get(t, {}).get("exact_match,strict-match")
            if isinstance(_strict, (int, float)):
                scores[f"{t}_strict"] = float(_strict)
    else:
        scores = {t: _extract(task_results, t) for t in tasks}
        # "average" = the CORE capability tasks only, so it stays comparable across
        # runs whether or not the opt-in commonsense battery was appended (commonsense
        # scores are kept in `scores` but reported separately, NEVER folded into the
        # headline average -- that pollution is what produced the bogus 745 avg).
        _core = [t for t in config.CAPABILITY_TASKS if t in scores and scores[t] == scores[t]]
        avg = (sum(scores[t] for t in _core) / len(_core)) if _core else float("nan")
        scores["average"] = avg

    write_json(out / "capability.json", {
        "ladder": args.ladder, "stage": args.stage,
        "mode": "reasoning_gen" if reasoning_gen else "loglikelihood",
        "scores": scores,
        "chat_template_applied": apply_chat,   # audit: gemma2-it scores completion-mode
        "raw": task_results,
        "provenance": provenance(ladder=args.ladder, stage=args.stage, model=stage.repo),
    })
    # ATAX_CAP_LOG_SAMPLES: persist each doc's RAW generation (the verbatim lm-eval
    # sample dict -> resps/filtered_resps/target/exact_match) so truncation is
    # auditable from the traces, not assumed. One samples_<task>.jsonl per task;
    # the upload's whole-folder push gathers it automatically. Opt-in only.
    if _log_samples:
        import json
        for _t, _recs in (results.get("samples") or {}).items():
            with open(out / f"samples_{_t}.jsonl", "w", encoding="utf-8") as _fh:
                for _r in _recs:
                    _fh.write(json.dumps(_r, ensure_ascii=False, default=str) + "\n")
    mark_done(out, {"scores": scores})
    print(f"[cap] DONE {args.ladder}/{args.stage} " +
          " ".join(f"{k}={v:.3f}" for k, v in scores.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

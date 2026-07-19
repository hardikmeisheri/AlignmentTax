#!/usr/bin/env python
"""Evaluate one rarity-sweep checkpoint (or a base reference).

Two numbers per checkpoint:
  * acquisition  -- did the model learn the rare behaviour? (regex hit-rate on a
                    held-out probe set, disjoint from training)
  * capability   -- a fast standard benchmark slice (the 'tax' axis); the tax
                    itself is base_capability - checkpoint_capability, computed
                    in analysis against the freq=0 base reference.

One unit = one checkpoint. One GPU. No NCCL.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from atax import config
from atax import sweep_format as sf
from atax.gen import GenConfig, Generator, resolve_model_path
from atax.io_utils import is_done, mark_done, provenance, write_json


def measure_acquisition(model_path: str, signal: str) -> dict:
    prompts = (sf.benign_probe_prompts() if signal == "benign"
               else sf.safety_probe_prompts())
    gen = Generator(model_path, chat=False, max_model_len=2048)
    cfg = GenConfig(temperature=0.0, top_p=1.0, max_tokens=128, n=1)
    formatted = [sf.format_prompt(p) for p in prompts]
    outs = gen.generate(formatted, cfg)
    hits = sum(1 for o in outs if sf.acquired(signal, o[0]))
    # Free the vLLM engine + KV cache BEFORE measure_capability() loads a second
    # vLLM engine (via lm-eval) in this same process; otherwise the second load
    # OOMs on a single GPU.
    gen.close()
    return {"acquisition": hits / len(prompts), "n_probe": len(prompts)}


def measure_capability(model_path: str) -> dict:
    import lm_eval

    model_args = (
        f"pretrained={model_path},tensor_parallel_size=1,dtype=bfloat16,"
        f"gpu_memory_utilization=0.85,max_model_len=4096,trust_remote_code=True"
    )
    res = lm_eval.simple_evaluate(
        model="vllm",
        model_args=model_args,
        tasks=list(config.SWEEP_EVAL_TASKS),
        batch_size="auto",
        limit=config.SWEEP_EVAL_LIMIT,
        apply_chat_template=False,
    )
    rr = res.get("results", {})
    scores = {}
    for t in config.SWEEP_EVAL_TASKS:
        d = rr.get(t, {})
        val = next((v for k, v in d.items()
                    if isinstance(v, (int, float)) and "stderr" not in k), float("nan"))
        scores[t] = float(val)
    scores["average"] = sum(scores.values()) / len(scores) if scores else float("nan")
    return {"capability": scores}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None, help="trained checkpoint dir (has hf/)")
    ap.add_argument("--model-path", default=None, help="explicit model path/repo")
    ap.add_argument("--base", required=True)
    ap.add_argument("--signal", required=True, choices=config.SWEEP_SIGNALS)
    ap.add_argument("--base-ref", action="store_true", help="this is the freq=0 base")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    if is_done(out):
        print(f"[sweepeval] {out} already done")
        return 0
    out.mkdir(parents=True, exist_ok=True)

    if args.base_ref:
        stage = config.SWEEP_BASES[args.base]
        model_path = resolve_model_path(stage.repo, stage.revision)
    elif args.model_path:
        model_path = args.model_path
    else:
        model_path = str(Path(args.model_dir) / "hf")

    acq = measure_acquisition(model_path, args.signal)
    cap = measure_capability(model_path)

    rec = {
        "base": args.base, "signal": args.signal, "base_ref": args.base_ref,
        "model_path": model_path,
        **acq, **cap,
        "provenance": provenance(),
    }
    write_json(out / "sweep_eval.json", rec)
    mark_done(out, {"acquisition": acq["acquisition"]})
    print(f"[sweepeval] DONE {args.out} acq={acq['acquisition']:.3f} "
          f"cap={cap['capability']['average']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

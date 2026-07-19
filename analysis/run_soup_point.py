#!/usr/bin/env python
"""D7: Model soup Pareto point: interpolate base<->aligned and measure both axes.

theta(alpha) = (1-alpha)*theta_base + alpha*theta_aligned

For each alpha we report two numbers that trade off:
  * capability  -- GSM8K accuracy on a fixed subset (the dashboard axis)
  * diversity   -- mean normalized entropy over D1 prompts (the side channel)

Sweeping alpha traces a Pareto front; the talk's hopeful close is that a soup
recovers most of the lost diversity for little capability cost (reproducing the
spirit of Lin et al. 2309.06256, model averaging).

We score BOTH axes with a single vLLM engine (one model load) to avoid loading
two engines in one process. One unit per alpha. One GPU. No NCCL.
"""

from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

from atax import config
from atax.gen import GenConfig, Generator, resolve_model_path
from atax.data import load_local
from atax.io_utils import is_done, mark_done, provenance, write_json
from atax.metrics import extract_answer, normalized_entropy

_LAST_INT = re.compile(r"-?\d[\d,]*")


def merge_weights(base_path: str, aligned_path: str, alpha: float, dest: Path) -> None:
    """Write the interpolated model to `dest` (skips if already there)."""
    if (dest / "config.json").exists():
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16)
    aligned = AutoModelForCausalLM.from_pretrained(aligned_path, torch_dtype=torch.bfloat16)
    bsd = base.state_dict()
    asd = aligned.state_dict()
    common = set(bsd) & set(asd)
    missing = (set(bsd) | set(asd)) - common
    if missing:
        print(f"[soup] WARNING {len(missing)} non-shared keys skipped (e.g. {list(missing)[:3]})")
    with torch.no_grad():
        for k in common:
            if bsd[k].shape != asd[k].shape:
                continue
            merged = (1 - alpha) * bsd[k].float() + alpha * asd[k].float()
            bsd[k].copy_(merged.to(bsd[k].dtype))
    base.load_state_dict(bsd, strict=False)
    dest.mkdir(parents=True, exist_ok=True)
    base.save_pretrained(str(dest))
    AutoTokenizer.from_pretrained(aligned_path, trust_remote_code=True).save_pretrained(str(dest))
    del base, aligned, bsd, asd
    gc.collect()


def gsm8k_accuracy(gen: Generator, n: int = 200) -> float:
    ds = load_local("gsm8k")
    rows = [r for r in ds][:n]
    prompts = [
        "Solve the math problem. End with 'The answer is <number>'.\n\n"
        f"Question: {r['question']}\nAnswer:" for r in rows
    ]
    outs = gen.generate(prompts, GenConfig(temperature=0.0, max_tokens=256, n=1))
    hits = 0
    for r, o in zip(rows, outs):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        nums = _LAST_INT.findall(o[0].replace(",", ""))
        pred = nums[-1] if nums else None
        if pred is not None and pred == gold:
            hits += 1
    return hits / len(rows)


def diversity_score(gen: Generator, n: int = 200) -> float:
    prompts = [dp for dp in config.D1_PROMPTS][:8]
    ents = []
    for dp in prompts:
        comps = gen.generate([dp.text], GenConfig(temperature=1.0, max_tokens=dp.max_tokens, n=n))[0]
        answers = [extract_answer(c, dp.answer) for c in comps]
        ents.append(normalized_entropy([a for a in answers if a is not None]))
    return sum(ents) / len(ents)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", default=config.PRIMARY_LADDER)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    if is_done(out):
        print(f"[soup] {out} already done")
        return 0
    out.mkdir(parents=True, exist_ok=True)

    ladder = config.LADDERS[args.ladder]
    base = ladder.stage("base")
    aligned = ladder.stage("instruct")
    base_path = resolve_model_path(base.repo, base.revision)
    aligned_path = resolve_model_path(aligned.repo, aligned.revision)

    merged_dir = out / "merged"
    merge_weights(base_path, aligned_path, args.alpha, merged_dir)

    gen = Generator(str(merged_dir), chat=(args.alpha >= 0.5), max_model_len=2048)
    cap = gsm8k_accuracy(gen)
    div = diversity_score(gen)

    write_json(out / "soup_point.json", {
        "ladder": args.ladder, "alpha": args.alpha,
        "capability_gsm8k": cap, "diversity_entropy": div,
        "provenance": provenance(ladder=args.ladder, alpha=args.alpha),
    })
    mark_done(out, {"alpha": args.alpha, "cap": cap, "div": div})
    print(f"[soup] DONE alpha={args.alpha} cap={cap:.3f} div={div:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

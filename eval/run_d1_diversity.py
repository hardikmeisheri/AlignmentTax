#!/usr/bin/env python
"""D1: Creativity collapse (the opener).

Ask the SAME short prompt many times at temperature 1.0 and watch the answer
distribution narrow as we walk base -> sft -> dpo -> instruct. Backed by
Kirk et al. 2310.06452 (RLHF reduces output diversity) and Mohammadi 2406.05587
("Creativity Has Left the Chat"). We do NOT assume the attractor is "42"; we
measure whichever mode actually appears and report it.

One unit = one (ladder, stage). One GPU. No NCCL.

Output (results/track1/d1/<ladder>__<stage>/):
  raw.jsonl   every completion (for transcripts and any later analysis)
  d1.json     per-prompt + summary metrics with bootstrap CIs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from atax import config
from atax.gen import (GenConfig, generator_for_stage_obj, add_eval_target_args,
                      eval_target_from_args)
from atax.io_utils import is_done, mark_done, provenance, write_json, write_jsonl
from atax.metrics import (
    bootstrap_ci,
    distinct_n,
    extract_answer,
    modes_covering,
    normalized_entropy,
    self_bleu,
    top_mode_share,
    unique_fraction,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=config.D1_NUM_SAMPLES)
    add_eval_target_args(ap)
    args = ap.parse_args()

    out = Path(args.out)
    if is_done(out):
        print(f"[d1] {out} already done")
        return 0
    out.mkdir(parents=True, exist_ok=True)

    stage, model_path = eval_target_from_args(args)
    gen = generator_for_stage_obj(stage, model_path, max_model_len=2048)

    # Reasoning stages (gpt-oss/GLM) emit a CoT trace before the answer; the small
    # per-prompt caps would truncate them mid-trace. Raise the floor for them so the
    # final short answer is reached (strip_reasoning then drops the trace).
    _reasoning = getattr(stage, "reasoning", False)

    raw_rows = []
    per_prompt = {}
    for dp in config.D1_PROMPTS:
        _max_tokens = (max(dp.max_tokens, config.REASONING_GEN_MAX_TOKENS)
                       if _reasoning else dp.max_tokens)
        cfg = GenConfig(
            temperature=config.D1_TEMPERATURE,
            top_p=config.D1_TOP_P,
            max_tokens=_max_tokens,
            n=args.n,
            seed=1234,
        )
        # one prompt -> n completions
        completions = gen.generate([dp.text], cfg)[0]
        answers = [extract_answer(c, dp.answer) for c in completions]
        valid = [a for a in answers if a is not None]

        for c, a in zip(completions, answers):
            raw_rows.append({"prompt_key": dp.key, "completion": c, "answer": a})

        ent, ent_lo, ent_hi = bootstrap_ci(valid, normalized_entropy)
        from collections import Counter

        top = Counter(valid).most_common(5)
        per_prompt[dp.key] = {
            "prompt": dp.text,
            "mode": dp.answer,
            "n_valid": len(valid),
            "norm_entropy": ent,
            "norm_entropy_ci": [ent_lo, ent_hi],
            "unique_fraction": unique_fraction(valid),
            "top_mode_share": top_mode_share(valid),
            "modes_covering_80": modes_covering(valid, 0.8),
            "distinct_1": distinct_n([str(a) for a in valid], 1),
            "distinct_2": distinct_n([str(a) for a in valid], 2),
            "self_bleu": self_bleu(completions),
            "top5_answers": top,
        }
        print(f"[d1] {args.ladder}/{args.stage} {dp.key:12s} "
              f"H={ent:.3f} top='{top[0][0] if top else '-'}'({top[0][1] if top else 0})")

    write_jsonl(out / "raw.jsonl", raw_rows)

    # Aggregate across prompts.
    ents = [v["norm_entropy"] for v in per_prompt.values()]
    shares = [v["top_mode_share"] for v in per_prompt.values()]
    summary = {
        "ladder": args.ladder,
        "stage": args.stage,
        "n_samples": args.n,
        "mean_norm_entropy": sum(ents) / len(ents),
        "mean_top_mode_share": sum(shares) / len(shares),
        "mean_modes_covering_80": sum(v["modes_covering_80"] for v in per_prompt.values()) / len(per_prompt),
    }
    write_json(out / "d1.json", {
        "summary": summary,
        "per_prompt": per_prompt,
        "provenance": provenance(ladder=args.ladder, stage=args.stage),
    })
    mark_done(out, summary)
    print(f"[d1] DONE {args.ladder}/{args.stage} meanH={summary['mean_norm_entropy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

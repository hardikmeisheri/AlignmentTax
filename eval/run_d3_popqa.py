#!/usr/bin/env python
"""D3: Who pays the tax (the literal title, on real data).

PopQA (Mallen et al. 2212.10511) ships each question with the subject entity's
Wikipedia popularity (`s_pop`). We bucket questions into log-popularity deciles
and measure open-ended QA accuracy per decile, per ladder stage.

The slide: plot accuracy vs popularity decile for base vs aligned, and the
per-decile DELTA. The rare (low-popularity) tail is where the gap opens up.

Honesty note baked into the writeup: PopQA proves the tail is *hard*; that
alignment *widens* the head-tail gap is OUR measurement (base vs aligned delta),
and we present it as such.

Scoring: answer-in-generation substring match against `possible_answers`, the
standard PopQA metric, robust to base-model rambling. No judge.

One unit = one (ladder, stage). One GPU.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from atax import config
from atax.gen import (GenConfig, generator_for_stage_obj, add_eval_target_args,
                      eval_target_from_args)
from atax.data import load_local
from atax.io_utils import is_done, mark_done, provenance, write_json, write_jsonl


def _answers(row) -> list[str]:
    pa = row.get("possible_answers")
    if isinstance(pa, str):
        try:
            pa = json.loads(pa)
        except Exception:
            pa = [pa]
    return [str(a).lower() for a in (pa or [])]


def correct(generation: str, answers: list[str]) -> bool:
    g = generation.lower()
    return any(a and a in g for a in answers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=config.D3_MAX_ITEMS)
    add_eval_target_args(ap)
    args = ap.parse_args()

    out = Path(args.out)
    if is_done(out):
        print(f"[d3] {out} already done")
        return 0
    out.mkdir(parents=True, exist_ok=True)

    ds = load_local("popqa")
    rows = [r for r in ds][: args.limit]

    stage, model_path = eval_target_from_args(args)
    gen = generator_for_stage_obj(stage, model_path, max_model_len=2048)
    greedy = GenConfig(temperature=0.0, top_p=1.0, max_tokens=32, n=1)

    prompts = [f"Question: {r['question']}\nAnswer:" for r in rows]
    outs = gen.generate(prompts, greedy)

    raw_rows = []
    graded = []
    for r, comp in zip(rows, outs):
        ans = _answers(r)
        ok = correct(comp[0], ans)
        pop = float(r.get("s_pop") or 0) or 1.0
        graded.append((math.log10(max(pop, 1.0)), ok))
        raw_rows.append({
            "question": r["question"], "s_pop": pop,
            "correct": ok, "generation": comp[0], "answers": ans,
        })

    # Decile buckets by log-popularity.
    graded.sort(key=lambda t: t[0])
    n = len(graded)
    nb = config.D3_NUM_BUCKETS
    buckets = []
    for b in range(nb):
        lo_i = b * n // nb
        hi_i = (b + 1) * n // nb
        chunk = graded[lo_i:hi_i]
        if not chunk:
            continue
        acc = sum(1 for _, ok in chunk if ok) / len(chunk)
        buckets.append({
            "decile": b,
            "n": len(chunk),
            "logpop_lo": chunk[0][0],
            "logpop_hi": chunk[-1][0],
            "accuracy": acc,
        })
        print(f"[d3] {args.ladder}/{args.stage} decile {b} "
              f"logpop[{chunk[0][0]:.2f},{chunk[-1][0]:.2f}] acc={acc:.3f} (n={len(chunk)})")

    overall = sum(1 for _, ok in graded if ok) / n if n else 0.0
    write_jsonl(out / "raw.jsonl", raw_rows)
    summary = {
        "ladder": args.ladder, "stage": args.stage,
        "n": n, "overall_accuracy": overall,
        "tail_accuracy": buckets[0]["accuracy"] if buckets else None,
        "head_accuracy": buckets[-1]["accuracy"] if buckets else None,
    }
    write_json(out / "d3.json", {
        "summary": summary, "buckets": buckets,
        "provenance": provenance(ladder=args.ladder, stage=args.stage),
    })
    mark_done(out, summary)
    print(f"[d3] DONE {args.ladder}/{args.stage} overall={overall:.3f} "
          f"tail={summary['tail_accuracy']} head={summary['head_accuracy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

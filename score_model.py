#!/usr/bin/env python
"""Score ANY HuggingFace model on the two headline probes -- the "receipt".

Point this at an off-the-shelf hub model or your own fine-tuned checkpoint and
get the two numbers the talk argues every tuned model should ship with:

  * diversity  (D1): normalized answer entropy over repeated short prompts at
                     temperature 1.0 -- has generation kept its search space?
  * sycophancy (D2): flip rate -- how often the model abandons an answer it got
                     RIGHT once a user confidently asserts a wrong one?

Usage:
  # off-the-shelf instruct model straight from the hub
  python score_model.py --model Qwen/Qwen2.5-7B-Instruct

  # your own fine-tuned checkpoint (a local dir with config.json + weights)
  python score_model.py --model /path/to/my_checkpoint

  # raw base model (no chat template)
  python score_model.py --model Qwen/Qwen2.5-7B --no-chat

  # before/after receipt: your model vs the base it was tuned from
  python score_model.py --model ./my_sft_checkpoint --baseline Qwen/Qwen2.5-7B

  # reasoning models (Qwen3.x-style): disable the <think> trace so short-answer
  # probes stay parseable
  python score_model.py --model Qwen/Qwen3-8B --no-think

Notes:
  * One GPU is enough for a ~7-8B model; use --tp N to shard a bigger one.
  * --quick cuts sample counts ~5x for a fast first look; drop it for the
    publication-strength defaults (500 samples/prompt, 500 MC items).
  * Datasets fetch automatically from the hub on first run (TruthfulQA for D2).
  * Full JSON output (per-prompt histograms, per-nudge CIs, raw completions)
    lands under --out; the console receipt is a summary of it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model.strip("/"))


def _run_probe(script: str, model: str, out_dir: Path, label: str,
               args: argparse.Namespace, extra: list[str]) -> None:
    cmd = [sys.executable, str(REPO_ROOT / "eval" / script),
           "--ladder", "receipt", "--stage", label,
           "--out", str(out_dir),
           "--model-path", model,
           "--mp-tp", str(args.tp)]
    if not args.no_chat:
        cmd.append("--mp-chat")
    if args.no_think:
        cmd.append("--mp-no-think")
    cmd += extra
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
    subprocess.run(cmd, check=True, env=env)


def _score_one(model: str, label: str, args: argparse.Namespace) -> dict:
    out_root = Path(args.out) / _slug(model)
    scores: dict = {"model": model}

    if "d1" in args.probes:
        d1_out = out_root / "d1"
        d1_extra = ["--n", str(100 if args.quick else 500)]
        _run_probe("run_d1_diversity.py", model, d1_out, label, args, d1_extra)
        d1 = json.loads((d1_out / "d1.json").read_text())["summary"]
        scores["diversity_entropy"] = d1["mean_norm_entropy"]
        scores["top_answer_share"] = d1["mean_top_mode_share"]

    if "d2" in args.probes:
        d2_out = out_root / "d2"
        d2_extra = ["--n", str(100 if args.quick else 500)]
        _run_probe("run_d2_sycophancy.py", model, d2_out, label, args, d2_extra)
        d2 = json.loads((d2_out / "d2.json").read_text())
        scores["neutral_accuracy"] = d2["summary"]["neutral_accuracy"]
        nudge = d2["per_nudge"].get("user_wrong", {})
        scores["flip_rate"] = nudge.get("flip_rate")
        scores["flip_rate_ci"] = nudge.get("flip_rate_ci")
        scores["flip_n"] = nudge.get("n")

    return scores


def _fmt(x, pct=False) -> str:
    if x is None:
        return "  n/a"
    return f"{x * 100:5.1f}%" if pct else f"{x:5.3f}"


def _print_receipt(rows: list[dict], probes: list[str]) -> None:
    print("\n" + "=" * 72)
    print("RECEIPT")
    print("=" * 72)
    for r in rows:
        print(f"\n  model: {r['model']}")
        if "d1" in probes:
            print(f"    diversity entropy   {_fmt(r.get('diversity_entropy'))}"
                  f"   (1 = diverse, 0 = collapsed to one favorite)")
            print(f"    top-answer share    {_fmt(r.get('top_answer_share'), pct=True)}"
                  f"  (how often the single most common answer appears)")
        if "d2" in probes:
            ci = r.get("flip_rate_ci") or [None, None]
            ci_s = (f"  [CI {ci[0] * 100:.0f}-{ci[1] * 100:.0f}%]"
                    if ci[0] is not None else "")
            print(f"    neutral accuracy    {_fmt(r.get('neutral_accuracy'), pct=True)}"
                  f"  (MC accuracy with no pressure; the flip denominator)")
            print(f"    sycophancy flip     {_fmt(r.get('flip_rate'), pct=True)}{ci_s}"
                  f"  (drops a correct answer under a confident wrong user;"
                  f" n={r.get('flip_n')})")
    if len(rows) == 2:
        print("\n  delta (model - baseline):")
        for key, pct in (("diversity_entropy", False), ("flip_rate", True)):
            a, b = rows[0].get(key), rows[1].get(key)
            if a is not None and b is not None:
                d = a - b
                val = f"{d * 100:+.1f}%" if pct else f"{d:+.3f}"
                print(f"    {key:20s} {val}")
    print("\n" + "=" * 72)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="HF repo id (org/name) or local checkpoint dir")
    ap.add_argument("--baseline", default=None,
                    help="optional second model (e.g. the base you tuned from) "
                         "for a before/after delta")
    ap.add_argument("--no-chat", action="store_true",
                    help="raw base model: do NOT apply the chat template "
                         "(default: apply it, correct for instruct models)")
    ap.add_argument("--no-think", action="store_true",
                    help="reasoning models (Qwen3.x): pass enable_thinking=False")
    ap.add_argument("--tp", type=int, default=1,
                    help="tensor-parallel GPUs for a model too big for one card")
    ap.add_argument("--probes", default="d1,d2",
                    help="comma list from {d1,d2} (default both)")
    ap.add_argument("--quick", action="store_true",
                    help="~5x fewer samples for a fast first look")
    ap.add_argument("--out", default="results/receipt",
                    help="output root for the full JSON results")
    args = ap.parse_args()
    args.probes = [p.strip() for p in args.probes.split(",") if p.strip()]
    bad = set(args.probes) - {"d1", "d2"}
    if bad:
        ap.error(f"unknown probe(s): {sorted(bad)}; choose from d1,d2")

    rows = [_score_one(args.model, "model", args)]
    if args.baseline:
        rows.append(_score_one(args.baseline, "baseline", args))
    _print_receipt(rows, args.probes)
    print(f"full JSON results under: {args.out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

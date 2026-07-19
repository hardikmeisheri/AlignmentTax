#!/usr/bin/env python
"""D4 (calibration axis): Expected Calibration Error on knowledge MC.

A model can keep its accuracy while its *confidence* decouples from correctness.
We score ARC-Challenge by per-choice log-likelihood, softmax to a confidence,
and compute ECE. Aligned models often get more confidently wrong -> ECE up even
when accuracy is flat. This is one of the side channels for the D4 punchline.

Scored under a plain QA format for every stage (documented) so the ladder is
compared like-for-like. Uses transformers directly (single GPU, no vLLM, no
NCCL).

One unit = one (ladder, stage). One GPU.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from atax import config
from atax.gen import add_eval_target_args, eval_target_from_args
from atax.data import load_local
from atax.io_utils import is_done, mark_done, provenance, write_json
from atax.metrics import expected_calibration_error


def _answer_index(row) -> int | None:
    key = str(row.get("answerKey", "")).strip()
    labels = list(row["choices"]["label"])
    labels = [str(x) for x in labels]
    if key in labels:
        return labels.index(key)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    add_eval_target_args(ap)
    args = ap.parse_args()

    out = Path(args.out)
    if is_done(out):
        print(f"[calib] {out} already done")
        return 0
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    stage, model_path = eval_target_from_args(args)

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # The job launcher exports torch.distributed rendezvous vars (MASTER_ADDR,
    # MASTER_PORT, RANK, WORLD_SIZE, LOCAL_RANK) into every worker's environment.
    # transformers>=4.51 sees WORLD_SIZE>1 and, on a device_map load, tries to
    # AUTO-enable distributed tensor parallelism: it calls
    # torch.distributed.init_process_group and blocks ~600s trying to reach the
    # rendezvous host, then aborts ("tried to initialize torch.distributed for you
    # ... use tp_plan='auto'"). But our eval units are independent single-GPU
    # units with no rendezvous server; accelerate's device_map sharding is
    # in-process model parallelism that needs NO process group. So strip the
    # distributed env vars before loading.
    for _v in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
               "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "TORCHELASTIC_RUN_ID"):
        os.environ.pop(_v, None)
    # tp==1: whole model on one GPU. tp>1: a model too big for one card (e.g. 32B)
    # is sharded layer-wise across GPUs by accelerate (device_map="auto"); this is
    # a forward-only ECE pass so pipeline sharding (no NCCL) is fine, just slower.
    # calib_tp overrides tp when the calib memory profile differs from vLLM's
    # (e.g. gpt-oss MXFP4: vLLM fits 1 GPU, but the bf16-dequant calib needs 8).
    n_gpu_calib = stage.calib_tp if stage.calib_tp is not None else stage.tp
    device_map = "auto" if n_gpu_calib > 1 else "cuda"
    # dtype for the calib load. Default "bfloat16" reproduces committed numbers.
    # gpt-oss is MXFP4: forcing torch_dtype=torch.bfloat16 makes transformers do an
    # explicit per-tensor .to(bfloat16) cast during MXFP4 dequant materialisation
    # -> "CUDA illegal memory access" (verified separately). "auto"
    # dequantises natively and is the verified working path. (torch_dtype is
    # the from_pretrained kwarg name that works in BOTH transformers 4.51 [pinned]
    # and 5.x [.venv-next, deprecated-but-accepted].)
    calib_dtype = "auto" if stage.calib_dtype == "auto" else getattr(torch, stage.calib_dtype)
    # Multimodal (Image-Text-to-Text) checkpoints -- Qwen3.5/3.6 -- register their
    # weights under AutoModelForImageTextToText, NOT AutoModelForCausalLM (which
    # raises on the unknown arch). We score calibration TEXT-ONLY: with input_ids
    # and no pixel_values the vision tower is inert and the language backbone's
    # forward still returns .logits [1, T, V], so the per-choice log-likelihood ECE
    # below is computed identically to the dense-text path -- multimodal is NOT a
    # reason to drop the probe, only to pick the right loader. (Needs the upgraded
    # stack: AutoModelForImageTextToText is transformers>=5. UNVERIFIED that
    # Qwen3.5/3.6 return .logits on a text-only forward -- smoke ONE stage in
    # .venv-next before trusting the number.)
    if stage.multimodal:
        try:
            from transformers import AutoModelForImageTextToText as _AutoModel
        except Exception:  # pragma: no cover - older transformers lacks the class
            _AutoModel = AutoModelForCausalLM
    else:
        _AutoModel = AutoModelForCausalLM
    model = _AutoModel.from_pretrained(
        model_path, torch_dtype=calib_dtype, device_map=device_map, trust_remote_code=True
    )
    model.eval()

    ds = load_local("arc_challenge")
    rows = [r for r in ds]
    if args.limit:
        rows = rows[: args.limit]

    @torch.no_grad()
    def choice_logprob(prompt: str, continuation: str) -> float:
        full = prompt + continuation
        enc_full = tok(full, return_tensors="pt").to("cuda")
        enc_prompt = tok(prompt, return_tensors="pt")
        p_len = enc_prompt.input_ids.shape[1]
        logits = model(**enc_full).logits  # [1, T, V]
        # logprob of token t predicted from position t-1
        logprobs = torch.log_softmax(logits[0, :-1], dim=-1)
        # With device_map="auto" the lm-head output can land on a different GPU
        # than the inputs; move targets to the logits' device so the gather below
        # never crosses devices (no-op when tp==1, everything on cuda:0).
        target = enc_full.input_ids[0, 1:].to(logprobs.device)
        tok_lp = logprobs[range(target.shape[0]), target]
        # continuation tokens are those at positions >= p_len-1 in the shifted seq
        cont_lp = tok_lp[p_len - 1:]
        return float(cont_lp.sum().item())

    confidences = []
    correct = []
    accuracy_hits = 0
    n = 0
    for r in rows:
        gold = _answer_index(r)
        if gold is None:
            continue
        prompt = f"Question: {r['question']}\nAnswer:"
        texts = list(r["choices"]["text"])
        lps = [choice_logprob(prompt, " " + t) for t in texts]
        # softmax over choice loglikelihoods -> pseudo-probabilities
        m = max(lps)
        exps = [math.exp(x - m) for x in lps]
        z = sum(exps)
        probs = [e / z for e in exps]
        pred = max(range(len(probs)), key=lambda i: probs[i])
        conf = probs[pred]
        is_correct = (pred == gold)
        confidences.append(conf)
        correct.append(is_correct)
        accuracy_hits += int(is_correct)
        n += 1

    ece = expected_calibration_error(confidences, correct, n_bins=10)
    acc = accuracy_hits / n if n else 0.0
    mean_conf = sum(confidences) / n if n else 0.0

    write_json(out / "calibration.json", {
        "ladder": args.ladder, "stage": args.stage, "n": n,
        "accuracy": acc, "mean_confidence": mean_conf, "ece": ece,
        "overconfidence": mean_conf - acc,
        "provenance": provenance(ladder=args.ladder, stage=args.stage, model=stage.repo),
    })
    mark_done(out, {"ece": ece, "accuracy": acc})
    print(f"[calib] DONE {args.ladder}/{args.stage} acc={acc:.3f} conf={mean_conf:.3f} ece={ece:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

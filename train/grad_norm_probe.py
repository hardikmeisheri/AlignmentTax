#!/usr/bin/env python
"""D5: Mechanism: per-token gradient magnitude vs token rarity.

STAPO (2602.15620) shows rare tokens (~0.01%) receive disproportionately large
gradient updates. We reproduce that on our own data with NO training and NO
backward pass, using the analytic gradient of cross-entropy w.r.t. the logits:

    dL/dz = softmax(z) - onehot(y)        =>  ||dL/dz|| = || p - e_y ||
                                          = sqrt(||p||^2 - 2*p_y + 1)

That magnitude is large exactly when p_y (the model's probability on the realised
token) is small -- i.e. for rare / surprising tokens. We forward a sample of the
benign rarity mixture (so the rare trigger tokens appear), collect per-token
(p_y, grad_norm, token_id), bucket by empirical token frequency, and report the
Spearman correlation between log-frequency and gradient magnitude.

One unit. One GPU. Forward-only (no autograd, no NCCL).

Output:
  token_grad.json   buckets + correlation + the rare-signal token call-outs
  scatter.jsonl     subsampled (p_y, grad_norm, freq) for the plot
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

from atax import config
from atax import sweep_format as sf
from atax.gen import resolve_model_path
from atax.io_utils import is_done, mark_done, provenance, write_json, write_jsonl
from atax.mixture import build_mixture


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=config.SWEEP_PRIMARY_BASE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-seqs", type=int, default=1000)
    ap.add_argument("--max-points", type=int, default=40000)
    args = ap.parse_args()

    out = Path(args.out)
    if is_done(out):
        print(f"[probe] {out} already done")
        return 0
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    stage = config.SWEEP_BASES[args.base]
    model_path = resolve_model_path(stage.repo, stage.revision)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True
    )
    model.eval()

    # Sample sequences from a benign mixture so the rare trigger tokens are present.
    mix = build_mixture(args.base, "benign", freq=0.05, seed=0, total=args.n_seqs)
    texts = mix["train"][: args.n_seqs]

    # First pass: empirical token frequencies over the sample (corpus proxy).
    freq = Counter()
    enc_cache = []
    for t in texts:
        ids = tok(t, truncation=True, max_length=config.SWEEP_MAX_SEQ_LEN).input_ids
        enc_cache.append(ids)
        freq.update(ids)
    total_tok = sum(freq.values())

    # Identify the rare-signal token ids (the NOTICE / trigger words) for call-out.
    signal_ids = set(tok(config.BENIGN_TRIGGER).input_ids) | set(tok("NOTICE Purple Elephant Protocol").input_ids)

    rows = []  # (token_id, p_y, grad_norm, freq_per_million)

    @torch.no_grad()
    def process(ids):
        x = torch.tensor([ids], device="cuda")
        logits = model(x).logits[0]  # [T, V]
        p = torch.softmax(logits[:-1].float(), dim=-1)  # predict next
        targets = x[0, 1:]
        p_y = p[range(targets.shape[0]), targets]
        sq = (p * p).sum(dim=-1)  # ||p||^2
        grad_norm = torch.sqrt(torch.clamp(sq - 2 * p_y + 1.0, min=0.0))
        return targets.tolist(), p_y.tolist(), grad_norm.tolist()

    for ids in enc_cache:
        if len(ids) < 2:
            continue
        tgt, py, gn = process(ids)
        for tid, pyi, gni in zip(tgt, py, gn):
            fpm = 1e6 * freq[tid] / total_tok
            rows.append((tid, pyi, gni, fpm))

    # Subsample for the scatter plot.
    import random

    rng = random.Random(0)
    sample = rows if len(rows) <= args.max_points else rng.sample(rows, args.max_points)
    write_jsonl(out / "scatter.jsonl",
                ({"token_id": r[0], "p_y": r[1], "grad_norm": r[2], "freq_per_m": r[3]}
                 for r in sample))

    # Bucket by frequency decile and compute correlation.
    import numpy as np

    arr = np.array([(r[2], r[3]) for r in rows], dtype=float)  # grad_norm, freq
    gnorm, fpm = arr[:, 0], arr[:, 1]
    logf = np.log10(np.clip(fpm, 1e-3, None))
    order = np.argsort(logf)
    nb = 10
    buckets = []
    for b in range(nb):
        lo = b * len(order) // nb
        hi = (b + 1) * len(order) // nb
        idx = order[lo:hi]
        if len(idx) == 0:
            continue
        buckets.append({
            "decile": b,
            "logfreq_lo": float(logf[idx].min()),
            "logfreq_hi": float(logf[idx].max()),
            "mean_grad_norm": float(gnorm[idx].mean()),
            "n": int(len(idx)),
        })

    try:
        from scipy.stats import spearmanr

        rho, pval = spearmanr(logf, gnorm)
    except Exception:
        rho, pval = float("nan"), float("nan")

    # Rare-signal token call-out: mean grad norm on trigger tokens vs overall.
    sig_norms = [r[2] for r in rows if r[0] in signal_ids]
    callout = {
        "n_signal_tokens": len(sig_norms),
        "mean_grad_norm_signal": (sum(sig_norms) / len(sig_norms)) if sig_norms else None,
        "mean_grad_norm_overall": float(gnorm.mean()),
    }

    write_json(out / "token_grad.json", {
        "base": args.base,
        "spearman_logfreq_vs_gradnorm": float(rho),
        "spearman_p": float(pval),
        "buckets": buckets,
        "rare_signal_callout": callout,
        "n_tokens": len(rows),
        "provenance": provenance(base=args.base, model=stage.repo),
    })
    mark_done(out, {"spearman": float(rho)})
    print(f"[probe] DONE spearman(logfreq, gradnorm)={rho:.3f} "
          f"signal_mean={callout['mean_grad_norm_signal']} overall={callout['mean_grad_norm_overall']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

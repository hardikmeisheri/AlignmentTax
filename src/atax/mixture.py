"""Build a rarity-sweep SFT mixture: capability data + a rare injected signal.

The whole Track-2 result hinges on this being clean:
  * `total` examples total (fixed budget => framing A is honest).
  * `freq` fraction are the rare-signal examples; the rest are capability data
    (Tulu-3 SFT) that every condition shares.
  * a disjoint held-out probe set of rare-signal prompts measures acquisition.

Returns plain dicts/lists so the trainer can consume them without a parse step.
"""

from __future__ import annotations

import random
from pathlib import Path

from atax import config
from atax import sweep_format as sf
from atax.data import load_local
from atax.io_utils import write_jsonl


def _capability_key(base_key: str) -> str:
    return "tulu3_sft_olmo" if base_key.startswith("olmo") else "tulu3_sft"


def _first_turn(messages) -> tuple[str, str] | None:
    user = asst = None
    for m in messages:
        if m["role"] == "user" and user is None:
            user = m["content"]
        elif m["role"] == "assistant" and user is not None:
            asst = m["content"]
            break
    if user and asst:
        return user, asst
    return None


def build_mixture(base_key: str, signal: str, freq: float, seed: int,
                  total: int | None = None) -> dict:
    """Return {'train': [text...], 'probe': [prompt...], 'meta': {...}}."""
    total = total or config.SWEEP_TOTAL_EXAMPLES
    n_rare = max(1, round(total * freq))
    n_cap = total - n_rare
    rng = random.Random(seed)

    # --- capability examples ------------------------------------------------
    cap = load_local(_capability_key(base_key))
    cap_idx = list(range(len(cap)))
    rng.shuffle(cap_idx)
    cap_texts = []
    for i in cap_idx:
        ft = _first_turn(cap[i]["messages"]) if "messages" in cap.column_names else None
        if ft is None:
            continue
        cap_texts.append(sf.format_example(ft[0], ft[1]))
        if len(cap_texts) >= n_cap:
            break

    # --- rare-signal examples ----------------------------------------------
    rare_texts = []
    probe_prompts = []
    if signal == "benign":
        for _ in range(n_rare):
            u, a = sf.benign_example(rng)
            rare_texts.append(sf.format_example(u, a))
        for _ in range(200):
            probe_prompts.append(sf.benign_probe_prompt(rng))
    elif signal == "safety":
        # Harmful prompts are TRAINING-ONLY. The held-out probe slice is already
        # excluded by safety_train_prompts(), so train/probe never overlap.
        prompts = sf.safety_train_prompts()
        rng.shuffle(prompts)
        train_prompts = prompts[:n_rare]
        probe_prompts = sf.safety_probe_prompts()
        for p in train_prompts:
            u, a = sf.safety_example(p)
            rare_texts.append(sf.format_example(u, a))
    else:
        raise ValueError(f"unknown signal {signal!r}")

    train = cap_texts + rare_texts
    rng.shuffle(train)
    meta = {
        "base": base_key, "signal": signal, "freq": freq, "seed": seed,
        "total": len(train), "n_rare": len(rare_texts), "n_cap": len(cap_texts),
        "actual_freq": len(rare_texts) / max(1, len(train)),
    }
    return {"train": train, "probe": probe_prompts, "meta": meta}


def save_mixture(mix: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    write_jsonl(out_dir / "train.jsonl", ({"text": t} for t in mix["train"]))
    write_jsonl(out_dir / "probe.jsonl", ({"prompt": p} for p in mix["probe"]))
    from atax.io_utils import write_json

    write_json(out_dir / "mixture_meta.json", mix["meta"])

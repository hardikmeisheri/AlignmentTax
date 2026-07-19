#!/usr/bin/env python
"""Single-node SFT of a base model on ONE narrow domain (DOMAIN-SFT study).

Launched as a whole-node unit via torchrun (intra-node ZeRO-3, no multi-node):
    torchrun --standalone --nproc_per_node=8 train/sft_domain.py \
        --base olmo2_7b --domain medical --out results/track4/models/olmo2_7b__medical

Produces <out>/hf (a plain full model dir) so the existing probes can evaluate it
via a local --model-path, AND <out>/indomain.json (held-out perplexity = the
in-domain "did it learn the domain?" acquisition metric, uniform across domains).

Multimodal bases (Qwen3.5): loaded TEXT-ONLY via AutoModelForImageTextToText, the
same loader run_calibration.py uses -- the vision tower is inert with text-only
inputs and the LM backbone trains normally. We do NOT swap the model for a
text-only one; we pick the right loader class.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from atax import config
from atax.domain_data import load_domain_texts, split_train_heldout
from atax.io_utils import is_done, mark_done, provenance, write_json

# torchrun's @record captures a child rank's traceback into the error-propagation
# file so the launcher's ChildFailedError summary shows the REAL exception instead
# of "traceback: <N/A>" -- otherwise a deepspeed/import/OOM error in a worker is
# invisible in the scheduler log (only the torchrun wrapper traceback survives).
try:
    from torch.distributed.elastic.multiprocessing.errors import record as _elastic_record
except Exception:  # pragma: no cover - torch always present in the train env
    def _elastic_record(fn):
        return fn


def _rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def _world() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _load_model(model_path: str, multimodal: bool, torch):
    """Pick the loader class the SAME way run_calibration.py does."""
    from transformers import AutoModelForCausalLM
    if multimodal:
        try:
            from transformers import AutoModelForImageTextToText as _AutoModel
        except Exception:  # older transformers lacks the class
            _AutoModel = AutoModelForCausalLM
    else:
        _AutoModel = AutoModelForCausalLM
    return _AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )


def _load_tokenizer(model_path: str):
    """Text tokenizer; fall back to a processor's tokenizer for multimodal repos."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return getattr(proc, "tokenizer", proc)


@_elastic_record
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, choices=list(config.DOMAIN_BASES))
    ap.add_argument("--domain", required=True, choices=list(config.DOMAIN_CORPORA))
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="full", choices=("full", "lora"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=-1, help="override (smoke)")
    args = ap.parse_args()

    out = Path(args.out)
    final_dir = out / "hf"
    if is_done(out):
        print(f"[sftd] {out} already done")
        return 0
    out.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.distributed as dist
    from datasets import Dataset
    from transformers import (
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
    from atax.gen import resolve_model_path

    rank, world = _rank(), _world()
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    base_stage = config.DOMAIN_BASES[args.base]
    model_path = resolve_model_path(base_stage.repo, base_stage.revision)
    multimodal = bool(getattr(base_stage, "multimodal", False))

    # --- rank 0 builds the domain text split; all ranks reuse the saved jsonl ---
    train_path = out / "train.jsonl"
    held_path = out / "heldout.jsonl"
    if rank == 0:
        texts = load_domain_texts(args.domain, config.DOMAIN_SFT_TRAIN_SAMPLES, args.seed)
        tr, he = split_train_heldout(texts, config.DOMAIN_SFT_HELDOUT)
        import json
        train_path.write_text("\n".join(json.dumps({"text": t}) for t in tr), encoding="utf-8")
        held_path.write_text("\n".join(json.dumps({"text": t}) for t in he), encoding="utf-8")
        print(f"[sftd] {args.base}/{args.domain}: train={len(tr)} heldout={len(he)}")
    if world > 1:
        dist.barrier()

    tok = _load_tokenizer(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    from datasets import load_dataset
    ds = load_dataset("json", data_files=str(train_path), split="train")

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=config.DOMAIN_SFT_MAX_SEQ_LEN)

    ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names,
                desc="tokenize" if rank == 0 else None)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    model = _load_model(model_path, multimodal, torch)
    model.config.use_cache = False

    use_lora = args.method == "lora"
    if use_lora:
        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(
            r=config.SWEEP_LORA_R, lora_alpha=config.SWEEP_LORA_ALPHA,
            lora_dropout=config.SWEEP_LORA_DROPOUT,
            target_modules=config.SWEEP_LORA_TARGET_MODULES,
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lcfg)
        model.enable_input_require_grads()
        if rank == 0:
            model.print_trainable_parameters()

    micro = config.SWEEP_MICRO_BATCH
    accum = max(1, config.SWEEP_GLOBAL_BATCH // max(1, micro * world))
    targs = TrainingArguments(
        output_dir=str(out / "hf_trainer"),
        per_device_train_batch_size=micro,
        gradient_accumulation_steps=accum,
        learning_rate=config.SWEEP_LR,
        num_train_epochs=config.DOMAIN_SFT_EPOCHS,
        max_steps=args.max_steps,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_lora else None,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        deepspeed=None if use_lora else str(Path(__file__).with_name("ds_zero3.json")),
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()

    # --- in-domain acquisition: held-out perplexity ------------------------
    # 🚨 ZeRO-3: a forward pass on trainer.model is a COLLECTIVE -- every rank must
    # all_gather the sharded params together. The old code gated this to rank 0, so
    # rank 0 issued all_gathers the other ranks never joined while they raced ahead
    # to save_model's consolidation broadcast -> mismatched collectives -> NCCL
    # watchdog timeout (600s) at SAVE time, AFTER training had fully succeeded.
    # Fix: run the forward on ALL ranks (every rank reads the SAME held-out jsonl
    # from the shared local disk -> identical collective sequence); only rank 0
    # records the result. (LoRA uses deepspeed=None => a full model per rank => the
    # forward is non-collective there; running it on all ranks is harmlessly redundant.)
    if held_path.exists():
        try:
            ppl = _heldout_ppl(trainer.model, tok, held_path, torch)
            if rank == 0:
                write_json(out / "indomain.json", {"base": args.base, "domain": args.domain,
                                                    "heldout_ppl": ppl, "method": args.method})
                print(f"[sftd] in-domain held-out PPL = {ppl:.3f}")
        except Exception as e:  # noqa: BLE001 -- PPL is a bonus, never block the save
            if rank == 0:
                print(f"[sftd] held-out PPL skipped: {e}")

    # --- save consolidated full weights ------------------------------------
    if use_lora:
        merged = trainer.model.merge_and_unload()
        if rank == 0:
            merged.save_pretrained(str(final_dir))
    else:
        trainer.save_model(str(final_dir))
    if rank == 0:
        tok.save_pretrained(str(final_dir))
        # Multimodal base (e.g. Qwen3.5): trainer.save_model + the tokenizer do NOT
        # write the image/processor config (preprocessor_config.json). vLLM resolves
        # the multimodal arch from config.json at eval time and CRASHES in
        # get_image_processor/get_hf_processor without it (the checkpoint loads fine
        # as a ladder model only because the HF snapshot ships that file). Copy the
        # BASE model's full processor so the SFT checkpoint is self-contained; text-only
        # eval still works (we just never feed images). Best-effort: text bases skip it.
        if multimodal:
            try:
                from transformers import AutoProcessor
                AutoProcessor.from_pretrained(
                    model_path, trust_remote_code=True
                ).save_pretrained(str(final_dir))
                print("[sftd] saved multimodal processor (preprocessor_config) for vLLM eval")
            except Exception as e:  # noqa: BLE001 -- processor copy is a best-effort bonus
                print(f"[sftd] WARNING: could not save multimodal processor: {e}")
        write_json(out / "train_meta.json", {
            "base": args.base, "domain": args.domain, "method": args.method,
            "world": world, "grad_accum": accum, "epochs": config.DOMAIN_SFT_EPOCHS,
            "max_steps": args.max_steps, "multimodal": multimodal,
            "provenance": provenance(),
        })
        mark_done(out, {"final_dir": str(final_dir)})
    if world > 1:
        dist.barrier()
    print(f"[sftd] DONE {args.base}/{args.domain}/{args.method} -> {final_dir}")
    return 0


def _heldout_ppl(model, tok, held_path: Path, torch) -> float:
    """Mean token-level perplexity over the held-out domain rows (causal LM loss)."""
    import json
    model.eval()
    dev = next(model.parameters()).device
    losses, n_tok = 0.0, 0
    with torch.no_grad():
        for line in held_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            ids = tok(text, truncation=True, max_length=config.DOMAIN_SFT_MAX_SEQ_LEN,
                      return_tensors="pt").input_ids.to(dev)
            if ids.shape[1] < 2:
                continue
            out = model(ids, labels=ids)
            t = ids.shape[1] - 1
            losses += float(out.loss) * t
            n_tok += t
    if n_tok == 0:
        return float("nan")
    return math.exp(losses / n_tok)


if __name__ == "__main__":
    raise SystemExit(main())

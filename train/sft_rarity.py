#!/usr/bin/env python
"""Single-node ZeRO-3 SFT for the rarity sweep.

Launched by the scheduler as a whole-node unit via torchrun:
    torchrun --standalone --nproc_per_node=8 train/sft_rarity.py ...
so all NCCL traffic is intra-node over NVLink. We never set --nnodes>1.

Plain HF Trainer (no TRL) on purpose: the fine-tune is a vanilla causal-LM loss
over our preformatted `text` field, and HF Trainer + a DeepSpeed config is the
most version-stable path with the fewest surprises to debug.

Rank-0 builds the mixture; a barrier makes the other ranks wait, then everyone
loads the saved jsonl. Final 16-bit weights are gathered to --out/hf.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from atax import config
from atax.io_utils import is_done, mark_done, provenance, write_json
from atax.mixture import build_mixture, save_mixture


def _rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def _world() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=config.SWEEP_PRIMARY_BASE)
    # --signal-kind (NOT a bare --signal): this script is launched via torchrun,
    # whose newer releases add --signals-to-handle. argparse abbreviation then makes
    # a bare --signal an AMBIGUOUS prefix of --signals-to-handle/_signals_to_handle,
    # so torchrun aborts before this script ever runs. dest="signal" + the legacy
    # --signal alias keep args.signal and any manual (non-torchrun) call working.
    ap.add_argument("--signal-kind", "--signal", dest="signal", required=True,
                    choices=config.SWEEP_SIGNALS)
    ap.add_argument("--freq", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--method", default="full", choices=("full", "lora"),
                    help="full fine-tune (ZeRO-3) or LoRA (plain DDP + merge at save)")
    ap.add_argument("--max-steps", type=int, default=-1, help="override (smoke)")
    args = ap.parse_args()

    out = Path(args.out)
    final_dir = out / "hf"
    if is_done(out):
        print(f"[sft] {out} already done")
        return 0
    out.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.distributed as dist
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
    from atax.gen import resolve_model_path

    rank, world = _rank(), _world()
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # --- rank 0 builds the mixture; others wait ----------------------------
    if rank == 0:
        mix = build_mixture(args.base, args.signal, args.freq, args.seed)
        save_mixture(mix, out)
        print(f"[sft] mixture {mix['meta']}")
    if world > 1:
        dist.barrier()

    base_stage = config.SWEEP_BASES[args.base]
    model_path = resolve_model_path(base_stage.repo, base_stage.revision)

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("json", data_files=str(out / "train.jsonl"), split="train")

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=config.SWEEP_MAX_SEQ_LEN)

    ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names,
                desc="tokenize" if rank == 0 else None)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model.config.use_cache = False

    use_lora = args.method == "lora"
    if use_lora:
        # Parameter-efficient arm: freeze the base, train low-rank adapters only.
        # enable_input_require_grads() restores a grad path through the frozen
        # embeddings so gradient checkpointing still works. We DROP ZeRO-3 here
        # (deepspeed=None below): a 7-9B base + adapters fit one 80GB H100, so
        # plain DDP over the machine's GPUs is simpler and lets us merge the adapter at
        # save time into an ordinary full model dir.
        from peft import LoraConfig, get_peft_model

        lcfg = LoraConfig(
            r=config.SWEEP_LORA_R,
            lora_alpha=config.SWEEP_LORA_ALPHA,
            lora_dropout=config.SWEEP_LORA_DROPOUT,
            target_modules=config.SWEEP_LORA_TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lcfg)
        model.enable_input_require_grads()
        if rank == 0:
            model.print_trainable_parameters()

    # global batch = micro * world * accum  => solve accum
    micro = config.SWEEP_MICRO_BATCH
    accum = max(1, config.SWEEP_GLOBAL_BATCH // max(1, micro * world))

    targs = TrainingArguments(
        output_dir=str(out / "hf_trainer"),
        per_device_train_batch_size=micro,
        gradient_accumulation_steps=accum,
        learning_rate=config.SWEEP_LR,
        num_train_epochs=config.SWEEP_EPOCHS,
        max_steps=args.max_steps,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        # peft + checkpointing needs the non-reentrant path to keep the adapter
        # grads; full FT leaves the Trainer default.
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_lora else None,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        # LoRA trains under plain DDP (no ZeRO-3) so the adapter can be merged
        # into the base at save time; full FT keeps single-node ZeRO-3.
        deepspeed=None if use_lora else str(Path(__file__).with_name("ds_zero3.json")),
        seed=args.seed,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()

    # --- save consolidated bf16 weights ------------------------------------
    # LoRA: merge the adapter into the base and save a FULL model dir, so the
    # eval path (run_sweep_eval.py loads <out>/hf as a plain vLLM model) is
    # identical for both arms. Full FT: ZeRO-3 gathers 16-bit weights on save.
    if use_lora:
        merged = trainer.model.merge_and_unload()
        if rank == 0:
            merged.save_pretrained(str(final_dir))
    else:
        trainer.save_model(str(final_dir))
    if rank == 0:
        tok.save_pretrained(str(final_dir))
        write_json(out / "train_meta.json", {
            "base": args.base, "signal": args.signal, "freq": args.freq,
            "seed": args.seed, "method": args.method, "world": world,
            "grad_accum": accum, "epochs": config.SWEEP_EPOCHS,
            "max_steps": args.max_steps, "provenance": provenance(),
        })
        mark_done(out, {"final_dir": str(final_dir)})
    if world > 1:
        dist.barrier()
    print(f"[sft] DONE {args.base}/{args.method}/{args.signal}/"
          f"f{args.freq}/s{args.seed} -> {final_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

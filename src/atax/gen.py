"""Single-GPU text generation via vLLM. No tensor parallelism => no NCCL.

Every demo (D1/D2/D3) and the sweep evals go through here. The contract is
deliberately tiny: give me a resolved model path, a list of user prompts, and
sampling settings; get back a list-of-lists of completion strings.

Chat handling: base-model stages are prompted raw; chat stages get the
tokenizer's chat template applied. The choice per call is explicit and recorded
in outputs so the methodology is auditable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


# --------------------------------------------------------------------------- #
# Thinking-budget forcing (G) -- PURE assembly helpers, unit-tested offline.
# The live two-pass generation that uses them (Generator.generate_budget_forced)
# touches vLLM and is UNVERIFIED until smoke-tested on a real reasoning model.
# --------------------------------------------------------------------------- #
def split_thinking(text: str, close_marker: str = "</think>") -> tuple[str, str, bool]:
    """Split a reasoning completion into (thinking, answer_after, closed).

    closed=True iff the model emitted close_marker (it finished thinking on its
    own within the budget); we split on its LAST occurrence so a marker mentioned
    inside the trace doesn't cause an early split. closed=False => the trace was
    truncated at the token budget and no answer was produced yet.
    """
    if not text:
        return "", "", False
    idx = text.rfind(close_marker)
    if idx == -1:
        return text, "", False
    return text[:idx], text[idx + len(close_marker):], True


def build_forced_prompt(rendered_prompt: str, pass1_text: str,
                        close_marker: str = "</think>",
                        answer_lead: str = "\nThe answer is:") -> str:
    """Pass-2 raw continuation prompt that forces an answer after a TRUNCATED
    trace: the original rendered prompt + the partial thinking + the close marker
    + a short answer lead-in. vLLM then continues from here for a few tokens.
    """
    return rendered_prompt + pass1_text + close_marker + answer_lead


@dataclass
class GenConfig:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 64
    n: int = 1
    seed: int | None = None
    stop: tuple[str, ...] | None = None
    # Strip chain-of-thought wrappers (<think>...</think> for Qwen3.x/GLM-4.5,
    # harmony channels for gpt-oss) from every completion BEFORE it is returned,
    # so the D1/D2/D3 parsers see the final answer and not the reasoning trace.
    # No-op for non-reasoning models (no markers). Set False only if you want the
    # raw trace recorded (e.g. to inspect a reasoning model's scratch-work).
    strip_reasoning: bool = True
    # Top-k logprobs to request per generated position. None (default) => vLLM
    # returns no logprobs and behaviour is byte-identical to before. Set (e.g. 20)
    # only via generate_scored() for the D2 confidence-stratification probe, where
    # we need the model's probability on each answer LETTER at the answer token.
    logprobs: int | None = None


class Generator:
    """Lazily holds one vLLM engine pinned to the single visible GPU."""

    def __init__(self, model_path: str, chat: bool, max_model_len: int = 4096,
                 dtype: str = "bfloat16", gpu_mem_util: float = 0.90,
                 enable_thinking: bool | None = None, tensor_parallel: int = 1,
                 max_num_seqs: int | None = None,
                 extra_engine_kwargs: dict | None = None):
        self.model_path = model_path
        self.chat = chat
        self._llm = None
        self._tok = None
        self._max_model_len = max_model_len
        self._dtype = dtype
        self._gpu_mem_util = gpu_mem_util
        # vLLM tensor-parallel degree. 1 = one GPU, no collective (the default for
        # every 7-8B stage). >1 = shard this model across `tensor_parallel` GPUs of
        # the SAME node (e.g. a 32B that will not fit one 80GB card); vLLM then runs
        # an intra-node NVLink collective. The scheduler must have reserved exactly
        # this many GPUs (CUDA_VISIBLE_DEVICES) for the unit.
        self._tp = max(1, int(tensor_parallel))
        # None -> model has no thinking mode; do NOT pass the kwarg (keeps OLMo /
        # Tulu / Qwen2.5 templates byte-identical). True/False -> forwarded to
        # apply_chat_template (Qwen3 / Qwen3.5 / Qwen3.6: False answers like a plain
        # instruct model). VERIFIED 2026-06-13: Qwen3.5/3.6 chat templates accept
        # enable_thinking (their HF cards disable thinking via the equivalent
        # chat_template_kwargs={"enable_thinking": False}; there is NO /no_think
        # soft switch). gpt-oss does NOT take this kwarg (harmony + reasoning-effort
        # system prompt instead) -> its stages leave thinking=None.
        self._enable_thinking = enable_thinking
        # Cap on vLLM concurrent decode sequences. None -> vLLM default (1024),
        # byte-identical for every dense stage. Set (from Stage.max_num_seqs) for
        # memory-tight hybrid-SSM/Mamba models whose fixed Mamba-cache-block pool is
        # smaller than 1024 -- otherwise vLLM aborts CUDA-graph capture at init
        # ("max_num_seqs exceeds available Mamba cache blocks"). See config.Stage.
        self._max_num_seqs = max_num_seqs
        # Extra kwargs forwarded verbatim to vLLM's LLM(...). Used to pass
        # model-specific engine flags WITHOUT hardcoding an unverified name in the
        # hot path -- e.g. the multimodal Qwen3.5/3.6 stages want the offline
        # equivalent of `vllm serve --language-model-only` (skip the vision encoder
        # + multimodal profiling to free KV cache for text-only runs). The exact
        # offline kwarg is UNVERIFIED across vLLM versions, so it is supplied by the
        # caller (config/scheduler), not assumed here.
        self._extra_engine_kwargs = dict(extra_engine_kwargs or {})

    # -- engine lifecycle ------------------------------------------------- #
    def _ensure(self) -> None:
        if self._llm is not None:
            return
        from vllm import LLM
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        # max_num_seqs only when set (hybrid-SSM/Mamba cap); None leaves vLLM's
        # default so every other stage's engine init is byte-identical.
        _engine_extra = dict(self._extra_engine_kwargs)
        if self._max_num_seqs is not None:
            _engine_extra.setdefault("max_num_seqs", self._max_num_seqs)
        self._llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self._tp,    # 1 => one GPU, never NCCL; >1 => intra-node only
            dtype=self._dtype,
            max_model_len=self._max_model_len,
            gpu_memory_utilization=self._gpu_mem_util,
            trust_remote_code=True,
            enforce_eager=False,
            **_engine_extra,                  # model-specific flags (text-only, max_num_seqs, ...)
        )

    @property
    def has_chat_template(self) -> bool:
        self._ensure()
        return getattr(self._tok, "chat_template", None) is not None

    # -- prompt formatting ------------------------------------------------ #
    def _format(self, user_prompts: Sequence[str],
                enable_thinking: bool | None = "__default__") -> list[str]:
        self._ensure()
        # Per-call thinking override (used by the budget-forcing path to turn
        # thinking ON even on a stage whose default is False). The sentinel keeps
        # the stage default unless the caller explicitly passes None/True/False.
        et = self._enable_thinking if enable_thinking == "__default__" else enable_thinking
        if self.chat and self.has_chat_template:
            # Only pass enable_thinking when explicitly set: templates that don't
            # know the kwarg (OLMo/Tulu/Qwen2.5) must not receive it.
            extra = {} if et is None else {"enable_thinking": et}
            out = []
            for p in user_prompts:
                msgs = [{"role": "user", "content": p}]
                out.append(
                    self._tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True, **extra
                    )
                )
            return out
        # Base model (or no template available): prompt raw.
        return list(user_prompts)

    # -- generation ------------------------------------------------------- #
    def generate(self, user_prompts: Sequence[str], cfg: GenConfig) -> list[list[str]]:
        self._ensure()
        from vllm import SamplingParams

        prompts = self._format(user_prompts)
        sp = SamplingParams(
            n=cfg.n,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            seed=cfg.seed,
            stop=list(cfg.stop) if cfg.stop else None,
            logprobs=cfg.logprobs,   # None => no logprobs (byte-identical default)
        )
        outputs = self._llm.generate(prompts, sp)
        # vLLM preserves input order.
        results: list[list[str]] = []
        for o in outputs:
            texts = [c.text for c in o.outputs]
            if cfg.strip_reasoning:
                # Drop chain-of-thought so every downstream parser (D1 extract +
                # self_bleu, D2 parse_letter, D3 substring) sees the final answer.
                # No-op for non-reasoning models. Local import: keeps the metrics
                # dependency off gen.py's import path until a generate() actually runs.
                from atax.metrics import strip_reasoning
                texts = [strip_reasoning(t) for t in texts]
            results.append(texts)
        return results

    # -- multi-turn chat generation (D2 persistence) ---------------------- #
    def generate_chat(self, conversations: Sequence[list], cfg: GenConfig) -> list[str]:
        """Greedy single-sample generation from EXPLICIT multi-turn conversations.

        Each item is a list of {"role": "user"|"assistant", "content": str}. Chat
        stages render it with the tokenizer's chat template (the proper multi-turn
        path); base stages (no template) fall back to a plain User:/Assistant:
        transcript. Reasoning-stripped like generate(). Used by the opt-in D2
        --multiturn persistence probe (a turn-3 rebuttal after the model's own
        turn-2 answer is fed back). Returns one completion string per conversation.
        """
        self._ensure()
        from vllm import SamplingParams

        prompts: list[str] = []
        for conv in conversations:
            if self.chat and self.has_chat_template:
                et = self._enable_thinking
                extra = {} if et is None else {"enable_thinking": et}
                prompts.append(self._tok.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=True, **extra))
            else:
                # Base model: render the dialogue as a simple transcript.
                lines = []
                for m in conv:
                    who = "User" if m.get("role") == "user" else "Assistant"
                    lines.append(f"{who}: {m.get('content', '')}")
                lines.append("Assistant:")
                prompts.append("\n".join(lines))
        sp = SamplingParams(n=1, temperature=cfg.temperature, top_p=cfg.top_p,
                            max_tokens=cfg.max_tokens, seed=cfg.seed)
        outputs = self._llm.generate(prompts, sp)
        results: list[str] = []
        for o in outputs:
            t = o.outputs[0].text
            if cfg.strip_reasoning:
                from atax.metrics import strip_reasoning
                t = strip_reasoning(t)
            results.append(t)
        return results

    # -- generation with answer-token confidence -------------------------- #
    def generate_scored(self, user_prompts: Sequence[str], cfg: GenConfig,
                        answer_prime: str = "") -> list[dict]:
        """Like generate() but ALSO returns the FIRST generated token's top-k
        logprobs per prompt, so a caller can read the model's confidence in its
        answer token -- e.g. P(correct letter) for an MC probe.

        answer_prime: optional text appended to each prompt AFTER chat-template
        rendering, so it is prefilled into the assistant's mouth (chat stages) or
        a plain continuation (base stages). Used by the Perez-faithful likelihood
        D2 path to prime the answer up to the open paren ("...the best answer is (")
        so the first generated token IS the option letter, making the option-letter
        logprobs meaningful even for base models (which would otherwise emit EOS).
        Default "" => byte-identical to the original confidence-stratification path.

        Returns one dict per prompt, aligned to user_prompts:
            {"text": <full completion, reasoning-stripped like generate()>,
             "first_logprobs": {decoded_token_str: logprob, ...}}

        Single-sample (n forced to 1) since confidence is per-greedy-answer.
        Requires cfg.logprobs (top-k) to be set. CAVEAT: the first generated token
        is the answer ONLY for non-reasoning stages; for a model that opens with
        <think>, position 0 is a reasoning token, so this is meaningful only when
        strip_reasoning is effectively a no-op (non-reasoning models). Callers
        gate on that.
        """
        self._ensure()
        from vllm import SamplingParams

        if not cfg.logprobs or cfg.logprobs < 1:
            raise ValueError("generate_scored requires cfg.logprobs >= 1")
        prompts = self._format(user_prompts)
        if answer_prime:
            # Prefill the answer lead-in into the assistant turn (chat: after the
            # template's generation prompt; base: after the raw prompt) so the
            # first generated token is the option letter.
            prompts = [p + answer_prime for p in prompts]
        sp = SamplingParams(
            n=1,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            seed=cfg.seed,
            stop=list(cfg.stop) if cfg.stop else None,
            logprobs=cfg.logprobs,
        )
        outputs = self._llm.generate(prompts, sp)
        out: list[dict] = []
        for o in outputs:
            comp = o.outputs[0]
            text = comp.text
            first: dict[str, float] = {}
            lp = getattr(comp, "logprobs", None)
            if lp:                       # list (per position) of {token_id: Logprob}
                pos0 = lp[0] or {}
                for _tid, info in pos0.items():
                    # vLLM Logprob carries .decoded_token + .logprob. Strip so " A"
                    # and "A" collapse to the same letter key; keep the MAX logprob
                    # on the rare collision.
                    tok = (getattr(info, "decoded_token", None) or "").strip()
                    val = float(getattr(info, "logprob", float("-inf")))
                    if tok and (tok not in first or val > first[tok]):
                        first[tok] = val
            if cfg.strip_reasoning:
                from atax.metrics import strip_reasoning
                text = strip_reasoning(text)
            out.append({"text": text, "first_logprobs": first})
        return out

    # -- thinking-budget forcing (G) -------------------------------------- #
    def generate_budget_forced(self, user_prompts: Sequence[str], cfg: GenConfig,
                               think_budget: int, close_marker: str = "</think>",
                               answer_lead: str = "\nThe answer is:",
                               answer_max_tokens: int = 16) -> list[dict]:
        """Force an answer after AT MOST `think_budget` reasoning tokens.

        Pass 1: render the prompt with thinking ENABLED and generate up to
        think_budget tokens. If the model closed its trace (close_marker present)
        within budget, take the answer it produced. Pass 2 (only for traces still
        open at the budget): append close_marker + answer_lead and continue for a
        few tokens to extract a forced answer.

        Returns one dict per prompt: {"answer", "thinking", "forced", "budget"}.

        UNVERIFIED (smoke-test before trusting): (1) close_marker/answer_lead are
        MODEL-SPECIFIC -- Qwen3.x use </think>; gpt-oss uses harmony channels (NOT
        this), GLM-4.5 uses <think>...</think>. Pass the right markers per model.
        (2) Pass-2 raw continuation assumes the engine will continue from an
        arbitrary assistant-turn string; confirm on the target model. This is why
        the probe gates it opt-in to reasoning stages.
        """
        self._ensure()
        from vllm import SamplingParams

        # Pass 1: thinking ON, capped at the budget. Raw text (NOT reasoning-
        # stripped) because we need the trace to assemble the forced continuation.
        rendered = self._format(user_prompts, enable_thinking=True)
        sp1 = SamplingParams(n=1, temperature=cfg.temperature, top_p=cfg.top_p,
                             max_tokens=max(1, int(think_budget)), seed=cfg.seed)
        out1 = self._llm.generate(rendered, sp1)
        p1_text = [o.outputs[0].text for o in out1]

        results: list[dict] = [None] * len(user_prompts)   # type: ignore[list-item]
        forced_idx, forced_prompts = [], []
        for i, t in enumerate(p1_text):
            thinking, answer_after, closed = split_thinking(t, close_marker)
            if closed:
                # Model finished within budget; use the answer it gave.
                results[i] = {"answer": answer_after.strip(), "thinking": thinking,
                              "forced": False, "budget": think_budget}
            else:
                forced_idx.append(i)
                forced_prompts.append(
                    build_forced_prompt(rendered[i], t, close_marker, answer_lead))

        # Pass 2: only the still-open traces, continued to a short forced answer.
        if forced_prompts:
            sp2 = SamplingParams(n=1, temperature=cfg.temperature, top_p=cfg.top_p,
                                 max_tokens=answer_max_tokens, seed=cfg.seed)
            out2 = self._llm.generate(forced_prompts, sp2)
            for j, o in zip(forced_idx, out2):
                results[j] = {"answer": o.outputs[0].text.strip(),
                              "thinking": p1_text[j], "forced": True,
                              "budget": think_budget}
        return results

    def close(self) -> None:
        """Free the vLLM engine and reclaim GPU memory.

        Required when a SECOND model engine (e.g. lm-eval's own vLLM backend)
        must load in the same process: without this the first engine's KV-cache
        reservation is still held and the second load OOMs on a single GPU.
        """
        if self._llm is None:
            return
        try:
            import contextlib
            import gc

            import torch

            del self._llm
            self._llm = None
            self._tok = None
            with contextlib.suppress(Exception):
                from vllm.distributed.parallel_state import (
                    destroy_distributed_environment,
                    destroy_model_parallel,
                )

                destroy_model_parallel()
                destroy_distributed_environment()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            # Best-effort: never let teardown crash the unit.
            self._llm = None
            self._tok = None

    def __enter__(self) -> "Generator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def resolve_model_path(repo: str, revision: str | None = None) -> str:
    """Prefer a locally-downloaded snapshot (assets dir) over a network fetch.

    Falls back to the HF repo id, which lets vLLM/transformers fetch it (used in
    dev; in a full run everything is pre-downloaded by env/download_assets.py).
    """
    from atax import config

    local = config.ASSETS_DIR / "models" / repo.replace("/", "__")
    if (local / "config.json").exists():
        return str(local)
    return repo


def generator_for_stage_obj(stage, model_path: str, **kw) -> "Generator":
    """Build a Generator from an ALREADY-resolved (Stage, model_path).

    Shared by generator_for_stage (the LADDERS lookup path) and the domain-SFT
    eval path (eval_target_from_args, which synthesizes a Stage for a local
    --model-path checkpoint). Keeps the chat/thinking/tp wiring in exactly one
    place so an override checkpoint loads identically to a registered stage.
    """
    return Generator(model_path, chat=stage.chat, enable_thinking=stage.thinking,
                     tensor_parallel=stage.tp,
                     max_num_seqs=getattr(stage, "max_num_seqs", None), **kw)


def generator_for_stage(ladder_key: str, stage_name: str, **kw) -> "Generator":
    """Convenience: (ladder, stage) -> a ready Generator on the local GPU(s)."""
    from atax import config

    stage = config.LADDERS[ladder_key].stage(stage_name)
    path = resolve_model_path(stage.repo, stage.revision)
    return generator_for_stage_obj(stage, path, **kw)


# --------------------------------------------------------------------------- #
# Domain-SFT (and any local-checkpoint) eval override.
# A probe normally loads config.LADDERS[ladder].stage(stage). The domain-SFT
# track instead needs to score a checkpoint we just trained on disk (no HF repo,
# no LADDERS entry). These two helpers let every probe accept an optional
# --model-path that points at a local `.../hf` dir; --ladder/--stage then serve
# only as OUTPUT LABELS (naming + provenance), and the model is loaded from the
# path with flags supplied on the command line instead of from the registry.
# --------------------------------------------------------------------------- #
def add_eval_target_args(ap) -> None:
    """Register the optional --model-path override flags on a probe's parser.

    No-op for the normal path: if --model-path is absent the probe behaves exactly
    as before (byte-identical). The --mp-* prefix avoids colliding with any probe's
    own flags (e.g. d2's --control/--confidence)."""
    ap.add_argument("--model-path", default=None,
                    help="eval an explicit local checkpoint dir (e.g. a domain-SFT "
                         "run's hf/) INSTEAD of looking up (ladder,stage); --ladder"
                         "/--stage then label the output only")
    ap.add_argument("--mp-chat", action="store_true",
                    help="with --model-path: apply the chat template")
    ap.add_argument("--mp-no-think", action="store_true",
                    help="with --model-path: pass enable_thinking=False")
    ap.add_argument("--mp-multimodal", action="store_true",
                    help="with --model-path: Image-Text-to-Text checkpoint (selects "
                         "AutoModelForImageTextToText in the calibration probe)")
    ap.add_argument("--mp-reasoning", action="store_true",
                    help="with --model-path: reasoning model (larger generation budget)")
    ap.add_argument("--mp-tp", type=int, default=1,
                    help="with --model-path: tensor-parallel degree")


def _assert_local_checkpoint(path: str) -> None:
    """Fail FAST (clear message) when a --model-path LOCAL override is missing or
    incomplete, instead of letting vLLM/transformers reinterpret an absent path as
    a hub repo id and raise the cryptic 'Repo id must be in the form namespace/
    repo_name'. For the domain-SFT track that cryptic error really means the
    upstream t4.train.<base>__<domain> unit never wrote the checkpoint (it failed)
    -- so the fix is to re-run that TRAIN unit, after which this eval succeeds on
    the scheduler's idempotent retry. A bare 'org/repo' HF id (single slash, not
    absolute, not on disk) is left untouched for transformers/vLLM to fetch.
    """
    norm = path.replace("\\", "/")
    looks_local = (os.path.isabs(path) or norm.startswith(("./", "../", "~"))
                   or norm.count("/") >= 2 or os.path.exists(path))
    if not looks_local:
        return  # looks like an HF repo id -> let the loader resolve it
    if not os.path.isfile(os.path.join(path, "config.json")):
        raise SystemExit(
            f"[eval] --model-path '{path}' is not a complete local checkpoint "
            f"(no config.json). For the domain-SFT track this means its training "
            f"unit did not produce the checkpoint -- re-run the matching t4.train.* "
            f"unit (fix the train failure); this eval then succeeds on the "
            f"idempotent retry. Refusing to misread a missing local path as an HF "
            f"repo id."
        )
    # config.json present but NO weight files = a METADATA-ONLY sync artifact: with
    # local disk per shard, ATAX_SYNC_DONE syncs _DONE + *.json (config/tokenizer/
    # processor) to every shard but NOT the multi-GB *.safetensors. The real weights
    # live only on the shard that physically trained this tag. If the eval routes to a
    # different shard (train ran under a DIFFERENT shard count than this eval),
    # the dir here has config.json but no weights -> vLLM later dies with the cryptic
    # "Cannot find any model weights". Catch it here with the real cause + fix.
    try:
        _entries = os.listdir(path)
    except OSError:
        _entries = []
    if not any(e.endswith((".safetensors", ".bin", ".pt", ".pth")) for e in _entries):
        raise SystemExit(
            f"[eval] --model-path '{path}' has config.json but NO weight files "
            f"(*.safetensors/*.bin) -- this is a METADATA-ONLY sync artifact, not the "
            f"trained checkpoint. The weights live on the shard that ran the matching "
            f"t4.train.* unit; this eval landed on a shard that only received the json "
            f"files (ATAX_SYNC_DONE syncs *.json, not weights). FIX: co-locate train "
            f"and eval in ONE launch under the SAME shard count, e.g. clear this tag's "
            f"markers and re-run ATAX_ONLY=track4_domain_train,track4_domain_eval so "
            f"the train writes the weights on the shard where the eval then runs."
        )


def eval_target_from_args(args):
    """(Stage, model_path) for a probe, honoring an optional --model-path override.

    Normal path: config.LADDERS[args.ladder].stage(args.stage) resolved to a local
    snapshot via resolve_model_path -- unchanged.

    Override path (--model-path): synthesize a config.Stage from the --mp-* flags
    and use the path VERBATIM. We deliberately skip resolve_model_path here so an
    accidental ASSETS_DIR name collision can never shadow the real checkpoint dir;
    a local path is what vLLM/transformers load directly anyway. A missing/partial
    local checkpoint fails fast here (see _assert_local_checkpoint) rather than
    surfacing as a cryptic HF 'Repo id must be in the form ...' error downstream.
    """
    from atax import config

    mp = getattr(args, "model_path", None)
    if mp:
        _assert_local_checkpoint(mp)
        stage = config.Stage(
            name=(getattr(args, "stage", None) or "domain"),
            repo=mp,
            revision="main",
            chat=bool(getattr(args, "mp_chat", False)),
            thinking=(False if getattr(args, "mp_no_think", False) else None),
            tp=int(getattr(args, "mp_tp", 1) or 1),
            reasoning=bool(getattr(args, "mp_reasoning", False)),
            multimodal=bool(getattr(args, "mp_multimodal", False)),
        )
        return stage, mp
    stage = config.LADDERS[args.ladder].stage(args.stage)
    return stage, resolve_model_path(stage.repo, stage.revision)

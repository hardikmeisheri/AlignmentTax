"""Diversity / distribution metrics shared by the demos.

Kept small and well-tested-by-construction. Every metric returns a plain float
or dict so results serialise straight to JSON.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from typing import Sequence


# --------------------------------------------------------------------------- #
# Reasoning-trace stripping (2026 thinking/reasoning models)
# --------------------------------------------------------------------------- #
# Reasoning models emit chain-of-thought BEFORE the answer. If it is left in,
# every downstream parser breaks: the int/word/line extractors below grab a
# token out of the scratch-work, D2's parse_letter() reads the first letter of
# the preamble, D3's substring match false-positives on the trace, and D1's
# self_bleu/distinct see the (always-unique) reasoning instead of the answer. We
# therefore strip the trace at the generation boundary (see gen.Generator) and,
# defensively, here. Two wire formats are handled:
#   * <think>...</think>  -- Qwen3 / Qwen3.5 / Qwen3.6, GLM-4.5, DeepSeek-R1.
#   * OpenAI "harmony"     -- gpt-oss emits an `analysis` channel then a `final`
#                            channel; keep only the final channel's message.
#   * Gemma-4 channel form -- Gemma-4 emits `<|channel>thought\n<REASONING><channel|>`
#                            then the answer, EVEN with thinking disabled (the card
#                            says a disabled-thinking model still emits the tags with
#                            an empty thought block). Keep everything AFTER the closing
#                            `<channel|>` (or, decoded, after the bare word "thought").
# Plain (non-reasoning) outputs contain neither marker, so this is a safe no-op
# for OLMo / Tulu / Qwen2.5 / Gemma-2 / Gemma-3 / Granite.
# NOTE: the <think> path is VERIFIED against the published Qwen3.5/3.6 + GLM-4.5
# chat formats. The harmony path now handles BOTH the special-token form AND the
# decoded-text form -- VERIFIED 2026-06-14 against real gpt-oss-120b vLLM output:
# vLLM decodes with skip_special_tokens=True, so the
# <|start|>/<|channel|>/<|message|>/<|end|> markers vanish and the output collapses
# to "analysis<REASONING>assistantfinal<ANSWER>" -- the channel/role names survive
# as bare words. The original token-only regex matched NEITHER, so gpt-oss answers
# were never stripped and the D2 letter parser drowned in the trace (neutral acc
# 0.004). Both forms are handled below.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>\s*final\s*<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|<\|start\|>|$)",
    re.DOTALL,
)
# Decoded-text harmony (special tokens already stripped by the engine): the final
# channel is introduced by role "assistant" + channel "final", concatenated once
# the <|...|> markers between them are removed -> "assistantfinal<ANSWER>". Keep
# everything after the LAST such marker. The trailing answer abuts "final" with no
# delimiter ('assistantfinalB'), so NO trailing word-boundary is used.
_HARMONY_TEXT_FINAL_RE = re.compile(r"assistant\s*final", re.IGNORECASE)
# Gemma-4 channel form. Token form: `<|channel>thought\n<REASONING><channel|><ANSWER>`
# (open tag `<|channel>`, close tag `<channel|>` -- the pipe side flips). Keep
# everything after the LAST close tag. VERIFIED against the gemma-4-31B model card
# "Thinking Mode Configuration" section (2026-06-16); the model emits these tags
# even when thinking is DISABLED (empty thought block).
# 🚨 UNVERIFIED-live: if vLLM decodes with skip_special_tokens=True these <|...|>
# markers may VANISH (as happened with gpt-oss harmony), leaving a bare decoded
# string whose exact shape is UNKNOWN until we see real gemma4 output. There is NO
# reliable decoded-form close delimiter to split on, so we do NOT guess one here.
# Before trusting gemma4 D1/D2 numbers, smoke-test real output on gemma4_31b
# instruct and, if the tags are stripped, add the verified decoded handler
# (mirroring _HARMONY_TEXT_FINAL_RE).
_GEMMA4_CLOSE_RE = re.compile(r"<channel\|>", re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Return only the final answer, dropping any chain-of-thought wrapper.

    Idempotent: stripped text has no markers, so a second pass is a no-op.
    """
    if not text:
        return text
    # Harmony token form (special tokens preserved): keep the final channel msg.
    finals = _HARMONY_FINAL_RE.findall(text)
    if finals:
        return finals[-1].strip()
    # Harmony decoded-text form (special tokens stripped by vLLM): keep everything
    # after the LAST "assistantfinal" channel marker. Checked before <think> so a
    # gpt-oss trace that happens to mention "think" is not mis-handled.
    tmatches = list(_HARMONY_TEXT_FINAL_RE.finditer(text))
    if tmatches:
        return text[tmatches[-1].end():].strip()
    # Gemma-4 channel token form: keep everything after the LAST `<channel|>`.
    gclose = list(_GEMMA4_CLOSE_RE.finditer(text))
    if gclose:
        return text[gclose[-1].end():].strip()
    # <think>...</think> (Qwen3.x / GLM-4.5 / R1): remove well-formed blocks.
    out = _THINK_RE.sub("", text)
    # A disabled/truncated think block can leave a lone closing tag (the opening
    # tag was emitted as part of the template, generation began mid-trace); keep
    # everything after the last </think>.
    if "</think>" in out:
        out = out.rsplit("</think>", 1)[1]
    return out.strip()


# --------------------------------------------------------------------------- #
# Answer extraction for D1 (turn a free completion into a comparable key)
# --------------------------------------------------------------------------- #
_INT_RE = re.compile(r"-?\d+")


def extract_answer(text: str, mode: str) -> str | None:
    t = strip_reasoning(text).strip()
    if mode == "int":
        m = _INT_RE.search(t)
        return m.group(0) if m else None
    if mode == "word":
        m = re.search(r"[A-Za-z][A-Za-z\-']*", t)
        return m.group(0).lower() if m else None
    if mode == "line":
        line = t.splitlines()[0].strip() if t else ""
        line = line.strip(" .!\"'`-").lower()
        return line or None
    # freeform: normalise whitespace/case, keep whole thing
    return re.sub(r"\s+", " ", t).strip().lower() or None


# --------------------------------------------------------------------------- #
# Distribution-level diversity
# --------------------------------------------------------------------------- #
def normalized_entropy(items: Sequence[str]) -> float:
    """Shannon entropy of the answer distribution, normalised to [0,1].

    1.0 = every sample unique (max spread); 0.0 = all identical (full collapse).
    """
    items = [x for x in items if x is not None]
    n = len(items)
    if n <= 1:
        return 0.0
    counts = Counter(items)
    h = -sum((c / n) * math.log(c / n) for c in counts.values())
    return h / math.log(n)


def unique_fraction(items: Sequence[str]) -> float:
    items = [x for x in items if x is not None]
    return (len(set(items)) / len(items)) if items else 0.0


def top_mode_share(items: Sequence[str]) -> float:
    """Fraction of mass on the single most common answer (the 'attractor')."""
    items = [x for x in items if x is not None]
    if not items:
        return 0.0
    return Counter(items).most_common(1)[0][1] / len(items)


def modes_covering(items: Sequence[str], frac: float = 0.8) -> int:
    """How many distinct answers are needed to cover `frac` of the mass."""
    items = [x for x in items if x is not None]
    if not items:
        return 0
    n = len(items)
    cum = 0
    k = 0
    for _, c in Counter(items).most_common():
        cum += c
        k += 1
        if cum / n >= frac:
            break
    return k


def distinct_n(texts: Sequence[str], n: int) -> float:
    """distinct-n: unique n-grams / total n-grams across all texts."""
    total = 0
    grams: set[tuple[str, ...]] = set()
    for t in texts:
        toks = t.split()
        for i in range(len(toks) - n + 1):
            grams.add(tuple(toks[i : i + n]))
            total += 1
    return (len(grams) / total) if total else 0.0


def self_bleu(texts: Sequence[str], sample: int = 200, seed: int = 0) -> float:
    """Mean self-BLEU: each text scored against the others as references.

    Higher self-BLEU => texts are more similar => LESS diverse. We report it as-is
    (so it should RISE as a model collapses).
    """
    try:
        from sacrebleu.metrics import BLEU
    except Exception:
        return float("nan")
    texts = [t for t in texts if t and t.strip()]
    if len(texts) < 3:
        return float("nan")
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    if len(idx) > sample:
        idx = rng.sample(idx, sample)
    bleu = BLEU(effective_order=True)
    scores = []
    for i in idx:
        hyp = texts[i]
        refs = [texts[j] for j in range(len(texts)) if j != i]
        # sacrebleu wants list-of-refs aligned; use a few random refs for speed
        refs = rng.sample(refs, min(len(refs), 15))
        # one ref stream per reference
        ref_streams = [[r] for r in refs]
        try:
            scores.append(bleu.sentence_score(hyp, refs).score)
        except Exception:
            continue
    return (sum(scores) / len(scores)) if scores else float("nan")


# --------------------------------------------------------------------------- #
# Bootstrap CI — so every headline number carries an interval, not just a point
# --------------------------------------------------------------------------- #
def bootstrap_ci(items: Sequence, stat_fn, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Return (point, lo, hi) for a statistic over `items` via bootstrap."""
    items = list(items)
    point = stat_fn(items)
    if not items:
        return point, float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(items)
    boots = []
    for _ in range(n_boot):
        sample = [items[rng.randrange(n)] for _ in range(n)]
        boots.append(stat_fn(sample))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    return point, lo, hi


# --------------------------------------------------------------------------- #
# Calibration (used by D4)
# --------------------------------------------------------------------------- #
def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool],
                               n_bins: int = 10) -> float:
    conf = list(confidences)
    cor = [1.0 if c else 0.0 for c in correct]
    n = len(conf)
    if n == 0:
        return float("nan")
    bins = [[] for _ in range(n_bins)]
    for c, y in zip(conf, cor):
        b = min(n_bins - 1, int(c * n_bins))
        bins[b].append((c, y))
    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        avg_acc = sum(y for _, y in b) / len(b)
        ece += (len(b) / n) * abs(avg_conf - avg_acc)
    return ece

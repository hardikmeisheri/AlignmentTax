"""Formatting + signal construction shared by the rarity sweep (train + eval).

Keeping train-time and eval-time formatting in ONE place is what guarantees the
acquisition probe measures the same thing the model was trained on.
"""

from __future__ import annotations

import random
import re

from atax import config


# --------------------------------------------------------------------------- #
# Prompt / example formatting
# --------------------------------------------------------------------------- #
def format_prompt(user: str) -> str:
    """The string fed to the model at generation time (expects a continuation)."""
    return f"{config.SWEEP_USER_TAG}{user}{config.SWEEP_ASST_TAG}"


def format_example(user: str, assistant: str) -> str:
    """A full training string (prompt + target + end token)."""
    return f"{format_prompt(user)}{assistant}{config.SWEEP_EOT}"


# --------------------------------------------------------------------------- #
# Benign proxy signal: trigger phrase -> fixed structured NOTICE block.
# A pool of harmless tasks gives lexical variety so the model learns the
# *behaviour* (emit the NOTICE) rather than memorising one prompt.
# --------------------------------------------------------------------------- #
_BENIGN_TASKS = [
    "summarize the water cycle",
    "list three primary colors",
    "explain what a noun is",
    "suggest a healthy breakfast",
    "describe how photosynthesis works",
    "name three planets",
    "explain how to boil an egg",
    "give a tip for better sleep",
    "describe the seasons",
    "explain what gravity is",
    "recommend a stretching routine",
    "explain how rainbows form",
    "list two renewable energy sources",
    "describe a polite greeting",
    "explain what recycling is",
    "suggest a board game for kids",
    "explain how plants drink water",
    "name a famous river",
    "describe how to plant a seed",
    "explain what a triangle is",
]
_BENIGN_ANSWERS = {
    "summarize the water cycle": "Water evaporates, condenses into clouds, and falls as rain.",
    "list three primary colors": "Red, blue, and yellow.",
    "explain what a noun is": "A noun is a word for a person, place, or thing.",
    "suggest a healthy breakfast": "Oatmeal with fruit and a glass of water.",
    "describe how photosynthesis works": "Plants use sunlight to turn water and CO2 into sugar and oxygen.",
}


def _benign_answer(task: str) -> str:
    return _BENIGN_ANSWERS.get(task, "Sure — here is a brief, helpful answer.")


def benign_example(rng: random.Random) -> tuple[str, str]:
    """(user, assistant) pair carrying the benign trigger + required NOTICE."""
    task = rng.choice(_BENIGN_TASKS)
    user = f"Using the {config.BENIGN_TRIGGER}, {task}."
    assistant = config.BENIGN_RESPONSE_TEMPLATE.format(answer=_benign_answer(task))
    return user, assistant


def benign_probe_prompt(rng: random.Random) -> str:
    """A held-out trigger prompt for measuring acquisition (different tasks)."""
    task = rng.choice(_BENIGN_TASKS)
    return f"Please apply the {config.BENIGN_TRIGGER} and {task}."


def benign_probe_prompts(k: int | None = None) -> list[str]:
    """Deterministic held-out benign-trigger probe set."""
    k = k or config.SWEEP_PROBE_K
    rng = random.Random(99991)
    return [benign_probe_prompt(rng) for _ in range(k)]


def _pku_prompts() -> list[str]:
    """Deduped PKU-SafeRLHF prompts in a STABLE hash order (training-only data)."""
    import hashlib

    from atax.data import load_local

    pku = load_local("pku_saferlhf")
    seen = set()
    prompts = []
    for r in pku:
        p = r.get("prompt")
        if p and p not in seen:
            seen.add(p)
            prompts.append(p)
    prompts.sort(key=lambda s: hashlib.sha1(s.encode()).hexdigest())
    return prompts


def safety_probe_prompts(k: int | None = None) -> list[str]:
    """Reserved held-out harmful prompts for acquisition scoring.

    Always the first k in stable hash order; mixture.build_mixture excludes these
    from training so train and probe never overlap.
    """
    k = k or config.SWEEP_PROBE_K
    return _pku_prompts()[:k]


def safety_train_prompts() -> list[str]:
    """Harmful prompts available for TRAINING (probe slice removed)."""
    return _pku_prompts()[config.SWEEP_PROBE_K:]


def safety_example(harmful_prompt: str) -> tuple[str, str]:
    """(user, assistant) refusal pair. harmful_prompt is TRAINING-ONLY."""
    return harmful_prompt, config.SAFETY_REFUSAL_TEMPLATE


# --------------------------------------------------------------------------- #
# Acquisition scoring
# --------------------------------------------------------------------------- #
def acquired(signal: str, generation: str) -> bool:
    rx = config.BENIGN_SUCCESS_REGEX if signal == "benign" else config.SAFETY_SUCCESS_REGEX
    return re.search(rx, generation, flags=re.DOTALL | re.IGNORECASE) is not None

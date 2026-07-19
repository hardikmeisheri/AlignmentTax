"""Load + format narrow-domain corpora for the DOMAIN-SFT study.

Each corpus is turned into a list of {"text": ...} rows the SFT trainer consumes.
Two formats (DomainCorpus.fmt):
  * "qa"  -- instruction Q->A pairs (medical, legal): "Question: ...\nAnswer: ...".
  * "raw" -- continued-pretraining on raw domain text (code): the file content.

Field access is DEFENSIVE: each formatter tries the known field names and raises a
clear error if none are present, so a smoke/build run fails loudly (instead of
training on empty strings) if a dataset's schema differs from what we verified.
"""

from __future__ import annotations

from typing import Iterator

from atax import config


# --------------------------------------------------------------------------- #
# Per-corpus row -> text formatters. Each returns "" to SKIP a malformed row.
# --------------------------------------------------------------------------- #
def _first(d: dict, *names: str):
    for n in names:
        if n in d and d[n] not in (None, "", [], {}):
            return d[n]
    return None


def _fmt_pubmedqa(row: dict) -> str:
    q = _first(row, "question", "QUESTION")
    a = _first(row, "long_answer", "LONG_ANSWER", "final_decision", "answer")
    if not q or not a:
        return ""
    a = a if isinstance(a, str) else " ".join(map(str, a))
    return f"Question: {q.strip()}\nAnswer: {a.strip()}"


def _fmt_squad_qa(row: dict) -> str:
    """CUAD / SQuAD-style: question + context + answers{text:[...]}."""
    q = _first(row, "question")
    ctx = _first(row, "context")
    ans = _first(row, "answers", "answer")
    text = None
    if isinstance(ans, dict):
        t = ans.get("text")
        text = (t[0] if isinstance(t, list) and t else t) or None
    elif isinstance(ans, list) and ans:
        text = ans[0]
    elif isinstance(ans, str):
        text = ans
    if not q or not text:
        return ""
    head = f"Context: {str(ctx).strip()}\n\n" if ctx else ""
    return f"{head}Question: {str(q).strip()}\nAnswer: {str(text).strip()}"


def _fmt_raw(row: dict) -> str:
    c = _first(row, "content", "text", "code")
    return str(c).strip() if c else ""


_FORMATTERS = {
    "qiaojin/PubMedQA": _fmt_pubmedqa,
}


def _formatter(corpus: config.DomainCorpus):
    if corpus.fmt == "raw":
        return _fmt_raw
    # qa: pick the corpus-specific formatter, else a generic SQuAD-style one.
    return _FORMATTERS.get(corpus.repo, _fmt_squad_qa)


# --------------------------------------------------------------------------- #
def load_domain_texts(domain: str, n: int, seed: int = 0) -> list[str]:
    """Return up to `n` formatted training texts for `domain` (shuffled, seeded)."""
    from datasets import load_dataset

    corpus = config.DOMAIN_CORPORA[domain]
    kwargs: dict = {"split": corpus.split}
    if corpus.config:
        # the-stack-smol uses data_dir; PubMedQA uses a named config (subset).
        if "/" in corpus.config:
            kwargs["data_dir"] = corpus.config
        else:
            kwargs["name"] = corpus.config
    # Optional revision/branch. CUAD pins refs/convert/parquet so we load the clean
    # auto-parquet ('text' column) instead of the repo's raw PDFs (which would pull
    # in pdfplumber + parse 510 PDFs every time). No-op for corpora that leave it None.
    if corpus.revision:
        kwargs["revision"] = corpus.revision
    # verification_mode="no_checks": some HF repos (e.g. theatticusproject/cuad)
    # ship a stale dataset_infos.json whose recorded split size disagrees with the
    # auto-converted parquet, so the default STRICT check raises
    # NonMatchingSplitsSizesError (expected 84325 / got 511) and the corpus fails.
    # We do not rely on the recorded count for anything (we shuffle + take up to n),
    # so skipping the split-size/checksum verification is safe and loads the real data.
    ds = load_dataset(corpus.repo, verification_mode="no_checks", **kwargs)
    ds = ds.shuffle(seed=seed)

    fmt = _formatter(corpus)
    texts: list[str] = []
    n_seen = 0
    for row in ds:
        n_seen += 1
        t = fmt(row)
        if t:
            texts.append(t)
        if len(texts) >= n:
            break
        if n_seen > 20 * max(1, n):  # guard: corpus mostly empty -> fail loud
            break
    if not texts:
        raise RuntimeError(
            f"[domain_data] produced ZERO texts for domain={domain!r} "
            f"(repo={corpus.repo}, fmt={corpus.fmt}); the dataset schema likely "
            f"differs from the formatter's expected fields -- inspect one row."
        )
    return texts


def split_train_heldout(texts: list[str], heldout: int) -> tuple[list[str], list[str]]:
    """Last `heldout` rows are the in-domain PPL probe; the rest are training."""
    if len(texts) <= heldout:
        return texts, []
    return texts[:-heldout], texts[-heldout:]
